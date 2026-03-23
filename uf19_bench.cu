/*
 * UF19 Custom 2-Rank AllReduce Benchmark
 *
 * Bypasses NCCL's Ring protocol for TP=2 with direct ibverbs RDMA writes.
 * Uses mapped pinned memory so GPU and NIC share the same buffers.
 *
 * Architecture:
 *   GPU writes partial_sum → mapped send_buf → CPU posts RDMA Write →
 *   NIC DMAs to remote recv_buf → remote GPU polls flag → adds local + received
 *
 * Build:
 *   nvcc -O2 -arch=sm_121 -o uf19_bench uf19_bench.cu -libverbs -lpthread
 *
 * Run (bidirectional, on both nodes simultaneously):
 *   Node 0 (DGX):  ./uf19_bench 0 192.168.0.116 rocep1s0f0 3
 *   Node 1 (PGX):  ./uf19_bench 1 192.168.0.117 rocep1s0f0 3
 *
 * Args: <rank> <peer_ip> <ib_dev> <gid_index>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define NUMEL       2048        // BF16 elements (= 4 KiB, Qwen3-Coder hidden_size)
#define DATA_BYTES  (NUMEL * 2) // 4096 bytes
#define TCP_PORT    18600
#define WARMUP      500
#define ITERS       5000

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_NULL(ptr, msg) do { \
    if (!(ptr)) { fprintf(stderr, "Error: %s (errno=%d)\n", msg, errno); exit(1); } \
} while(0)

// --- QP connection info exchanged via TCP ---
struct qp_info {
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey_data;
    uint32_t rkey_flag;
    uint64_t addr_data;     // remote recv_buf address
    uint64_t addr_flag;     // remote recv_flag address
    union ibv_gid gid;
};

// --- CUDA Kernels ---

// Kernel 1: Copy device partial_sum → mapped send_buf, then set send_flag
__global__ void prepare_send_kernel(
    const __nv_bfloat16* __restrict__ src,   // device memory
    __nv_bfloat16* __restrict__ send_buf,     // mapped pinned memory
    volatile int* send_flag,                  // mapped pinned memory
    int step, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        send_buf[idx] = src[idx];
    }
    // Last thread signals readiness to CPU
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();  // ensure all writes visible to CPU/NIC
        *send_flag = step;
    }
}

// Kernel 2: Poll recv_flag, then add local + received → output
__global__ void poll_and_add_kernel(
    const __nv_bfloat16* __restrict__ local_data,  // device memory (my partial sum)
    const __nv_bfloat16* __restrict__ recv_buf,     // mapped pinned (peer's data)
    __nv_bfloat16* __restrict__ output,             // device memory (result)
    volatile int* recv_flag,                         // mapped pinned
    int step, int n)
{
    // Thread 0 polls until data arrives
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (*recv_flag < step) {
            // spin — NIC writes this via RDMA
        }
        __threadfence_system();  // ensure we see the data written before the flag
    }
    __syncthreads();

    // All threads add
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = __bfloat162float(local_data[idx]);
        float b = __bfloat162float(recv_buf[idx]);
        output[idx] = __float2bfloat16(a + b);
    }
}

// Kernel 3: Just the add (for measuring add-only latency)
__global__ void add_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
    }
}

// --- ibverbs helpers ---

static struct ibv_context* open_ib_device(const char* dev_name) {
    int num_devices;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    CHECK_NULL(dev_list, "ibv_get_device_list");

    struct ibv_device* dev = NULL;
    for (int i = 0; i < num_devices; i++) {
        if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) {
            dev = dev_list[i];
            break;
        }
    }
    CHECK_NULL(dev, "IB device not found");

    struct ibv_context* ctx = ibv_open_device(dev);
    CHECK_NULL(ctx, "ibv_open_device");
    ibv_free_device_list(dev_list);
    return ctx;
}

static union ibv_gid get_gid(struct ibv_context* ctx, int port, int gid_idx) {
    union ibv_gid gid;
    if (ibv_query_gid(ctx, port, gid_idx, &gid)) {
        fprintf(stderr, "ibv_query_gid failed\n");
        exit(1);
    }
    return gid;
}

// --- TCP exchange ---

static void tcp_exchange(int rank, const char* peer_ip,
                         struct qp_info* local, struct qp_info* remote) {
    int sock;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT);

    if (rank == 0) {
        // Server: listen and accept
        int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(listen_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind"); exit(1);
        }
        listen(listen_sock, 1);
        printf("  Rank 0: waiting for connection on port %d...\n", TCP_PORT);
        sock = accept(listen_sock, NULL, NULL);
        close(listen_sock);
    } else {
        // Client: connect
        sock = socket(AF_INET, SOCK_STREAM, 0);
        inet_pton(AF_INET, peer_ip, &addr.sin_addr);
        printf("  Rank 1: connecting to %s:%d...\n", peer_ip, TCP_PORT);
        while (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            usleep(100000);  // retry every 100ms
        }
    }

    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // Exchange: send local, recv remote
    if (rank == 0) {
        write(sock, local, sizeof(*local));
        read(sock, remote, sizeof(*remote));
    } else {
        read(sock, remote, sizeof(*remote));
        write(sock, local, sizeof(*local));
    }

    // Barrier: both sides ready
    char c = 'R';
    write(sock, &c, 1);
    read(sock, &c, 1);

    close(sock);
    printf("  QP info exchanged. Remote QPN=0x%x\n", remote->qpn);
}

static void modify_qp_to_init(struct ibv_qp* qp) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        fprintf(stderr, "modify_qp to INIT failed\n"); exit(1);
    }
}

static void modify_qp_to_rtr(struct ibv_qp* qp, struct qp_info* remote, int gid_idx) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote->qpn;
    attr.rq_psn = remote->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;

    attr.ah_attr.dlid = 0;  // RoCE
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remote->gid;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.sgid_index = gid_idx;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = 0;

    if (ibv_modify_qp(qp, &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        fprintf(stderr, "modify_qp to RTR failed\n"); exit(1);
    }
}

static void modify_qp_to_rts(struct ibv_qp* qp, uint32_t psn) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = psn;
    attr.max_rd_atomic = 1;

    if (ibv_modify_qp(qp, &attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        fprintf(stderr, "modify_qp to RTS failed\n"); exit(1);
    }
}

// --- Timing helpers ---

static double time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <rank 0|1> <peer_ip> <ib_dev> <gid_index>\n", argv[0]);
        return 1;
    }

    int rank = atoi(argv[1]);
    const char* peer_ip = argv[2];
    const char* ib_dev = argv[3];
    int gid_idx = atoi(argv[4]);

    printf("\n=== UF19 Custom AllReduce Benchmark ===\n");
    printf("  Rank: %d, Peer: %s, Dev: %s, GID: %d\n", rank, peer_ip, ib_dev, gid_idx);
    printf("  Data: %d BF16 elements = %d bytes\n", NUMEL, DATA_BYTES);

    // --- CUDA setup ---
    CHECK_CUDA(cudaSetDevice(0));

    // Device memory (simulates GEMM output / AllReduce result)
    __nv_bfloat16 *d_partial, *d_output;
    CHECK_CUDA(cudaMalloc(&d_partial, DATA_BYTES));
    CHECK_CUDA(cudaMalloc(&d_output, DATA_BYTES));

    // Initialize with rank-dependent values
    {
        __nv_bfloat16* h_init = (__nv_bfloat16*)malloc(DATA_BYTES);
        for (int i = 0; i < NUMEL; i++)
            h_init[i] = __float2bfloat16((float)(rank + 1) * 0.1f + i * 0.001f);
        CHECK_CUDA(cudaMemcpy(d_partial, h_init, DATA_BYTES, cudaMemcpyHostToDevice));
        free(h_init);
    }

    // Mapped pinned memory: accessible by GPU (device pointer) AND CPU/NIC (host pointer)
    __nv_bfloat16 *h_send_buf, *h_recv_buf;
    int *h_send_flag, *h_recv_flag;

    CHECK_CUDA(cudaHostAlloc(&h_send_buf, DATA_BYTES,
                             cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc(&h_recv_buf, DATA_BYTES,
                             cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc(&h_send_flag, sizeof(int),
                             cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc(&h_recv_flag, sizeof(int),
                             cudaHostAllocMapped | cudaHostAllocPortable));

    // Get device pointers for mapped memory
    __nv_bfloat16 *d_send_buf, *d_recv_buf;
    int *d_send_flag, *d_recv_flag;
    CHECK_CUDA(cudaHostGetDevicePointer(&d_send_buf, h_send_buf, 0));
    CHECK_CUDA(cudaHostGetDevicePointer(&d_recv_buf, h_recv_buf, 0));
    CHECK_CUDA(cudaHostGetDevicePointer(&d_send_flag, h_send_flag, 0));
    CHECK_CUDA(cudaHostGetDevicePointer(&d_recv_flag, h_recv_flag, 0));

    memset(h_send_buf, 0, DATA_BYTES);
    memset(h_recv_buf, 0, DATA_BYTES);
    *h_send_flag = 0;
    *h_recv_flag = 0;
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- ibverbs setup ---
    struct ibv_context* ctx = open_ib_device(ib_dev);
    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    CHECK_NULL(pd, "ibv_alloc_pd");

    struct ibv_cq* cq = ibv_create_cq(ctx, 128, NULL, NULL, 0);
    CHECK_NULL(cq, "ibv_create_cq");

    struct ibv_qp_init_attr qp_init;
    memset(&qp_init, 0, sizeof(qp_init));
    qp_init.send_cq = cq;
    qp_init.recv_cq = cq;
    qp_init.qp_type = IBV_QPT_RC;
    qp_init.cap.max_send_wr = 128;
    qp_init.cap.max_recv_wr = 4;
    qp_init.cap.max_send_sge = 1;
    qp_init.cap.max_recv_sge = 1;
    qp_init.cap.max_inline_data = 64;
    qp_init.sq_sig_all = 0;  // only signal when requested

    struct ibv_qp* qp = ibv_create_qp(pd, &qp_init);
    CHECK_NULL(qp, "ibv_create_qp");

    // Register memory with ibverbs for RDMA
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    struct ibv_mr* mr_send = ibv_reg_mr(pd, h_send_buf, DATA_BYTES, mr_flags);
    CHECK_NULL(mr_send, "ibv_reg_mr send_buf");
    struct ibv_mr* mr_recv = ibv_reg_mr(pd, h_recv_buf, DATA_BYTES, mr_flags);
    CHECK_NULL(mr_recv, "ibv_reg_mr recv_buf");
    struct ibv_mr* mr_flag = ibv_reg_mr(pd, h_recv_flag, sizeof(int), mr_flags);
    CHECK_NULL(mr_flag, "ibv_reg_mr recv_flag");

    // Small buffer for flag source (holds the step number to write remotely)
    int *h_flag_src;
    CHECK_CUDA(cudaHostAlloc(&h_flag_src, sizeof(int), cudaHostAllocMapped));
    struct ibv_mr* mr_flag_src = ibv_reg_mr(pd, h_flag_src, sizeof(int), mr_flags);
    CHECK_NULL(mr_flag_src, "ibv_reg_mr flag_src");

    // --- Connect QPs ---
    modify_qp_to_init(qp);

    uint32_t psn = rand() & 0xFFFFFF;
    union ibv_gid local_gid = get_gid(ctx, 1, gid_idx);

    struct qp_info local_info, remote_info;
    local_info.qpn = qp->qp_num;
    local_info.psn = psn;
    local_info.rkey_data = mr_recv->rkey;    // peer writes to MY recv_buf
    local_info.rkey_flag = mr_flag->rkey;    // peer writes to MY recv_flag
    local_info.addr_data = (uint64_t)h_recv_buf;
    local_info.addr_flag = (uint64_t)h_recv_flag;
    local_info.gid = local_gid;

    tcp_exchange(rank, peer_ip, &local_info, &remote_info);

    modify_qp_to_rtr(qp, &remote_info, gid_idx);
    modify_qp_to_rts(qp, psn);
    printf("  QP connected (RTS). Ready for RDMA.\n\n");

    // --- Pre-build RDMA Work Requests ---
    // WR1: RDMA Write data (send_buf → remote recv_buf)
    struct ibv_sge sge_data;
    sge_data.addr = (uint64_t)h_send_buf;
    sge_data.length = DATA_BYTES;
    sge_data.lkey = mr_send->lkey;

    struct ibv_send_wr wr_data, *bad_wr;
    memset(&wr_data, 0, sizeof(wr_data));
    wr_data.wr_id = 1;
    wr_data.sg_list = &sge_data;
    wr_data.num_sge = 1;
    wr_data.opcode = IBV_WR_RDMA_WRITE;
    wr_data.send_flags = 0;  // no signal, no inline
    wr_data.wr.rdma.remote_addr = remote_info.addr_data;
    wr_data.wr.rdma.rkey = remote_info.rkey_data;

    // WR2: RDMA Write flag (flag_src → remote recv_flag), signaled
    struct ibv_sge sge_flag;
    sge_flag.addr = (uint64_t)h_flag_src;
    sge_flag.length = sizeof(int);
    sge_flag.lkey = mr_flag_src->lkey;

    struct ibv_send_wr wr_flag;
    memset(&wr_flag, 0, sizeof(wr_flag));
    wr_flag.wr_id = 2;
    wr_flag.sg_list = &sge_flag;
    wr_flag.num_sge = 1;
    wr_flag.opcode = IBV_WR_RDMA_WRITE;
    wr_flag.send_flags = IBV_SEND_SIGNALED;  // signal completion
    wr_flag.wr.rdma.remote_addr = remote_info.addr_flag;
    wr_flag.wr.rdma.rkey = remote_info.rkey_flag;

    // Chain: data first, then flag (IB guarantees ordering on same QP)
    wr_data.next = &wr_flag;
    wr_flag.next = NULL;

    dim3 block(256);
    dim3 grid((NUMEL + 255) / 256);

    // ============================================================
    // Benchmark 1: GPU add-only latency (baseline)
    // ============================================================
    {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // Warmup
        for (int i = 0; i < WARMUP; i++)
            add_kernel<<<grid, block>>>(d_partial, d_partial, d_output, NUMEL);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            add_kernel<<<grid, block>>>(d_partial, d_partial, d_output, NUMEL);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  GPU add kernel:              %7.2f us/call\n", ms / ITERS * 1000);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // ============================================================
    // Benchmark 2: GPU write to mapped memory latency
    // ============================================================
    {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            prepare_send_kernel<<<grid, block>>>(d_partial, d_send_buf, d_send_flag, i+1, NUMEL);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        *h_send_flag = 0;  // reset

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++) {
            prepare_send_kernel<<<grid, block>>>(d_partial, d_send_buf, d_send_flag, i+1, NUMEL);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  GPU→mapped write + fence:    %7.2f us/call\n", ms / ITERS * 1000);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // ============================================================
    // Benchmark 3: CPU ibv_post_send overhead (no actual RDMA, just post timing)
    // ============================================================
    {
        // First, fill send_buf so posts are valid
        *h_send_flag = WARMUP + ITERS;  // from benchmark 2
        *h_flag_src = 1;

        double t0 = time_us();
        for (int i = 0; i < ITERS; i++) {
            *h_flag_src = i + 1;
            int ret = ibv_post_send(qp, &wr_data, &bad_wr);
            if (ret) { fprintf(stderr, "ibv_post_send failed: %d\n", ret); exit(1); }

            // Must poll CQ to avoid SQ overflow
            struct ibv_wc wc;
            int ne;
            do { ne = ibv_poll_cq(cq, 1, &wc); } while (ne == 0);
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "WC error: %s\n", ibv_wc_status_str(wc.status));
                exit(1);
            }
        }
        double t1 = time_us();
        printf("  RDMA post+poll (data+flag):  %7.2f us/call  (CPU wall-clock)\n",
               (t1 - t0) / ITERS);
    }

    // Small barrier via TCP before full benchmark
    {
        int barrier_sock;
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(TCP_PORT + 1);

        if (rank == 0) {
            barrier_sock = socket(AF_INET, SOCK_STREAM, 0);
            int opt = 1;
            setsockopt(barrier_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            addr.sin_addr.s_addr = INADDR_ANY;
            bind(barrier_sock, (struct sockaddr*)&addr, sizeof(addr));
            listen(barrier_sock, 1);
            int c = accept(barrier_sock, NULL, NULL);
            char ch = 'G';
            write(c, &ch, 1);
            read(c, &ch, 1);
            close(c);
            close(barrier_sock);
        } else {
            barrier_sock = socket(AF_INET, SOCK_STREAM, 0);
            inet_pton(AF_INET, peer_ip, &addr.sin_addr);
            while (connect(barrier_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0)
                usleep(50000);
            char ch;
            read(barrier_sock, &ch, 1);
            write(barrier_sock, &ch, 1);
            close(barrier_sock);
        }
    }

    // ============================================================
    // Benchmark 4: Full custom AllReduce pipeline
    //   GPU prepare → CPU poll → RDMA post → GPU poll+add
    // ============================================================
    {
        *h_send_flag = 0;
        *h_recv_flag = 0;
        CHECK_CUDA(cudaDeviceSynchronize());

        // Warmup
        for (int step = 1; step <= WARMUP; step++) {
            // GPU: copy to mapped send_buf + set flag
            prepare_send_kernel<<<grid, block>>>(d_partial, d_send_buf, d_send_flag, step, NUMEL);

            // CPU: wait for GPU to finish writing
            while (*(volatile int*)h_send_flag < step) { /* spin */ }

            // CPU: post RDMA writes (data + flag)
            *h_flag_src = step;
            int ret = ibv_post_send(qp, &wr_data, &bad_wr);
            if (ret) { fprintf(stderr, "RDMA post failed: %d\n", ret); exit(1); }

            // GPU: poll recv_flag + add
            poll_and_add_kernel<<<grid, block>>>(d_partial, d_recv_buf, d_output,
                                                  d_recv_flag, step, NUMEL);

            // Wait for everything
            CHECK_CUDA(cudaDeviceSynchronize());

            // Poll CQ
            struct ibv_wc wc;
            int ne;
            do { ne = ibv_poll_cq(cq, 1, &wc); } while (ne == 0);
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "Warmup WC error: %s (step=%d)\n",
                        ibv_wc_status_str(wc.status), step);
                exit(1);
            }
        }

        printf("\n  Warmup done (%d iters). Starting timed benchmark...\n\n", WARMUP);

        // Timed run
        int base_step = WARMUP;
        double t_total = 0;
        double t_gpu_prep = 0, t_cpu_poll = 0, t_rdma_post = 0, t_gpu_wait = 0;

        for (int i = 0; i < ITERS; i++) {
            int step = base_step + i + 1;

            double t0 = time_us();

            // 1. GPU: prepare send
            prepare_send_kernel<<<grid, block>>>(d_partial, d_send_buf, d_send_flag, step, NUMEL);

            // 2. CPU: poll send_flag
            while (*(volatile int*)h_send_flag < step) { /* spin */ }
            double t1 = time_us();

            // 3. CPU: post RDMA (data + flag, chained)
            *h_flag_src = step;
            ibv_post_send(qp, &wr_data, &bad_wr);
            double t2 = time_us();

            // 4. GPU: poll recv_flag + add (launched now, executes async)
            poll_and_add_kernel<<<grid, block>>>(d_partial, d_recv_buf, d_output,
                                                  d_recv_flag, step, NUMEL);

            // 5. Wait for GPU kernel to complete (includes RDMA wait time)
            CHECK_CUDA(cudaDeviceSynchronize());
            double t3 = time_us();

            // 6. Poll CQ (RDMA completion)
            struct ibv_wc wc;
            int ne;
            do { ne = ibv_poll_cq(cq, 1, &wc); } while (ne == 0);

            t_gpu_prep  += (t1 - t0);
            t_rdma_post += (t2 - t1);
            t_gpu_wait  += (t3 - t2);
            t_total     += (t3 - t0);
        }

        printf("  === Full Custom AllReduce (%d iters) ===\n", ITERS);
        printf("  GPU prepare + CPU poll:      %7.2f us/call\n", t_gpu_prep / ITERS);
        printf("  CPU RDMA post:               %7.2f us/call\n", t_rdma_post / ITERS);
        printf("  GPU poll+add (incl. wait):   %7.2f us/call\n", t_gpu_wait / ITERS);
        printf("  ─────────────────────────────────────────\n");
        printf("  TOTAL:                       %7.2f us/call\n", t_total / ITERS);
        printf("\n");
        printf("  vs NCCL AllReduce:            19.0 us/call\n");
        printf("  vs Raw RDMA Write (CPU):       3.2 us/call\n");
        printf("\n");

        double savings_per_call = 19.0 - (t_total / ITERS);
        double savings_per_token = savings_per_call * 97;  // 97 AllReduces/token
        double base_ms = 1000.0 / 116.3;  // current tok/s
        double new_ms = base_ms - savings_per_token / 1000.0;
        double new_tok_s = 1000.0 / new_ms;

        printf("  === Projected Impact (97 AllReduces/Token) ===\n");
        printf("  NCCL total:     97 × 19.0 = %5.0f us = %.2f ms\n", 97*19.0, 97*19.0/1000);
        printf("  Custom total:   97 × %.1f = %5.0f us = %.2f ms\n",
               t_total/ITERS, 97*t_total/ITERS, 97*t_total/ITERS/1000);
        printf("  Savings:        %.0f us/token = %.2f ms/token\n",
               savings_per_token, savings_per_token / 1000);
        if (new_ms > 0 && savings_per_call > 0) {
            printf("  At 116.3 tok/s → %.1f tok/s (+%.1f%%)\n",
                   new_tok_s, (new_tok_s / 116.3 - 1) * 100);
        } else {
            printf("  No improvement (custom is slower than NCCL)\n");
        }
    }

    // Verify correctness
    {
        __nv_bfloat16* h_out = (__nv_bfloat16*)malloc(DATA_BYTES);
        CHECK_CUDA(cudaMemcpy(h_out, d_output, DATA_BYTES, cudaMemcpyDeviceToHost));

        printf("\n  Verify: output[0]=%.4f, output[1]=%.4f (expect sum of both ranks)\n",
               __bfloat162float(h_out[0]), __bfloat162float(h_out[1]));
        free(h_out);
    }

    // --- Cleanup ---
    ibv_dereg_mr(mr_send);
    ibv_dereg_mr(mr_recv);
    ibv_dereg_mr(mr_flag);
    ibv_dereg_mr(mr_flag_src);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);

    CHECK_CUDA(cudaFreeHost(h_send_buf));
    CHECK_CUDA(cudaFreeHost(h_recv_buf));
    CHECK_CUDA(cudaFreeHost(h_send_flag));
    CHECK_CUDA(cudaFreeHost(h_recv_flag));
    CHECK_CUDA(cudaFreeHost(h_flag_src));
    CHECK_CUDA(cudaFree(d_partial));
    CHECK_CUDA(cudaFree(d_output));

    printf("\n  Done.\n\n");
    return 0;
}
