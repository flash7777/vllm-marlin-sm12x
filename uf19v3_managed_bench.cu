/*
 * UF19v3: cudaMallocManaged + ibverbs Benchmark
 *
 * Tests whether ibv_reg_mr() works on cudaMallocManaged() memory on GB10.
 * If yes, benchmarks AllReduce with GPU-direct access (no staging copies).
 *
 * Compares 4 memory strategies:
 *   A) cudaMallocManaged   — Unified Memory, GPU+CPU+NIC direct
 *   B) cudaHostAllocMapped — Pinned host, mapped to GPU (UF19 v1)
 *   C) posix_memalign      — Plain heap, GB10 unified makes it GPU-visible
 *   D) cudaMalloc           — Device memory (expected to fail ibv_reg_mr)
 *
 * Build:
 *   nvcc -O2 -arch=sm_121 -o uf19v3_bench uf19v3_managed_bench.cu -libverbs -lpthread
 *
 * Run (both nodes simultaneously):
 *   Node 0 (DGX):  ./uf19v3_bench 0 192.168.0.116 rocep1s0f0 3
 *   Node 1 (PGX):  ./uf19v3_bench 1 192.168.0.117 rocep1s0f0 3
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

#define NUMEL       2048
#define DATA_BYTES  (NUMEL * 2)
#define TCP_PORT    18700
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

// ============================================================
// CUDA Kernels
// ============================================================

// Copy input → send_buf + set flag (system fence for NIC visibility)
__global__ void prepare_send_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ send_buf,
    volatile int* send_flag,
    int step, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) send_buf[idx] = src[idx];
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        *send_flag = step;
    }
}

// Poll recv_flag via system-scope atomic, then add
__global__ void poll_and_add_kernel(
    const __nv_bfloat16* __restrict__ local_data,
    const __nv_bfloat16* __restrict__ recv_buf,
    __nv_bfloat16* __restrict__ output,
    int* recv_flag,
    int step, int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // system-scope atomic bypasses GPU L2 cache — sees NIC DMA writes
        while (atomicAdd_system(recv_flag, 0) < step) {
            __nanosleep(50);
        }
        __threadfence_system();
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = __bfloat162float(local_data[idx]);
        float b = __bfloat162float(recv_buf[idx]);
        output[idx] = __float2bfloat16(a + b);
    }
}

__global__ void add_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
}

// ============================================================
// ibverbs helpers (same as uf19_bench.cu)
// ============================================================

struct qp_info {
    uint32_t qpn, psn;
    uint32_t rkey_data, rkey_flag;
    uint64_t addr_data, addr_flag;
    union ibv_gid gid;
};

static struct ibv_context* open_ib_device(const char* dev_name) {
    int num_devices;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) { fprintf(stderr, "ibv_get_device_list failed\n"); exit(1); }
    struct ibv_device* dev = NULL;
    for (int i = 0; i < num_devices; i++)
        if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) { dev = dev_list[i]; break; }
    if (!dev) { fprintf(stderr, "IB device '%s' not found\n", dev_name); exit(1); }
    struct ibv_context* ctx = ibv_open_device(dev);
    ibv_free_device_list(dev_list);
    return ctx;
}

static void modify_qp_to_init(struct ibv_qp* qp) {
    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static void modify_qp_to_rtr(struct ibv_qp* qp, struct qp_info* remote, int gid_idx) {
    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote->qpn;
    attr.rq_psn = remote->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remote->gid;
    attr.ah_attr.grh.sgid_index = gid_idx;
    attr.ah_attr.grh.hop_limit = 64;
    ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static void modify_qp_to_rts(struct ibv_qp* qp, uint32_t psn) {
    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = psn;
    attr.max_rd_atomic = 1;
    ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
        IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

static void tcp_exchange(int rank, const char* peer_ip,
                         struct qp_info* local, struct qp_info* remote) {
    int sock;
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT);

    if (rank == 0) {
        int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        addr.sin_addr.s_addr = INADDR_ANY;
        bind(listen_sock, (struct sockaddr*)&addr, sizeof(addr));
        listen(listen_sock, 1);
        printf("  Rank 0: waiting on port %d...\n", TCP_PORT);
        sock = accept(listen_sock, NULL, NULL);
        close(listen_sock);
    } else {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        inet_pton(AF_INET, peer_ip, &addr.sin_addr);
        printf("  Rank 1: connecting to %s:%d...\n", peer_ip, TCP_PORT);
        while (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) usleep(100000);
    }
    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    if (rank == 0) {
        write(sock, local, sizeof(*local));
        read(sock, remote, sizeof(*remote));
    } else {
        read(sock, remote, sizeof(*remote));
        write(sock, local, sizeof(*local));
    }
    char c = 'R';
    write(sock, &c, 1);
    read(sock, &c, 1);
    close(sock);
    printf("  QP info exchanged. Remote QPN=0x%x\n", remote->qpn);
}

static void tcp_barrier(int rank, const char* peer_ip, int port_offset) {
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT + port_offset);

    if (rank == 0) {
        int ls = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        addr.sin_addr.s_addr = INADDR_ANY;
        bind(ls, (struct sockaddr*)&addr, sizeof(addr));
        listen(ls, 1);
        int c = accept(ls, NULL, NULL);
        char ch = 'G';
        write(c, &ch, 1);
        read(c, &ch, 1);
        close(c); close(ls);
    } else {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        inet_pton(AF_INET, peer_ip, &addr.sin_addr);
        while (connect(s, (struct sockaddr*)&addr, sizeof(addr)) < 0) usleep(50000);
        char ch;
        read(s, &ch, 1);
        write(s, &ch, 1);
        close(s);
    }
}

static double time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// ============================================================
// Phase 1: Test ibv_reg_mr on different memory types
// ============================================================

static void test_reg_mr(struct ibv_pd* pd) {
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    size_t sz = DATA_BYTES;

    printf("\n=== Phase 1: ibv_reg_mr compatibility test ===\n\n");

    // A) cudaMallocManaged
    {
        void* ptr = NULL;
        CHECK_CUDA(cudaMallocManaged(&ptr, sz));
        memset(ptr, 0, sz);  // touch pages (ensure allocation)
        struct ibv_mr* mr = ibv_reg_mr(pd, ptr, sz, mr_flags);
        if (mr) {
            printf("  [A] cudaMallocManaged:    OK  (mr=%p, rkey=0x%08x)\n", mr, mr->rkey);
            ibv_dereg_mr(mr);
        } else {
            printf("  [A] cudaMallocManaged:    FAILED (errno=%d: %s)\n", errno, strerror(errno));
        }
        cudaFree(ptr);
    }

    // B) cudaHostAllocMapped
    {
        void* ptr = NULL;
        CHECK_CUDA(cudaHostAlloc(&ptr, sz, cudaHostAllocMapped | cudaHostAllocPortable));
        struct ibv_mr* mr = ibv_reg_mr(pd, ptr, sz, mr_flags);
        if (mr) {
            printf("  [B] cudaHostAllocMapped:  OK  (mr=%p, rkey=0x%08x)\n", mr, mr->rkey);
            ibv_dereg_mr(mr);
        } else {
            printf("  [B] cudaHostAllocMapped:  FAILED (errno=%d: %s)\n", errno, strerror(errno));
        }
        cudaFreeHost(ptr);
    }

    // C) posix_memalign
    {
        void* ptr = NULL;
        posix_memalign(&ptr, 4096, sz);
        memset(ptr, 0, sz);
        struct ibv_mr* mr = ibv_reg_mr(pd, ptr, sz, mr_flags);
        if (mr) {
            printf("  [C] posix_memalign:       OK  (mr=%p, rkey=0x%08x)\n", mr, mr->rkey);
            ibv_dereg_mr(mr);
        } else {
            printf("  [C] posix_memalign:       FAILED (errno=%d: %s)\n", errno, strerror(errno));
        }
        free(ptr);
    }

    // D) cudaMalloc (device memory — expected to fail without nvidia-peermem)
    {
        void* ptr = NULL;
        CHECK_CUDA(cudaMalloc(&ptr, sz));
        struct ibv_mr* mr = ibv_reg_mr(pd, ptr, sz, mr_flags);
        if (mr) {
            printf("  [D] cudaMalloc (device):  OK  (mr=%p, rkey=0x%08x) — GPUDirect works!\n", mr, mr->rkey);
            ibv_dereg_mr(mr);
        } else {
            printf("  [D] cudaMalloc (device):  FAILED (errno=%d: %s) — expected, no GPU BAR\n", errno, strerror(errno));
        }
        cudaFree(ptr);
    }

    printf("\n");
}

// ============================================================
// Phase 2: Full AllReduce benchmark with cudaMallocManaged
// ============================================================

static void bench_allreduce(int rank, const char* peer_ip, const char* ib_dev,
                            int gid_idx, struct ibv_context* ib_ctx, struct ibv_pd* pd)
{
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    printf("=== Phase 2: Full AllReduce benchmark (cudaMallocManaged) ===\n\n");

    // Allocate ALL buffers with cudaMallocManaged
    __nv_bfloat16 *send_buf, *recv_buf;
    int *send_flag, *recv_flag, *flag_src;
    __nv_bfloat16 *d_partial, *d_output;

    CHECK_CUDA(cudaMallocManaged(&send_buf, DATA_BYTES));
    CHECK_CUDA(cudaMallocManaged(&recv_buf, DATA_BYTES));
    CHECK_CUDA(cudaMallocManaged(&send_flag, sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&recv_flag, sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&flag_src, sizeof(int)));

    // Input/output remain device memory (simulates GEMM output)
    CHECK_CUDA(cudaMalloc(&d_partial, DATA_BYTES));
    CHECK_CUDA(cudaMalloc(&d_output, DATA_BYTES));

    memset(send_buf, 0, DATA_BYTES);
    memset(recv_buf, 0, DATA_BYTES);
    *send_flag = 0;
    *recv_flag = 0;
    *flag_src = 0;

    // Initialize input
    {
        __nv_bfloat16* h_init = (__nv_bfloat16*)malloc(DATA_BYTES);
        for (int i = 0; i < NUMEL; i++)
            h_init[i] = __float2bfloat16((float)(rank + 1) * 0.1f + i * 0.001f);
        CHECK_CUDA(cudaMemcpy(d_partial, h_init, DATA_BYTES, cudaMemcpyHostToDevice));
        free(h_init);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Register managed memory with ibverbs
    struct ibv_mr* mr_send = ibv_reg_mr(pd, send_buf, DATA_BYTES, mr_flags);
    struct ibv_mr* mr_recv = ibv_reg_mr(pd, recv_buf, DATA_BYTES, mr_flags);
    struct ibv_mr* mr_flag = ibv_reg_mr(pd, recv_flag, sizeof(int), mr_flags);
    struct ibv_mr* mr_flag_src = ibv_reg_mr(pd, flag_src, sizeof(int), mr_flags);

    if (!mr_send || !mr_recv || !mr_flag || !mr_flag_src) {
        printf("  ibv_reg_mr on managed memory FAILED — cannot run benchmark\n");
        return;
    }
    printf("  All ibv_reg_mr on cudaMallocManaged succeeded\n");

    // Create QP
    struct ibv_cq* cq = ibv_create_cq(ib_ctx, 256, NULL, NULL, 0);
    struct ibv_qp_init_attr qp_init = {};
    qp_init.send_cq = cq;
    qp_init.recv_cq = cq;
    qp_init.qp_type = IBV_QPT_RC;
    qp_init.cap.max_send_wr = 256;
    qp_init.cap.max_recv_wr = 4;
    qp_init.cap.max_send_sge = 1;
    qp_init.cap.max_recv_sge = 1;
    qp_init.cap.max_inline_data = 64;
    struct ibv_qp* qp = ibv_create_qp(pd, &qp_init);

    // Connect QP
    modify_qp_to_init(qp);
    uint32_t psn = (rank * 12345 + 7) & 0xFFFFFF;
    union ibv_gid local_gid;
    ibv_query_gid(ib_ctx, 1, gid_idx, &local_gid);

    struct qp_info local_info = {}, remote_info = {};
    local_info.qpn = qp->qp_num;
    local_info.psn = psn;
    local_info.rkey_data = mr_recv->rkey;
    local_info.rkey_flag = mr_flag->rkey;
    local_info.addr_data = (uint64_t)recv_buf;   // managed ptr = valid for RDMA
    local_info.addr_flag = (uint64_t)recv_flag;
    local_info.gid = local_gid;

    tcp_exchange(rank, peer_ip, &local_info, &remote_info);
    modify_qp_to_rtr(qp, &remote_info, gid_idx);
    modify_qp_to_rts(qp, psn);
    printf("  QP connected (RTS)\n\n");

    // Pre-build WRs
    struct ibv_sge sge_data = {};
    sge_data.addr = (uint64_t)send_buf;
    sge_data.length = DATA_BYTES;
    sge_data.lkey = mr_send->lkey;

    struct ibv_send_wr wr_data = {}, wr_flag_wr = {}, *bad_wr;
    wr_data.wr_id = 1;
    wr_data.sg_list = &sge_data;
    wr_data.num_sge = 1;
    wr_data.opcode = IBV_WR_RDMA_WRITE;
    wr_data.wr.rdma.remote_addr = remote_info.addr_data;
    wr_data.wr.rdma.rkey = remote_info.rkey_data;

    struct ibv_sge sge_flag = {};
    sge_flag.addr = (uint64_t)flag_src;
    sge_flag.length = sizeof(int);
    sge_flag.lkey = mr_flag_src->lkey;

    wr_flag_wr.wr_id = 2;
    wr_flag_wr.sg_list = &sge_flag;
    wr_flag_wr.num_sge = 1;
    wr_flag_wr.opcode = IBV_WR_RDMA_WRITE;
    wr_flag_wr.send_flags = IBV_SEND_SIGNALED;
    wr_flag_wr.wr.rdma.remote_addr = remote_info.addr_flag;
    wr_flag_wr.wr.rdma.rkey = remote_info.rkey_flag;

    wr_data.next = &wr_flag_wr;
    wr_flag_wr.next = NULL;

    dim3 block(256);
    dim3 grid((NUMEL + 255) / 256);

    // --- GPU add baseline ---
    {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
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

    // --- GPU write to managed memory ---
    {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        for (int i = 0; i < WARMUP; i++)
            prepare_send_kernel<<<grid, block>>>(d_partial, send_buf, send_flag, i+1, NUMEL);
        CHECK_CUDA(cudaDeviceSynchronize());
        *send_flag = 0;
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < ITERS; i++)
            prepare_send_kernel<<<grid, block>>>(d_partial, send_buf, send_flag, i+1, NUMEL);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        printf("  GPU→managed write + fence:   %7.2f us/call\n", ms / ITERS * 1000);
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // --- Barrier before full benchmark ---
    tcp_barrier(rank, peer_ip, 1);

    // --- Full AllReduce ---
    {
        *send_flag = 0;
        *recv_flag = 0;
        CHECK_CUDA(cudaDeviceSynchronize());

        // Warmup
        for (int step = 1; step <= WARMUP; step++) {
            prepare_send_kernel<<<grid, block>>>(d_partial, send_buf, send_flag, step, NUMEL);
            while (*(volatile int*)send_flag < step) {}
            *flag_src = step;
            ibv_post_send(qp, &wr_data, &bad_wr);
            poll_and_add_kernel<<<grid, block>>>(d_partial, recv_buf, d_output,
                                                  recv_flag, step, NUMEL);
            CHECK_CUDA(cudaDeviceSynchronize());
            struct ibv_wc wc;
            int ne;
            do { ne = ibv_poll_cq(cq, 1, &wc); } while (ne == 0);
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "  Warmup WC error: %s (step=%d)\n",
                        ibv_wc_status_str(wc.status), step);
                return;
            }
        }
        printf("\n  Warmup done (%d iters). Starting timed run...\n\n", WARMUP);

        // Timed run
        int base = WARMUP;
        double t_total = 0, t_gpu_prep = 0, t_rdma_post = 0, t_gpu_wait = 0;

        for (int i = 0; i < ITERS; i++) {
            int step = base + i + 1;
            double t0 = time_us();

            prepare_send_kernel<<<grid, block>>>(d_partial, send_buf, send_flag, step, NUMEL);
            while (*(volatile int*)send_flag < step) {}
            double t1 = time_us();

            *flag_src = step;
            ibv_post_send(qp, &wr_data, &bad_wr);
            double t2 = time_us();

            poll_and_add_kernel<<<grid, block>>>(d_partial, recv_buf, d_output,
                                                  recv_flag, step, NUMEL);
            CHECK_CUDA(cudaDeviceSynchronize());
            double t3 = time_us();

            struct ibv_wc wc;
            int ne;
            do { ne = ibv_poll_cq(cq, 1, &wc); } while (ne == 0);

            t_gpu_prep  += (t1 - t0);
            t_rdma_post += (t2 - t1);
            t_gpu_wait  += (t3 - t2);
            t_total     += (t3 - t0);
        }

        printf("  === UF19v3 AllReduce: cudaMallocManaged (%d iters) ===\n", ITERS);
        printf("  GPU prepare + CPU poll:      %7.2f us/call\n", t_gpu_prep / ITERS);
        printf("  CPU RDMA post:               %7.2f us/call\n", t_rdma_post / ITERS);
        printf("  GPU poll+add (incl. wait):   %7.2f us/call\n", t_gpu_wait / ITERS);
        printf("  ─────────────────────────────────────────\n");
        printf("  TOTAL:                       %7.2f us/call\n", t_total / ITERS);
        printf("\n");
        printf("  Reference:\n");
        printf("    UF19 v1 (cudaHostAllocMapped):  12.1 us/call\n");
        printf("    NCCL AllReduce:                  19.4 us/call\n");
        printf("    Raw RDMA Write (ib_write_lat):    3.2 us/call\n");
        printf("\n");

        double savings = 19.4 - (t_total / ITERS);
        double savings_per_token = savings * 97;
        double base_ms = 1000.0 / 118.0;
        double new_ms = base_ms - savings_per_token / 1000.0;
        printf("  === Projected Impact (97 calls/token) ===\n");
        printf("  NCCL:    97 × 19.4 = %.0f us = %.2f ms\n", 97*19.4, 97*19.4/1000);
        printf("  UF19v3:  97 × %.1f = %.0f us = %.2f ms\n",
               t_total/ITERS, 97*t_total/ITERS, 97*t_total/ITERS/1000);
        if (savings > 0) {
            printf("  Savings: %.0f us/token = %.2f ms/token\n", savings_per_token, savings_per_token/1000);
            printf("  118 tok/s → %.1f tok/s (+%.1f%%)\n", 1000/new_ms, (1000/new_ms/118-1)*100);
        } else {
            printf("  No improvement vs NCCL\n");
        }
    }

    // Verify
    {
        __nv_bfloat16* h = (__nv_bfloat16*)malloc(DATA_BYTES);
        CHECK_CUDA(cudaMemcpy(h, d_output, DATA_BYTES, cudaMemcpyDeviceToHost));
        printf("\n  Verify: out[0]=%.4f, out[1]=%.4f\n",
               __bfloat162float(h[0]), __bfloat162float(h[1]));
        free(h);
    }

    // Cleanup
    ibv_dereg_mr(mr_send);
    ibv_dereg_mr(mr_recv);
    ibv_dereg_mr(mr_flag);
    ibv_dereg_mr(mr_flag_src);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    cudaFree(send_buf);
    cudaFree(recv_buf);
    cudaFree(send_flag);
    cudaFree(recv_flag);
    cudaFree(flag_src);
    cudaFree(d_partial);
    cudaFree(d_output);
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

    printf("\n=== UF19v3: cudaMallocManaged + ibverbs Benchmark ===\n");
    printf("  Rank: %d, Peer: %s, Dev: %s, GID: %d\n", rank, peer_ip, ib_dev, gid_idx);
    printf("  Data: %d BF16 elements = %d bytes\n\n", NUMEL, DATA_BYTES);

    CHECK_CUDA(cudaSetDevice(0));

    // Open IB device + PD (shared for all tests)
    struct ibv_context* ib_ctx = open_ib_device(ib_dev);
    struct ibv_pd* pd = ibv_alloc_pd(ib_ctx);

    // Phase 1: Test which memory types work with ibv_reg_mr
    test_reg_mr(pd);

    // Phase 2: Full benchmark (only if managed memory works)
    bench_allreduce(rank, peer_ip, ib_dev, gid_idx, ib_ctx, pd);

    ibv_dealloc_pd(pd);
    ibv_close_device(ib_ctx);

    printf("\n  Done.\n\n");
    return 0;
}
