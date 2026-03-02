/*
 * UF19: Custom 2-Rank AllReduce via raw ibverbs RDMA
 *
 * Shared library for vLLM integration. Bypasses NCCL for TP=2 AllReduce
 * on RoCE networks. Reduces per-call latency from 19µs (NCCL) to ~12µs
 * (mapped memory) or ~6-7µs (GPUDirect RDMA with nvidia-peermem).
 *
 * Exported functions:
 *   uf19_init(rank, world_size, peer_ip, ib_dev, gid_idx, numel) → 0 on success
 *   uf19_step(input_ptr, output_ptr, numel) → 0 on success
 *   uf19_cleanup()
 *
 * Build:
 *   nvcc -O2 -arch=sm_121 -shared -Xcompiler -fPIC \
 *       -o libuf19_rdma.so uf19_rdma.cu -libverbs -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TCP_PORT 18600
#define CQ_DRAIN_INTERVAL 64
#define CQ_SIZE 256

// --- Error handling ---
#define LOG_PREFIX "[UF19] "

static void log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, LOG_PREFIX);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

static void log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, LOG_PREFIX "ERROR: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// --- CUDA Device Helpers ---

// Load bf16 bypassing ALL GPU caches (ld.global.cv = cache volatile).
// On GB10, NIC RDMA writes go directly to DRAM, bypassing GPU L1+L2.
// Normal loads (or even ld.cs streaming loads) may hit stale L2 entries.
// ld.global.cv guarantees every read goes to DRAM — sees NIC DMA data.
__device__ __forceinline__ __nv_bfloat16 load_cv(const __nv_bfloat16* ptr) {
    unsigned short val;
    asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(val) : "l"(ptr));
    __nv_bfloat16 result;
    memcpy(&result, &val, sizeof(result));
    return result;
}

// Store bf16 with write-through (st.global.wt = write-through to DRAM).
// Ensures GPU writes to send_buf are visible in DRAM for NIC DMA reads.
// __threadfence_system() orders but may not flush L2 on GB10 unified memory.
__device__ __forceinline__ void store_wt(__nv_bfloat16* ptr, __nv_bfloat16 val) {
    unsigned short bits;
    memcpy(&bits, &val, sizeof(bits));
    asm volatile("st.global.wt.u16 [%0], %1;" :: "l"(ptr), "h"(bits));
}

// --- CUDA Kernels ---

// Copy input → send buffer + set flag.
// Regular stores go through GPU L2 cache. __threadfence_system() flushes
// dirty L2 lines to DRAM, making data visible to NIC for RDMA read.
__global__ void uf19_prepare_send(
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

// Poll recv_flag via system-scope atomic (bypasses GPU L2 cache, sees NIC DMA).
// On GB10 unified memory, volatile reads serve stale L2 cache values and never
// see NIC PCIe DMA writes. atomicAdd_system goes through the full memory fabric.
// Launched as <<<1, 1>>> — stream ordering blocks subsequent kernels until done.
__global__ void uf19_poll_recv(int* recv_flag, int step)
{
    while (atomicAdd_system(recv_flag, 0) < step) {
        __nanosleep(100);
    }
}

// Add local + remote → output (runs after staging copy)
// Both local_data and recv_staging use normal loads — staging buffer
// was populated by cudaMemcpyAsync which handles cache coherence.
__global__ void uf19_add_kernel(
    const __nv_bfloat16* __restrict__ local_data,
    const __nv_bfloat16* __restrict__ recv_staging,
    __nv_bfloat16* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = __bfloat162float(local_data[idx]);
        float b = __bfloat162float(recv_staging[idx]);
        output[idx] = __float2bfloat16(a + b);
    }
}

// --- Global state ---
struct uf19_state {
    int initialized;
    int rank;
    int max_numel;    // buffer capacity (elements)
    int max_data_bytes;
    int use_gdr;  // 1 = GPUDirect RDMA, 0 = mapped pinned
    int step;     // monotonic counter
    int active;   // 0 = skip (use NCCL), 1 = active (use UF19 RDMA)

    // ibverbs
    struct ibv_context* ib_ctx;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_mr* mr_send;
    struct ibv_mr* mr_recv;
    struct ibv_mr* mr_flag;
    struct ibv_mr* mr_flag_src;

    // GPU device memory (GDR path) or mapped pointers
    __nv_bfloat16* gpu_send;    // GPU-accessible send buffer
    __nv_bfloat16* gpu_recv;    // GPU-accessible recv buffer

    // Host pointers (for mapped path cleanup)
    __nv_bfloat16* h_send_buf;  // NULL if GDR
    __nv_bfloat16* h_recv_buf;  // NULL if GDR

    // RDMA buffer pointers (what ibv_reg_mr was called on)
    void* buf_send;
    void* buf_recv;

    // GDR path: device memory for send/recv
    __nv_bfloat16* d_send_buf;
    __nv_bfloat16* d_recv_buf;

    // Flags: always in mapped pinned memory
    int* h_send_flag;
    int* h_recv_flag;
    int* d_send_flag;  // device pointer to h_send_flag
    int* d_recv_flag;  // device pointer to h_recv_flag
    int* h_flag_src;   // source for RDMA flag write

    // Pre-built RDMA WRs (sge_data.length updated per call)
    struct ibv_sge sge_data;
    struct ibv_sge sge_flag;
    struct ibv_send_wr wr_data;
    struct ibv_send_wr wr_flag;

    // Remote RDMA addresses (for dynamic length adjustment)
    uint64_t remote_addr_data;
    uint32_t remote_rkey_data;

    // Stream
    cudaStream_t stream;

    // Staging buffer: cudaMemcpy from NIC-written recv_buf to this buffer
    // before add_kernel. Bypasses GPU L2 cache staleness on GB10 unified memory.
    __nv_bfloat16* d_recv_staging;

    // CPU-side buffers for fully CPU-based addition (debug mode)
    __nv_bfloat16* h_input_copy;   // CPU copy of input tensor
    __nv_bfloat16* h_output_buf;   // CPU addition result

    // Mode: 0=C (sync), 1=B (CPU recv poll), 2=A (separate poll stream)
    int mode;
    cudaStream_t poll_stream;  // mode A only
    cudaEvent_t poll_event;    // mode A only
};

static struct uf19_state g_state = {0};

// --- QP info for TCP exchange ---
struct qp_info {
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey_data;
    uint32_t rkey_flag;
    uint64_t addr_data;
    uint64_t addr_flag;
    union ibv_gid gid;
};

// --- ibverbs helpers ---

static struct ibv_context* open_ib_device(const char* dev_name) {
    int num_devices;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) { log_error("ibv_get_device_list failed"); return NULL; }
    struct ibv_device* dev = NULL;
    for (int i = 0; i < num_devices; i++) {
        if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) {
            dev = dev_list[i]; break;
        }
    }
    if (!dev) { log_error("IB device '%s' not found", dev_name); ibv_free_device_list(dev_list); return NULL; }
    struct ibv_context* ctx = ibv_open_device(dev);
    ibv_free_device_list(dev_list);
    return ctx;
}

static int modify_qp_to_init(struct ibv_qp* qp) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
}

static int modify_qp_to_rtr(struct ibv_qp* qp, struct qp_info* remote, int gid_idx) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote->qpn;
    attr.rq_psn = remote->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remote->gid;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.sgid_index = gid_idx;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = 0;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int modify_qp_to_rts(struct ibv_qp* qp, uint32_t psn) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = psn;
    attr.max_rd_atomic = 1;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
        IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

// Reliable full read/write — TCP can return partial data!
static int full_write(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = write(fd, p, remaining);
        if (n <= 0) { log_error("full_write: %s", strerror(errno)); return -1; }
        p += n; remaining -= n;
    }
    return 0;
}

static int full_read(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = read(fd, p, remaining);
        if (n <= 0) { log_error("full_read: %s (got %zu/%zu)", strerror(errno), len-remaining, len); return -1; }
        p += n; remaining -= n;
    }
    return 0;
}

static int tcp_exchange(int rank, const char* peer_ip,
                        struct qp_info* local, struct qp_info* remote) {
    int sock;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT);

    if (rank == 0) {
        int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_sock < 0) { log_error("socket: %s", strerror(errno)); return -1; }
        int opt = 1;
        setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(listen_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            log_error("bind: %s", strerror(errno)); close(listen_sock); return -1;
        }
        listen(listen_sock, 1);
        log_info("Rank 0: waiting for peer on port %d...", TCP_PORT);
        sock = accept(listen_sock, NULL, NULL);
        close(listen_sock);
        if (sock < 0) { log_error("accept: %s", strerror(errno)); return -1; }
    } else {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) { log_error("socket: %s", strerror(errno)); return -1; }
        inet_pton(AF_INET, peer_ip, &addr.sin_addr);
        log_info("Rank 1: connecting to %s:%d...", peer_ip, TCP_PORT);
        int retries = 0;
        while (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            if (++retries > 300) {  // 30 seconds
                log_error("connect timeout after 30s"); close(sock); return -1;
            }
            usleep(100000);
        }
    }
    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    // Log what we're sending
    log_info("TCP exchange: LOCAL  qpn=%u psn=%u rkey_data=0x%08x rkey_flag=0x%08x "
             "addr_data=0x%lx addr_flag=0x%lx",
             local->qpn, local->psn, local->rkey_data, local->rkey_flag,
             (unsigned long)local->addr_data, (unsigned long)local->addr_flag);

    int ret;
    if (rank == 0) {
        ret = full_write(sock, local, sizeof(*local));
        if (ret) { close(sock); return -1; }
        ret = full_read(sock, remote, sizeof(*remote));
        if (ret) { close(sock); return -1; }
    } else {
        ret = full_read(sock, remote, sizeof(*remote));
        if (ret) { close(sock); return -1; }
        ret = full_write(sock, local, sizeof(*local));
        if (ret) { close(sock); return -1; }
    }

    // Log what we received
    log_info("TCP exchange: REMOTE qpn=%u psn=%u rkey_data=0x%08x rkey_flag=0x%08x "
             "addr_data=0x%lx addr_flag=0x%lx",
             remote->qpn, remote->psn, remote->rkey_data, remote->rkey_flag,
             (unsigned long)remote->addr_data, (unsigned long)remote->addr_flag);

    // Sanity check
    if (remote->addr_data == 0 || remote->addr_flag == 0 ||
        remote->rkey_data == 0 || remote->rkey_flag == 0) {
        log_error("TCP exchange: INVALID remote info (zeroes detected!)");
        close(sock);
        return -1;
    }

    // Barrier
    char c = 'R';
    if (full_write(sock, &c, 1) || full_read(sock, &c, 1)) {
        log_error("TCP exchange barrier failed");
        close(sock);
        return -1;
    }
    close(sock);
    return 0;
}

// --- Exported API ---

extern "C" {

int uf19_init(int rank, int world_size,
              const char* peer_ip, const char* ib_dev,
              int gid_idx, int numel)
{
    if (g_state.initialized) {
        log_error("already initialized");
        return -1;
    }
    if (world_size != 2) {
        log_error("only world_size=2 supported (got %d)", world_size);
        return -1;
    }

    memset(&g_state, 0, sizeof(g_state));
    g_state.rank = rank;
    g_state.max_numel = numel;
    g_state.max_data_bytes = numel * sizeof(__nv_bfloat16);
    g_state.step = 0;

    // Parse mode: C (default) = sync after step, B = CPU recv poll, A = separate poll stream
    g_state.mode = 0;
    const char* mode_env = getenv("VLLM_UF19_MODE");
    if (mode_env) {
        if (mode_env[0] == 'B' || mode_env[0] == 'b') g_state.mode = 1;
        else if (mode_env[0] == 'A' || mode_env[0] == 'a') g_state.mode = 2;
    }
    const char* mode_names[] = {"C (sync)", "B (CPU recv poll)", "A (separate poll stream)"};
    log_info("Init: rank=%d, peer=%s, dev=%s, gid=%d, max_numel=%d (%d bytes), mode=%s",
             rank, peer_ip, ib_dev, gid_idx, numel, g_state.max_data_bytes,
             mode_names[g_state.mode]);

    // CUDA stream (use default stream = 0 for vLLM compat)
    g_state.stream = 0;

    // Mode A: create separate stream + event for poll_recv
    if (g_state.mode == 2) {
        cudaStreamCreate(&g_state.poll_stream);
        cudaEventCreateWithFlags(&g_state.poll_event, cudaEventDisableTiming);
    }

    // --- ibverbs setup ---
    g_state.ib_ctx = open_ib_device(ib_dev);
    if (!g_state.ib_ctx) return -1;

    g_state.pd = ibv_alloc_pd(g_state.ib_ctx);
    if (!g_state.pd) { log_error("ibv_alloc_pd failed"); return -1; }

    g_state.cq = ibv_create_cq(g_state.ib_ctx, CQ_SIZE, NULL, NULL, 0);
    if (!g_state.cq) { log_error("ibv_create_cq failed"); return -1; }

    struct ibv_qp_init_attr qp_init;
    memset(&qp_init, 0, sizeof(qp_init));
    qp_init.send_cq = g_state.cq;
    qp_init.recv_cq = g_state.cq;
    qp_init.qp_type = IBV_QPT_RC;
    qp_init.cap.max_send_wr = CQ_SIZE;
    qp_init.cap.max_recv_wr = 4;
    qp_init.cap.max_send_sge = 1;
    qp_init.cap.max_recv_sge = 1;
    qp_init.cap.max_inline_data = 64;
    qp_init.sq_sig_all = 0;

    g_state.qp = ibv_create_qp(g_state.pd, &qp_init);
    if (!g_state.qp) { log_error("ibv_create_qp failed"); return -1; }

    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    // --- Try GPUDirect RDMA ---
    cudaMalloc(&g_state.d_send_buf, g_state.max_data_bytes);
    cudaMalloc(&g_state.d_recv_buf, g_state.max_data_bytes);
    cudaMemset(g_state.d_send_buf, 0, g_state.max_data_bytes);
    cudaMemset(g_state.d_recv_buf, 0, g_state.max_data_bytes);

    g_state.use_gdr = 0;
    struct ibv_mr* test_mr = ibv_reg_mr(g_state.pd, g_state.d_send_buf, g_state.max_data_bytes, mr_flags);
    if (test_mr) {
        ibv_dereg_mr(test_mr);
        struct ibv_mr* test_mr2 = ibv_reg_mr(g_state.pd, g_state.d_recv_buf, g_state.max_data_bytes, mr_flags);
        if (test_mr2) {
            ibv_dereg_mr(test_mr2);
            g_state.use_gdr = 1;
            log_info("GPUDirect RDMA (nvidia-peermem): ACTIVE");
        }
    }

    if (g_state.use_gdr) {
        // GDR path: RDMA directly on device memory
        g_state.buf_send = g_state.d_send_buf;
        g_state.buf_recv = g_state.d_recv_buf;
        g_state.gpu_send = g_state.d_send_buf;
        g_state.gpu_recv = g_state.d_recv_buf;
        g_state.h_send_buf = NULL;
        g_state.h_recv_buf = NULL;
    } else {
        // Heap-only path (ibverbs-compatible on GB10 unified memory)
        // cudaHostAlloc/cudaHostRegister alter page tables in ways that break
        // ibv_reg_mr DMA mapping on GB10. Plain posix_memalign works because
        // GB10 unified memory makes ALL host memory GPU-accessible by default.
        log_info("GPUDirect not available, using heap memory (no cudaHostRegister)");
        if (posix_memalign((void**)&g_state.h_send_buf, 4096, g_state.max_data_bytes) ||
            posix_memalign((void**)&g_state.h_recv_buf, 4096, g_state.max_data_bytes)) {
            log_error("posix_memalign failed for data buffers"); return -1;
        }
        memset(g_state.h_send_buf, 0, g_state.max_data_bytes);
        memset(g_state.h_recv_buf, 0, g_state.max_data_bytes);
        // On GB10 unified memory, host pointers ARE device pointers (no registration needed)
        g_state.gpu_send = g_state.h_send_buf;
        g_state.gpu_recv = g_state.h_recv_buf;
        g_state.buf_send = g_state.h_send_buf;
        g_state.buf_recv = g_state.h_recv_buf;
    }

    // Staging buffer: cudaMalloc'd device memory, separate from NIC-written recv_buf.
    // After recv_flag confirms data arrived, cudaMemcpyAsync copies recv_buf → staging.
    // add_kernel reads from staging with normal loads (no L2 cache coherency issues).
    cudaMalloc(&g_state.d_recv_staging, g_state.max_data_bytes);
    cudaMemset(g_state.d_recv_staging, 0, g_state.max_data_bytes);
    log_info("Allocated recv staging buffer: %p (%d bytes)", g_state.d_recv_staging, g_state.max_data_bytes);

    // CPU buffers for full-CPU addition path
    g_state.h_input_copy = (__nv_bfloat16*)malloc(g_state.max_data_bytes);
    g_state.h_output_buf = (__nv_bfloat16*)malloc(g_state.max_data_bytes);
    if (!g_state.h_input_copy || !g_state.h_output_buf) {
        log_error("malloc failed for CPU buffers"); return -1;
    }

    // Flags: plain heap (same reason — no cudaHostRegister on GB10)
    if (posix_memalign((void**)&g_state.h_send_flag, 64, sizeof(int)) ||
        posix_memalign((void**)&g_state.h_recv_flag, 64, sizeof(int)) ||
        posix_memalign((void**)&g_state.h_flag_src, 64, sizeof(int))) {
        log_error("posix_memalign failed for flags"); return -1;
    }
    *g_state.h_send_flag = 0;
    *g_state.h_recv_flag = 0;
    // On GB10 unified memory, host pointers ARE device pointers
    g_state.d_send_flag = g_state.h_send_flag;
    g_state.d_recv_flag = g_state.h_recv_flag;
    cudaDeviceSynchronize();

    // Register RDMA memory regions
    g_state.mr_send = ibv_reg_mr(g_state.pd, g_state.buf_send, g_state.max_data_bytes, mr_flags);
    if (!g_state.mr_send) { log_error("ibv_reg_mr send failed (errno=%d)", errno); return -1; }

    g_state.mr_recv = ibv_reg_mr(g_state.pd, g_state.buf_recv, g_state.max_data_bytes, mr_flags);
    if (!g_state.mr_recv) { log_error("ibv_reg_mr recv failed"); return -1; }

    g_state.mr_flag = ibv_reg_mr(g_state.pd, g_state.h_recv_flag, sizeof(int), mr_flags);
    if (!g_state.mr_flag) { log_error("ibv_reg_mr flag failed"); return -1; }

    g_state.mr_flag_src = ibv_reg_mr(g_state.pd, g_state.h_flag_src, sizeof(int), mr_flags);
    if (!g_state.mr_flag_src) { log_error("ibv_reg_mr flag_src failed"); return -1; }

    // Log MR details for debugging RDMA address issues
    log_info("MR send:  addr=%p len=%d lkey=0x%08x rkey=0x%08x",
             g_state.mr_send->addr, (int)g_state.mr_send->length,
             g_state.mr_send->lkey, g_state.mr_send->rkey);
    log_info("MR recv:  addr=%p len=%d lkey=0x%08x rkey=0x%08x",
             g_state.mr_recv->addr, (int)g_state.mr_recv->length,
             g_state.mr_recv->lkey, g_state.mr_recv->rkey);
    log_info("MR flag:  addr=%p len=%d lkey=0x%08x rkey=0x%08x",
             g_state.mr_flag->addr, (int)g_state.mr_flag->length,
             g_state.mr_flag->lkey, g_state.mr_flag->rkey);
    log_info("buf_send=%p buf_recv=%p h_recv_flag=%p",
             g_state.buf_send, g_state.buf_recv, g_state.h_recv_flag);

    // QP state machine: RESET → INIT → RTR → RTS
    if (modify_qp_to_init(g_state.qp)) { log_error("QP → INIT failed"); return -1; }

    uint32_t psn = (uint32_t)(rank * 12345 + 1) & 0xFFFFFF;
    union ibv_gid local_gid;
    if (ibv_query_gid(g_state.ib_ctx, 1, gid_idx, &local_gid)) {
        log_error("ibv_query_gid failed"); return -1;
    }

    struct qp_info local_info, remote_info;
    local_info.qpn = g_state.qp->qp_num;
    local_info.psn = psn;
    local_info.rkey_data = g_state.mr_recv->rkey;
    local_info.rkey_flag = g_state.mr_flag->rkey;
    local_info.addr_data = (uint64_t)g_state.buf_recv;
    local_info.addr_flag = (uint64_t)g_state.h_recv_flag;
    local_info.gid = local_gid;

    if (tcp_exchange(rank, peer_ip, &local_info, &remote_info)) {
        log_error("TCP exchange failed"); return -1;
    }

    if (modify_qp_to_rtr(g_state.qp, &remote_info, gid_idx)) {
        log_error("QP → RTR failed"); return -1;
    }
    if (modify_qp_to_rts(g_state.qp, psn)) {
        log_error("QP → RTS failed"); return -1;
    }

    // Store remote addresses for dynamic WR updates
    g_state.remote_addr_data = remote_info.addr_data;
    g_state.remote_rkey_data = remote_info.rkey_data;

    // Pre-build RDMA Work Requests (sge_data.length updated per call)
    memset(&g_state.sge_data, 0, sizeof(g_state.sge_data));
    g_state.sge_data.addr = (uint64_t)g_state.buf_send;
    g_state.sge_data.length = 0;  // set per call
    g_state.sge_data.lkey = g_state.mr_send->lkey;

    memset(&g_state.wr_data, 0, sizeof(g_state.wr_data));
    g_state.wr_data.wr_id = 1;
    g_state.wr_data.sg_list = &g_state.sge_data;
    g_state.wr_data.num_sge = 1;
    g_state.wr_data.opcode = IBV_WR_RDMA_WRITE;
    g_state.wr_data.send_flags = 0;
    g_state.wr_data.wr.rdma.remote_addr = remote_info.addr_data;
    g_state.wr_data.wr.rdma.rkey = remote_info.rkey_data;

    memset(&g_state.sge_flag, 0, sizeof(g_state.sge_flag));
    g_state.sge_flag.addr = (uint64_t)g_state.h_flag_src;
    g_state.sge_flag.length = sizeof(int);
    g_state.sge_flag.lkey = g_state.mr_flag_src->lkey;

    memset(&g_state.wr_flag, 0, sizeof(g_state.wr_flag));
    g_state.wr_flag.wr_id = 2;
    g_state.wr_flag.sg_list = &g_state.sge_flag;
    g_state.wr_flag.num_sge = 1;
    g_state.wr_flag.opcode = IBV_WR_RDMA_WRITE;
    g_state.wr_flag.send_flags = IBV_SEND_SIGNALED;
    g_state.wr_flag.wr.rdma.remote_addr = remote_info.addr_flag;
    g_state.wr_flag.wr.rdma.rkey = remote_info.rkey_flag;

    // Chain: data WR → flag WR (IB ordering guarantees data arrives before flag)
    g_state.wr_data.next = &g_state.wr_flag;
    g_state.wr_flag.next = NULL;

    // Start inactive — Python calls uf19_activate() after warmup/capture is done.
    // During warmup+capture, torch.distributed barriers can deadlock with UF19 polling.
    g_state.active = 0;

    g_state.initialized = 1;
    log_info("Init complete (rank=%d, %s, max_numel=%d, INACTIVE until uf19_activate())",
             rank, g_state.use_gdr ? "GPUDirect" : "mapped", numel);
    return 0;
}


int uf19_step(void* input_ptr, void* output_ptr, int numel)
{
    if (!g_state.initialized) {
        log_error("not initialized");
        return -1;
    }
    if (numel > g_state.max_numel) {
        return -2;
    }

    g_state.step++;
    int step = g_state.step;
    int data_bytes = numel * (int)sizeof(__nv_bfloat16);

    // Inactive until uf19_activate() is called (after warmup+capture)
    if (!g_state.active) {
        return -2;  // signal Python to use NCCL
    }

    int verbose = (step <= 10 || (step % 500 == 0 && step <= 5000) || step % 5000 == 0);
    if (verbose)
        log_info("step %d: numel=%d (%d bytes)", step, numel, data_bytes);

    __nv_bfloat16* input = (__nv_bfloat16*)input_ptr;
    __nv_bfloat16* output = (__nv_bfloat16*)output_ptr;

    // Wait for previous RDMA to complete before overwriting send_buf.
    // The NIC reads send_buf asynchronously after ibv_post_send. Without this
    // wait, prepare_send overwrites send_buf before the NIC finishes reading,
    // causing the peer to receive corrupted (mixed old+new) data.
    if (step > 1) {
        struct ibv_wc wc;
        int ne;
        long long cq_polls = 0;
        while ((ne = ibv_poll_cq(g_state.cq, 1, &wc)) == 0) {
            cq_polls++;
            if (cq_polls > 5000000000LL) {
                log_error("step %d: CQ poll timeout waiting for prev RDMA", step);
                return -4;
            }
        }
        if (ne < 0) {
            log_error("step %d: ibv_poll_cq error: %d", step, ne);
            return -1;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            log_error("step %d: prev RDMA failed: status=%d (%s) wr_id=%lu",
                      step, wc.status, ibv_wc_status_str(wc.status), wc.wr_id);
            return -1;
        }
        if (verbose) log_info("step %d: prev RDMA CQ ok (polls=%lld)", step, cq_polls);
    }

    // FULL CPU PATH: No GPU kernels for data movement or addition.
    // Eliminates ALL GPU cache coherence issues on GB10.

    // Phase 1: Sync GPU, copy input to CPU
    cudaDeviceSynchronize();
    cudaMemcpy(g_state.h_input_copy, input, data_bytes, cudaMemcpyDeviceToHost);
    // Copy to send_buf (CPU-to-CPU, no GPU caches involved)
    memcpy(g_state.buf_send, g_state.h_input_copy, data_bytes);
    if (verbose) log_info("step %d: send_buf prepared (CPU, %d bytes)", step, data_bytes);

    // Phase 2: Post RDMA
    g_state.sge_data.length = data_bytes;
    *g_state.h_flag_src = step;
    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(g_state.qp, &g_state.wr_data, &bad_wr);
    if (ret) {
        log_error("step %d: ibv_post_send failed: %d (errno=%d: %s)",
                  step, ret, errno, strerror(errno));
        return -1;
    }

    // Phase 3: CPU polls recv_flag
    {
        long long recv_polls = 0;
        while (*(volatile int*)g_state.h_recv_flag < step) {
            recv_polls++;
            if (recv_polls > 5000000000LL) {
                log_error("step %d: h_recv_flag timeout (cur=%d, need=%d, polls=%lld)",
                          step, *g_state.h_recv_flag, step, recv_polls);
                struct ibv_wc wc;
                int ne = ibv_poll_cq(g_state.cq, 1, &wc);
                if (ne > 0 && wc.status != IBV_WC_SUCCESS) {
                    log_error("step %d: CQ error: status=%d (%s) wr_id=%lu",
                              step, wc.status, ibv_wc_status_str(wc.status), wc.wr_id);
                }
                return -4;
            }
        }
        if (verbose) log_info("step %d: recv_flag ok (polls=%lld)", step, recv_polls);
    }

    // Phase 4: CPU addition — no GPU kernels involved
    {
        const __nv_bfloat16* h_in = g_state.h_input_copy;
        const __nv_bfloat16* h_rv = (__nv_bfloat16*)g_state.buf_recv;
        __nv_bfloat16* h_out = g_state.h_output_buf;
        for (int i = 0; i < numel; i++) {
            float a = __bfloat162float(h_in[i]);
            float b = __bfloat162float(h_rv[i]);
            h_out[i] = __float2bfloat16(a + b);
        }
        // Diagnostic: log first few values on early steps
        if (step <= 5) {
            log_info("step %d: in[0..3]={%.4f,%.4f,%.4f,%.4f} rv[0..3]={%.4f,%.4f,%.4f,%.4f} out[0..3]={%.4f,%.4f,%.4f,%.4f}",
                step,
                __bfloat162float(h_in[0]), __bfloat162float(h_in[1]),
                __bfloat162float(h_in[2]), __bfloat162float(h_in[3]),
                __bfloat162float(h_rv[0]), __bfloat162float(h_rv[1]),
                __bfloat162float(h_rv[2]), __bfloat162float(h_rv[3]),
                __bfloat162float(h_out[0]), __bfloat162float(h_out[1]),
                __bfloat162float(h_out[2]), __bfloat162float(h_out[3]));
        }
    }

    // Phase 5: Copy result to GPU
    cudaMemcpy(output, g_state.h_output_buf, data_bytes, cudaMemcpyHostToDevice);

    if (verbose) log_info("step %d: done (mode %c)", step,
                          g_state.mode == 0 ? 'C' : (g_state.mode == 1 ? 'B' : 'A'));

    return 0;
}


void uf19_activate(void)
{
    if (!g_state.initialized) return;
    if (g_state.active) return;
    g_state.active = 1;
    g_state.step = 0;  // reset step counter for clean start
    log_info("ACTIVATED — UF19 RDMA AllReduce now handling traffic (rank=%d)", g_state.rank);
}

void uf19_cleanup(void)
{
    if (!g_state.initialized) return;

    log_info("Cleanup (rank=%d, %d steps completed)", g_state.rank, g_state.step);

    // Drain remaining CQ entries
    if (g_state.cq) {
        struct ibv_wc wc[16];
        while (ibv_poll_cq(g_state.cq, 16, wc) > 0) {}
    }

    // ibverbs cleanup
    if (g_state.mr_send) ibv_dereg_mr(g_state.mr_send);
    if (g_state.mr_recv) ibv_dereg_mr(g_state.mr_recv);
    if (g_state.mr_flag) ibv_dereg_mr(g_state.mr_flag);
    if (g_state.mr_flag_src) ibv_dereg_mr(g_state.mr_flag_src);
    if (g_state.qp) ibv_destroy_qp(g_state.qp);
    if (g_state.cq) ibv_destroy_cq(g_state.cq);
    if (g_state.pd) ibv_dealloc_pd(g_state.pd);
    if (g_state.ib_ctx) ibv_close_device(g_state.ib_ctx);

    // CUDA cleanup
    if (g_state.mode == 2) {
        if (g_state.poll_stream) cudaStreamDestroy(g_state.poll_stream);
        if (g_state.poll_event) cudaEventDestroy(g_state.poll_event);
    }
    // Heap path: just free (no cudaHostUnregister needed)
    if (g_state.h_send_buf) free(g_state.h_send_buf);
    if (g_state.h_recv_buf) free(g_state.h_recv_buf);
    if (g_state.d_send_buf) cudaFree(g_state.d_send_buf);
    if (g_state.d_recv_buf) cudaFree(g_state.d_recv_buf);
    if (g_state.d_recv_staging) cudaFree(g_state.d_recv_staging);
    if (g_state.h_input_copy) free(g_state.h_input_copy);
    if (g_state.h_output_buf) free(g_state.h_output_buf);
    if (g_state.h_send_flag) free(g_state.h_send_flag);
    if (g_state.h_recv_flag) free(g_state.h_recv_flag);
    if (g_state.h_flag_src) free(g_state.h_flag_src);

    memset(&g_state, 0, sizeof(g_state));
}

}  // extern "C"
