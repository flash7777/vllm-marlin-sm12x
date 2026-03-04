/*
 * UF19v4: CUDA-Graph-Compatible 2-Rank AllReduce via ibverbs RDMA
 *
 * Architecture:
 *   GPU kernels (graph-capturable):
 *     1. wait_send_done <<<1,1>>>  — polls send_done (proxy completed prev RDMA)
 *     2. copy_to_send   <<<N,256>>> — copies input→send_buf
 *     3. signal_send    <<<1,1>>>  — __threadfence_system + increment send_flag
 *     4. poll_recv      <<<1,1>>>  — polls recv_flag (peer's data arrived)
 *     5. add_recv       <<<N,256>>> — adds local + recv_buf → output (load_cv)
 *
 *   CPU proxy thread (background):
 *     Polls send_flag → posts RDMA WRITE (data+flag) → polls CQ → sets send_done
 *
 * The GPU never calls ibverbs or cudaDeviceSynchronize. All 4 kernels are
 * captured in vLLM's CUDA graph. The proxy thread runs independently.
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
#include <pthread.h>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TCP_PORT 18600
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
// NIC RDMA writes go directly to DRAM, bypassing GPU L1+L2.
// ld.global.cv guarantees every read goes to DRAM — sees NIC DMA data.
__device__ __forceinline__ __nv_bfloat16 load_cv(const __nv_bfloat16* ptr) {
    unsigned short val;
    asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(val) : "l"(ptr));
    __nv_bfloat16 result;
    memcpy(&result, &val, sizeof(result));
    return result;
}

// --- v4 CUDA Kernels (all graph-capturable) ---

// K0: Wait for proxy to complete previous RDMA (send_buf safe to overwrite).
// Reads send_flag (= how many steps launched) and send_done (= how many completed).
// First step: send_flag=0, send_done=0 → returns immediately.
__global__ void uf19_wait_send_done(int* send_flag, int* send_done)
{
    int launched = atomicAdd_system(send_flag, 0);
    while (atomicAdd_system(send_done, 0) < launched) {
        __nanosleep(100);
    }
}

// K1a: Copy input → send_buf (multi-block, no signaling).
__global__ void uf19_copy_to_send(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ send_buf, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) send_buf[idx] = src[idx];
}

// K1b: Signal proxy that send_buf is ready (single thread).
// Launched AFTER uf19_copy_to_send on same stream → CUDA stream ordering
// guarantees all K1a writes are complete. __threadfence_system() then
// flushes this thread's view to system memory before incrementing send_flag.
__global__ void uf19_signal_send(int* send_flag)
{
    __threadfence_system();
    atomicAdd_system(send_flag, 1);
}

// K2: Poll recv_flag until peer's data arrives.
// Reads send_flag to get target step (set by prepare_send on same stream).
// recv_flag is written by peer's NIC via RDMA WRITE — use atomicAdd_system
// to bypass GPU L2 cache and read from DRAM.
__global__ void uf19_poll_recv(int* recv_flag, int* send_flag)
{
    int target = atomicAdd_system(send_flag, 0);
    while (atomicAdd_system(recv_flag, 0) < target) {
        __nanosleep(100);
    }
}

// K3: Add local input + received data → output.
// local_data: GPU tensor (normal loads, coherent).
// recv_buf: NIC-written mapped memory — use load_cv to bypass stale L2.
__global__ void uf19_add_recv(
    const __nv_bfloat16* __restrict__ local_data,
    const __nv_bfloat16* __restrict__ recv_buf,
    __nv_bfloat16* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = __bfloat162float(local_data[idx]);
        float b = __bfloat162float(load_cv(&recv_buf[idx]));
        output[idx] = __float2bfloat16(a + b);
    }
}

// --- Global state ---
struct uf19_state {
    int initialized;
    int rank;
    int max_numel;
    int max_data_bytes;
    int use_gdr;
    int active;
    int host_step;  // CPU-side counter for logging only

    // ibverbs
    struct ibv_context* ib_ctx;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_mr* mr_send;
    struct ibv_mr* mr_recv;
    struct ibv_mr* mr_flag;
    struct ibv_mr* mr_flag_src;

    // GPU-accessible send/recv buffers
    __nv_bfloat16* gpu_send;
    __nv_bfloat16* gpu_recv;

    // Host pointers (for cleanup)
    __nv_bfloat16* h_send_buf;
    __nv_bfloat16* h_recv_buf;

    // RDMA buffer pointers (what ibv_reg_mr was called on)
    void* buf_send;
    void* buf_recv;

    // GDR path device memory
    __nv_bfloat16* d_send_buf;
    __nv_bfloat16* d_recv_buf;

    // Flags (mapped memory, GPU + CPU accessible)
    int* h_send_flag;   // GPU increments, proxy reads
    int* h_recv_flag;   // peer's NIC writes, GPU reads
    int* h_send_done;   // proxy increments, GPU reads
    int* h_flag_src;    // proxy writes step value, NIC reads for RDMA

    // Device pointers to flags (= host pointers on GB10 unified memory)
    int* d_send_flag;
    int* d_recv_flag;
    int* d_send_done;

    // Pre-built RDMA WRs
    struct ibv_sge sge_data;
    struct ibv_sge sge_flag;
    struct ibv_send_wr wr_data;
    struct ibv_send_wr wr_flag;

    // Stream (default = 0 for vLLM compat)
    cudaStream_t stream;

    // Proxy thread
    pthread_t proxy_tid;
    volatile int proxy_running;
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

    // List available devices for debugging
    log_info("Available IB devices (%d):", num_devices);
    for (int i = 0; i < num_devices; i++)
        log_info("  [%d] %s", i, ibv_get_device_name(dev_list[i]));

    struct ibv_device* dev = NULL;
    for (int i = 0; i < num_devices; i++) {
        if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) {
            dev = dev_list[i]; break;
        }
    }
    // Fallback: if not found by exact name, try substring match
    if (!dev) {
        for (int i = 0; i < num_devices; i++) {
            if (strstr(ibv_get_device_name(dev_list[i]), dev_name)) {
                dev = dev_list[i];
                log_info("Matched device '%s' by substring", ibv_get_device_name(dev));
                break;
            }
        }
    }
    // Last resort: use first device
    if (!dev && num_devices > 0) {
        dev = dev_list[0];
        log_info("Using first available device: %s", ibv_get_device_name(dev));
    }
    if (!dev) { log_error("No IB devices found"); ibv_free_device_list(dev_list); return NULL; }
    log_info("Opening IB device: %s", ibv_get_device_name(dev));
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
            if (++retries > 300) {
                log_error("connect timeout after 30s"); close(sock); return -1;
            }
            usleep(100000);
        }
    }
    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

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

    log_info("TCP exchange: REMOTE qpn=%u psn=%u rkey_data=0x%08x rkey_flag=0x%08x "
             "addr_data=0x%lx addr_flag=0x%lx",
             remote->qpn, remote->psn, remote->rkey_data, remote->rkey_flag,
             (unsigned long)remote->addr_data, (unsigned long)remote->addr_flag);

    if (remote->addr_data == 0 || remote->addr_flag == 0 ||
        remote->rkey_data == 0 || remote->rkey_flag == 0) {
        log_error("TCP exchange: INVALID remote info (zeroes detected!)");
        close(sock);
        return -1;
    }

    char c = 'R';
    if (full_write(sock, &c, 1) || full_read(sock, &c, 1)) {
        log_error("TCP exchange barrier failed");
        close(sock);
        return -1;
    }
    close(sock);
    return 0;
}

// --- Proxy thread ---
// Runs in background. Polls send_flag for new data, posts RDMA WRITE,
// polls CQ for completion, then sets send_done to signal GPU.

static void* proxy_thread_fn(void* arg) {
    (void)arg;
    struct uf19_state* s = &g_state;
    int last_completed = 0;

    log_info("Proxy thread started (rank=%d)", s->rank);

    while (__atomic_load_n(&s->proxy_running, __ATOMIC_RELAXED)) {
        // Wait for GPU to signal new data
        int current = __atomic_load_n(s->h_send_flag, __ATOMIC_ACQUIRE);
        if (current <= last_completed) continue;

        // Post RDMA WRITE: send_buf → peer's recv_buf, then flag → peer's recv_flag
        *s->h_flag_src = current;
        struct ibv_send_wr* bad_wr;
        int ret = ibv_post_send(s->qp, &s->wr_data, &bad_wr);
        if (ret) {
            log_error("Proxy: ibv_post_send failed: %d (errno=%d: %s)",
                      ret, errno, strerror(errno));
            break;
        }

        // Wait for CQ completion (NIC finished reading send_buf)
        struct ibv_wc wc;
        while (ibv_poll_cq(s->cq, 1, &wc) == 0) {
            if (!__atomic_load_n(&s->proxy_running, __ATOMIC_RELAXED))
                goto done;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            log_error("Proxy: RDMA failed: status=%d (%s) wr_id=%lu step=%d",
                      wc.status, ibv_wc_status_str(wc.status), wc.wr_id, current);
            break;
        }

        // Signal GPU: send_buf is now free to overwrite
        last_completed = current;
        __atomic_store_n(s->h_send_done, current, __ATOMIC_RELEASE);

        if (current <= 10 || current % 5000 == 0) {
            log_info("Proxy: step %d RDMA complete", current);
        }
    }

done:
    log_info("Proxy thread exiting (rank=%d, completed=%d steps)", s->rank, last_completed);
    return NULL;
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
    g_state.host_step = 0;
    g_state.stream = 0;

    log_info("v4 Init: rank=%d, peer=%s, dev=%s, gid=%d, max_numel=%d (%d bytes)",
             rank, peer_ip, ib_dev, gid_idx, numel, g_state.max_data_bytes);

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
        g_state.buf_send = g_state.d_send_buf;
        g_state.buf_recv = g_state.d_recv_buf;
        g_state.gpu_send = g_state.d_send_buf;
        g_state.gpu_recv = g_state.d_recv_buf;
        g_state.h_send_buf = NULL;
        g_state.h_recv_buf = NULL;
    } else {
        log_info("GPUDirect not available, using heap memory");
        if (posix_memalign((void**)&g_state.h_send_buf, 4096, g_state.max_data_bytes) ||
            posix_memalign((void**)&g_state.h_recv_buf, 4096, g_state.max_data_bytes)) {
            log_error("posix_memalign failed for data buffers"); return -1;
        }
        memset(g_state.h_send_buf, 0, g_state.max_data_bytes);
        memset(g_state.h_recv_buf, 0, g_state.max_data_bytes);
        g_state.gpu_send = g_state.h_send_buf;
        g_state.gpu_recv = g_state.h_recv_buf;
        g_state.buf_send = g_state.h_send_buf;
        g_state.buf_recv = g_state.h_recv_buf;
    }

    // Flags: plain heap (posix_memalign — GB10 unified memory = GPU accessible)
    if (posix_memalign((void**)&g_state.h_send_flag, 64, sizeof(int)) ||
        posix_memalign((void**)&g_state.h_recv_flag, 64, sizeof(int)) ||
        posix_memalign((void**)&g_state.h_send_done, 64, sizeof(int)) ||
        posix_memalign((void**)&g_state.h_flag_src, 64, sizeof(int))) {
        log_error("posix_memalign failed for flags"); return -1;
    }
    *g_state.h_send_flag = 0;
    *g_state.h_recv_flag = 0;
    *g_state.h_send_done = 0;
    *g_state.h_flag_src = 0;
    // On GB10 unified memory, host pointers ARE device pointers
    g_state.d_send_flag = g_state.h_send_flag;
    g_state.d_recv_flag = g_state.h_recv_flag;
    g_state.d_send_done = g_state.h_send_done;
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

    log_info("MR send:  addr=%p len=%d lkey=0x%08x rkey=0x%08x",
             g_state.mr_send->addr, (int)g_state.mr_send->length,
             g_state.mr_send->lkey, g_state.mr_send->rkey);
    log_info("MR recv:  addr=%p len=%d lkey=0x%08x rkey=0x%08x",
             g_state.mr_recv->addr, (int)g_state.mr_recv->length,
             g_state.mr_recv->lkey, g_state.mr_recv->rkey);

    // QP state machine: RESET → INIT → RTR → RTS
    if (modify_qp_to_init(g_state.qp)) { log_error("QP → INIT failed"); return -1; }

    uint32_t psn = (uint32_t)(rank * 12345 + 1) & 0xFFFFFF;
    union ibv_gid local_gid;

    // Auto-detect GID index if requested (-1) or verify given index
    if (gid_idx < 0) {
        // Scan GID table for a RoCEv2 entry (IPv4-mapped)
        gid_idx = 3; // reasonable default
        for (int i = 0; i < 16; i++) {
            union ibv_gid g;
            if (ibv_query_gid(g_state.ib_ctx, 1, i, &g) == 0) {
                // IPv4-mapped: first 10 bytes zero, bytes 10-11 = 0xffff
                if (g.raw[0] == 0 && g.raw[1] == 0 && g.raw[10] == 0xff && g.raw[11] == 0xff) {
                    gid_idx = i;
                    log_info("Auto-detected RoCEv2 GID index: %d", gid_idx);
                    break;
                }
            }
        }
    }
    if (ibv_query_gid(g_state.ib_ctx, 1, gid_idx, &local_gid)) {
        log_error("ibv_query_gid failed (index=%d)", gid_idx); return -1;
    }
    log_info("Using GID[%d]: %02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x",
             gid_idx,
             local_gid.raw[0], local_gid.raw[1], local_gid.raw[2], local_gid.raw[3],
             local_gid.raw[4], local_gid.raw[5], local_gid.raw[6], local_gid.raw[7],
             local_gid.raw[8], local_gid.raw[9], local_gid.raw[10], local_gid.raw[11],
             local_gid.raw[12], local_gid.raw[13], local_gid.raw[14], local_gid.raw[15]);

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

    // Pre-build RDMA Work Requests (always send max_data_bytes)
    memset(&g_state.sge_data, 0, sizeof(g_state.sge_data));
    g_state.sge_data.addr = (uint64_t)g_state.buf_send;
    g_state.sge_data.length = g_state.max_data_bytes;
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

    // Chain: data WR → flag WR (IB ordering: data arrives before flag)
    g_state.wr_data.next = &g_state.wr_flag;
    g_state.wr_flag.next = NULL;

    // Start proxy thread
    g_state.proxy_running = 1;
    if (pthread_create(&g_state.proxy_tid, NULL, proxy_thread_fn, NULL)) {
        log_error("pthread_create failed for proxy thread");
        return -1;
    }

    // Start inactive — Python calls uf19_activate() after warmup/capture
    g_state.active = 0;
    g_state.initialized = 1;
    log_info("v4 Init complete (rank=%d, %s, max_numel=%d, INACTIVE until uf19_activate())",
             rank, g_state.use_gdr ? "GPUDirect" : "mapped", numel);
    return 0;
}


// Graph-capturable uf19_step: only launches GPU kernels, no CPU blocking.
int uf19_step(void* input_ptr, void* output_ptr, int numel)
{
    if (!g_state.initialized) return -1;
    if (numel > g_state.max_numel) return -2;
    if (!g_state.active) return -2;

    g_state.host_step++;
    if (g_state.host_step <= 10 || g_state.host_step % 5000 == 0)
        log_info("step %d: numel=%d (%d bytes)", g_state.host_step, numel,
                 (int)(numel * sizeof(__nv_bfloat16)));

    int blocks = (numel + 255) / 256;
    cudaStream_t s = g_state.stream;

    // K0: Wait for proxy to finish previous RDMA (send_buf safe to overwrite)
    uf19_wait_send_done<<<1, 1, 0, s>>>(g_state.d_send_flag, g_state.d_send_done);

    // K1a: Copy input → send_buf
    uf19_copy_to_send<<<blocks, 256, 0, s>>>(
        (const __nv_bfloat16*)input_ptr, g_state.gpu_send, numel);

    // K1b: Signal proxy that send_buf is ready
    uf19_signal_send<<<1, 1, 0, s>>>(g_state.d_send_flag);

    // K2: Poll recv_flag until peer's data arrives
    uf19_poll_recv<<<1, 1, 0, s>>>(g_state.d_recv_flag, g_state.d_send_flag);

    // K3: Add local input + received data → output
    uf19_add_recv<<<blocks, 256, 0, s>>>(
        (const __nv_bfloat16*)input_ptr, g_state.gpu_recv,
        (__nv_bfloat16*)output_ptr, numel);

    return 0;
}


void uf19_activate(void)
{
    if (!g_state.initialized) return;
    if (g_state.active) return;
    g_state.active = 1;
    g_state.host_step = 0;
    // Reset mapped counters for clean start
    *g_state.h_send_flag = 0;
    *g_state.h_recv_flag = 0;
    *g_state.h_send_done = 0;
    __sync_synchronize();
    log_info("ACTIVATED — UF19v4 RDMA AllReduce now handling traffic (rank=%d)", g_state.rank);
}

void uf19_cleanup(void)
{
    if (!g_state.initialized) return;

    log_info("Cleanup (rank=%d, %d steps)", g_state.rank, g_state.host_step);

    // Stop proxy thread
    __atomic_store_n(&g_state.proxy_running, 0, __ATOMIC_RELAXED);
    pthread_join(g_state.proxy_tid, NULL);

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

    // Memory cleanup
    if (g_state.h_send_buf) free(g_state.h_send_buf);
    if (g_state.h_recv_buf) free(g_state.h_recv_buf);
    if (g_state.d_send_buf) cudaFree(g_state.d_send_buf);
    if (g_state.d_recv_buf) cudaFree(g_state.d_recv_buf);
    if (g_state.h_send_flag) free(g_state.h_send_flag);
    if (g_state.h_recv_flag) free(g_state.h_recv_flag);
    if (g_state.h_send_done) free(g_state.h_send_done);
    if (g_state.h_flag_src) free(g_state.h_flag_src);

    memset(&g_state, 0, sizeof(g_state));
}

}  // extern "C"
