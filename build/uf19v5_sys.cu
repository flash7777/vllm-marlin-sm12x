/*
 * UF19v5: System-Scope Minimal Proxy AllReduce
 *
 * Replaces NCCL AllReduce (~18 µs) with a single GPU kernel using
 * system-scope PTX memory ordering + a minimal CPU proxy for ibverbs RDMA.
 *
 * Key innovation: st.release.sys.global.b32 / ld.acquire.sys.global.b32
 * correctly handle GB10 SM121 L2 cache coherence — unlike ld.global.cv (v4)
 * or cudaMemcpyAsync (4.4 µs fixed API overhead).
 *
 * NOTE: Persistent kernel approach tested and rejected — GB10 CUDA runtime
 * deadlocks on cudaFree/cudaDeviceSynchronize/cudaStreamSynchronize(0)
 * when any non-blocking stream has a persistent (never-finishing) kernel.
 * See test_persistent_isolate.cu for proof.
 *
 * Architecture:
 *   GPU Kernel (1 launch per step, on CUDA stream):
 *     0. Wait send_done >= prev step (proxy freed send_buf)
 *     1. st.release.sys: input (device) → send_buf (pinned)
 *     2. atomicExch_system(send_flag, step)
 *     3. Poll recv_flag via atomicAdd_system
 *     4. ld.acquire.sys: recv_buf (pinned) → reduce + output (device)
 *
 *   CPU Proxy Thread (pinned to core, high priority):
 *     Poll send_flag → ibv_post_send (data+flag) → ibv_poll_cq → send_done
 *
 * Build:
 *   nvcc -O2 -arch=sm_121 -shared -Xcompiler -fPIC \
 *       -o libuf19v5_sys.so uf19v5_sys.cu \
 *       -L/usr/lib/aarch64-linux-gnu -libverbs -lpthread
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
#include <sched.h>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TCP_PORT 18605   /* Different from v4 (18600) for coexistence */
#define CQ_SIZE  256

/* ================================================================
 * Logging
 * ================================================================ */

#define LOG_PREFIX "[UF19v5] "

static void log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, LOG_PREFIX);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(args);
}

static void log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, LOG_PREFIX "ERROR: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(args);
}

/* ================================================================
 * GPU Kernel: Single-launch AllReduce with system-scope PTX
 *
 * Launched as <<<1, 256>>> on stream 0. Single block avoids the
 * need for inter-block sync. 256 threads handle up to 65K uint32
 * words (128K bf16 elements) via looping.
 *
 * Optimization: No __nanosleep in polling loops — pure busy-wait
 * for minimum latency (~1 µs saved vs 50ns nanosleep).
 * ================================================================ */

extern "C" __global__
void uf19v5_kernel(
    const uint32_t* __restrict__ input,    /* device memory (bf16 packed as u32) */
    uint32_t* __restrict__ output,          /* device memory (bf16 packed as u32) */
    uint32_t* send_buf,                     /* pinned host (cudaHostAlloc) */
    const uint32_t* recv_buf,               /* pinned host (NIC writes here) */
    volatile uint32_t* send_flag,           /* pinned: GPU increments, proxy reads */
    volatile uint32_t* recv_flag,           /* pinned: NIC writes (peer RDMA) */
    volatile uint32_t* send_done,           /* pinned: proxy writes, GPU reads */
    int n_words)                            /* numel/2 (uint32 = 2x bf16) */
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    /* --- Phase 0: Determine step & wait for previous proxy completion ---
     * Use ld.acquire.sys instead of atomicAdd_system(..., 0) — avoids
     * expensive atomic RMW cache line bounce, just does a sys-scope load. */
    __shared__ uint32_t s_step;
    if (tid == 0) {
        uint32_t cur;
        asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
                     : "=r"(cur) : "l"(send_flag));
        s_step = cur + 1;
        uint32_t prev = cur;
        if (prev > 0) {
            uint32_t val;
            do {
                asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
                             : "=r"(val) : "l"(send_done));
            } while (val < prev);
        }
    }
    __syncthreads();
    uint32_t step = s_step;

    /* --- Phase 1: Copy input → send_buf ---
     * Use normal vectorized stores (128-bit uint4) — fully pipelined and
     * coalesced. Then one __threadfence_system() makes all writes visible
     * to CPU/NIC. Much faster than per-element st.release.sys stores. */
    {
        const uint4* in4 = (const uint4*)input;
        uint4* sb4 = (uint4*)send_buf;
        int n4 = n_words / 4;  /* each uint4 = 4 uint32 = 8 bf16 */
        for (int i = tid; i < n4; i += nthreads)
            sb4[i] = in4[i];
        /* Handle remainder (n_words not multiple of 4) */
        int base = n4 * 4;
        for (int i = base + tid; i < n_words; i += nthreads)
            send_buf[i] = input[i];
    }
    __syncthreads();
    __threadfence_system();

    /* --- Phase 2: Signal proxy that send_buf is ready --- */
    if (tid == 0) {
        atomicExch_system((uint32_t*)send_flag, step);
    }

    /* --- Phase 3: Wait for peer data (NIC RDMA → recv_buf + recv_flag) ---
     * Use ld.acquire.sys instead of atomicAdd_system for polling. */
    if (tid == 0) {
        uint32_t val;
        do {
            asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
                         : "=r"(val) : "l"(recv_flag));
        } while (val < step);
    }
    __syncthreads();

    /* --- Phase 4: Read recv with ld.acquire.sys, reduce (bf16 add), output ---
     * Use 64-bit sys-scope loads to halve the number of expensive loads.
     * Process 4 bf16 values per iteration per thread. */
    {
        int n2 = n_words / 2;  /* pairs of uint32 = 64-bit chunks */
        for (int i = tid; i < n2; i += nthreads) {
            /* 64-bit sys-scope load from recv_buf (NIC-written pinned mem) */
            uint64_t rv64;
            asm volatile("ld.acquire.sys.global.b64 %0, [%1];"
                         : "=l"(rv64) : "l"((const uint64_t*)recv_buf + i));
            uint32_t rlo = (uint32_t)rv64;
            uint32_t rhi = (uint32_t)(rv64 >> 32);

            /* 64-bit normal load from input (device VRAM — fast) */
            uint64_t lv64 = ((const uint64_t*)input)[i];
            uint32_t llo = (uint32_t)lv64;
            uint32_t lhi = (uint32_t)(lv64 >> 32);

            /* Reduce 4 bf16 values: out = local + recv */
            __nv_bfloat16 r0, r1, r2, r3, l0, l1, l2, l3;
            memcpy(&r0, &rlo, 2);
            memcpy(&r1, ((char*)&rlo) + 2, 2);
            memcpy(&r2, &rhi, 2);
            memcpy(&r3, ((char*)&rhi) + 2, 2);
            memcpy(&l0, &llo, 2);
            memcpy(&l1, ((char*)&llo) + 2, 2);
            memcpy(&l2, &lhi, 2);
            memcpy(&l3, ((char*)&lhi) + 2, 2);

            __nv_bfloat16 o0 = __float2bfloat16(__bfloat162float(l0) + __bfloat162float(r0));
            __nv_bfloat16 o1 = __float2bfloat16(__bfloat162float(l1) + __bfloat162float(r1));
            __nv_bfloat16 o2 = __float2bfloat16(__bfloat162float(l2) + __bfloat162float(r2));
            __nv_bfloat16 o3 = __float2bfloat16(__bfloat162float(l3) + __bfloat162float(r3));

            uint32_t out_lo, out_hi;
            memcpy(&out_lo, &o0, 2);
            memcpy(((char*)&out_lo) + 2, &o1, 2);
            memcpy(&out_hi, &o2, 2);
            memcpy(((char*)&out_hi) + 2, &o3, 2);

            /* 64-bit store to output (device VRAM) */
            uint64_t ov64 = (uint64_t)out_lo | ((uint64_t)out_hi << 32);
            ((uint64_t*)output)[i] = ov64;
        }
        /* Handle remainder if n_words is odd */
        int base = n2 * 2;
        for (int i = base + tid; i < n_words; i += nthreads) {
            uint32_t recv_val;
            asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
                         : "=r"(recv_val) : "l"(recv_buf + i));
            __nv_bfloat16 r0, r1;
            memcpy(&r0, &recv_val, 2);
            memcpy(&r1, ((char*)&recv_val) + 2, 2);
            uint32_t local_val = input[i];
            __nv_bfloat16 l0, l1;
            memcpy(&l0, &local_val, 2);
            memcpy(&l1, ((char*)&local_val) + 2, 2);
            __nv_bfloat16 o0 = __float2bfloat16(__bfloat162float(l0) + __bfloat162float(r0));
            __nv_bfloat16 o1 = __float2bfloat16(__bfloat162float(l1) + __bfloat162float(r1));
            uint32_t out_val;
            memcpy(&out_val, &o0, 2);
            memcpy(((char*)&out_val) + 2, &o1, 2);
            output[i] = out_val;
        }
    }
}

/* ================================================================
 * Global State
 * ================================================================ */

struct uf19v5_state {
    int initialized;
    int rank;
    int max_numel;
    int max_data_bytes;   /* max_numel * sizeof(bf16) */
    int active;
    int host_step;

    /* ibverbs */
    struct ibv_context* ib_ctx;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_mr* mr_send;
    struct ibv_mr* mr_recv;
    struct ibv_mr* mr_flag;
    struct ibv_mr* mr_flag_src;

    /* Pinned buffers (cudaHostAlloc — GPU + CPU + NIC accessible) */
    void* send_buf;
    void* recv_buf;

    /* Flags (pinned, 4 bytes each) */
    uint32_t* send_flag;
    uint32_t* recv_flag;
    uint32_t* send_done;
    uint32_t* flag_src;

    /* Pre-built RDMA Work Requests */
    struct ibv_sge sge_data;
    struct ibv_sge sge_flag;
    struct ibv_send_wr wr_data;
    struct ibv_send_wr wr_flag;

    cudaStream_t stream;

    /* Dynamic RDMA data size (updated per step by host, read by proxy) */
    volatile int current_data_bytes;

    /* Proxy thread */
    pthread_t proxy_tid;
    volatile int proxy_running;
};

static struct uf19v5_state g_state = {0};

/* ================================================================
 * QP info exchanged via TCP
 * ================================================================ */

struct qp_info {
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey_data;
    uint32_t rkey_flag;
    uint64_t addr_data;
    uint64_t addr_flag;
    union ibv_gid gid;
};

/* ================================================================
 * ibverbs Helpers
 * ================================================================ */

static struct ibv_context* open_ib_device(const char* dev_name) {
    int num_devices;
    struct ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) { log_error("ibv_get_device_list failed"); return NULL; }

    log_info("IB devices (%d):", num_devices);
    for (int i = 0; i < num_devices; i++)
        log_info("  [%d] %s", i, ibv_get_device_name(dev_list[i]));

    struct ibv_device* dev = NULL;
    for (int i = 0; i < num_devices; i++) {
        if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) {
            dev = dev_list[i]; break;
        }
    }
    if (!dev) {
        for (int i = 0; i < num_devices; i++) {
            if (strstr(ibv_get_device_name(dev_list[i]), dev_name)) {
                dev = dev_list[i];
                log_info("Matched '%s' by substring", ibv_get_device_name(dev));
                break;
            }
        }
    }
    if (!dev && num_devices > 0) {
        dev = dev_list[0];
        log_info("Fallback to first device: %s", ibv_get_device_name(dev));
    }
    if (!dev) { log_error("No IB devices"); ibv_free_device_list(dev_list); return NULL; }

    log_info("Opening: %s", ibv_get_device_name(dev));
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

/* ================================================================
 * TCP Exchange
 * ================================================================ */

static int full_write(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    size_t rem = len;
    while (rem > 0) {
        ssize_t n = write(fd, p, rem);
        if (n <= 0) { log_error("write: %s", strerror(errno)); return -1; }
        p += n; rem -= n;
    }
    return 0;
}

static int full_read(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    size_t rem = len;
    while (rem > 0) {
        ssize_t n = read(fd, p, rem);
        if (n <= 0) { log_error("read: %s (got %zu/%zu)", strerror(errno), len-rem, len); return -1; }
        p += n; rem -= n;
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
        int lsock = socket(AF_INET, SOCK_STREAM, 0);
        if (lsock < 0) { log_error("socket: %s", strerror(errno)); return -1; }
        int opt = 1;
        setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(lsock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            log_error("bind: %s", strerror(errno)); close(lsock); return -1;
        }
        listen(lsock, 1);
        log_info("Rank 0: listening on port %d...", TCP_PORT);
        sock = accept(lsock, NULL, NULL);
        close(lsock);
        if (sock < 0) { log_error("accept: %s", strerror(errno)); return -1; }
    } else {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) { log_error("socket: %s", strerror(errno)); return -1; }
        inet_pton(AF_INET, peer_ip, &addr.sin_addr);
        log_info("Rank 1: connecting to %s:%d...", peer_ip, TCP_PORT);
        for (int retry = 0; retry < 300; retry++) {
            if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) break;
            if (retry == 299) { log_error("connect timeout"); close(sock); return -1; }
            usleep(100000);
        }
    }
    int flag = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    log_info("TCP LOCAL  qpn=%u psn=%u rkey_d=0x%08x rkey_f=0x%08x addr_d=0x%lx addr_f=0x%lx",
             local->qpn, local->psn, local->rkey_data, local->rkey_flag,
             (unsigned long)local->addr_data, (unsigned long)local->addr_flag);

    int ret;
    if (rank == 0) {
        ret = full_write(sock, local, sizeof(*local));
        if (!ret) ret = full_read(sock, remote, sizeof(*remote));
    } else {
        ret = full_read(sock, remote, sizeof(*remote));
        if (!ret) ret = full_write(sock, local, sizeof(*local));
    }
    if (ret) { close(sock); return -1; }

    log_info("TCP REMOTE qpn=%u psn=%u rkey_d=0x%08x rkey_f=0x%08x addr_d=0x%lx addr_f=0x%lx",
             remote->qpn, remote->psn, remote->rkey_data, remote->rkey_flag,
             (unsigned long)remote->addr_data, (unsigned long)remote->addr_flag);

    if (remote->addr_data == 0 || remote->addr_flag == 0 ||
        remote->rkey_data == 0 || remote->rkey_flag == 0) {
        log_error("TCP: invalid remote info (zeroes)");
        close(sock); return -1;
    }

    char c = 'R';
    if (full_write(sock, &c, 1) || full_read(sock, &c, 1)) {
        log_error("TCP barrier failed"); close(sock); return -1;
    }
    close(sock);
    return 0;
}

/* ================================================================
 * CPU Proxy Thread
 *
 * Pinned to a dedicated CPU core for minimum latency.
 * Polls send_flag, posts chained RDMA WRITE, waits CQ, sets send_done.
 * ================================================================ */

static void* proxy_thread_fn(void* arg) {
    (void)arg;
    struct uf19v5_state* s = &g_state;
    uint32_t last_done = 0;

    /* Pin proxy to a dedicated core (last available core).
     * GB10 has 10 cores — reserve one for proxy. */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    int proxy_core = (ncpus > 2) ? ncpus - 1 : 0;
    CPU_SET(proxy_core, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0)
        log_info("Proxy pinned to core %d (of %d)", proxy_core, ncpus);
    else
        log_info("Proxy core pinning failed (non-fatal), core=%d", proxy_core);

    /* Set real-time scheduling for minimum latency */
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) == 0)
        log_info("Proxy set to SCHED_FIFO priority %d", param.sched_priority);

    log_info("Proxy started (rank=%d)", s->rank);

    while (__atomic_load_n(&s->proxy_running, __ATOMIC_RELAXED)) {
        uint32_t current = __atomic_load_n(s->send_flag, __ATOMIC_ACQUIRE);
        if (current <= last_done) continue;

        __atomic_store_n(s->flag_src, current, __ATOMIC_RELEASE);

        /* Update RDMA data length to match this step's actual numel */
        s->sge_data.length = __atomic_load_n(&s->current_data_bytes, __ATOMIC_ACQUIRE);

        struct ibv_send_wr* bad_wr;
        int ret = ibv_post_send(s->qp, &s->wr_data, &bad_wr);
        if (ret) {
            log_error("Proxy: ibv_post_send: %d (errno=%d: %s)", ret, errno, strerror(errno));
            break;
        }

        struct ibv_wc wc;
        while (ibv_poll_cq(s->cq, 1, &wc) == 0) {
            if (!__atomic_load_n(&s->proxy_running, __ATOMIC_RELAXED))
                goto done;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            log_error("Proxy: RDMA status=%d (%s) wr_id=%lu step=%u",
                      wc.status, ibv_wc_status_str(wc.status), wc.wr_id, current);
            break;
        }

        last_done = current;
        __atomic_store_n(s->send_done, current, __ATOMIC_RELEASE);

        if (current <= 10 || current % 5000 == 0)
            log_info("Proxy: step %u RDMA done", current);
    }

done:
    log_info("Proxy exiting (rank=%d, done=%u steps)", s->rank, last_done);
    return NULL;
}

/* ================================================================
 * Exported C API
 * ================================================================ */

extern "C" {

int uf19v5_init(int rank, int world_size,
                const char* peer_ip, const char* ib_dev,
                int gid_idx, int numel)
{
    if (g_state.initialized) { log_error("already initialized"); return -1; }
    if (world_size != 2) { log_error("only world_size=2 (got %d)", world_size); return -1; }

    memset(&g_state, 0, sizeof(g_state));
    g_state.rank = rank;
    g_state.max_numel = numel;
    g_state.max_data_bytes = numel * sizeof(__nv_bfloat16);
    g_state.stream = 0;

    log_info("Init: rank=%d peer=%s dev=%s gid=%d max_numel=%d (%d bytes)",
             rank, peer_ip, ib_dev, gid_idx, numel, g_state.max_data_bytes);

    /* --- ibverbs setup --- */
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

    /* --- Allocate pinned buffers (cudaHostAlloc) --- */
    cudaError_t ce;
    ce = cudaHostAlloc(&g_state.send_buf, g_state.max_data_bytes, cudaHostAllocDefault);
    if (ce) { log_error("cudaHostAlloc send_buf: %s", cudaGetErrorString(ce)); return -1; }

    ce = cudaHostAlloc(&g_state.recv_buf, g_state.max_data_bytes, cudaHostAllocDefault);
    if (ce) { log_error("cudaHostAlloc recv_buf: %s", cudaGetErrorString(ce)); return -1; }

    ce = cudaHostAlloc((void**)&g_state.send_flag, sizeof(uint32_t), cudaHostAllocDefault);
    if (ce) { log_error("cudaHostAlloc send_flag: %s", cudaGetErrorString(ce)); return -1; }

    ce = cudaHostAlloc((void**)&g_state.recv_flag, sizeof(uint32_t), cudaHostAllocDefault);
    if (ce) { log_error("cudaHostAlloc recv_flag: %s", cudaGetErrorString(ce)); return -1; }

    ce = cudaHostAlloc((void**)&g_state.send_done, sizeof(uint32_t), cudaHostAllocDefault);
    if (ce) { log_error("cudaHostAlloc send_done: %s", cudaGetErrorString(ce)); return -1; }

    ce = cudaHostAlloc((void**)&g_state.flag_src, sizeof(uint32_t), cudaHostAllocDefault);
    if (ce) { log_error("cudaHostAlloc flag_src: %s", cudaGetErrorString(ce)); return -1; }

    memset(g_state.send_buf, 0, g_state.max_data_bytes);
    memset(g_state.recv_buf, 0, g_state.max_data_bytes);
    *g_state.send_flag = 0;
    *g_state.recv_flag = 0;
    *g_state.send_done = 0;
    *g_state.flag_src = 0;

    /* --- Register RDMA memory regions --- */
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    g_state.mr_send = ibv_reg_mr(g_state.pd, g_state.send_buf, g_state.max_data_bytes, mr_flags);
    if (!g_state.mr_send) { log_error("ibv_reg_mr send: errno=%d", errno); return -1; }

    g_state.mr_recv = ibv_reg_mr(g_state.pd, g_state.recv_buf, g_state.max_data_bytes, mr_flags);
    if (!g_state.mr_recv) { log_error("ibv_reg_mr recv: errno=%d", errno); return -1; }

    g_state.mr_flag = ibv_reg_mr(g_state.pd, g_state.recv_flag, sizeof(uint32_t), mr_flags);
    if (!g_state.mr_flag) { log_error("ibv_reg_mr flag: errno=%d", errno); return -1; }

    g_state.mr_flag_src = ibv_reg_mr(g_state.pd, g_state.flag_src, sizeof(uint32_t), mr_flags);
    if (!g_state.mr_flag_src) { log_error("ibv_reg_mr flag_src: errno=%d", errno); return -1; }

    log_info("MR send: %p len=%d lkey=0x%08x rkey=0x%08x",
             g_state.mr_send->addr, (int)g_state.mr_send->length,
             g_state.mr_send->lkey, g_state.mr_send->rkey);
    log_info("MR recv: %p len=%d lkey=0x%08x rkey=0x%08x",
             g_state.mr_recv->addr, (int)g_state.mr_recv->length,
             g_state.mr_recv->lkey, g_state.mr_recv->rkey);

    /* --- QP state machine --- */
    if (modify_qp_to_init(g_state.qp)) { log_error("QP INIT failed"); return -1; }

    uint32_t psn = (uint32_t)(rank * 12345 + 1) & 0xFFFFFF;
    union ibv_gid local_gid;

    if (gid_idx < 0) {
        gid_idx = 3;
        for (int i = 0; i < 16; i++) {
            union ibv_gid g;
            if (ibv_query_gid(g_state.ib_ctx, 1, i, &g) == 0) {
                if (g.raw[0] == 0 && g.raw[1] == 0 &&
                    g.raw[10] == 0xff && g.raw[11] == 0xff) {
                    gid_idx = i;
                    log_info("Auto-detected RoCEv2 GID: %d", gid_idx);
                    break;
                }
            }
        }
    }
    if (ibv_query_gid(g_state.ib_ctx, 1, gid_idx, &local_gid)) {
        log_error("ibv_query_gid (idx=%d) failed", gid_idx); return -1;
    }
    log_info("GID[%d]: %02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x",
             gid_idx,
             local_gid.raw[0], local_gid.raw[1], local_gid.raw[2], local_gid.raw[3],
             local_gid.raw[4], local_gid.raw[5], local_gid.raw[6], local_gid.raw[7],
             local_gid.raw[8], local_gid.raw[9], local_gid.raw[10], local_gid.raw[11],
             local_gid.raw[12], local_gid.raw[13], local_gid.raw[14], local_gid.raw[15]);

    /* --- TCP handshake --- */
    struct qp_info local_info, remote_info;
    local_info.qpn = g_state.qp->qp_num;
    local_info.psn = psn;
    local_info.rkey_data = g_state.mr_recv->rkey;
    local_info.rkey_flag = g_state.mr_flag->rkey;
    local_info.addr_data = (uint64_t)g_state.recv_buf;
    local_info.addr_flag = (uint64_t)g_state.recv_flag;
    local_info.gid = local_gid;

    if (tcp_exchange(rank, peer_ip, &local_info, &remote_info)) {
        log_error("TCP exchange failed"); return -1;
    }

    if (modify_qp_to_rtr(g_state.qp, &remote_info, gid_idx)) {
        log_error("QP RTR failed"); return -1;
    }
    if (modify_qp_to_rts(g_state.qp, psn)) {
        log_error("QP RTS failed"); return -1;
    }

    /* --- Pre-build chained RDMA Work Requests --- */
    memset(&g_state.sge_data, 0, sizeof(g_state.sge_data));
    g_state.sge_data.addr = (uint64_t)g_state.send_buf;
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
    g_state.sge_flag.addr = (uint64_t)g_state.flag_src;
    g_state.sge_flag.length = sizeof(uint32_t);
    g_state.sge_flag.lkey = g_state.mr_flag_src->lkey;

    memset(&g_state.wr_flag, 0, sizeof(g_state.wr_flag));
    g_state.wr_flag.wr_id = 2;
    g_state.wr_flag.sg_list = &g_state.sge_flag;
    g_state.wr_flag.num_sge = 1;
    g_state.wr_flag.opcode = IBV_WR_RDMA_WRITE;
    g_state.wr_flag.send_flags = IBV_SEND_SIGNALED;
    g_state.wr_flag.wr.rdma.remote_addr = remote_info.addr_flag;
    g_state.wr_flag.wr.rdma.rkey = remote_info.rkey_flag;

    g_state.wr_data.next = &g_state.wr_flag;
    g_state.wr_flag.next = NULL;

    /* --- Start proxy thread --- */
    g_state.proxy_running = 1;
    if (pthread_create(&g_state.proxy_tid, NULL, proxy_thread_fn, NULL)) {
        log_error("pthread_create failed"); return -1;
    }

    g_state.active = 0;
    g_state.initialized = 1;
    log_info("Init complete (rank=%d, max_numel=%d, INACTIVE until activate())", rank, numel);
    return 0;
}


int uf19v5_step(void* input_ptr, void* output_ptr, int numel, uint64_t stream_handle)
{
    if (!g_state.initialized) return -1;
    if (numel > g_state.max_numel) return -2;
    if (!g_state.active) return -2;

    g_state.host_step++;
    if (g_state.host_step <= 10 || g_state.host_step % 5000 == 0)
        log_info("step %d: numel=%d", g_state.host_step, numel);

    /* Set RDMA data size for this step (proxy reads before ibv_post_send) */
    __atomic_store_n(&g_state.current_data_bytes, numel * (int)sizeof(__nv_bfloat16), __ATOMIC_RELEASE);

    int n_words = numel / 2;
    cudaStream_t stream = (cudaStream_t)stream_handle;

    uf19v5_kernel<<<1, 256, 0, stream>>>(
        (const uint32_t*)input_ptr,
        (uint32_t*)output_ptr,
        (uint32_t*)g_state.send_buf,
        (const uint32_t*)g_state.recv_buf,
        g_state.send_flag,
        g_state.recv_flag,
        g_state.send_done,
        n_words);

    return 0;
}


void uf19v5_activate(void)
{
    if (!g_state.initialized || g_state.active) return;
    g_state.active = 1;
    g_state.host_step = 0;
    *g_state.send_flag = 0;
    *g_state.recv_flag = 0;
    *g_state.send_done = 0;
    __sync_synchronize();
    log_info("ACTIVATED — sys-scope AllReduce live (rank=%d)", g_state.rank);
}


void uf19v5_cleanup(void)
{
    if (!g_state.initialized) return;

    log_info("Cleanup (rank=%d, %d steps)", g_state.rank, g_state.host_step);

    __atomic_store_n(&g_state.proxy_running, 0, __ATOMIC_RELAXED);
    pthread_join(g_state.proxy_tid, NULL);

    if (g_state.cq) {
        struct ibv_wc wc[16];
        while (ibv_poll_cq(g_state.cq, 16, wc) > 0) {}
    }

    if (g_state.mr_send) ibv_dereg_mr(g_state.mr_send);
    if (g_state.mr_recv) ibv_dereg_mr(g_state.mr_recv);
    if (g_state.mr_flag) ibv_dereg_mr(g_state.mr_flag);
    if (g_state.mr_flag_src) ibv_dereg_mr(g_state.mr_flag_src);
    if (g_state.qp) ibv_destroy_qp(g_state.qp);
    if (g_state.cq) ibv_destroy_cq(g_state.cq);
    if (g_state.pd) ibv_dealloc_pd(g_state.pd);
    if (g_state.ib_ctx) ibv_close_device(g_state.ib_ctx);

    if (g_state.send_buf) cudaFreeHost(g_state.send_buf);
    if (g_state.recv_buf) cudaFreeHost(g_state.recv_buf);
    if (g_state.send_flag) cudaFreeHost(g_state.send_flag);
    if (g_state.recv_flag) cudaFreeHost(g_state.recv_flag);
    if (g_state.send_done) cudaFreeHost(g_state.send_done);
    if (g_state.flag_src) cudaFreeHost(g_state.flag_src);

    memset(&g_state, 0, sizeof(g_state));
}

}  /* extern "C" */
