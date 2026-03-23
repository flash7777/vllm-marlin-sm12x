/*
 * C-level correctness + latency test for UF19v5 with real RDMA.
 * Tests:
 *   A. Extensive correctness (100 consecutive steps, various patterns)
 *   B. Per-call kernel launch latency
 *   C. Persistent kernel latency
 *
 * Usage: Run on both ranks simultaneously:
 *   Head:   ./test_uf19v5_clevel 0 192.168.0.116
 *   Worker: ./test_uf19v5_clevel 1 192.168.0.117
 *
 * Build: nvcc -O2 -arch=sm_121 -o test_uf19v5_clevel test_uf19v5_clevel.cu \
 *        -L/usr/lib/aarch64-linux-gnu -libverbs -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TCP_PORT 18606
#define CQ_SIZE 256
#define NUMEL 4096
#define N_WORDS (NUMEL / 2)
#define DATA_BYTES (NUMEL * sizeof(__nv_bfloat16))

static inline double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* === Single-call kernel (production version with bf16 reduce) === */
extern "C" __global__
void uf19v5_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t* send_buf,
    const uint32_t* recv_buf,
    volatile uint32_t* send_flag,
    volatile uint32_t* recv_flag,
    volatile uint32_t* send_done,
    int n_words)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    __shared__ uint32_t s_step;
    if (tid == 0) {
        s_step = atomicAdd_system((uint32_t*)send_flag, 0) + 1;
        uint32_t prev = s_step - 1;
        if (prev > 0) {
            while (atomicAdd_system((uint32_t*)send_done, 0) < prev) {
                __nanosleep(50);
            }
        }
    }
    __syncthreads();
    uint32_t step = s_step;

    for (int i = tid; i < n_words; i += nthreads) {
        uint32_t v = input[i];
        asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(send_buf + i), "r"(v));
    }
    __syncthreads();
    __threadfence_system();

    if (tid == 0) atomicExch_system((uint32_t*)send_flag, step);

    if (tid == 0) {
        while (atomicAdd_system((uint32_t*)recv_flag, 0) < step) {
            __nanosleep(50);
        }
    }
    __syncthreads();

    /* bf16 reduce */
    for (int i = tid; i < n_words; i += nthreads) {
        uint32_t recv_val;
        asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(recv_val) : "l"(recv_buf + i));

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

/* === Persistent kernel (internal loop) === */
extern "C" __global__
void uf19v5_persistent(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t* send_buf,
    const uint32_t* recv_buf,
    volatile uint32_t* send_flag,
    volatile uint32_t* recv_flag,
    volatile uint32_t* send_done,
    int n_words,
    int n_iters)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    for (int step = 1; step <= n_iters; step++) {
        if (tid == 0 && step > 1) {
            while (atomicAdd_system((uint32_t*)send_done, 0) < (uint32_t)(step - 1)) {
                __nanosleep(50);
            }
        }
        __syncthreads();

        for (int i = tid; i < n_words; i += nthreads) {
            uint32_t v = input[i];
            asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(send_buf + i), "r"(v));
        }
        __syncthreads();
        __threadfence_system();

        if (tid == 0) atomicExch_system((uint32_t*)send_flag, (uint32_t)step);

        if (tid == 0) {
            while (atomicAdd_system((uint32_t*)recv_flag, 0) < (uint32_t)step) {
                __nanosleep(50);
            }
        }
        __syncthreads();

        for (int i = tid; i < n_words; i += nthreads) {
            uint32_t recv_val;
            asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(recv_val) : "l"(recv_buf + i));

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
        __syncthreads();
    }
}

/* === ibverbs + proxy === */

struct rdma_ctx {
    struct ibv_context* ib_ctx;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    struct ibv_mr *mr_send, *mr_recv, *mr_flag, *mr_flag_src;

    void* send_buf;
    void* recv_buf;
    uint32_t* send_flag;
    uint32_t* recv_flag;
    uint32_t* send_done;
    uint32_t* flag_src;

    struct ibv_sge sge_data, sge_flag;
    struct ibv_send_wr wr_data, wr_flag;

    volatile int proxy_running;
    pthread_t proxy_tid;
    int rank;
};

static struct rdma_ctx g_ctx;

static void* proxy_fn(void* arg) {
    struct rdma_ctx* c = (struct rdma_ctx*)arg;
    uint32_t last = 0;
    while (__atomic_load_n(&c->proxy_running, __ATOMIC_RELAXED)) {
        uint32_t cur = __atomic_load_n(c->send_flag, __ATOMIC_ACQUIRE);
        if (cur <= last) continue;
        __atomic_store_n(c->flag_src, cur, __ATOMIC_RELEASE);
        struct ibv_send_wr* bad;
        if (ibv_post_send(c->qp, &c->wr_data, &bad)) {
            fprintf(stderr, "ibv_post_send failed\n"); break;
        }
        struct ibv_wc wc;
        while (ibv_poll_cq(c->cq, 1, &wc) == 0) {
            if (!__atomic_load_n(&c->proxy_running, __ATOMIC_RELAXED)) goto out;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "RDMA failed: %s\n", ibv_wc_status_str(wc.status)); break;
        }
        last = cur;
        __atomic_store_n(c->send_done, cur, __ATOMIC_RELEASE);
    }
out:
    return NULL;
}

struct qp_info {
    uint32_t qpn, psn, rkey_data, rkey_flag;
    uint64_t addr_data, addr_flag;
    union ibv_gid gid;
};

static int full_rw(int fd, void* buf, size_t len, int wr) {
    char* p = (char*)buf; size_t rem = len;
    while (rem > 0) {
        ssize_t n = wr ? write(fd, p, rem) : read(fd, p, rem);
        if (n <= 0) return -1;
        p += n; rem -= n;
    }
    return 0;
}

static void reset_flags() {
    *g_ctx.send_flag = 0; *g_ctx.recv_flag = 0; *g_ctx.send_done = 0;
    *g_ctx.flag_src = 0;
    memset(g_ctx.send_buf, 0, DATA_BYTES);
    memset(g_ctx.recv_buf, 0, DATA_BYTES);
    __sync_synchronize();
    usleep(5000);
}

static int setup_rdma(int rank, const char* peer_ip) {
    struct rdma_ctx* c = &g_ctx;
    c->rank = rank;

    int nd; struct ibv_device** dl = ibv_get_device_list(&nd);
    if (!dl || nd == 0) { fprintf(stderr, "No IB devices\n"); return -1; }
    c->ib_ctx = ibv_open_device(dl[0]);
    ibv_free_device_list(dl);
    if (!c->ib_ctx) return -1;

    c->pd = ibv_alloc_pd(c->ib_ctx);
    c->cq = ibv_create_cq(c->ib_ctx, CQ_SIZE, NULL, NULL, 0);

    struct ibv_qp_init_attr qi = {};
    qi.send_cq = c->cq; qi.recv_cq = c->cq;
    qi.qp_type = IBV_QPT_RC;
    qi.cap.max_send_wr = CQ_SIZE; qi.cap.max_recv_wr = 4;
    qi.cap.max_send_sge = 1; qi.cap.max_recv_sge = 1;
    qi.cap.max_inline_data = 64;
    c->qp = ibv_create_qp(c->pd, &qi);

    cudaHostAlloc(&c->send_buf, DATA_BYTES, cudaHostAllocDefault);
    cudaHostAlloc(&c->recv_buf, DATA_BYTES, cudaHostAllocDefault);
    cudaHostAlloc((void**)&c->send_flag, 4, cudaHostAllocDefault);
    cudaHostAlloc((void**)&c->recv_flag, 4, cudaHostAllocDefault);
    cudaHostAlloc((void**)&c->send_done, 4, cudaHostAllocDefault);
    cudaHostAlloc((void**)&c->flag_src, 4, cudaHostAllocDefault);
    memset(c->send_buf, 0, DATA_BYTES);
    memset(c->recv_buf, 0, DATA_BYTES);
    *c->send_flag = 0; *c->recv_flag = 0; *c->send_done = 0; *c->flag_src = 0;

    int mf = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    c->mr_send = ibv_reg_mr(c->pd, c->send_buf, DATA_BYTES, mf);
    c->mr_recv = ibv_reg_mr(c->pd, c->recv_buf, DATA_BYTES, mf);
    c->mr_flag = ibv_reg_mr(c->pd, c->recv_flag, 4, mf);
    c->mr_flag_src = ibv_reg_mr(c->pd, c->flag_src, 4, mf);

    struct ibv_qp_attr a = {}; a.qp_state = IBV_QPS_INIT; a.port_num = 1;
    a.qp_access_flags = mf;
    ibv_modify_qp(c->qp, &a, IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS);

    uint32_t psn = (rank * 12345 + 1) & 0xFFFFFF;
    union ibv_gid lgid;
    int gid_idx = 3;
    for (int i = 0; i < 16; i++) {
        union ibv_gid g;
        if (ibv_query_gid(c->ib_ctx, 1, i, &g) == 0 &&
            g.raw[0]==0 && g.raw[1]==0 && g.raw[10]==0xff && g.raw[11]==0xff) {
            gid_idx = i; break;
        }
    }
    ibv_query_gid(c->ib_ctx, 1, gid_idx, &lgid);

    struct qp_info li = {}, ri;
    li.qpn = c->qp->qp_num; li.psn = psn;
    li.rkey_data = c->mr_recv->rkey; li.rkey_flag = c->mr_flag->rkey;
    li.addr_data = (uint64_t)c->recv_buf; li.addr_flag = (uint64_t)c->recv_flag;
    li.gid = lgid;

    struct sockaddr_in sa = {}; sa.sin_family = AF_INET; sa.sin_port = htons(TCP_PORT);
    int sock;
    if (rank == 0) {
        int ls = socket(AF_INET, SOCK_STREAM, 0);
        int o=1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &o, sizeof(o));
        sa.sin_addr.s_addr = INADDR_ANY;
        bind(ls, (struct sockaddr*)&sa, sizeof(sa)); listen(ls, 1);
        printf("[R%d] Waiting on port %d...\n", rank, TCP_PORT);
        sock = accept(ls, NULL, NULL); close(ls);
    } else {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        inet_pton(AF_INET, peer_ip, &sa.sin_addr);
        printf("[R%d] Connecting to %s:%d...\n", rank, peer_ip, TCP_PORT);
        for (int r=0; r<300; r++) {
            if (connect(sock, (struct sockaddr*)&sa, sizeof(sa))==0) break;
            usleep(100000);
        }
    }
    int f=1; setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &f, sizeof(f));
    if (rank == 0) { full_rw(sock, &li, sizeof(li), 1); full_rw(sock, &ri, sizeof(ri), 0); }
    else { full_rw(sock, &ri, sizeof(ri), 0); full_rw(sock, &li, sizeof(li), 1); }
    char b='R'; full_rw(sock, &b, 1, 1); full_rw(sock, &b, 1, 0);
    close(sock);

    memset(&a, 0, sizeof(a)); a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_4096;
    a.dest_qp_num = ri.qpn; a.rq_psn = ri.psn; a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12; a.ah_attr.port_num = 1; a.ah_attr.is_global = 1;
    a.ah_attr.grh.dgid = ri.gid; a.ah_attr.grh.sgid_index = gid_idx; a.ah_attr.grh.hop_limit = 64;
    ibv_modify_qp(c->qp, &a, IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|
                  IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER);

    memset(&a, 0, sizeof(a)); a.qp_state = IBV_QPS_RTS; a.timeout = 14;
    a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = psn; a.max_rd_atomic = 1;
    ibv_modify_qp(c->qp, &a, IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|
                  IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC);

    c->sge_data.addr = (uint64_t)c->send_buf; c->sge_data.length = DATA_BYTES;
    c->sge_data.lkey = c->mr_send->lkey;
    c->wr_data.wr_id = 1; c->wr_data.sg_list = &c->sge_data; c->wr_data.num_sge = 1;
    c->wr_data.opcode = IBV_WR_RDMA_WRITE;
    c->wr_data.wr.rdma.remote_addr = ri.addr_data; c->wr_data.wr.rdma.rkey = ri.rkey_data;

    c->sge_flag.addr = (uint64_t)c->flag_src; c->sge_flag.length = 4;
    c->sge_flag.lkey = c->mr_flag_src->lkey;
    c->wr_flag.wr_id = 2; c->wr_flag.sg_list = &c->sge_flag; c->wr_flag.num_sge = 1;
    c->wr_flag.opcode = IBV_WR_RDMA_WRITE; c->wr_flag.send_flags = IBV_SEND_SIGNALED;
    c->wr_flag.wr.rdma.remote_addr = ri.addr_flag; c->wr_flag.wr.rdma.rkey = ri.rkey_flag;

    c->wr_data.next = &c->wr_flag; c->wr_flag.next = NULL;

    c->proxy_running = 1;
    pthread_create(&c->proxy_tid, NULL, proxy_fn, c);
    printf("[R%d] RDMA ready\n", rank);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("Usage: %s <rank> <peer_ip>\n", argv[0]); return 1; }
    int rank = atoi(argv[1]);
    const char* peer = argv[2];

    printf("=== UF19v5 Correctness + Latency Test (rank=%d) ===\n", rank);

    if (setup_rdma(rank, peer)) { printf("RDMA setup failed\n"); return 1; }

    struct rdma_ctx* c = &g_ctx;

    /* Alloc device buffers */
    __nv_bfloat16 *d_input, *d_output;
    cudaMalloc(&d_input, DATA_BYTES);
    cudaMalloc(&d_output, DATA_BYTES);

    __nv_bfloat16* h_input = (__nv_bfloat16*)malloc(DATA_BYTES);
    __nv_bfloat16* h_output = (__nv_bfloat16*)malloc(DATA_BYTES);

    /* ============================================================
     * TEST A: Correctness — 100 consecutive steps, per-call kernel
     * Tests for even/odd bugs, stale data, accumulation errors
     * ============================================================ */
    printf("\n=== TEST A: Correctness (100 steps, per-call kernel) ===\n");
    reset_flags();
    {
        int total_fail = 0;
        for (int step = 0; step < 100; step++) {
            /* Pattern: rank0 sends (step+1)*i, rank1 sends (step+1)*i*10
             * Expected result on both: (step+1)*i*11 */
            float scale = (rank == 0) ? 1.0f : 10.0f;
            for (int i = 0; i < NUMEL; i++)
                h_input[i] = __float2bfloat16((float)(i + 1) * (step + 1) * scale);
            cudaMemcpy(d_input, h_input, DATA_BYTES, cudaMemcpyHostToDevice);

            uf19v5_kernel<<<1, 256>>>(
                (const uint32_t*)d_input, (uint32_t*)d_output,
                (uint32_t*)c->send_buf, (const uint32_t*)c->recv_buf,
                c->send_flag, c->recv_flag, c->send_done, N_WORDS);
            cudaDeviceSynchronize();

            cudaMemcpy(h_output, d_output, DATA_BYTES, cudaMemcpyDeviceToHost);

            int errs = 0;
            float maxdiff = 0;
            for (int i = 0; i < NUMEL; i++) {
                float got = __bfloat162float(h_output[i]);
                float exp = (float)(i + 1) * (step + 1) * 11.0f;
                float diff = fabsf(got - exp);
                if (diff > maxdiff) maxdiff = diff;
                /* BF16 relative tolerance: ~0.8% + small absolute */
                float thr = fabsf(exp) * 0.01f + 1.0f;
                if (diff > thr) errs++;
            }
            const char* tag = errs ? "FAIL" : "OK";
            if (errs || step < 10 || step == 49 || step == 50 || step == 99 ||
                step % 10 == 0) {
                printf("  step %3d: maxdiff=%8.1f errs=%d/%d %s %s\n",
                       step, maxdiff, errs, NUMEL, tag,
                       (step % 2 == 0) ? "(even)" : "(odd)");
            }
            if (errs) {
                total_fail++;
                /* Show first 3 bad elements */
                int shown = 0;
                for (int i = 0; i < NUMEL && shown < 3; i++) {
                    float got = __bfloat162float(h_output[i]);
                    float exp = (float)(i + 1) * (step + 1) * 11.0f;
                    float thr = fabsf(exp) * 0.01f + 1.0f;
                    if (fabsf(got - exp) > thr) {
                        printf("    [%d] got=%.1f exp=%.1f diff=%.1f\n",
                               i, got, exp, got - exp);
                        shown++;
                    }
                }
            }
        }
        printf("  TOTAL: %d/100 steps failed\n", total_fail);
    }

    /* ============================================================
     * TEST B: Correctness — constant data, 50 consecutive steps
     * Catches stale buffer issues where previous step's data leaks
     * ============================================================ */
    printf("\n=== TEST B: Stale data test (50 steps, constant pattern per step) ===\n");
    reset_flags();
    {
        int total_fail = 0;
        for (int step = 0; step < 50; step++) {
            /* Rank 0: all 1.0, Rank 1: all 2.0 → expected: all 3.0 */
            float val = (rank == 0) ? 1.0f : 2.0f;
            for (int i = 0; i < NUMEL; i++)
                h_input[i] = __float2bfloat16(val);
            cudaMemcpy(d_input, h_input, DATA_BYTES, cudaMemcpyHostToDevice);

            uf19v5_kernel<<<1, 256>>>(
                (const uint32_t*)d_input, (uint32_t*)d_output,
                (uint32_t*)c->send_buf, (const uint32_t*)c->recv_buf,
                c->send_flag, c->recv_flag, c->send_done, N_WORDS);
            cudaDeviceSynchronize();

            cudaMemcpy(h_output, d_output, DATA_BYTES, cudaMemcpyDeviceToHost);

            int errs = 0;
            for (int i = 0; i < NUMEL; i++) {
                float got = __bfloat162float(h_output[i]);
                if (fabsf(got - 3.0f) > 0.1f) errs++;
            }
            if (errs || step < 5 || step == 49) {
                printf("  step %2d: errs=%d/%d %s %s\n",
                       step, errs, NUMEL, errs?"FAIL":"OK",
                       (step%2==0)?"(even)":"(odd)");
            }
            if (errs) total_fail++;
        }
        printf("  TOTAL: %d/50 steps failed\n", total_fail);
    }

    /* ============================================================
     * TEST C: Correctness — alternating patterns (catch even/odd bugs)
     * ============================================================ */
    printf("\n=== TEST C: Alternating patterns (50 steps) ===\n");
    reset_flags();
    {
        int total_fail = 0;
        for (int step = 0; step < 50; step++) {
            /* Even steps: rank0=1.0, rank1=100.0 → 101.0
             * Odd steps:  rank0=5.0, rank1=500.0 → 505.0 */
            float r0_val = (step % 2 == 0) ? 1.0f : 5.0f;
            float r1_val = (step % 2 == 0) ? 100.0f : 500.0f;
            float val = (rank == 0) ? r0_val : r1_val;
            float expected = r0_val + r1_val;

            for (int i = 0; i < NUMEL; i++)
                h_input[i] = __float2bfloat16(val);
            cudaMemcpy(d_input, h_input, DATA_BYTES, cudaMemcpyHostToDevice);

            uf19v5_kernel<<<1, 256>>>(
                (const uint32_t*)d_input, (uint32_t*)d_output,
                (uint32_t*)c->send_buf, (const uint32_t*)c->recv_buf,
                c->send_flag, c->recv_flag, c->send_done, N_WORDS);
            cudaDeviceSynchronize();

            cudaMemcpy(h_output, d_output, DATA_BYTES, cudaMemcpyDeviceToHost);

            int errs = 0;
            for (int i = 0; i < NUMEL; i++) {
                float got = __bfloat162float(h_output[i]);
                if (fabsf(got - expected) > 1.0f) errs++;
            }
            if (errs || step < 10 || step == 49) {
                printf("  step %2d (%s): exp=%.0f errs=%d/%d %s\n",
                       step, (step%2==0)?"even":"odd ",
                       expected, errs, NUMEL, errs?"FAIL":"OK");
            }
            if (errs) {
                total_fail++;
                float got = __bfloat162float(h_output[0]);
                printf("    [0] got=%.1f exp=%.1f\n", got, expected);
            }
        }
        printf("  TOTAL: %d/50 steps failed\n", total_fail);
    }

    /* ============================================================
     * TEST D: Latency — per-call kernel launch
     * ============================================================ */
    printf("\n=== TEST D: Latency — per-call kernel launch ===\n");
    {
        for (int i = 0; i < NUMEL; i++)
            h_input[i] = __float2bfloat16((float)(i + 1));
        cudaMemcpy(d_input, h_input, DATA_BYTES, cudaMemcpyHostToDevice);

        reset_flags();
        int warmup = 200;
        for (int i = 0; i < warmup; i++) {
            uf19v5_kernel<<<1, 256>>>(
                (const uint32_t*)d_input, (uint32_t*)d_output,
                (uint32_t*)c->send_buf, (const uint32_t*)c->recv_buf,
                c->send_flag, c->recv_flag, c->send_done, N_WORDS);
        }
        cudaDeviceSynchronize();

        reset_flags();
        int bench = 5000;
        double t0 = now_us();
        for (int i = 0; i < bench; i++) {
            uf19v5_kernel<<<1, 256>>>(
                (const uint32_t*)d_input, (uint32_t*)d_output,
                (uint32_t*)c->send_buf, (const uint32_t*)c->recv_buf,
                c->send_flag, c->recv_flag, c->send_done, N_WORDS);
        }
        cudaDeviceSynchronize();
        double elapsed = now_us() - t0;
        printf("  %d iters: %.1f µs/iter (wall clock)\n", bench, elapsed / bench);

        /* Also with CUDA events */
        reset_flags();
        cudaEvent_t ev0, ev1; float ms;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        for (int i = 0; i < bench; i++) {
            uf19v5_kernel<<<1, 256>>>(
                (const uint32_t*)d_input, (uint32_t*)d_output,
                (uint32_t*)c->send_buf, (const uint32_t*)c->recv_buf,
                c->send_flag, c->recv_flag, c->send_done, N_WORDS);
        }
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&ms, ev0, ev1);
        printf("  %d iters: %.1f µs/iter (CUDA events)\n", bench, ms * 1000.0f / bench);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }

    /* ============================================================
     * TEST E: Latency — persistent kernel (single launch)
     * ============================================================ */
    printf("\n=== TEST E: Latency — persistent kernel ===\n");
    {
        for (int i = 0; i < NUMEL; i++)
            h_input[i] = __float2bfloat16((float)(i + 1));
        cudaMemcpy(d_input, h_input, DATA_BYTES, cudaMemcpyHostToDevice);

        reset_flags();
        uf19v5_persistent<<<1, 256>>>(
            (const uint32_t*)d_input, (uint32_t*)d_output,
            (uint32_t*)c->send_buf, (const uint32_t*)c->recv_buf,
            c->send_flag, c->recv_flag, c->send_done, N_WORDS, 200);
        cudaDeviceSynchronize();

        reset_flags();
        int bench = 5000;
        double t0 = now_us();
        uf19v5_persistent<<<1, 256>>>(
            (const uint32_t*)d_input, (uint32_t*)d_output,
            (uint32_t*)c->send_buf, (const uint32_t*)c->recv_buf,
            c->send_flag, c->recv_flag, c->send_done, N_WORDS, bench);
        cudaDeviceSynchronize();
        double elapsed = now_us() - t0;
        printf("  %d iters: %.1f µs/iter (wall, single launch)\n", bench, elapsed / bench);
    }

    printf("\n--- Summary ---\n");
    printf("  NCCL AllReduce 4K bf16: ~18 µs\n");
    printf("  bench_sys.cu (local no RDMA): 3.5 µs\n");

    __atomic_store_n(&c->proxy_running, 0, __ATOMIC_RELAXED);
    pthread_join(c->proxy_tid, NULL);
    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output);

    printf("\n[R%d] Done\n", rank);
    return 0;
}
