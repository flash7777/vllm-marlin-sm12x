/*
 * Simple UF19v5 correctness + latency test. No flag resets between phases.
 * Tests: 100 correctness steps (various patterns) → 200 warmup → 5000 bench
 * All using continuous step counter (no reset).
 *
 * Build: nvcc -O2 -arch=sm_121 -o test_uf19v5_simple test_uf19v5_simple.cu \
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

#define TCP_PORT 18607
#define CQ_SIZE 256
#define NUMEL 4096
#define N_WORDS (NUMEL / 2)
#define DATA_BYTES (NUMEL * sizeof(__nv_bfloat16))

static inline double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* === GPU kernels === */

extern "C" __global__
void v5_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t* send_buf, const uint32_t* recv_buf,
    volatile uint32_t* send_flag, volatile uint32_t* recv_flag,
    volatile uint32_t* send_done, int n_words)
{
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ uint32_t s_step;
    if (tid == 0) {
        s_step = atomicAdd_system((uint32_t*)send_flag, 0) + 1;
        uint32_t prev = s_step - 1;
        if (prev > 0)
            while (atomicAdd_system((uint32_t*)send_done, 0) < prev) __nanosleep(50);
    }
    __syncthreads();
    uint32_t step = s_step;

    for (int i = tid; i < n_words; i += nt) {
        uint32_t v = input[i];
        asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(send_buf + i), "r"(v));
    }
    __syncthreads(); __threadfence_system();
    if (tid == 0) atomicExch_system((uint32_t*)send_flag, step);

    if (tid == 0)
        while (atomicAdd_system((uint32_t*)recv_flag, 0) < step) __nanosleep(50);
    __syncthreads();

    for (int i = tid; i < n_words; i += nt) {
        uint32_t rv;
        asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(rv) : "l"(recv_buf + i));
        __nv_bfloat16 r0, r1; memcpy(&r0, &rv, 2); memcpy(&r1, ((char*)&rv)+2, 2);
        uint32_t lv = input[i];
        __nv_bfloat16 l0, l1; memcpy(&l0, &lv, 2); memcpy(&l1, ((char*)&lv)+2, 2);
        __nv_bfloat16 o0 = __float2bfloat16(__bfloat162float(l0) + __bfloat162float(r0));
        __nv_bfloat16 o1 = __float2bfloat16(__bfloat162float(l1) + __bfloat162float(r1));
        uint32_t ov; memcpy(&ov, &o0, 2); memcpy(((char*)&ov)+2, &o1, 2);
        output[i] = ov;
    }
}

extern "C" __global__
void v5_persistent(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t* send_buf, const uint32_t* recv_buf,
    volatile uint32_t* send_flag, volatile uint32_t* recv_flag,
    volatile uint32_t* send_done, int n_words, int n_iters)
{
    int tid = threadIdx.x, nt = blockDim.x;
    for (int step = 1; step <= n_iters; step++) {
        if (tid == 0 && step > 1)
            while (atomicAdd_system((uint32_t*)send_done, 0) < (uint32_t)(step-1)) __nanosleep(50);
        __syncthreads();
        for (int i = tid; i < n_words; i += nt) {
            uint32_t v = input[i];
            asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(send_buf + i), "r"(v));
        }
        __syncthreads(); __threadfence_system();
        if (tid == 0) atomicExch_system((uint32_t*)send_flag, (uint32_t)step);
        if (tid == 0)
            while (atomicAdd_system((uint32_t*)recv_flag, 0) < (uint32_t)step) __nanosleep(50);
        __syncthreads();
        for (int i = tid; i < n_words; i += nt) {
            uint32_t rv;
            asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(rv) : "l"(recv_buf + i));
            __nv_bfloat16 r0, r1; memcpy(&r0, &rv, 2); memcpy(&r1, ((char*)&rv)+2, 2);
            uint32_t lv = input[i];
            __nv_bfloat16 l0, l1; memcpy(&l0, &lv, 2); memcpy(&l1, ((char*)&lv)+2, 2);
            __nv_bfloat16 o0 = __float2bfloat16(__bfloat162float(l0) + __bfloat162float(r0));
            __nv_bfloat16 o1 = __float2bfloat16(__bfloat162float(l1) + __bfloat162float(r1));
            uint32_t ov; memcpy(&ov, &o0, 2); memcpy(((char*)&ov)+2, &o1, 2);
            output[i] = ov;
        }
        __syncthreads();
    }
}

/* === ibverbs / proxy (inline) === */
struct ctx {
    struct ibv_context* ib; struct ibv_pd* pd; struct ibv_cq* cq; struct ibv_qp* qp;
    struct ibv_mr *ms, *mr, *mf, *mfs;
    void *sb, *rb; uint32_t *sf, *rf, *sd, *fs;
    struct ibv_sge sgd, sgf; struct ibv_send_wr wd, wf;
    volatile int run; pthread_t tid; int rank;
};
static struct ctx C;

static void* proxy(void* a) {
    struct ctx* c = (struct ctx*)a; uint32_t last = 0;
    while (__atomic_load_n(&c->run, __ATOMIC_RELAXED)) {
        uint32_t cur = __atomic_load_n(c->sf, __ATOMIC_ACQUIRE);
        if (cur <= last) continue;
        __atomic_store_n(c->fs, cur, __ATOMIC_RELEASE);
        struct ibv_send_wr* bad;
        if (ibv_post_send(c->qp, &c->wd, &bad)) { fprintf(stderr,"post_send fail\n"); break; }
        struct ibv_wc wc;
        while (ibv_poll_cq(c->cq, 1, &wc) == 0)
            if (!__atomic_load_n(&c->run, __ATOMIC_RELAXED)) goto out;
        if (wc.status) { fprintf(stderr,"RDMA err: %s\n", ibv_wc_status_str(wc.status)); break; }
        last = cur; __atomic_store_n(c->sd, cur, __ATOMIC_RELEASE);
    }
out: return NULL;
}

struct qi { uint32_t qpn,psn,rkd,rkf; uint64_t ad,af; union ibv_gid gid; };

static int frw(int fd, void* b, size_t n, int w) {
    char* p=(char*)b; size_t r=n;
    while(r>0){ssize_t x=w?write(fd,p,r):read(fd,p,r); if(x<=0)return -1; p+=x;r-=x;}
    return 0;
}

static int setup(int rank, const char* peer) {
    struct ctx* c=&C; c->rank=rank;
    setlinebuf(stdout);  /* IMPORTANT: unbuffered output */
    int nd; struct ibv_device** dl=ibv_get_device_list(&nd);
    if(!dl||!nd){fprintf(stderr,"No IB\n");return-1;}
    c->ib=ibv_open_device(dl[0]); ibv_free_device_list(dl);
    c->pd=ibv_alloc_pd(c->ib); c->cq=ibv_create_cq(c->ib,CQ_SIZE,0,0,0);
    struct ibv_qp_init_attr qi={}; qi.send_cq=c->cq; qi.recv_cq=c->cq;
    qi.qp_type=IBV_QPT_RC; qi.cap.max_send_wr=CQ_SIZE; qi.cap.max_recv_wr=4;
    qi.cap.max_send_sge=1; qi.cap.max_recv_sge=1; qi.cap.max_inline_data=64;
    c->qp=ibv_create_qp(c->pd,&qi);

    cudaHostAlloc(&c->sb,DATA_BYTES,0); cudaHostAlloc(&c->rb,DATA_BYTES,0);
    cudaHostAlloc((void**)&c->sf,4,0); cudaHostAlloc((void**)&c->rf,4,0);
    cudaHostAlloc((void**)&c->sd,4,0); cudaHostAlloc((void**)&c->fs,4,0);
    memset(c->sb,0,DATA_BYTES); memset(c->rb,0,DATA_BYTES);
    *c->sf=0; *c->rf=0; *c->sd=0; *c->fs=0;

    int mf=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
    c->ms=ibv_reg_mr(c->pd,c->sb,DATA_BYTES,mf); c->mr=ibv_reg_mr(c->pd,c->rb,DATA_BYTES,mf);
    c->mf=ibv_reg_mr(c->pd,c->rf,4,mf); c->mfs=ibv_reg_mr(c->pd,c->fs,4,mf);

    struct ibv_qp_attr a={}; a.qp_state=IBV_QPS_INIT; a.port_num=1; a.qp_access_flags=mf;
    ibv_modify_qp(c->qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS);

    uint32_t psn=(rank*12345+1)&0xFFFFFF; union ibv_gid lg; int gi=3;
    for(int i=0;i<16;i++){union ibv_gid g;
        if(!ibv_query_gid(c->ib,1,i,&g)&&!g.raw[0]&&!g.raw[1]&&g.raw[10]==0xff&&g.raw[11]==0xff){gi=i;break;}}
    ibv_query_gid(c->ib,1,gi,&lg);

    struct qi li={},ri; li.qpn=c->qp->qp_num; li.psn=psn;
    li.rkd=c->mr->rkey; li.rkf=c->mf->rkey;
    li.ad=(uint64_t)c->rb; li.af=(uint64_t)c->rf; li.gid=lg;

    struct sockaddr_in sa={}; sa.sin_family=AF_INET; sa.sin_port=htons(TCP_PORT);
    int sock;
    if(rank==0){int ls=socket(AF_INET,SOCK_STREAM,0);int o=1;
        setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&o,4);
        sa.sin_addr.s_addr=INADDR_ANY; bind(ls,(struct sockaddr*)&sa,sizeof(sa));
        listen(ls,1); printf("[R%d] Listening port %d...\n",rank,TCP_PORT);
        sock=accept(ls,0,0); close(ls);
    }else{sock=socket(AF_INET,SOCK_STREAM,0); inet_pton(AF_INET,peer,&sa.sin_addr);
        printf("[R%d] Connecting %s:%d...\n",rank,peer,TCP_PORT);
        for(int r=0;r<300;r++){if(!connect(sock,(struct sockaddr*)&sa,sizeof(sa)))break;usleep(100000);}
    }
    int f=1; setsockopt(sock,IPPROTO_TCP,TCP_NODELAY,&f,4);
    if(rank==0){frw(sock,&li,sizeof(li),1);frw(sock,&ri,sizeof(ri),0);}
    else{frw(sock,&ri,sizeof(ri),0);frw(sock,&li,sizeof(li),1);}
    char b='R'; frw(sock,&b,1,1); frw(sock,&b,1,0); close(sock);

    memset(&a,0,sizeof(a)); a.qp_state=IBV_QPS_RTR; a.path_mtu=IBV_MTU_4096;
    a.dest_qp_num=ri.qpn; a.rq_psn=ri.psn; a.max_dest_rd_atomic=1; a.min_rnr_timer=12;
    a.ah_attr.port_num=1; a.ah_attr.is_global=1; a.ah_attr.grh.dgid=ri.gid;
    a.ah_attr.grh.sgid_index=gi; a.ah_attr.grh.hop_limit=64;
    ibv_modify_qp(c->qp,&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|
        IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER);
    memset(&a,0,sizeof(a)); a.qp_state=IBV_QPS_RTS; a.timeout=14; a.retry_cnt=7;
    a.rnr_retry=7; a.sq_psn=psn; a.max_rd_atomic=1;
    ibv_modify_qp(c->qp,&a,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|
        IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC);

    c->sgd.addr=(uint64_t)c->sb; c->sgd.length=DATA_BYTES; c->sgd.lkey=c->ms->lkey;
    c->wd.wr_id=1; c->wd.sg_list=&c->sgd; c->wd.num_sge=1;
    c->wd.opcode=IBV_WR_RDMA_WRITE; c->wd.wr.rdma.remote_addr=ri.ad; c->wd.wr.rdma.rkey=ri.rkd;
    c->sgf.addr=(uint64_t)c->fs; c->sgf.length=4; c->sgf.lkey=c->mfs->lkey;
    c->wf.wr_id=2; c->wf.sg_list=&c->sgf; c->wf.num_sge=1;
    c->wf.opcode=IBV_WR_RDMA_WRITE; c->wf.send_flags=IBV_SEND_SIGNALED;
    c->wf.wr.rdma.remote_addr=ri.af; c->wf.wr.rdma.rkey=ri.rkf;
    c->wd.next=&c->wf; c->wf.next=0;

    c->run=1; pthread_create(&c->tid,0,proxy,c);
    printf("[R%d] RDMA ready\n",rank); return 0;
}

#define LAUNCH(in,out) v5_kernel<<<1,256>>>((const uint32_t*)(in),(uint32_t*)(out),\
    (uint32_t*)C.sb,(const uint32_t*)C.rb,C.sf,C.rf,C.sd,N_WORDS)

int main(int argc, char** argv) {
    if(argc<3){printf("Usage: %s rank peer_ip\n",argv[0]);return 1;}
    int rank=atoi(argv[1]); const char* peer=argv[2];
    setlinebuf(stdout);

    printf("=== UF19v5 Test (rank=%d, numel=%d) ===\n", rank, NUMEL);
    if(setup(rank,peer)) return 1;

    __nv_bfloat16 *d_in, *d_out;
    cudaMalloc(&d_in, DATA_BYTES); cudaMalloc(&d_out, DATA_BYTES);
    __nv_bfloat16* h_in = (__nv_bfloat16*)malloc(DATA_BYTES);
    __nv_bfloat16* h_out = (__nv_bfloat16*)malloc(DATA_BYTES);

    int total_fail = 0;

    /* === PHASE 1: Correctness — 100 steps with varying patterns === */
    printf("\n--- Phase 1: Correctness (100 steps) ---\n");
    for (int s = 0; s < 100; s++) {
        /* Pattern varies each step to catch stale/even-odd bugs */
        float scale;
        if (s % 3 == 0) {
            /* Ascending: rank0=i*(s+1), rank1=i*(s+1)*10 → sum = i*(s+1)*11 */
            for (int i = 0; i < NUMEL; i++)
                h_in[i] = __float2bfloat16((float)(i+1) * (s+1) * (rank==0 ? 1.0f : 10.0f));
        } else if (s % 3 == 1) {
            /* Constant: rank0=s+1, rank1=(s+1)*100 */
            float v = (rank==0) ? (float)(s+1) : (float)(s+1)*100.0f;
            for (int i = 0; i < NUMEL; i++) h_in[i] = __float2bfloat16(v);
        } else {
            /* Alternating: even elements one val, odd another */
            float v0 = (rank==0) ? 1.0f : 7.0f;
            float v1 = (rank==0) ? 3.0f : 11.0f;
            for (int i = 0; i < NUMEL; i++)
                h_in[i] = __float2bfloat16((i%2==0) ? v0*(s+1) : v1*(s+1));
        }
        cudaMemcpy(d_in, h_in, DATA_BYTES, cudaMemcpyHostToDevice);

        LAUNCH(d_in, d_out);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, DATA_BYTES, cudaMemcpyDeviceToHost);

        /* Compute expected and check */
        int errs = 0; float maxdiff = 0;
        for (int i = 0; i < NUMEL; i++) {
            float exp_val;
            if (s % 3 == 0)
                exp_val = (float)(i+1) * (s+1) * 11.0f;
            else if (s % 3 == 1)
                exp_val = (float)(s+1) * 101.0f;
            else
                exp_val = (i%2==0) ? 8.0f*(s+1) : 14.0f*(s+1);

            float got = __bfloat162float(h_out[i]);
            float diff = fabsf(got - exp_val);
            if (diff > maxdiff) maxdiff = diff;
            float thr = fabsf(exp_val) * 0.01f + 1.0f;
            if (diff > thr) errs++;
        }
        int show = (errs || s < 10 || s >= 95 || s == 49 || s == 50);
        if (show)
            printf("  step %3d [%s %s]: maxdiff=%8.1f errs=%d/%d %s\n",
                   s, (s%2==0)?"even":"odd ",
                   (s%3==0)?"asc ":(s%3==1)?"const":"alt  ",
                   maxdiff, errs, NUMEL, errs?"FAIL":"OK");
        if (errs) {
            total_fail++;
            for (int i = 0; i < NUMEL && i < 3; i++) {
                float got = __bfloat162float(h_out[i]);
                float exp_val;
                if (s%3==0) exp_val=(float)(i+1)*(s+1)*11.0f;
                else if (s%3==1) exp_val=(float)(s+1)*101.0f;
                else exp_val=(i%2==0)?8.0f*(s+1):14.0f*(s+1);
                printf("    [%d] got=%.1f exp=%.1f\n", i, got, exp_val);
            }
        }
    }
    printf("  TOTAL: %d/100 failed\n", total_fail);

    /* === PHASE 2: Latency — per-call kernel (5000 iters, continuous counter) === */
    printf("\n--- Phase 2: Per-call kernel latency ---\n");
    {
        for (int i = 0; i < NUMEL; i++) h_in[i] = __float2bfloat16((float)(i+1));
        cudaMemcpy(d_in, h_in, DATA_BYTES, cudaMemcpyHostToDevice);

        /* 200 warmup (counter continues from phase 1) */
        for (int i = 0; i < 200; i++) { LAUNCH(d_in, d_out); }
        cudaDeviceSynchronize();

        /* Timed */
        cudaEvent_t e0, e1; float ms;
        cudaEventCreate(&e0); cudaEventCreate(&e1);

        int N = 5000;
        double t0 = now_us();
        cudaEventRecord(e0);
        for (int i = 0; i < N; i++) { LAUNCH(d_in, d_out); }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        double wall = now_us() - t0;
        cudaEventElapsedTime(&ms, e0, e1);

        printf("  %d iters: %.1f µs/iter (wall), %.1f µs/iter (GPU events)\n",
               N, wall/N, ms*1000.0f/N);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    /* === PHASE 3: Latency — persistent kernel === */
    printf("\n--- Phase 3: Persistent kernel latency ---\n");
    {
        /* Need fresh flags for persistent kernel (it counts from 1) */
        /* Both ranks sync here via the step counter — we can't reset without barrier.
         * Instead, use the persistent kernel with its own independent flag set. */
        uint32_t *sf2, *rf2, *sd2;
        void *sb2, *rb2;
        uint32_t *fs2;
        cudaHostAlloc(&sb2, DATA_BYTES, 0); cudaHostAlloc(&rb2, DATA_BYTES, 0);
        cudaHostAlloc((void**)&sf2, 4, 0); cudaHostAlloc((void**)&rf2, 4, 0);
        cudaHostAlloc((void**)&sd2, 4, 0); cudaHostAlloc((void**)&fs2, 4, 0);
        memset(sb2,0,DATA_BYTES); memset(rb2,0,DATA_BYTES);
        *sf2=0; *rf2=0; *sd2=0; *fs2=0;

        int mf2 = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
        struct ibv_mr *ms2 = ibv_reg_mr(C.pd, sb2, DATA_BYTES, mf2);
        struct ibv_mr *mr2 = ibv_reg_mr(C.pd, rb2, DATA_BYTES, mf2);
        struct ibv_mr *mf2r = ibv_reg_mr(C.pd, rf2, 4, mf2);
        struct ibv_mr *mfs2 = ibv_reg_mr(C.pd, fs2, 4, mf2);

        if (!ms2 || !mr2 || !mf2r || !mfs2) {
            printf("  SKIP: could not register persistent MRs\n");
        } else {
            /* Need a second proxy for the persistent test's separate buffers.
             * Too complex for this test — skip persistent and just report per-call. */
            printf("  (Persistent kernel needs separate proxy — skipping for now)\n");
            printf("  Per-call result from Phase 2 is the production-relevant number.\n");
        }

        /* Cleanup persistent buffers */
        if (ms2) ibv_dereg_mr(ms2);
        if (mr2) ibv_dereg_mr(mr2);
        if (mf2r) ibv_dereg_mr(mf2r);
        if (mfs2) ibv_dereg_mr(mfs2);
        cudaFreeHost(sb2); cudaFreeHost(rb2);
        cudaFreeHost(sf2); cudaFreeHost(rf2);
        cudaFreeHost(sd2); cudaFreeHost(fs2);
    }

    printf("\n=== Summary ===\n");
    printf("  Correctness: %d/100 failed\n", total_fail);
    printf("  NCCL AllReduce 4K bf16: ~18 µs (reference)\n");
    printf("  bench_sys.cu local (no wire): 3.5 µs\n");

    __atomic_store_n(&C.run, 0, __ATOMIC_RELAXED);
    pthread_join(C.tid, NULL);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    printf("\n[R%d] Done\n", rank);
    return 0;
}
