#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#include <cuda_runtime.h>

static inline double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

/* GPU kernel: write data with st.global.sys, then signal flag */
extern "C" __global__
void kern_write_sys(uint32_t* buf, volatile uint32_t* flag, int n_words, int step)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        uint32_t val = (uint32_t)(i + 1) * (step + 1);
        asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(buf + i), "r"(val));
    }
    __syncthreads();
    __threadfence_system();
    if (idx == 0) atomicExch_system((uint32_t*)flag, (uint32_t)(step + 1));
}

/* GPU kernel: wait for flag, then read with ld.global.sys */
extern "C" __global__
void kern_read_sys(const uint32_t* buf, float* results,
                   volatile uint32_t* flag, int n_words, int step)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (atomicAdd_system((uint32_t*)flag, 0) < (uint32_t)(step + 1)) {}
    }
    __syncthreads();
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        uint32_t val;
        asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(val) : "l"(buf + i));
        uint32_t expected = (uint32_t)(i + 1) * (step + 1);
        results[i] = (float)((int)val - (int)expected);
    }
}

/* GPU kernel: full proxy sim - write sys, wait, read sys, reduce */
extern "C" __global__
void kern_proxy_sim(uint32_t* send_buf, const uint32_t* recv_buf,
                    uint32_t* output, const uint32_t* input,
                    volatile uint32_t* send_flag, volatile uint32_t* recv_flag,
                    int n_words, int n_iters)
{
    for (int step = 1; step <= n_iters; step++) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        /* write input to send buffer with sys-scope */
        for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
            uint32_t v = input[i];
            asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(send_buf + i), "r"(v));
        }
        __syncthreads();
        __threadfence_system();
        if (idx == 0) atomicExch_system((uint32_t*)send_flag, (uint32_t)step);

        /* wait for recv data */
        if (idx == 0) {
            while (atomicAdd_system((uint32_t*)recv_flag, 0) < (uint32_t)step) {}
        }
        __syncthreads();

        /* read recv with sys-scope and reduce */
        for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
            uint32_t recv_val;
            asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(recv_val) : "l"(recv_buf + i));
            output[i] = input[i] + recv_val;
        }
        __syncthreads();
    }
}

/* sys-scope write timing kernel */
extern "C" __global__
void kern_sys_write(uint32_t* buf, int n_words)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        asm volatile("st.release.sys.global.b32 [%0], %1;" :: "l"(buf + i), "r"(i));
    }
    __threadfence_system();
}

/* sys-scope read timing kernel */
extern "C" __global__
void kern_sys_read(const uint32_t* buf, uint32_t* out, int n_words)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t sum = 0;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        uint32_t v;
        asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(buf + i));
        sum += v;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

/* normal write kernel for comparison */
extern "C" __global__
void kern_normal_write(uint32_t* buf, int n_words)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x)
        buf[i] = i;
}

/* normal read kernel for comparison */
extern "C" __global__
void kern_normal_read(const uint32_t* buf, uint32_t* out, int n_words)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t sum = 0;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x)
        sum += buf[i];
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

/* ---- CPU proxy thread ---- */
struct ProxyArgs {
    uint32_t* send_buf;
    uint32_t* recv_buf;
    volatile uint32_t* send_flag;
    volatile uint32_t* recv_flag;
    int n_words;
    int n_iters;
};

static void* proxy_thread(void* arg)
{
    ProxyArgs* a = (ProxyArgs*)arg;
    for (int step = 1; step <= a->n_iters; step++) {
        while (__atomic_load_n((uint32_t*)a->send_flag, __ATOMIC_ACQUIRE) < (uint32_t)step) {}
        memcpy((void*)a->recv_buf, (void*)a->send_buf, a->n_words * sizeof(uint32_t));
        __sync_synchronize();
        __atomic_store_n((uint32_t*)a->recv_flag, (uint32_t)step, __ATOMIC_RELEASE);
    }
    return NULL;
}

int main()
{
    int numel = 4096;
    int n_words = numel / 2;
    int bytes = n_words * sizeof(uint32_t);

    printf("=== System-Scope Memory Tests (GB10 SM121) ===\n");
    printf("numel=%d, n_words=%d, bytes=%d\n", numel, n_words, bytes);

    /* ---- Test A: GPU write (sys) -> CPU read ---- */
    printf("\n--- Test A: GPU st.global.sys -> CPU read ---\n");
    {
        uint32_t* buf; volatile uint32_t* flag;
        cudaHostAlloc((void**)&buf, bytes, cudaHostAllocDefault);
        cudaHostAlloc((void**)&flag, 4, cudaHostAllocDefault);
        *flag = 0; memset((void*)buf, 0, bytes);
        int blocks = (n_words + 255) / 256;
        for (int step = 0; step < 5; step++) {
            kern_write_sys<<<blocks, 256>>>(buf, (uint32_t*)flag, n_words, step);
            cudaDeviceSynchronize();
            int err = 0;
            for (int i = 0; i < n_words; i++) {
                if (buf[i] != (uint32_t)(i+1)*(step+1)) err++;
            }
            printf("  step %d: errors=%d/%d %s\n", step, err, n_words, err?"FAIL":"OK");
        }
        cudaFreeHost((void*)buf); cudaFreeHost((void*)flag);
    }

    /* ---- Test B: CPU write -> GPU read (sys) ---- */
    printf("\n--- Test B: CPU write -> GPU ld.global.sys ---\n");
    {
        uint32_t* buf; volatile uint32_t* flag;
        float* d_res; float h_res[2048];
        cudaHostAlloc((void**)&buf, bytes, cudaHostAllocDefault);
        cudaHostAlloc((void**)&flag, 4, cudaHostAllocDefault);
        cudaMalloc(&d_res, n_words * sizeof(float));
        *flag = 0; memset((void*)buf, 0, bytes);
        int blocks = (n_words + 255) / 256;
        for (int step = 0; step < 5; step++) {
            for (int i = 0; i < n_words; i++)
                buf[i] = (uint32_t)(i+1) * (step+1);
            __sync_synchronize();
            __atomic_store_n((uint32_t*)flag, (uint32_t)(step+1), __ATOMIC_RELEASE);
            kern_read_sys<<<blocks, 256>>>(buf, d_res, flag, n_words, step);
            cudaDeviceSynchronize();
            cudaMemcpy(h_res, d_res, n_words*sizeof(float), cudaMemcpyDeviceToHost);
            int err = 0;
            for (int i = 0; i < n_words; i++)
                if (fabsf(h_res[i]) > 0) err++;
            printf("  step %d: errors=%d/%d %s\n", step, err, n_words, err?"FAIL":"OK");
        }
        cudaFree(d_res);
        cudaFreeHost((void*)buf); cudaFreeHost((void*)flag);
    }

    /* ---- Timing: sys vs normal ---- */
    printf("\n--- Timing: sys-scope vs normal (pinned mem) ---\n");
    {
        uint32_t* buf; uint32_t* d_buf; uint32_t* d_out;
        cudaHostAlloc((void**)&buf, 32768, cudaHostAllocDefault);
        cudaMalloc(&d_buf, 32768);
        cudaMalloc(&d_out, 256*4);
        cudaEvent_t t0, t1; float ms;
        cudaEventCreate(&t0); cudaEventCreate(&t1);

        int sizes[] = {2048, 4096};
        for (int s = 0; s < 2; s++) {
            int nw = sizes[s];
            int bl = (nw + 255) / 256;
            int N = 10000;

            cudaEventRecord(t0);
            for (int i = 0; i < N; i++) kern_sys_write<<<bl, 256>>>(buf, nw);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms, t0, t1);
            float sw = ms * 1000.0f / N;

            cudaEventRecord(t0);
            for (int i = 0; i < N; i++) kern_normal_write<<<bl, 256>>>(d_buf, nw);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms, t0, t1);
            float nw2 = ms * 1000.0f / N;

            cudaEventRecord(t0);
            for (int i = 0; i < N; i++) kern_sys_read<<<bl, 256>>>(buf, d_out, nw);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms, t0, t1);
            float sr = ms * 1000.0f / N;

            cudaEventRecord(t0);
            for (int i = 0; i < N; i++) kern_normal_read<<<bl, 256>>>(d_buf, d_out, nw);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms, t0, t1);
            float nr = ms * 1000.0f / N;

            printf("  %d words (%dKB): sys_wr=%.1f norm_wr=%.1f  sys_rd=%.1f norm_rd=%.1f µs\n",
                   nw, nw*4/1024, sw, nw2, sr, nr);
        }
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFreeHost((void*)buf); cudaFree(d_buf); cudaFree(d_out);
    }

    /* ---- Full proxy sim with sys-scope (no cudaMemcpy) ---- */
    printf("\n--- Proxy sim: sys-scope (no cudaMemcpy, no network) ---\n");
    {
        uint32_t *sbuf, *rbuf, *d_in, *d_out;
        volatile uint32_t *sf, *rf;
        cudaHostAlloc((void**)&sbuf, bytes, cudaHostAllocDefault);
        cudaHostAlloc((void**)&rbuf, bytes, cudaHostAllocDefault);
        cudaHostAlloc((void**)&sf, 4, cudaHostAllocDefault);
        cudaHostAlloc((void**)&rf, 4, cudaHostAllocDefault);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        *sf = 0; *rf = 0;
        memset((void*)sbuf, 0, bytes);
        memset((void*)rbuf, 0, bytes);
        uint32_t* h = (uint32_t*)malloc(bytes);
        for (int i = 0; i < n_words; i++) h[i] = i + 1;
        cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);
        int blocks = (n_words + 255) / 256;

        /* warmup */
        ProxyArgs wa = {sbuf, rbuf, sf, rf, n_words, 200};
        pthread_t tid;
        pthread_create(&tid, NULL, proxy_thread, &wa);
        kern_proxy_sim<<<blocks, 256>>>(sbuf, rbuf, d_out, d_in, sf, rf, n_words, 200);
        cudaDeviceSynchronize();
        pthread_join(tid, NULL);

        /* bench */
        int N = 5000;
        *sf = 0; *rf = 0;
        ProxyArgs ba = {sbuf, rbuf, sf, rf, n_words, N};
        pthread_create(&tid, NULL, proxy_thread, &ba);
        double start = now_us();
        kern_proxy_sim<<<blocks, 256>>>(sbuf, rbuf, d_out, d_in, sf, rf, n_words, N);
        cudaDeviceSynchronize();
        double elapsed = now_us() - start;
        pthread_join(tid, NULL);

        double per = elapsed / N;
        printf("  4K elem (8KB): %.1f µs/iter (no network)\n", per);
        printf("  + ~3 µs wire = ~%.1f µs total\n", per + 3.0);
        printf("  vs cudaMemcpy proxy: 15.4 µs\n");
        printf("  vs NCCL AllReduce:   18.0 µs\n");

        /* verify */
        cudaMemcpy(h, d_out, bytes, cudaMemcpyDeviceToHost);
        int err = 0;
        for (int i = 0; i < n_words; i++)
            if (h[i] != (uint32_t)(i+1)*2) err++;
        printf("  Correctness: %d/%d %s\n", err, n_words, err?"FAIL":"OK");

        free(h);
        cudaFreeHost((void*)sbuf); cudaFreeHost((void*)rbuf);
        cudaFreeHost((void*)sf); cudaFreeHost((void*)rf);
        cudaFree(d_in); cudaFree(d_out);
    }

    printf("\n=== Done ===\n");
    return 0;
}
