// Test: system-scope loads/stores for bulk data transfer GPU↔CPU
// If this works, we can eliminate cudaMemcpyAsync (~9 µs overhead)
// and use direct coherent memory access (~1-2 µs)
// Compile: nvcc -O2 -arch=sm_121 -o bench_sys_scope_data bench_sys_scope_data.cu -lpthread
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define CHECK_CUDA(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        printf("CUDA error: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static inline double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// =============================================
// GPU kernel: write data using system-scope stores + fence + flag
// =============================================
__global__ void gpu_write_sys(uint32_t* buf, volatile uint32_t* flag,
                               int n_words, int step) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Write data with system-scope stores
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        uint32_t val = (uint32_t)(i + 1) * (step + 1);
        // st.global.sys — system-scope store
        asm volatile("st.global.sys.b32 [%0], %1;" :: "l"(buf + i), "r"(val));
    }
    __syncthreads();

    // System fence to ensure all stores are visible
    __threadfence_system();

    // Signal flag (system-scope atomic)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicExch_system((uint32_t*)flag, (uint32_t)(step + 1));
    }
}

// =============================================
// GPU kernel: read data using system-scope loads after CPU wrote
// =============================================
__global__ void gpu_read_sys(const uint32_t* buf, float* results,
                              volatile uint32_t* flag, int n_words, int step) {
    // Wait for CPU signal
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (atomicAdd_system((uint32_t*)flag, 0) < (uint32_t)(step + 1)) {}
    }
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Read data with system-scope loads
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        uint32_t val;
        // ld.global.sys — system-scope load
        asm volatile("ld.global.sys.b32 %0, [%1];" : "=r"(val) : "l"(buf + i));
        uint32_t expected = (uint32_t)(i + 1) * (step + 1);
        results[i] = (float)(int)(val - expected);  // difference
    }
}

// =============================================
// Test 1: GPU writes (sys-scope) → CPU reads
// =============================================
void test_gpu_write_cpu_read() {
    printf("\n--- Test A: GPU write (st.global.sys) → CPU read ---\n");

    int numel = 4096;  // 8KB BF16 = 4096 uint16 = 2048 uint32
    int n_words = numel / 2;  // uint32 words
    int bytes = n_words * sizeof(uint32_t);

    uint32_t* buf;
    volatile uint32_t* flag;
    CHECK_CUDA(cudaHostAlloc((void**)&buf, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&flag, sizeof(uint32_t), cudaHostAllocDefault));
    *flag = 0;
    memset((void*)buf, 0, bytes);

    int blocks = (n_words + 255) / 256;

    for (int step = 0; step < 10; step++) {
        // GPU writes with system-scope stores
        gpu_write_sys<<<blocks, 256>>>(buf, (uint32_t*)flag, n_words, step);
        CHECK_CUDA(cudaDeviceSynchronize());

        // CPU reads and verifies
        int errors = 0;
        for (int i = 0; i < n_words; i++) {
            uint32_t expected = (uint32_t)(i + 1) * (step + 1);
            if (buf[i] != expected) errors++;
        }
        printf("  step %d: errors=%d/%d %s\n", step, errors, n_words,
               errors == 0 ? "OK" : "FAIL");
    }

    cudaFreeHost((void*)buf);
    cudaFreeHost((void*)flag);
}

// =============================================
// Test 2: CPU writes → GPU reads (ld.global.sys)
// =============================================
struct CPUWriteArgs {
    uint32_t* buf;
    volatile uint32_t* flag;
    int n_words;
    int n_steps;
};

void* cpu_writer(void* arg) {
    CPUWriteArgs* a = (CPUWriteArgs*)arg;
    for (int step = 0; step < a->n_steps; step++) {
        // Wait for GPU to be ready (step 0 is always ready)
        // CPU writes data
        for (int i = 0; i < a->n_words; i++) {
            a->buf[i] = (uint32_t)(i + 1) * (step + 1);
        }
        // Memory barrier
        __sync_synchronize();
        // Signal GPU
        __atomic_store_n((uint32_t*)a->flag, (uint32_t)(step + 1), __ATOMIC_RELEASE);
    }
    return nullptr;
}

void test_cpu_write_gpu_read() {
    printf("\n--- Test B: CPU write → GPU read (ld.global.sys) ---\n");

    int numel = 4096;
    int n_words = numel / 2;
    int bytes = n_words * sizeof(uint32_t);

    uint32_t* buf;
    volatile uint32_t* flag;
    float* d_results;
    float* h_results;

    CHECK_CUDA(cudaHostAlloc((void**)&buf, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&flag, sizeof(uint32_t), cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(&d_results, n_words * sizeof(float)));
    h_results = (float*)malloc(n_words * sizeof(float));
    *flag = 0;
    memset((void*)buf, 0, bytes);

    int blocks = (n_words + 255) / 256;

    for (int step = 0; step < 10; step++) {
        // CPU writes data + sets flag
        for (int i = 0; i < n_words; i++) {
            buf[i] = (uint32_t)(i + 1) * (step + 1);
        }
        __sync_synchronize();
        __atomic_store_n((uint32_t*)flag, (uint32_t)(step + 1), __ATOMIC_RELEASE);

        // GPU reads with system-scope loads
        gpu_read_sys<<<blocks, 256>>>(buf, d_results, flag, n_words, step);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Check results
        CHECK_CUDA(cudaMemcpy(h_results, d_results, n_words * sizeof(float), cudaMemcpyDeviceToHost));
        int errors = 0;
        float maxdiff = 0;
        for (int i = 0; i < n_words; i++) {
            if (fabsf(h_results[i]) > 0) errors++;
            if (fabsf(h_results[i]) > maxdiff) maxdiff = fabsf(h_results[i]);
        }
        printf("  step %d: errors=%d/%d maxdiff=%.0f %s\n",
               step, errors, n_words, maxdiff, errors == 0 ? "OK" : "FAIL");
    }

    cudaFree(d_results);
    cudaFreeHost((void*)buf);
    cudaFreeHost((void*)flag);
    free(h_results);
}

// =============================================
// Test 3: Timing — system-scope bulk read/write vs cudaMemcpy
// =============================================
__global__ void sys_write_bulk(uint32_t* buf, int n_words) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        asm volatile("st.global.sys.b32 [%0], %1;" :: "l"(buf + i), "r"(i));
    }
    __threadfence_system();
}

__global__ void sys_read_bulk(const uint32_t* buf, uint32_t* out, int n_words) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t sum = 0;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        uint32_t val;
        asm volatile("ld.global.sys.b32 %0, [%1];" : "=r"(val) : "l"(buf + i));
        sum += val;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

__global__ void normal_write_bulk(uint32_t* buf, int n_words) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        buf[i] = i;
    }
}

__global__ void normal_read_bulk(const uint32_t* buf, uint32_t* out, int n_words) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t sum = 0;
    for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
        sum += buf[i];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

void bench_sys_scope_throughput() {
    printf("\n--- Timing: system-scope vs normal memory access ---\n");

    int sizes[] = {2048, 4096, 8192, 16384};  // uint32 words
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    uint32_t* buf;
    uint32_t* d_out;
    CHECK_CUDA(cudaHostAlloc((void**)&buf, 65536 * 4, cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(&d_out, 256 * sizeof(uint32_t)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int s = 0; s < n_sizes; s++) {
        int n_words = sizes[s];
        int bytes = n_words * sizeof(uint32_t);
        int blocks = (n_words + 255) / 256;
        int n_iter = 10000;
        float ms;

        // System-scope write
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < n_iter; i++)
            sys_write_bulk<<<blocks, 256>>>(buf, n_words);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float sys_write = ms * 1000.0f / n_iter;

        // Normal write (for comparison)
        uint32_t* d_buf;
        CHECK_CUDA(cudaMalloc(&d_buf, bytes));
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < n_iter; i++)
            normal_write_bulk<<<blocks, 256>>>(d_buf, n_words);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float normal_write = ms * 1000.0f / n_iter;

        // System-scope read
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < n_iter; i++)
            sys_read_bulk<<<blocks, 256>>>(buf, d_out, n_words);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float sys_read = ms * 1000.0f / n_iter;

        // Normal read
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < n_iter; i++)
            normal_read_bulk<<<blocks, 256>>>(d_buf, d_out, n_words);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float normal_read = ms * 1000.0f / n_iter;

        printf("  %5d words (%d KB): sys_wr=%.1f µs, norm_wr=%.1f µs (%.1fx)  "
               "sys_rd=%.1f µs, norm_rd=%.1f µs (%.1fx)\n",
               n_words, bytes / 1024, sys_write, normal_write, sys_write / normal_write,
               sys_read, normal_read, sys_read / normal_read);

        cudaFree(d_buf);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost((void*)buf);
    cudaFree(d_out);
}

// =============================================
// Test 4: Full simulated proxy with system-scope (no cudaMemcpy)
// GPU writes to shared buf (sys-scope) → CPU reads → CPU writes → GPU reads (sys-scope) → reduce
// =============================================
__global__ void proxy_sim_gpu(uint32_t* send_buf, const uint32_t* recv_buf,
                               uint32_t* output, const uint32_t* input,
                               volatile uint32_t* send_flag, volatile uint32_t* recv_flag,
                               int n_words, int n_iters) {
    for (int step = 1; step <= n_iters; step++) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Write input to send buffer (system-scope)
        for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
            uint32_t val = input[i];
            asm volatile("st.global.sys.b32 [%0], %1;" :: "l"(send_buf + i), "r"(val));
        }
        __syncthreads();
        __threadfence_system();

        // Signal CPU: data ready
        if (idx == 0) atomicExch_system((uint32_t*)send_flag, (uint32_t)step);

        // Wait for CPU to provide recv data
        if (idx == 0) {
            while (atomicAdd_system((uint32_t*)recv_flag, 0) < (uint32_t)step) {}
        }
        __syncthreads();

        // Read recv buffer (system-scope) and reduce
        for (int i = idx; i < n_words; i += blockDim.x * gridDim.x) {
            uint32_t local_val = input[i];
            uint32_t recv_val;
            asm volatile("ld.global.sys.b32 %0, [%1];" : "=r"(recv_val) : "l"(recv_buf + i));
            // For bf16: just store recv_val as-is (we're testing coherence, not math)
            output[i] = local_val + recv_val;
        }
        __syncthreads();
    }
}

struct ProxySimArgs {
    uint32_t* send_buf;
    uint32_t* recv_buf;
    volatile uint32_t* send_flag;
    volatile uint32_t* recv_flag;
    int n_words;
    int n_iters;
};

void* proxy_sim_cpu(void* arg) {
    ProxySimArgs* a = (ProxySimArgs*)arg;
    for (int step = 1; step <= a->n_iters; step++) {
        // Wait for GPU send signal
        while (__atomic_load_n((uint32_t*)a->send_flag, __ATOMIC_ACQUIRE) < (uint32_t)step) {}

        // "Process" data (simulate: read send buf, write to recv buf)
        // In real proxy: read send_buf → ibv_post_send → remote CQ → recv_buf
        memcpy((void*)a->recv_buf, (void*)a->send_buf, a->n_words * sizeof(uint32_t));
        __sync_synchronize();

        // Signal GPU: recv data ready
        __atomic_store_n((uint32_t*)a->recv_flag, (uint32_t)step, __ATOMIC_RELEASE);
    }
    return nullptr;
}

void bench_proxy_sys_scope() {
    printf("\n--- Full Proxy Sim with system-scope (no cudaMemcpy) ---\n");

    int numel = 4096;
    int n_words = numel / 2;  // uint32
    int bytes = n_words * sizeof(uint32_t);

    uint32_t *send_buf, *recv_buf;
    volatile uint32_t *send_flag, *recv_flag;
    uint32_t *d_input, *d_output;

    CHECK_CUDA(cudaHostAlloc((void**)&send_buf, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&recv_buf, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&send_flag, sizeof(uint32_t), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&recv_flag, sizeof(uint32_t), cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    *send_flag = 0;
    *recv_flag = 0;
    memset((void*)send_buf, 0, bytes);
    memset((void*)recv_buf, 0, bytes);

    // Init input
    uint32_t* h_tmp = (uint32_t*)malloc(bytes);
    for (int i = 0; i < n_words; i++) h_tmp[i] = i + 1;
    CHECK_CUDA(cudaMemcpy(d_input, h_tmp, bytes, cudaMemcpyHostToDevice));

    int blocks = (n_words + 255) / 256;
    int n_iters = 5000;

    // Warmup
    *send_flag = 0;
    *recv_flag = 0;
    ProxySimArgs warmup_args = { send_buf, recv_buf, send_flag, recv_flag, n_words, 200 };
    pthread_t tid;
    pthread_create(&tid, nullptr, proxy_sim_cpu, &warmup_args);
    proxy_sim_gpu<<<blocks, 256>>>(send_buf, recv_buf, d_output, d_input,
                                    send_flag, recv_flag, n_words, 200);
    CHECK_CUDA(cudaDeviceSynchronize());
    pthread_join(tid, nullptr);

    // Benchmark
    *send_flag = 0;
    *recv_flag = 0;
    ProxySimArgs args = { send_buf, recv_buf, send_flag, recv_flag, n_words, n_iters };
    pthread_create(&tid, nullptr, proxy_sim_cpu, &args);

    double t0 = now_us();
    proxy_sim_gpu<<<blocks, 256>>>(send_buf, recv_buf, d_output, d_input,
                                    send_flag, recv_flag, n_words, n_iters);
    CHECK_CUDA(cudaDeviceSynchronize());
    double elapsed = now_us() - t0;
    pthread_join(tid, nullptr);

    double per_iter = elapsed / n_iters;
    printf("  4K elem (8KB): sys-scope roundtrip = %.1f µs/call (no network)\n", per_iter);
    printf("  (Add ~3 µs for RDMA wire = ~%.1f µs total)\n", per_iter + 3.0);
    printf("  vs cudaMemcpy proxy: 15.4 µs (no network)\n");
    printf("  vs NCCL AllReduce: 18 µs\n");

    // Verify correctness
    CHECK_CUDA(cudaMemcpy(h_tmp, d_output, bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n_words; i++) {
        uint32_t expected = (i + 1) * 2;  // input + recv (which = input via memcpy)
        if (h_tmp[i] != expected) errors++;
    }
    printf("  Correctness: errors=%d/%d %s\n", errors, n_words,
           errors == 0 ? "OK" : "FAIL");

    free(h_tmp);
    cudaFreeHost((void*)send_buf);
    cudaFreeHost((void*)recv_buf);
    cudaFreeHost((void*)send_flag);
    cudaFreeHost((void*)recv_flag);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    printf("=== System-Scope Memory Access Tests (GB10 SM121) ===\n");

    test_gpu_write_cpu_read();
    test_cpu_write_gpu_read();
    bench_sys_scope_throughput();
    bench_proxy_sys_scope();

    return 0;
}
