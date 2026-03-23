// Benchmark individual proxy component latencies on GB10
// Components: cudaMemcpy D2H/H2D, system-scope atomics, kernel launch
// Compile: nvcc -O2 -arch=sm_121 -o bench_proxy_components bench_proxy_components.cu -lpthread
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

#define CHECK_CUDA(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        printf("CUDA error %d: %s @ %s:%d\n", e, cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static inline double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// =============================================
// Component 1: cudaMemcpyAsync D2H / H2D timing
// =============================================
void bench_memcpy() {
    printf("\n=== Component: cudaMemcpyAsync D2H / H2D ===\n");

    int sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < n_sizes; s++) {
        int bytes = sizes[s] * sizeof(__nv_bfloat16);
        __nv_bfloat16* d_buf;
        __nv_bfloat16* h_buf;
        CHECK_CUDA(cudaMalloc(&d_buf, bytes));
        CHECK_CUDA(cudaMallocHost(&h_buf, bytes));

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));

        // Warmup
        for (int i = 0; i < 100; i++) {
            cudaMemcpyAsync(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        // Benchmark D2H
        int n_iter = 5000;
        double t0 = now_us();
        for (int i = 0; i < n_iter; i++) {
            cudaMemcpyAsync(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        double d2h_us = (now_us() - t0) / n_iter;

        // Benchmark H2D
        t0 = now_us();
        for (int i = 0; i < n_iter; i++) {
            cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        double h2d_us = (now_us() - t0) / n_iter;

        printf("  %5d elem (%d KB): D2H=%.1f µs, H2D=%.1f µs, roundtrip=%.1f µs\n",
               sizes[s], bytes/1024, d2h_us, h2d_us, d2h_us + h2d_us);

        cudaStreamDestroy(stream);
        cudaFree(d_buf);
        cudaFreeHost(h_buf);
    }
}

// =============================================
// Component 2: GPU↔CPU flag signaling via system-scope atomics
// =============================================

// GPU kernel: write flag, wait for CPU to respond
__global__ void gpu_signal_and_wait(volatile uint32_t* gpu_flag, volatile uint32_t* cpu_flag,
                                     int n_roundtrips) {
    for (int i = 1; i <= n_roundtrips; i++) {
        // Signal CPU
        atomicExch_system((uint32_t*)gpu_flag, (uint32_t)i);
        // Wait for CPU response
        while (atomicAdd_system((uint32_t*)cpu_flag, 0) < (uint32_t)i) {}
    }
}

struct SignalArgs {
    volatile uint32_t* gpu_flag;
    volatile uint32_t* cpu_flag;
    int n_roundtrips;
};

void* cpu_responder(void* arg) {
    SignalArgs* a = (SignalArgs*)arg;
    for (int i = 1; i <= a->n_roundtrips; i++) {
        // Wait for GPU signal
        while (__atomic_load_n((uint32_t*)a->gpu_flag, __ATOMIC_ACQUIRE) < (uint32_t)i) {}
        // Respond to GPU
        __atomic_store_n((uint32_t*)a->cpu_flag, (uint32_t)i, __ATOMIC_RELEASE);
    }
    return nullptr;
}

void bench_flag_signaling() {
    printf("\n=== Component: GPU↔CPU Flag Roundtrip (system-scope atomics) ===\n");

    // Allocate flags in pinned memory (accessible by both GPU and CPU)
    volatile uint32_t* gpu_flag;
    volatile uint32_t* cpu_flag;
    CHECK_CUDA(cudaHostAlloc((void**)&gpu_flag, sizeof(uint32_t), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&cpu_flag, sizeof(uint32_t), cudaHostAllocDefault));
    *gpu_flag = 0;
    *cpu_flag = 0;

    int n_roundtrips = 10000;

    // Start CPU responder thread
    SignalArgs args = { gpu_flag, cpu_flag, n_roundtrips };
    pthread_t tid;
    pthread_create(&tid, nullptr, cpu_responder, &args);

    // Run GPU kernel
    double t0 = now_us();
    gpu_signal_and_wait<<<1, 1>>>(gpu_flag, cpu_flag, n_roundtrips);
    CHECK_CUDA(cudaDeviceSynchronize());
    double elapsed = now_us() - t0;

    pthread_join(tid, nullptr);

    printf("  %d roundtrips: %.1f µs total, %.2f µs/roundtrip\n",
           n_roundtrips, elapsed, elapsed / n_roundtrips);

    cudaFreeHost((void*)gpu_flag);
    cudaFreeHost((void*)cpu_flag);
}

// =============================================
// Component 3: cudaMemcpy overhead for flag-only (1 word)
// =============================================
void bench_flag_via_memcpy() {
    printf("\n=== Component: Flag via cudaMemcpy (1 word) ===\n");

    uint32_t* d_flag;
    uint32_t* h_flag;
    CHECK_CUDA(cudaMalloc(&d_flag, sizeof(uint32_t)));
    CHECK_CUDA(cudaMallocHost(&h_flag, sizeof(uint32_t)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < 100; i++) {
        cudaMemcpyAsync(h_flag, d_flag, 4, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    int n_iter = 10000;
    double t0 = now_us();
    for (int i = 0; i < n_iter; i++) {
        cudaMemcpyAsync(h_flag, d_flag, 4, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    double d2h_us = (now_us() - t0) / n_iter;

    t0 = now_us();
    for (int i = 0; i < n_iter; i++) {
        cudaMemcpyAsync(d_flag, h_flag, 4, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    double h2d_us = (now_us() - t0) / n_iter;

    printf("  D2H (4 bytes): %.1f µs\n", d2h_us);
    printf("  H2D (4 bytes): %.1f µs\n", h2d_us);

    cudaStreamDestroy(stream);
    cudaFree(d_flag);
    cudaFreeHost(h_flag);
}

// =============================================
// Component 4: Kernel launch overhead
// =============================================
__global__ void noop_kernel() {}

void bench_kernel_launch() {
    printf("\n=== Component: Kernel Launch Overhead ===\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < 100; i++) {
        noop_kernel<<<1, 1, 0, stream>>>();
        cudaStreamSynchronize(stream);
    }

    int n_iter = 10000;
    double t0 = now_us();
    for (int i = 0; i < n_iter; i++) {
        noop_kernel<<<1, 1, 0, stream>>>();
        cudaStreamSynchronize(stream);
    }
    double us = (now_us() - t0) / n_iter;

    printf("  Launch + sync: %.1f µs/call\n", us);

    cudaStreamDestroy(stream);
}

// =============================================
// Component 5: Full proxy simulation (no network)
// GPU writes data → D2H → CPU processes → H2D → GPU reads
// =============================================

__global__ void write_data(volatile __nv_bfloat16* buf, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) buf[idx] = __float2bfloat16(1.0f);
}

__global__ void reduce_data(__nv_bfloat16* output, const __nv_bfloat16* input,
                             const __nv_bfloat16* recv, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float a = __bfloat162float(input[idx]);
        float b = __bfloat162float(recv[idx]);
        output[idx] = __float2bfloat16(a + b);
    }
}

void bench_proxy_sim() {
    printf("\n=== Full Proxy Simulation (no network, local roundtrip) ===\n");

    int numel = 4096;
    int bytes = numel * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_input, *d_output, *d_recv;
    __nv_bfloat16 *h_send, *h_recv;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMalloc(&d_recv, bytes));
    CHECK_CUDA(cudaMallocHost(&h_send, bytes));
    CHECK_CUDA(cudaMallocHost(&h_recv, bytes));

    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));

    int blocks = (numel + 255) / 256;

    // Warmup
    for (int i = 0; i < 100; i++) {
        cudaMemcpyAsync(h_send, d_input, bytes, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        // Simulate "network" - just memcpy
        memcpy(h_recv, h_send, bytes);
        cudaMemcpyAsync(d_recv, h_recv, bytes, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
        reduce_data<<<blocks, 256, 0, s>>>(d_output, d_input, d_recv, numel);
        cudaStreamSynchronize(s);
    }

    int n_iter = 5000;
    double t0 = now_us();
    for (int i = 0; i < n_iter; i++) {
        // Step 1: D2H (GPU send buffer → host)
        cudaMemcpyAsync(h_send, d_input, bytes, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        // Step 2: CPU "processes" (simulates proxy reading data)
        memcpy(h_recv, h_send, bytes);
        // Step 3: H2D (host recv → GPU recv buffer)
        cudaMemcpyAsync(d_recv, h_recv, bytes, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
        // Step 4: GPU reduce
        reduce_data<<<blocks, 256, 0, s>>>(d_output, d_input, d_recv, numel);
        cudaStreamSynchronize(s);
    }
    double total_us = (now_us() - t0) / n_iter;

    printf("  4K elem (8KB): D2H + memcpy + H2D + reduce = %.1f µs/call\n", total_us);
    printf("  (Add ~3 µs for RDMA wire time = ~%.1f µs total estimate)\n", total_us + 3.0);

    cudaStreamDestroy(s);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_recv);
    cudaFreeHost(h_send);
    cudaFreeHost(h_recv);
}

int main() {
    printf("=== Proxy Component Benchmarks (GB10 SM121) ===\n");

    bench_memcpy();
    bench_flag_signaling();
    bench_flag_via_memcpy();
    bench_kernel_launch();
    bench_proxy_sim();

    printf("\n=== Summary ===\n");
    printf("These are the irreducible component costs.\n");
    printf("A minimal proxy = flag roundtrip + cudaMemcpy D2H + ibv_post_send + wire + CQ poll + cudaMemcpy H2D + reduce\n");
    printf("NCCL AllReduce 4K: ~18 µs for reference\n");
    return 0;
}
