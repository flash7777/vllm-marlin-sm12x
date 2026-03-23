// Test: cudaHostAlloc memory coherence between GPU writes and CPU reads on GB10
// This validates the key assumption: pinned host memory bypasses GPU L2 cache
// Compile: nvcc -O2 -arch=sm_121 -o test_pinned_coherence test_pinned_coherence.cu
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CHECK_CUDA(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        printf("CUDA error %d: %s @ %s:%d\n", e, cudaGetErrorString(e), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

// GPU kernel: write a pattern to pinned memory
__global__ void write_pattern(volatile __nv_bfloat16* buf, int numel, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Write a unique value per element per step
        float val = (float)(idx + 1) * (step + 1) * 0.01f;
        buf[idx] = __float2bfloat16(val);
    }
    // System fence to ensure visibility
    __threadfence_system();
}

// GPU kernel: read from pinned memory and write result to device memory
__global__ void read_and_check(volatile __nv_bfloat16* pinned_buf, float* results,
                                int numel, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float expected = (float)(idx + 1) * (step + 1) * 0.01f;
        float got = __bfloat162float(pinned_buf[idx]);
        results[idx] = got - expected;
    }
}

int main() {
    const int numel = 4096;  // 8KB BF16
    const int bytes = numel * sizeof(__nv_bfloat16);

    printf("=== GB10 Pinned Memory Coherence Test ===\n");
    printf("Testing cudaHostAlloc memory for GPU↔CPU coherence\n");
    printf("numel=%d, bytes=%d\n\n", numel, bytes);

    // Allocate pinned host memory (this is the key!)
    __nv_bfloat16* pinned_buf = nullptr;
    CHECK_CUDA(cudaHostAlloc(&pinned_buf, bytes, cudaHostAllocDefault));
    printf("cudaHostAlloc: %p (%d bytes)\n", pinned_buf, bytes);

    // Also allocate a regular cudaMalloc buffer for comparison
    __nv_bfloat16* device_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&device_buf, bytes));

    // Allocate results buffer on device
    float* d_results = nullptr;
    CHECK_CUDA(cudaMalloc(&d_results, numel * sizeof(float)));
    float* h_results = (float*)malloc(numel * sizeof(float));

    int blocks = (numel + 255) / 256;

    // Test 1: GPU writes to pinned memory, CPU reads
    printf("\n--- Test 1: GPU write → CPU read (pinned memory) ---\n");
    for (int step = 0; step < 10; step++) {
        // GPU writes pattern
        write_pattern<<<blocks, 256>>>(pinned_buf, numel, step);
        CHECK_CUDA(cudaDeviceSynchronize());

        // CPU reads and checks
        float maxdiff = 0;
        int errors = 0;
        for (int i = 0; i < numel; i++) {
            float expected = (float)(i + 1) * (step + 1) * 0.01f;
            float got = __bfloat162float(pinned_buf[i]);
            float diff = fabsf(got - expected);
            if (diff > maxdiff) maxdiff = diff;
            // BF16 has ~0.8% relative error, so use generous threshold
            if (diff > fabsf(expected) * 0.02f + 0.01f) errors++;
        }
        printf("  step %d: maxdiff=%.6f, errors=%d/%d %s\n",
               step, maxdiff, errors, numel, errors == 0 ? "OK" : "FAIL");
    }

    // Test 2: CPU writes to pinned memory, GPU reads
    printf("\n--- Test 2: CPU write → GPU read (pinned memory) ---\n");
    for (int step = 0; step < 10; step++) {
        // CPU writes pattern
        for (int i = 0; i < numel; i++) {
            float val = (float)(i + 1) * (step + 100) * 0.01f;
            pinned_buf[i] = __float2bfloat16(val);
        }

        // GPU reads and checks difference
        read_and_check<<<blocks, 256>>>(pinned_buf, d_results, numel, step + 99);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_results, d_results, numel * sizeof(float), cudaMemcpyDeviceToHost));

        float maxdiff = 0;
        int errors = 0;
        for (int i = 0; i < numel; i++) {
            float diff = fabsf(h_results[i]);
            if (diff > maxdiff) maxdiff = diff;
            if (diff > 0.1f) errors++;
        }
        printf("  step %d: maxdiff=%.6f, errors=%d/%d %s\n",
               step, maxdiff, errors, numel, errors == 0 ? "OK" : "FAIL");
    }

    // Test 3: GPU writes, then immediately GPU reads back (same pinned buffer)
    printf("\n--- Test 3: GPU write → GPU read back (pinned, no CPU involvement) ---\n");
    for (int step = 0; step < 10; step++) {
        write_pattern<<<blocks, 256>>>(pinned_buf, numel, step);
        read_and_check<<<blocks, 256>>>(pinned_buf, d_results, numel, step);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_results, d_results, numel * sizeof(float), cudaMemcpyDeviceToHost));

        float maxdiff = 0;
        int errors = 0;
        for (int i = 0; i < numel; i++) {
            float diff = fabsf(h_results[i]);
            if (diff > maxdiff) maxdiff = diff;
            if (diff > 0.1f) errors++;
        }
        printf("  step %d: maxdiff=%.6f, errors=%d/%d %s\n",
               step, maxdiff, errors, numel, errors == 0 ? "OK" : "FAIL");
    }

    // Test 4: Timing — GPU write to pinned vs device memory
    printf("\n--- Test 4: Latency — pinned vs device memory writes ---\n");
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int n_iter = 10000;

    // Pinned memory writes
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iter; i++) {
        write_pattern<<<blocks, 256>>>(pinned_buf, numel, i);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float pinned_ms;
    CHECK_CUDA(cudaEventElapsedTime(&pinned_ms, start, stop));

    // Device memory writes
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iter; i++) {
        write_pattern<<<blocks, 256>>>(device_buf, numel, i);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float device_ms;
    CHECK_CUDA(cudaEventElapsedTime(&device_ms, start, stop));

    printf("  Pinned writes: %.3f ms / %d = %.2f µs/call\n",
           pinned_ms, n_iter, pinned_ms * 1000.0f / n_iter);
    printf("  Device writes: %.3f ms / %d = %.2f µs/call\n",
           device_ms, n_iter, device_ms * 1000.0f / n_iter);
    printf("  Ratio: %.2fx\n", pinned_ms / device_ms);

    // Cleanup
    cudaFree(d_results);
    cudaFree(device_buf);
    cudaFreeHost(pinned_buf);
    free(h_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== Done ===\n");
    return 0;
}
