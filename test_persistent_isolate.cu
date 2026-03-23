/*
 * Isolate exactly what kills the persistent kernel on GB10:
 * Test each CUDA API call independently while persistent kernel runs.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

struct uf19v5_ctrl {
    uint64_t input_ptr;
    uint64_t output_ptr;
    uint32_t n_words;
    uint32_t work_step;
    uint32_t kernel_done;
    uint32_t shutdown;
};

extern "C" __global__
void test_persistent_kernel(struct uf19v5_ctrl* ctrl)
{
    int tid = threadIdx.x;
    __shared__ uint32_t s_step;
    uint32_t last_step = 0;

    if (tid == 0) {
        asm volatile("st.release.sys.global.b32 [%0], %1;"
            :: "l"(&ctrl->kernel_done), "r"((uint32_t)0xABCD));
    }

    while (1) {
        if (tid == 0) {
            while (1) {
                uint32_t sd;
                asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
                    : "=r"(sd) : "l"(&ctrl->shutdown));
                if (sd) { s_step = 0; break; }
                uint32_t ws;
                asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
                    : "=r"(ws) : "l"(&ctrl->work_step));
                if (ws > last_step) { s_step = ws; break; }
                __nanosleep(50);
            }
        }
        __syncthreads();
        if (s_step == 0) return;
        uint32_t step = s_step;
        last_step = step;

        if (tid == 0) {
            const uint32_t* src = (const uint32_t*)ctrl->input_ptr;
            uint32_t* dst = (uint32_t*)ctrl->output_ptr;
            for (uint32_t i = 0; i < ctrl->n_words && i < 8; i++)
                dst[i] = src[i] + step;
        }

        __threadfence();
        if (tid == 0) {
            asm volatile("st.release.sys.global.b32 [%0], %1;"
                :: "l"(&ctrl->kernel_done), "r"(step));
        }
        __syncthreads();
    }
}

/* Simple kernel on stream 0 */
__global__ void dummy_kernel(uint32_t* ptr, int val) {
    ptr[threadIdx.x] = val;
}

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int check_alive(struct uf19v5_ctrl* ctrl, uint32_t step, const char* label) {
    ctrl->n_words = 0;  /* no copy needed */
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&ctrl->work_step, step, __ATOMIC_RELEASE);
    double t0 = now_us();
    while (__atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE) < step) {
        if (now_us() - t0 > 2000000) {
            printf("  %s: DEAD (kernel_done=%u, want %u)\n", label,
                   __atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE), step);
            return 0;
        }
    }
    printf("  %s: ALIVE (%.1f µs)\n", label, now_us() - t0);
    return 1;
}

int main() {
    setlinebuf(stdout);

    /* Pre-allocate ALL memory before launching persistent kernel */
    struct uf19v5_ctrl* ctrl;
    cudaHostAlloc((void**)&ctrl, sizeof(struct uf19v5_ctrl), cudaHostAllocDefault);
    memset(ctrl, 0, sizeof(struct uf19v5_ctrl));

    uint32_t *d_a, *d_b;
    cudaMalloc(&d_a, 4096);
    cudaMalloc(&d_b, 4096);
    cudaMemset(d_a, 1, 4096);
    cudaDeviceSynchronize();  /* ensure all allocs done */

    printf("=== Persistent Kernel Isolation Test ===\n");
    printf("All memory pre-allocated. Launching persistent kernel...\n");

    cudaStream_t pstream;
    cudaStreamCreateWithFlags(&pstream, cudaStreamNonBlocking);
    test_persistent_kernel<<<1, 256, 0, pstream>>>(ctrl);

    /* Wait for alive */
    double t0 = now_us();
    while (__atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE) != 0xABCD) {
        if (now_us() - t0 > 5000000) {
            printf("TIMEOUT waiting for alive\n"); return 1;
        }
    }
    printf("Kernel alive.\n\n");
    __atomic_store_n(&ctrl->kernel_done, 0, __ATOMIC_RELEASE);

    uint32_t step = 0;

    /* Test 1: Baseline - no other ops */
    printf("Test 1: Baseline (no other CUDA ops)\n");
    if (!check_alive(ctrl, ++step, "baseline")) goto shutdown;

    /* Test 2: cudaMemset on stream 0 (legacy default stream) */
    printf("\nTest 2: cudaMemset on stream 0\n");
    cudaMemset(d_a, 0x42, 1024);
    printf("  cudaMemset returned\n");
    if (!check_alive(ctrl, ++step, "after cudaMemset")) goto shutdown;

    /* Test 3: Kernel launch on stream 0 */
    printf("\nTest 3: Kernel launch on stream 0\n");
    dummy_kernel<<<1, 32>>>(d_a, 123);
    printf("  kernel launched\n");
    if (!check_alive(ctrl, ++step, "after kernel launch")) goto shutdown;

    /* Test 4: cudaMalloc (new alloc while kernel running) — CRITICAL for PyTorch */
    printf("\nTest 4: cudaMalloc (new)\n");
    {
        uint32_t* d_new;
        cudaError_t e = cudaMalloc(&d_new, 1024);
        printf("  cudaMalloc returned: %s\n", e ? cudaGetErrorString(e) : "OK");
        if (!check_alive(ctrl, ++step, "after cudaMalloc")) goto shutdown;

        /* Test 5: cudaFree */
        printf("\nTest 5: cudaFree\n");
        e = cudaFree(d_new);
        printf("  cudaFree returned: %s\n", e ? cudaGetErrorString(e) : "OK");
        if (!check_alive(ctrl, ++step, "after cudaFree")) goto shutdown;
    }

    /* Test 6: Kernel on separate non-blocking stream */
    printf("\nTest 6: Kernel + sync on non-blocking stream\n");
    {
        cudaStream_t s2;
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        dummy_kernel<<<1, 32, 0, s2>>>(d_b, 456);
        cudaStreamSynchronize(s2);
        cudaStreamDestroy(s2);
    }
    printf("  non-blocking stream kernel done\n");
    if (!check_alive(ctrl, ++step, "after NB stream kernel")) goto shutdown;

    /* Test 7: cudaStreamSynchronize(0) — LIKELY BLOCKS */
    printf("\nTest 7: cudaStreamSynchronize(0)\n");
    cudaStreamSynchronize(0);
    printf("  cudaStreamSynchronize(0) returned\n");
    if (!check_alive(ctrl, ++step, "after StreamSync(0)")) goto shutdown;

    /* Test 8: cudaDeviceSynchronize — KNOWN to block */
    printf("\nTest 8: cudaDeviceSynchronize (EXPECTED TO BLOCK)\n");
    printf("  Skipping — known to deadlock with persistent kernel\n");

    printf("\n=== ALL TESTS PASSED ===\n");

shutdown:
    printf("\nShutting down...\n");
    __atomic_store_n(&ctrl->shutdown, 1, __ATOMIC_RELEASE);
    cudaStreamSynchronize(pstream);
    printf("Done.\n");
    cudaStreamDestroy(pstream);
    cudaFreeHost(ctrl);
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}
