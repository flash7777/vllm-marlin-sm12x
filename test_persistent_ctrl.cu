/*
 * Minimal test: persistent kernel ctrl struct communication.
 * No RDMA — just tests that host can write work_step and kernel sees it,
 * and kernel can write kernel_done and host sees it.
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

/* Minimal persistent kernel: just polls work_step, copies data, signals done */
extern "C" __global__
void test_persistent_kernel(struct uf19v5_ctrl* ctrl)
{
    int tid = threadIdx.x;
    __shared__ uint32_t s_step;
    uint32_t last_step = 0;

    /* Signal we're alive */
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
                if (ws > last_step) {
                    s_step = ws;
                    break;
                }
                __nanosleep(50);
            }
        }
        __syncthreads();

        if (s_step == 0) return;

        uint32_t step = s_step;
        last_step = step;

        /* Read input_ptr, write to output_ptr (simple copy test) */
        uint64_t inp = 0, outp = 0;
        uint32_t nw = 0;
        if (tid == 0) {
            asm volatile("ld.acquire.sys.global.b64 %0, [%1];"
                : "=l"(inp) : "l"(&ctrl->input_ptr));
            asm volatile("ld.acquire.sys.global.b64 %0, [%1];"
                : "=l"(outp) : "l"(&ctrl->output_ptr));
            asm volatile("ld.acquire.sys.global.b32 %0, [%1];"
                : "=r"(nw) : "l"(&ctrl->n_words));
        }
        /* Broadcast via shared memory would be needed for multi-thread,
         * but for this test we just use tid==0 for the copy */

        /* Simple: thread 0 copies n_words uint32_t from input to output */
        if (tid == 0 && inp && outp && nw > 0) {
            const uint32_t* src = (const uint32_t*)inp;
            uint32_t* dst = (uint32_t*)outp;
            for (uint32_t i = 0; i < nw; i++) {
                dst[i] = src[i] + step;  /* add step so we can verify */
            }
        }

        __threadfence();
        if (tid == 0) {
            asm volatile("st.release.sys.global.b32 [%0], %1;"
                :: "l"(&ctrl->kernel_done), "r"(step));
        }
        __syncthreads();
    }
}

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main() {
    setlinebuf(stdout);
    printf("=== Persistent Kernel Ctrl Test ===\n");

    struct uf19v5_ctrl* ctrl;
    cudaError_t ce;

    ce = cudaHostAlloc((void**)&ctrl, sizeof(struct uf19v5_ctrl), cudaHostAllocDefault);
    if (ce) { printf("cudaHostAlloc ctrl: %s\n", cudaGetErrorString(ce)); return 1; }
    memset(ctrl, 0, sizeof(struct uf19v5_ctrl));

    /* Allocate input/output in device memory */
    uint32_t* d_input;
    uint32_t* d_output;
    int nw = 2048;  /* 4096 bf16 = 2048 uint32 */
    ce = cudaMalloc(&d_input, nw * sizeof(uint32_t));
    if (ce) { printf("cudaMalloc input: %s\n", cudaGetErrorString(ce)); return 1; }
    ce = cudaMalloc(&d_output, nw * sizeof(uint32_t));
    if (ce) { printf("cudaMalloc output: %s\n", cudaGetErrorString(ce)); return 1; }

    /* Fill input with pattern */
    uint32_t* h_input = (uint32_t*)malloc(nw * sizeof(uint32_t));
    uint32_t* h_output = (uint32_t*)malloc(nw * sizeof(uint32_t));
    for (int i = 0; i < nw; i++) h_input[i] = i + 1;
    cudaMemcpy(d_input, h_input, nw * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* Launch persistent kernel on a dedicated stream */
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    printf("Launching persistent kernel...\n");
    test_persistent_kernel<<<1, 256, 0, stream>>>(ctrl);
    ce = cudaGetLastError();
    if (ce) { printf("Launch error: %s\n", cudaGetErrorString(ce)); return 1; }
    printf("Launched. Waiting for alive signal (NO other CUDA ops)...\n");

    /* Wait for kernel alive signal (0xABCD in kernel_done) */
    int timeout_ms = 5000;
    double t0 = now_us();
    while (__atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE) != 0xABCD) {
        if (now_us() - t0 > timeout_ms * 1000) {
            printf("TIMEOUT waiting for kernel alive signal! kernel_done=%u\n",
                   __atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE));
            return 1;
        }
    }
    printf("Kernel alive (kernel_done=0x%X). Resetting to 0.\n",
           __atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE));
    __atomic_store_n(&ctrl->kernel_done, 0, __ATOMIC_RELEASE);

    /* Test: does launching a kernel on stream 0 kill the persistent kernel? */
    printf("\n--- Interference Test: cudaMemset on stream 0 ---\n");
    void* test_ptr;
    ce = cudaMalloc(&test_ptr, 1024);
    printf("cudaMalloc: %s\n", ce ? cudaGetErrorString(ce) : "OK");
    if (!ce) {
        ce = cudaMemset(test_ptr, 0x42, 1024);
        printf("cudaMemset: %s\n", ce ? cudaGetErrorString(ce) : "OK");
        cudaFree(test_ptr);
    }
    printf("Now testing if persistent kernel still responds...\n");
    ctrl->input_ptr = (uint64_t)d_input;
    ctrl->output_ptr = (uint64_t)d_output;
    ctrl->n_words = nw;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&ctrl->work_step, 1, __ATOMIC_RELEASE);
    t0 = now_us();
    while (__atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE) < 1) {
        if (now_us() - t0 > 3000000) {
            printf("PERSISTENT KERNEL DEAD after stream-0 op! kernel_done=%u\n",
                   __atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE));
            goto shutdown;
        }
    }
    printf("Persistent kernel still alive! kernel_done=%u in %.1f µs\n",
           __atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE), now_us() - t0);

    {
    /* --- Test 1: Single step --- */
    printf("\n--- Test 1: Single step ---\n");
    ctrl->input_ptr = (uint64_t)d_input;
    ctrl->output_ptr = (uint64_t)d_output;
    ctrl->n_words = nw;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&ctrl->work_step, 1, __ATOMIC_RELEASE);
    printf("Wrote work_step=1, polling kernel_done...\n");

    t0 = now_us();
    while (__atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE) < 1) {
        if (now_us() - t0 > 5000000) {
            printf("TIMEOUT! kernel_done=%u\n",
                   __atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE));
            goto shutdown;
        }
    }
    double elapsed = now_us() - t0;
    printf("kernel_done=%u in %.1f µs\n",
           __atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE), elapsed);

    /* Verify output: should be input[i] + step */
    cudaMemcpy(h_output, d_output, nw * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    int ok = 1;
    for (int i = 0; i < nw; i++) {
        if (h_output[i] != h_input[i] + 1) {
            printf("  MISMATCH [%d]: got=%u exp=%u\n", i, h_output[i], h_input[i] + 1);
            ok = 0;
            if (i > 5) break;
        }
    }
    printf("  Verify: %s\n", ok ? "OK" : "FAIL");

    /* --- Test 2: Multiple steps + latency --- */
    printf("\n--- Test 2: 1000 steps latency ---\n");
    int N = 1000;
    t0 = now_us();
    for (int s = 2; s <= N + 1; s++) {
        ctrl->input_ptr = (uint64_t)d_input;
        ctrl->output_ptr = (uint64_t)d_output;
        ctrl->n_words = nw;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        __atomic_store_n(&ctrl->work_step, (uint32_t)s, __ATOMIC_RELEASE);

        while (__atomic_load_n(&ctrl->kernel_done, __ATOMIC_ACQUIRE) < (uint32_t)s) {}
    }
    elapsed = now_us() - t0;
    printf("  %d steps in %.1f µs = %.2f µs/step\n", N, elapsed, elapsed / N);

    /* Verify last step */
    cudaMemcpy(h_output, d_output, nw * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    ok = 1;
    uint32_t expect_step = N + 1;
    for (int i = 0; i < nw; i++) {
        if (h_output[i] != h_input[i] + expect_step) {
            printf("  MISMATCH [%d]: got=%u exp=%u\n", i, h_output[i], h_input[i] + expect_step);
            ok = 0;
            if (i > 5) break;
        }
    }
    printf("  Verify step %u: %s\n", expect_step, ok ? "OK" : "FAIL");
    }

shutdown:
    printf("\nShutting down...\n");
    __atomic_store_n(&ctrl->shutdown, 1, __ATOMIC_RELEASE);
    cudaStreamSynchronize(stream);
    printf("Kernel exited.\n");

    cudaStreamDestroy(stream);
    cudaFreeHost(ctrl);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    printf("Done.\n");
    return 0;
}
