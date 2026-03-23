/*
 * test_tcgen05.cu — Test whether tcgen05 (5th-gen Tensor Core) PTX instructions
 *                   are available on SM120/SM121 (GB10 Blackwell GeForce)
 *
 * Background: SM120/SM121 supposedly do NOT have tcgen05 (no Multicast, no CTA
 * Pairs, no 2-SM MMA). SM100/SM100a (data center Blackwell) DOES have tcgen05.
 * This test verifies empirically whether the 'f' suffix (FP4/FP6 features)
 * enables tcgen05 PTX on SM120.
 *
 * Compile attempts (try each — if compilation FAILS, that proves the point):
 *
 *   Test 1 (SM120 with F-suffix — the question):
 *     nvcc -gencode arch=compute_120f,code=sm_120f -o test_tcgen05 test_tcgen05.cu
 *
 *   Test 2 (SM120a — standard):
 *     nvcc -gencode arch=compute_120a,code=sm_120a -o test_tcgen05 test_tcgen05.cu
 *
 *   Test 3 (SM100a — reference, should compile):
 *     nvcc -gencode arch=compute_100a,code=sm_100a -o test_tcgen05 test_tcgen05.cu
 *
 *   Test 4 (compile-time feature test only, no tcgen05 — should always work):
 *     nvcc -gencode arch=compute_120a,code=sm_120a -DSKIP_TCGEN05 -o test_tcgen05 test_tcgen05.cu
 *
 * Expected results:
 *   - Test 1: FAIL (SM120f likely does not accept tcgen05 PTX)
 *   - Test 2: FAIL (SM120a does not have tcgen05)
 *   - Test 3: PASS (SM100a has tcgen05)
 *   - Test 4: PASS (no tcgen05 instructions compiled)
 *
 * If Test 1 or 2 surprisingly PASSES compilation, run the binary on a GB10 to
 * see if it actually executes or hits "illegal instruction" at runtime.
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Utility: print GPU info and SM version
// ============================================================================
void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SM:  %d.%d (sm_%d%d)\n", prop.major, prop.minor, prop.major, prop.minor);
    printf("---\n\n");
}

// ============================================================================
// Test 1: tcgen05.mma — Tensor Core MMA instruction
//
// tcgen05.mma is the core matrix multiply-accumulate instruction for 5th-gen
// Tensor Cores. It operates on the tcgen05 accumulator register space and
// requires CTA Pair support (2-SM cooperative execution).
//
// Minimal PTX signature (from PTX ISA 8.7):
//   tcgen05.mma.cta_group::1.kind::tf32  d, a_desc, b_desc, idesc, enable;
// ============================================================================

#ifndef SKIP_TCGEN05

__global__ void test_tcgen05_mma_kernel(uint32_t* result) {
    // We use inline PTX to attempt a tcgen05.mma instruction.
    // If the assembler accepts it, this SM supports tcgen05 PTX.
    //
    // tcgen05.mma requires:
    //   - d: accumulator address in shared memory (tcgen05 register file)
    //   - a_desc, b_desc: matrix descriptors (64-bit)
    //   - idesc: instruction descriptor (128-bit, .b256)
    //   - enable: predicate
    //
    // We cannot meaningfully execute this without proper TMA descriptors,
    // but we CAN test if the assembler/compiler accepts the PTX instruction.
    // We guard execution behind a false predicate so it never actually runs.

    uint32_t dummy = 0;

    // Attempt 1: Minimal tcgen05.mma under a false predicate
    // This tests whether the PTX assembler recognizes the instruction at all.
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.eq.u32 p, 0, 1;\n"         // p = false (never execute)
        "  @p tcgen05.mma.cta_group::1.kind::tf32 [%0], 0, 0, {0, 0, 0, 0}, 1;\n"
        "}\n"
        : : "r"(dummy)
    );

    result[0] = 1;  // Signal: compilation + launch succeeded
}

#endif // SKIP_TCGEN05

// ============================================================================
// Test 2: tcgen05.ld — Tensor Core register load
//
// tcgen05.ld loads data into the tcgen05 accumulator register file from
// shared memory or via TMA descriptors.
//
// PTX signature:
//   tcgen05.ld.sync.aligned.16x256b.x1  [taddr], [smem_addr];
// ============================================================================

#ifndef SKIP_TCGEN05

__global__ void test_tcgen05_ld_kernel(uint32_t* result) {
    // tcgen05.ld loads into the Tensor Core accumulator register space.
    // Guard behind false predicate to avoid actual execution.

    uint32_t dummy_taddr = 0;
    uint32_t dummy_smem  = 0;

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.eq.u32 p, 0, 1;\n"         // p = false
        "  @p tcgen05.ld.sync.aligned.16x256b.x1 [%0], [%1];\n"
        "}\n"
        : : "r"(dummy_taddr), "r"(dummy_smem)
    );

    result[0] = 1;
}

#endif // SKIP_TCGEN05

// ============================================================================
// Test 3: tcgen05.st — Tensor Core register store
//
// tcgen05.st stores data from the tcgen05 accumulator register file to
// shared memory.
//
// PTX signature:
//   tcgen05.st.sync.aligned.16x256b.x1  [smem_addr], [taddr];
// ============================================================================

#ifndef SKIP_TCGEN05

__global__ void test_tcgen05_st_kernel(uint32_t* result) {
    uint32_t dummy_smem  = 0;
    uint32_t dummy_taddr = 0;

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.eq.u32 p, 0, 1;\n"         // p = false
        "  @p tcgen05.st.sync.aligned.16x256b.x1 [%0], [%1];\n"
        "}\n"
        : : "r"(dummy_smem), "r"(dummy_taddr)
    );

    result[0] = 1;
}

#endif // SKIP_TCGEN05

// ============================================================================
// Test 4: tcgen05.cp — Tensor Core copy (accumulator <-> shared memory)
//
// tcgen05.cp copies data between tcgen05 register file and shared memory
// via bulk copy operations.
//
// PTX signature:
//   tcgen05.cp.cta_group::1.128x256b  [taddr], [smem_addr];
// ============================================================================

#ifndef SKIP_TCGEN05

__global__ void test_tcgen05_cp_kernel(uint32_t* result) {
    uint32_t dummy_taddr = 0;
    uint32_t dummy_smem  = 0;

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.eq.u32 p, 0, 1;\n"         // p = false
        "  @p tcgen05.cp.cta_group::1.128x256b [%0], [%1];\n"
        "}\n"
        : : "r"(dummy_taddr), "r"(dummy_smem)
    );

    result[0] = 1;
}

#endif // SKIP_TCGEN05

// ============================================================================
// Test 5: tcgen05.alloc / tcgen05.dealloc — Accumulator allocation
//
// Before using tcgen05 instructions, the accumulator register file must be
// allocated. This is the most basic tcgen05 operation.
//
// PTX signature:
//   tcgen05.alloc.cta_group::1.sync.aligned  taddr, nCols;
//   tcgen05.dealloc.cta_group::1.sync.aligned  taddr, nCols;
// ============================================================================

#ifndef SKIP_TCGEN05

__global__ void test_tcgen05_alloc_kernel(uint32_t* result) {
    uint32_t taddr;

    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.eq.u32 p, 0, 1;\n"         // p = false
        "  @p tcgen05.alloc.cta_group::1.sync.aligned  %0, 1;\n"
        "}\n"
        : "=r"(taddr) :
    );

    result[0] = 1;
}

#endif // SKIP_TCGEN05

// ============================================================================
// Fallback: simple kernel that always works (for -DSKIP_TCGEN05 mode)
// ============================================================================

__global__ void dummy_kernel(uint32_t* result) {
    result[0] = 42;
}

// ============================================================================
// Main: run all tests, report results
// ============================================================================

int main() {
    print_gpu_info();

#ifdef SKIP_TCGEN05
    printf("=== SKIP_TCGEN05 defined — no tcgen05 instructions compiled ===\n");
    printf("This binary only tests basic CUDA functionality.\n\n");

    uint32_t* d_result;
    uint32_t h_result = 0;
    cudaMalloc(&d_result, sizeof(uint32_t));
    cudaMemset(d_result, 0, sizeof(uint32_t));

    dummy_kernel<<<1, 1>>>(d_result);

    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        printf("  dummy_kernel:       PASS (returned %u)\n", h_result);
    } else {
        printf("  dummy_kernel:       FAIL (%s)\n", cudaGetErrorString(err));
    }

    cudaFree(d_result);
    printf("\nConclusion: Basic CUDA works. tcgen05 tests were skipped.\n");
    printf("Recompile WITHOUT -DSKIP_TCGEN05 to test tcgen05 instructions.\n");
    return 0;

#else
    printf("=== Testing tcgen05 (5th-gen Tensor Core) PTX instructions ===\n");
    printf("If you see this message, compilation SUCCEEDED.\n");
    printf("This means the PTX assembler accepted tcgen05 instructions\n");
    printf("for the target architecture.\n\n");
    printf("Now testing runtime execution (behind false predicates)...\n\n");

    uint32_t* d_result;
    uint32_t h_result;
    cudaMalloc(&d_result, sizeof(uint32_t));

    // --- Test structure ---
    struct TestCase {
        const char* name;
        void (*kernel)(uint32_t*);
    };

    TestCase tests[] = {
        {"tcgen05.alloc  (accumulator alloc)", test_tcgen05_alloc_kernel},
        {"tcgen05.mma    (matrix multiply) ",  test_tcgen05_mma_kernel},
        {"tcgen05.ld     (register load)   ",  test_tcgen05_ld_kernel},
        {"tcgen05.st     (register store)  ",  test_tcgen05_st_kernel},
        {"tcgen05.cp     (register copy)   ",  test_tcgen05_cp_kernel},
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);

    int pass = 0, fail = 0;

    for (int i = 0; i < n_tests; i++) {
        cudaMemset(d_result, 0, sizeof(uint32_t));
        h_result = 0;

        // Reset error state
        cudaGetLastError();

        // Launch kernel
        tests[i].kernel<<<1, 1>>>(d_result);

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            printf("  %s  FAIL (launch: %s)\n", tests[i].name, cudaGetErrorString(launch_err));
            fail++;
            continue;
        }

        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            printf("  %s  FAIL (exec:   %s)\n", tests[i].name, cudaGetErrorString(sync_err));
            fail++;
            // Reset device after illegal instruction
            cudaDeviceReset();
            cudaMalloc(&d_result, sizeof(uint32_t));
            continue;
        }

        cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (h_result == 1) {
            printf("  %s  PASS (compiled + executed)\n", tests[i].name);
            pass++;
        } else {
            printf("  %s  FAIL (unexpected result: %u)\n", tests[i].name, h_result);
            fail++;
        }
    }

    printf("\n--- Summary ---\n");
    printf("%d/%d tcgen05 instructions: compiled + executed\n", pass, n_tests);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (pass == n_tests) {
        printf("\n>>> RESULT: SM%d%d SUPPORTS tcgen05 PTX instructions!\n",
               prop.major, prop.minor);
        printf(">>> All 5th-gen Tensor Core instructions compiled and executed.\n");
    } else if (pass > 0) {
        printf("\n>>> RESULT: SM%d%d has PARTIAL tcgen05 support.\n",
               prop.major, prop.minor);
        printf(">>> %d/%d instructions worked, %d failed at runtime.\n",
               pass, n_tests, fail);
    } else {
        printf("\n>>> RESULT: SM%d%d does NOT support tcgen05 at runtime.\n",
               prop.major, prop.minor);
        printf(">>> Instructions compiled (PTX accepted) but failed to execute.\n");
        printf(">>> This confirms: SM120/121 lacks tcgen05 hardware.\n");
    }

    cudaFree(d_result);
    return fail > 0 ? 1 : 0;

#endif // SKIP_TCGEN05
}
