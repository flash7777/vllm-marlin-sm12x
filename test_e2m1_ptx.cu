/*
 * test_e2m1_ptx.cu — Proof that SM121 (GB10) supports cvt.rn.satfinite.e2m1x2.f32
 *
 * Compile: nvcc -arch=sm_121a -o test_e2m1_ptx test_e2m1_ptx.cu
 * Run:     ./test_e2m1_ptx
 *
 * E2M1 format (4-bit float): sign(1) + exponent(2) + mantissa(1)
 * Positive values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
 * Encoding:        0000 0001 0010 0011 0100 0101 0110 0111
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Convert 2 float32 values to 2 packed E2M1 values using hardware PTX
__device__ uint8_t float2_to_e2m1x2(float a, float b) {
    uint32_t result;
    asm volatile(
        "{\n"
        ".reg .b8 tmp;\n"
        "cvt.rn.satfinite.e2m1x2.f32 tmp, %2, %1;\n"
        "cvt.u32.u8 %0, tmp;\n"
        "}\n"
        : "=r"(result)
        : "f"(a), "f"(b)
    );
    return (uint8_t)result;
}

// Test kernel: convert float values to E2M1 and store results
__global__ void test_e2m1_kernel(float* input, uint8_t* output, int n_pairs) {
    int tid = threadIdx.x;
    if (tid < n_pairs) {
        output[tid] = float2_to_e2m1x2(input[tid * 2], input[tid * 2 + 1]);
    }
}

// Decode E2M1 nibble to float for verification
float e2m1_to_float(uint8_t bits) {
    // E2M1: 3 magnitude bits → 8 positive values
    static const float table[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,  // positive
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f   // negative
    };
    return table[bits & 0xF];
}

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Testing cvt.rn.satfinite.e2m1x2.f32 on SM%d%d...\n\n",
           prop.major, prop.minor);

    // Test values: all E2M1 representable values + edge cases
    float test_values[] = {
        // Pair 0: exact E2M1 values
        0.0f, 0.5f,
        // Pair 1: exact E2M1 values
        1.0f, 1.5f,
        // Pair 2: exact E2M1 values
        2.0f, 3.0f,
        // Pair 3: exact E2M1 values
        4.0f, 6.0f,
        // Pair 4: negative values
        -1.0f, -6.0f,
        // Pair 5: values between E2M1 steps (round-to-nearest)
        0.7f, 1.2f,
        // Pair 6: saturation (>6.0 clamps to 6.0)
        100.0f, -100.0f,
        // Pair 7: small values
        0.1f, 0.3f,
    };
    int n_values = sizeof(test_values) / sizeof(float);
    int n_pairs = n_values / 2;

    // Expected E2M1 packed bytes (low nibble = first value, high nibble = second)
    // cvt.rn.satfinite.e2m1x2.f32 packs: result = (b_e2m1 << 4) | a_e2m1
    struct Expected {
        float a, b;
        const char* desc;
    } expected[] = {
        {0.0f, 0.5f,   "exact: 0.0, 0.5"},
        {1.0f, 1.5f,   "exact: 1.0, 1.5"},
        {2.0f, 3.0f,   "exact: 2.0, 3.0"},
        {4.0f, 6.0f,   "exact: 4.0, 6.0"},
        {-1.0f, -6.0f, "negative: -1.0, -6.0"},
        {0.5f, 1.0f,   "round: 0.7->0.5, 1.2->1.0"},
        {6.0f, -6.0f,  "saturate: 100->6.0, -100->-6.0"},
        {0.0f, 0.5f,   "small: 0.1->0.0, 0.3->0.5"},
    };

    // Allocate device memory
    float *d_input;
    uint8_t *d_output;
    cudaMalloc(&d_input, n_values * sizeof(float));
    cudaMalloc(&d_output, n_pairs * sizeof(uint8_t));

    // Copy input to device
    cudaMemcpy(d_input, test_values, n_values * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    test_e2m1_kernel<<<1, n_pairs>>>(d_input, d_output, n_pairs);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KERNEL LAUNCH FAILED: %s\n", cudaGetErrorString(err));
        printf(">>> SM%d%d does NOT support cvt.rn.satfinite.e2m1x2.f32\n",
               prop.major, prop.minor);
        return 1;
    }

    // Synchronize and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("KERNEL EXECUTION FAILED: %s\n", cudaGetErrorString(err));
        printf(">>> SM%d%d does NOT support cvt.rn.satfinite.e2m1x2.f32\n",
               prop.major, prop.minor);
        return 1;
    }

    // Copy results back
    uint8_t results[n_pairs];
    cudaMemcpy(results, d_output, n_pairs * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Print results
    printf("%-35s | Input          | E2M1 packed | Decoded\n", "Test");
    printf("%-35s-+----------------+-------------+--------\n",
           "-----------------------------------");

    int pass = 0, fail = 0;
    for (int i = 0; i < n_pairs; i++) {
        uint8_t packed = results[i];
        uint8_t lo = packed & 0xF;        // first value (a)
        uint8_t hi = (packed >> 4) & 0xF;  // second value (b)
        float decoded_a = e2m1_to_float(lo);
        float decoded_b = e2m1_to_float(hi);

        // Check against expected
        bool ok_a = (decoded_a == expected[i].a);
        bool ok_b = (decoded_b == expected[i].b);
        bool ok = ok_a && ok_b;
        if (ok) pass++; else fail++;

        printf("%-35s | %+5.1f, %+6.1f | 0x%02X        | %+.1f, %+.1f  %s\n",
               expected[i].desc,
               test_values[i*2], test_values[i*2+1],
               packed,
               decoded_a, decoded_b,
               ok ? "OK" : "MISMATCH");
    }

    printf("\n%d/%d tests passed.\n", pass, n_pairs);

    if (fail == 0) {
        printf("\n>>> PROOF: SM%d%d (GB10) supports cvt.rn.satfinite.e2m1x2.f32\n",
               prop.major, prop.minor);
        printf(">>> Hardware E2M1 conversion works correctly.\n");
    } else {
        printf("\n>>> Some tests failed — check E2M1 rounding.\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return fail > 0 ? 1 : 0;
}
