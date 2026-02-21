# NVFP4 E2M1 Software Fallbacks for SM121 (Blackwell GB10)

## Problem

SM121 (DGX Spark / PGX ThinkStation, GB10) **lacks** the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction.
SM120 (RTX PRO 6000, Blackwell Desktop) **has** this instruction.

Both GPUs have FP4 Tensor Cores (`mma.e2m1`). Only the **float-to-E2M1 conversion**
(activation quantization) requires a software fallback on SM121.

When compiling for `sm_120a`, `__CUDA_ARCH__ == 1200` — hardware PTX is used.
When compiling for `sm_121a`, `__CUDA_ARCH__ == 1210` — software fallback is activated.

## Three Approaches

### 1. Avarok / turbo (Threshold if-else)

Specialized implementation using 7 precomputed thresholds.
Each E2M1 value has a defined "capture range" — a simple comparison suffices.

```cpp
// Source: turbo/patch_nvfp4_utils_sw_e2m1.py
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
__device__ __forceinline__ uint8_t _sw_float_to_e2m1(float x) {
    uint8_t sign = (uint8_t)((__float_as_uint(x) >> 28) & 8u);
    float ax = fabsf(x);
    uint8_t mag;
    if      (ax <= 0.25f)  mag = 0;   // -> 0.0
    else if (ax <  0.75f)  mag = 1;   // -> 0.5
    else if (ax <= 1.25f)  mag = 2;   // -> 1.0
    else if (ax <  1.75f)  mag = 3;   // -> 1.5
    else if (ax <= 2.5f)   mag = 4;   // -> 2.0
    else if (ax <  3.5f)   mag = 5;   // -> 3.0
    else if (ax <= 5.0f)   mag = 6;   // -> 4.0
    else                    mag = 7;   // -> 6.0 (satfinite)
    return sign | mag;
}
#endif
```

**Properties:**
- 7 branches (if-else chain)
- Compiler likely optimizes to predicated `FSETP` + `SEL` (no real branching in SASS)
- ~14 SASS instructions
- Round-to-nearest (not round-to-nearest-even)
- E2M1 only

### 2. turbo2 (Branchless Predicate Sum)

Optimized version: each comparison yields 0 or 1, the sum directly produces the E2M1 code.
Exploits the fact that the 8 E2M1 magnitudes (0-7) are sorted in ascending order.

```cpp
// Source: turbo2/patch_nvfp4_utils_sw_e2m1.py
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
__device__ __forceinline__ uint8_t _sw_float_to_e2m1(float x) {
    uint8_t sign = (uint8_t)((__float_as_uint(x) >> 28) & 8u);
    float ax = fabsf(x);
    uint8_t mag = (ax > 0.25f)
                + (ax >= 0.75f)
                + (ax > 1.25f)
                + (ax >= 1.75f)
                + (ax > 2.5f)
                + (ax >= 3.5f)
                + (ax > 5.0f);
    return sign | mag;
}
#endif
```

**Properties:**
- 0 branches — 7 predicate additions
- Guaranteed zero warp divergence
- ~14 SASS instructions (`FSETP.GT` / `FSETP.GE` + `IADD3`)
- Same thresholds as turbo, identical results
- Round-to-nearest (not round-to-nearest-even)
- E2M1 only

### 3. CUTLASS exmy_base.h (Generic IEEE Bit Manipulation)

CUTLASS' own software fallback — generic for **all** FP formats (E2M1, E2M3, E3M2,
E4M3, E5M2, ...). Activated when `CUDA_PTX_FP4FP6_CVT_ENABLED` is not defined.

```
Call chain:
  NumericArrayConverter<float_e2m1_t, float, N>::convert()
    -> NumericConverter<float_e2m1_t, float>::convert()
      -> static_cast<float_e2m1_t>(x)
        -> float_exmy_base::convert_from_float(x)
          -> FpBitRepresentation<uint8_t, 4, 2, 1, NONE>::convert_from(fp32_bits)
            -> FpBitRepresentation::convert(src_encoding, src_val, dst_encoding)
```

```
Pseudocode of convert() function (~200 lines template):

  sign, exp, mantissa = extract_ieee754_bits(float_input)
  if (is_nan)  -> nan         // E2M1: impossible (NanInfEncoding::NONE)
  if (is_inf)  -> satfinite   // E2M1: clamp to 6.0
  if (is_zero) -> signed_zero

  // Normalize denormals
  while (hidden_bit == 0) { mantissa <<= 1; exp--; }

  if (exp > MAX_EXP) -> satfinite  // Overflow -> 6.0

  // Adjust mantissa: shift = dst_mantissa_bits - src_mantissa_bits = 1 - 23 = -22
  mantissa = round_nearest_even(mantissa, shift_amount)
    // Guard/Round/Sticky bits:
    //   guard_bit  = bit at position shift_amount
    //   round_bit  = bit at position shift_amount - 1
    //   sticky_bit = OR of all bits below round_bit
    //   Round up when: (sticky && round) || (guard && round && !sticky)

  // Post-rounding overflow
  if (hidden_bits > 1) { mantissa >>= 1; exp++; }

  return pack_bits(sign, exp, mantissa)
```

**Properties:**
- ~8 branches + 1 while loop (denormal normalization)
- ~40+ SASS instructions (bit extraction, shifts, masks, rounding)
- **Round-to-nearest-even** (IEEE-754 compliant, guard/round/sticky)
- Generic for all ExMy formats
- Used internally by CUTLASS when `CUDA_PTX_FP4FP6_CVT_ENABLED == 0`

## Comparison Table

| Property | turbo (Avarok) | turbo2 (Branchless) | CUTLASS exmy_base |
|---|---|---|---|
| **Approach** | Specialized | Specialized | Generic |
| **Branches** | 7 if-else | **0** | ~8 + while |
| **SASS (estimated)** | ~14 | ~14 | ~40+ |
| **Warp divergence** | Compiler-dependent | **Guaranteed none** | Yes (while, if) |
| **Rounding** | Nearest | Nearest | Nearest-**even** |
| **Latency** | ~7 cycles | ~7 cycles | ~20+ cycles |
| **Code size** | 20 lines | 8 lines | ~200 lines |
| **Formats** | E2M1 only | E2M1 only | All ExMy |
| **IEEE compliant** | Approximate* | Approximate* | **Formally correct** |

\* At exact midpoints (0.75, 1.75, 3.5) turbo/turbo2 may differ by 1 ULP from
round-to-nearest-even. With FP4 having only 8 values this is practically irrelevant.

## Where Is Each Approach Used?

| Path | SM120a (HW PTX) | SM121a (SW Fallback) |
|---|---|---|
| vLLM `nvfp4_utils.cuh` (3 functions) | `cvt.rn.satfinite.e2m1x2.f32` | turbo / turbo2 |
| CUTLASS `NumericArrayConverter` | `cvt.rn.satfinite.e2m1x2.f32` | exmy_base.h |
| FlashInfer CUTLASS JIT | `cvt.rn.satfinite.e2m1x2.f32` | exmy_base.h |

## Activation

```cpp
// Guard in nvfp4_utils.cuh (turbo + turbo2):
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
  // Software E2M1 (turbo or turbo2)
#else
  // Hardware PTX: cvt.rn.satfinite.e2m1x2.f32
#endif

// Guard in CUTLASS float_subbyte.h (after patch):
// SM121 removed from condition -> CUDA_PTX_FP4FP6_CVT_ENABLED == 0 -> exmy_base.h
#if (defined(CUTLASS_ARCH_MMA_SM120A_ENABLED))  // SM121 REMOVED
#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1
#endif
```

## Build Configuration (Dual-Arch)

```dockerfile
ENV TORCH_CUDA_ARCH_LIST="12.0a;12.1a"
# SM120a: Hardware E2M1 + native CUTLASS PTX
# SM121a: Software E2M1 (turbo/turbo2) + CUTLASS exmy_base fallback
```
