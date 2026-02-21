#!/usr/bin/env python3
"""
Patch CUTLASS numeric_conversion.h: Add branchless E2M1 NumericConverter specialization.

Problem: When CUDA_PTX_FP4FP6_CVT_ENABLED is NOT defined (SM121a), CUTLASS falls back
to the generic IEEE-754 bit-manipulation in exmy_base.h (~200 lines, ~40+ SASS instructions).
This is significantly slower than the hardware PTX instruction on SM120a.

Fix: Add a partial specialization of NumericConverter<float_e2m1_t, float, Round> that uses
a branchless predicate-sum approach (~15 SASS instructions). This specialization is automatically
used by all NumericArrayConverter<float_e2m1_t, float, N> #else fallback paths.

For CUTLASS v4.3.5 (vllm-next).
"""

import sys
import os
import glob


def find_numeric_conversion():
    """Find CUTLASS numeric_conversion.h in vLLM build tree."""
    candidates = [
        "/opt/cutlass/include/cutlass/numeric_conversion.h",
        "/app/vllm/.deps/cutlass-src/include/cutlass/numeric_conversion.h",
    ]

    # Search .deps
    for g in glob.glob("/app/vllm/.deps/*/include/cutlass/numeric_conversion.h"):
        if g not in candidates:
            candidates.append(g)

    # VLLM_CUTLASS_SRC_DIR
    cutlass_src = os.environ.get("VLLM_CUTLASS_SRC_DIR", "")
    if cutlass_src:
        p = os.path.join(cutlass_src, "include/cutlass/numeric_conversion.h")
        if p not in candidates:
            candidates.append(p)

    found = [c for c in candidates if os.path.exists(c)]
    return found


# The branchless E2M1 specialization to insert
SPECIALIZATION = r"""
/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Branchless software E2M1 conversion (SM121 fallback)
// Replaces ~200-line generic exmy_base.h IEEE-754 bit-manipulation with ~15 SASS instructions.
// Only active when CUDA_PTX_FP4FP6_CVT_ENABLED is NOT defined (i.e., SM121a builds).
// SM120a builds use the hardware cvt.rn.satfinite.e2m1x2.f32 instruction instead.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(CUDA_PTX_FP4FP6_CVT_ENABLED)

/// Specialized NumericConverter for float => float_e2m1_t using branchless predicate-sum.
/// E2M1 has 8 magnitude levels: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
/// Each comparison yields 0 or 1; the running sum directly encodes the E2M1 magnitude bits.
template <FloatRoundStyle Round>
struct NumericConverter<float_e2m1_t, float, Round> {

  using result_type = float_e2m1_t;
  using source_type = float;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_HOST_DEVICE
  static result_type convert(source_type const & v) {
    float av = (v < 0.0f) ? -v : v;

    // Branchless magnitude computation via threshold comparisons.
    // Thresholds are midpoints between adjacent E2M1 representable values:
    //   0↔0.5: 0.25  |  0.5↔1.0: 0.75  |  1.0↔1.5: 1.25
    //   1.5↔2.0: 1.75  |  2.0↔3.0: 2.5  |  3.0↔4.0: 3.5  |  4.0↔6.0: 5.0
    // Values >= 6.0 saturate to max (0b111 = 6.0).
    uint8_t mag = static_cast<uint8_t>(
        (av > 0.25f) + (av >= 0.75f) + (av > 1.25f) +
        (av >= 1.75f) + (av > 2.5f) + (av >= 3.5f) + (av > 5.0f));

    // Sign bit: v < 0 sets bit 3 (E2M1 sign position)
    // Pure arithmetic — no __float_as_uint needed (HOST_DEVICE safe)
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;

    return float_e2m1_t::bitcast(static_cast<uint8_t>(sign | mag));
  }

  CUTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#endif // !CUDA_PTX_FP4FP6_CVT_ENABLED

"""

# Marker to find the correct insertion point:
# Insert BEFORE the first NumericArrayConverter<float_e2m1_t, ...> specialization
INSERTION_MARKER = "/// Partial specialization for Array<float_e2m1_t, 2> <= Array<float, 2>"


def patch_file(target):
    """Add branchless E2M1 NumericConverter specialization."""

    with open(target, "r") as f:
        content = f.read()

    # Check if already patched
    if "Branchless software E2M1 conversion" in content:
        print(f"    Already patched")
        return True

    # Check that E2M1 array converters exist (correct CUTLASS version)
    if "NumericArrayConverter<float_e2m1_t, float, 2" not in content:
        print(f"    WARNING: No float_e2m1_t array converters found (wrong CUTLASS version?)")
        return False

    # Find insertion point
    if INSERTION_MARKER not in content:
        print(f"    WARNING: Insertion marker not found: {INSERTION_MARKER!r}")
        return False

    # Insert the specialization BEFORE the array converter comment
    content = content.replace(
        INSERTION_MARKER,
        SPECIALIZATION + INSERTION_MARKER
    )

    with open(target, "w") as f:
        f.write(content)

    print(f"    Inserted branchless E2M1 NumericConverter specialization")
    return True


def main():
    print("=== Patch CUTLASS numeric_conversion.h: Branchless E2M1 Converter ===")
    print()

    files = find_numeric_conversion()
    if not files:
        print("ERROR: No numeric_conversion.h found")
        sys.exit(1)

    print(f"Found {len(files)} numeric_conversion.h file(s):")
    for f in files:
        print(f"  {f}")
    print()

    success = True
    for f in files:
        print(f"Patching: {f}")
        if not patch_file(f):
            success = False
        print()

    if success:
        print("Done. SM121a NumericConverter<float_e2m1_t, float> now uses branchless E2M1.")
        print("SM120a continues to use hardware cvt.rn.satfinite.e2m1x2.f32 (PTX path).")
    else:
        print("WARNING: Some files could not be patched.")
        sys.exit(1)


if __name__ == "__main__":
    main()
