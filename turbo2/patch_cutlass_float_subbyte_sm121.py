#!/usr/bin/env python3
"""
Patch vLLM's bundled CUTLASS float_subbyte.h to remove SM121 from PTX E2M1 path.

Problem: CUTLASS defines CUDA_PTX_FP4FP6_CVT_ENABLED for SM121A, but SM121 (GB10)
does NOT support the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction. This causes
ptxas to fail when compiling any CUTLASS code that uses NumericArrayConverter<float_e2m1_t>
for sm_121a target.

Fix: Remove CUTLASS_ARCH_MMA_SM121A_ENABLED (and SM121F) from the condition.
SM121a falls back to CUTLASS' own C++ software E2M1 conversion via exmy_base.h.
SM120a keeps using the hardware PTX instruction.

This is separate from fix_flashinfer_e2m1_sm121.py which patches FlashInfer's
bundled copy of CUTLASS. This patch targets vLLM's build-time CUTLASS source.

Needed for dual-arch builds (TORCH_CUDA_ARCH_LIST="12.0a;12.1a").
"""

import sys
import os
import glob


def find_float_subbyte():
    """Find vLLM's bundled CUTLASS float_subbyte.h"""
    candidates = [
        "/app/vllm/.deps/cutlass-src/include/cutlass/float_subbyte.h",
        # Fallback: search in common locations
    ]

    # Also search for any other copies in .deps
    for g in glob.glob("/app/vllm/.deps/*/include/cutlass/float_subbyte.h"):
        if g not in candidates:
            candidates.append(g)
    for g in glob.glob("/app/vllm/.deps/*/csrc/cutlass/include/cutlass/float_subbyte.h"):
        if g not in candidates:
            candidates.append(g)
    # CUTLASS source set via VLLM_CUTLASS_SRC_DIR
    cutlass_src = os.environ.get("VLLM_CUTLASS_SRC_DIR", "")
    if cutlass_src:
        p = os.path.join(cutlass_src, "include/cutlass/float_subbyte.h")
        if p not in candidates:
            candidates.append(p)

    found = [c for c in candidates if os.path.exists(c)]
    return found


def patch_file(target):
    """Remove SM121 from CUDA_PTX_FP4FP6_CVT_ENABLED in a single file."""

    with open(target, "r") as f:
        content = f.read()

    patched = False

    # Pattern 1: SM*A_ENABLED block (architecture-specific)
    old_a = (
        '#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM121A_ENABLED))\n'
        '#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1'
    )
    new_a = (
        '/* SM121 removed: cvt.rn.satfinite.e2m1x2.f32 not supported on GB10 */\n'
        '#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM120A_ENABLED))\n'
        '#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1'
    )

    # Pattern 2: SM*F_ENABLED block (feature-flag)
    old_f = (
        '#if (defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM103F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM110F_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM121F_ENABLED))\n'
        '#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1'
    )
    new_f = (
        '/* SM121 removed: cvt.rn.satfinite.e2m1x2.f32 not supported on GB10 */\n'
        '#if (defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM103F_ENABLED) || '
        'defined(CUTLASS_ARCH_MMA_SM110F_ENABLED) ||\\\n'
        '     defined(CUTLASS_ARCH_MMA_SM120F_ENABLED))\n'
        '#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1'
    )

    if old_a in content:
        content = content.replace(old_a, new_a)
        patched = True
        print(f"    Patched SM*A block: removed SM121A")

    if old_f in content:
        content = content.replace(old_f, new_f)
        patched = True
        print(f"    Patched SM*F block: removed SM121F")

    if not patched:
        if 'SM121 removed' in content:
            print(f"    Already patched")
        elif 'SM121' not in content:
            print(f"    No SM121 references found (older CUTLASS version)")
        else:
            print(f"    WARNING: SM121 found but pattern not matched")
            return False

    if patched:
        with open(target, "w") as f:
            f.write(content)

    return True


def main():
    print("=== Patch CUTLASS float_subbyte.h: Remove SM121 from PTX E2M1 ===")
    print()

    files = find_float_subbyte()
    if not files:
        print("ERROR: No float_subbyte.h found in vLLM deps")
        sys.exit(1)

    print(f"Found {len(files)} float_subbyte.h file(s):")
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
        print("Done. SM121a will use CUTLASS C++ software E2M1 (exmy_base.h).")
        print("SM120a continues to use hardware cvt.rn.satfinite.e2m1x2.f32.")
    else:
        print("WARNING: Some files could not be patched.")
        sys.exit(1)


if __name__ == "__main__":
    main()
