#!/usr/bin/env python3
"""
patch_fp8_cutlass_sm120.py — FP8 CUTLASS MoE SM120 Integration

Adds FP8 MoE grouped GEMM support for SM120/SM121 (GeForce Blackwell, GB10).
Dense FP8 SM120 already exists in vLLM — only MoE is missing.

Patches:
1. CMakeLists.txt: Add FP8 MoE SM120 build section
2. scaled_mm_entry.cu: Add SM120 MoE forward declaration + dispatch
3. _custom_ops.py: Fix Python gate that blocks SM120 (>= 110 → >= 130)

Run during container build BEFORE `pip install --no-build-isolation`.
Expects vLLM source at /tmp/vllm-build/.
"""

import os
import re
import sys

VLLM_ROOT = "/tmp/vllm-build"


def patch_cmake():
    """Add FP8 MoE SM120 section to CMakeLists.txt."""
    cmake_path = os.path.join(VLLM_ROOT, "CMakeLists.txt")
    with open(cmake_path, "r") as f:
        content = f.read()

    # Check if already patched
    if "grouped_mm_c3x_sm120.cu" in content:
        print("CMake: Already has grouped_mm_c3x_sm120.cu")
        return

    # Insert FP8 MoE SM120 section after the SM100 MoE section.
    # Find the end of SM100 MoE section (the endif() after ENABLE_CUTLASS_MOE_SM100)
    sm100_moe_pattern = r'(list\(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1"\).*?endif\(\))'
    match = re.search(sm100_moe_pattern, content, re.DOTALL)
    if not match:
        print("ERROR: Could not find SM100 MoE section in CMakeLists.txt")
        sys.exit(1)

    sm120_moe_section = """

  # FP8 MoE for SM120/SM121 (GeForce Blackwell)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SRCS "csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm120.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    # ENABLE_CUTLASS_MOE_SM120 may already be set by NVFP4 section —
    # adding it again is harmless (duplicate -D flags are OK)
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM120=1")
    message(STATUS "Building grouped_mm_c3x_sm120 (FP8 MoE) for archs: ${SCALED_MM_ARCHS}")
  else()
    message(STATUS "Not building FP8 MoE SM120 as no compatible archs found.")
  endif()"""

    insert_pos = match.end()
    content = content[:insert_pos] + sm120_moe_section + content[insert_pos:]

    with open(cmake_path, "w") as f:
        f.write(content)
    print("CMake: Added FP8 MoE SM120 section")


def patch_scaled_mm_entry():
    """Add SM120 MoE forward declaration and dispatch block."""
    entry_path = os.path.join(
        VLLM_ROOT, "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"
    )
    with open(entry_path, "r") as f:
        content = f.read()

    # Check if already patched
    if "cutlass_moe_mm_sm120" in content:
        print("scaled_mm_entry.cu: Already has cutlass_moe_mm_sm120")
        return

    # 1. Add forward declaration after SM100 MoE declaration
    sm100_moe_decl = """#if defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100
void cutlass_moe_mm_sm100(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch);
#endif"""

    sm120_moe_decl = """
#if defined ENABLE_CUTLASS_MOE_SM120 && ENABLE_CUTLASS_MOE_SM120
void cutlass_moe_mm_sm120(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch);
#endif"""

    if sm100_moe_decl in content:
        content = content.replace(
            sm100_moe_decl, sm100_moe_decl + "\n" + sm120_moe_decl
        )
        print("scaled_mm_entry.cu: Added SM120 MoE forward declaration")
    else:
        print("WARNING: Could not find SM100 MoE declaration pattern")
        print("  Trying regex fallback...")
        # Regex fallback: find the SM100 MoE declaration block
        pattern = r'(#if defined ENABLE_CUTLASS_MOE_SM100.*?#endif)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = content[:match.end()] + "\n" + sm120_moe_decl + content[match.end():]
            print("scaled_mm_entry.cu: Added SM120 MoE forward declaration (regex)")
        else:
            print("ERROR: Could not find SM100 MoE declaration")
            sys.exit(1)

    # 2. Add SM120 dispatch block in cutlass_moe_mm()
    # Insert BEFORE the SM100 block
    sm100_dispatch = """#if defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100
  if (version_num >= 100 && version_num < 110) {
    cutlass_moe_mm_sm100(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                         expert_offsets, problem_sizes, a_strides, b_strides,
                         c_strides, per_act_token, per_out_ch);
    return;
  }
#endif"""

    sm120_dispatch = """#if defined ENABLE_CUTLASS_MOE_SM120 && ENABLE_CUTLASS_MOE_SM120
  if (version_num >= 120) {
    cutlass_moe_mm_sm120(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                         expert_offsets, problem_sizes, a_strides, b_strides,
                         c_strides, per_act_token, per_out_ch);
    return;
  }
#endif
"""

    if sm100_dispatch in content:
        content = content.replace(sm100_dispatch, sm120_dispatch + sm100_dispatch)
        print("scaled_mm_entry.cu: Added SM120 MoE dispatch block")
    else:
        print("WARNING: Could not find SM100 MoE dispatch block")
        print("  Trying regex fallback...")
        pattern = r'(#if defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100\s*\n\s*if \(version_num >= 100)'
        match = re.search(pattern, content)
        if match:
            content = content[:match.start()] + sm120_dispatch + content[match.start():]
            print("scaled_mm_entry.cu: Added SM120 MoE dispatch block (regex)")
        else:
            print("ERROR: Could not find SM100 MoE dispatch block")
            sys.exit(1)

    # 3. Update error message to include 120
    content = content.replace(
        'Required capability: 90 or 100"',
        'Required capability: 90, 100, or 120"'
    )

    with open(entry_path, "w") as f:
        f.write(content)
    print("scaled_mm_entry.cu: Patching complete")


def patch_python_gate():
    """Fix Python gate that blocks SM120 from CUTLASS MoE."""
    ops_path = os.path.join(VLLM_ROOT, "vllm/_custom_ops.py")
    with open(ops_path, "r") as f:
        content = f.read()

    # Change >= 110 to >= 130 to allow SM120/SM121
    old = "cuda_device_capability < 90 or cuda_device_capability >= 110"
    new = "cuda_device_capability < 90 or cuda_device_capability >= 130"

    if old in content:
        content = content.replace(old, new)
        with open(ops_path, "w") as f:
            f.write(content)
        print("_custom_ops.py: Fixed Python gate (>= 110 → >= 130)")
    elif new in content:
        print("_custom_ops.py: Already patched")
    else:
        print("WARNING: Could not find Python gate pattern")
        print("  Looking for alternative patterns...")
        # Try to find and fix any pattern that blocks SM120
        if "cuda_device_capability >= 110" in content:
            content = content.replace(
                "cuda_device_capability >= 110",
                "cuda_device_capability >= 130"
            )
            with open(ops_path, "w") as f:
                f.write(content)
            print("_custom_ops.py: Fixed alternative pattern")
        else:
            print("ERROR: No blocking pattern found — may already be fixed upstream")


def main():
    print("=" * 70)
    print("FP8 CUTLASS MoE SM120 Integration Patch")
    print("=" * 70)
    print(f"vLLM root: {VLLM_ROOT}")
    print()

    if not os.path.exists(VLLM_ROOT):
        print(f"ERROR: vLLM root not found: {VLLM_ROOT}")
        sys.exit(1)

    patch_cmake()
    print()
    patch_scaled_mm_entry()
    print()
    patch_python_gate()

    print()
    print("=" * 70)
    print("FP8 CUTLASS MoE SM120 integration complete.")
    print("Dense FP8 SM120 was already present in vLLM — only MoE was added.")
    print("=" * 70)


if __name__ == "__main__":
    main()
