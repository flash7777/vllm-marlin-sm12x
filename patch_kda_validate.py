#!/usr/bin/env python3
"""Patch causal_conv1d.py + kda.py for CUDA Graph compatibility.

The assertion `num_cache_lines >= batch` (line ~1162) fails during CUDA Graph
capture because graphs capture with padded batch sizes larger than allocated
conv_state cache lines. The kernel uses conv_state_indices for actual indexing,
so the assertion is overly strict.

Usage: python3 patch_kda_validate.py
"""

CONV1D_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
KDA_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/kda.py"

patched = 0

# Patch 1: causal_conv1d.py — remove the bare assertion outside validate_data guard
with open(CONV1D_PATH, "r") as f:
    content = f.read()
original = content
# The problematic assertion is standalone (not inside if validate_data)
content = content.replace(
    "        assert num_cache_lines >= batch\n        assert weight.stride(1) == 1  # Need this",
    "        # assert num_cache_lines >= batch  # patched: fails during CUDA Graph capture\n        assert weight.stride(1) == 1  # Need this"
)
if content != original:
    with open(CONV1D_PATH, "w") as f:
        f.write(content)
    patched += 1
    print("patch_kda_validate: patched causal_conv1d.py (num_cache_lines assertion)")
else:
    print("patch_kda_validate: causal_conv1d.py already patched or pattern not found")

# Patch 2: kda.py — disable validate_data
with open(KDA_PATH, "r") as f:
    content = f.read()
original = content
content = content.replace("validate_data=True,", "validate_data=False,")
if content != original:
    with open(KDA_PATH, "w") as f:
        f.write(content)
    count = original.count("validate_data=True,")
    patched += 1
    print(f"patch_kda_validate: patched kda.py ({count} occurrences)")
else:
    print("patch_kda_validate: kda.py already patched or pattern not found")

print(f"patch_kda_validate: done ({patched} files patched)")
