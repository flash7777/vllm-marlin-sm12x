#!/usr/bin/env python3
"""Remove Qwen3_5MTPModel.load_weights — let AutoWeightsLoader handle it.

Bug: The manual load_weights in Qwen3_5MTPModel has fused_expert_params_mapping
that expects 'experts.gate_up_proj' checkpoint keys. Per-expert keys like
'experts.0.gate_proj.qweight' don't match, causing KeyError on w2_qweight.

Fix: Remove the manual loader. AutoWeightsLoader delegates to FusedMoE.weight_loader
which correctly handles per-expert keys for all quant formats (BF16, GPTQ, Marlin).

Usage: cat patch_mtp_remove_manual_loader.py | podman exec -i CONTAINER python3 -
"""

import re

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_mtp.py"

with open(path) as f:
    src = f.read()

# Find and remove load_fused_expert_weights + load_weights methods
# They sit between "return hidden_states" and "class Qwen3_5MTP"
old_marker = "    def load_fused_expert_weights("
end_marker = "            loaded_params.add(name)"

if old_marker in src:
    start = src.index(old_marker)
    end = src.index(end_marker) + len(end_marker)
    # Find the line before (should be blank line after return hidden_states)
    # Go back to find the preceding blank line
    while start > 0 and src[start-1] == '\n':
        start -= 1
    start += 1  # Keep one newline

    replacement = "\n    # No custom load_weights — AutoWeightsLoader handles per-expert\n    # keys via FusedMoE.weight_loader (works for BF16, GPTQ, Marlin)\n"
    src = src[:start] + replacement + src[end:]

    # Clean up unused imports
    src = src.replace("import typing\n", "")
    src = src.replace("from collections.abc import Callable, Iterable",
                       "from collections.abc import Iterable")

    with open(path, "w") as f:
        f.write(src)
    print("PATCHED: removed Qwen3_5MTPModel.load_weights (per-expert fix)")
elif "No custom load_weights" in src:
    print("ALREADY PATCHED")
else:
    print("PATTERN NOT FOUND — code may have changed")
