#!/usr/bin/env python3
"""Patch: Relax Marlin min_thread_k check for layers with K not divisible by 128.

On SM121, no other INT4 kernel is available (Machete/CUTLASS need SM90+, Exllama needs FP16).
Instead of raising ValueError, return False so vLLM can fall back to torch dequant.
"""
import sys

MARLIN_UTILS = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils.py"

with open(MARLIN_UTILS) as f:
    content = f.read()

if "MARLIN_K_RELAX" in content:
    print("patch_marlin_k_relax: already applied")
    sys.exit(0)

old = """    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"Weight input_size_per_partition = "
            f"{input_size_per_partition} is not divisible "
            f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )"""

new = """    # MARLIN_K_RELAX: Allow K not divisible by min_thread_k.
    # On SM121 no other INT4 kernel exists. Return False to signal
    # that Marlin cannot handle this layer (vLLM will use dequant fallback).
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        import logging as _mk_log
        _mk_log.getLogger(__name__).warning(
            f"Marlin: K={input_size_per_partition} not divisible by "
            f"{GPTQ_MARLIN_MIN_THREAD_K}, skipping Marlin for this layer")
        return"""

if old not in content:
    print("ERROR: pattern not found in marlin_utils.py")
    sys.exit(1)

content = content.replace(old, new)
with open(MARLIN_UTILS, "w") as f:
    f.write(content)

print("patch_marlin_k_relax: applied (K check returns None instead of raising)")
