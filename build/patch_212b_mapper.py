#!/usr/bin/env python3
"""Patch: Extend WeightsMapper to handle 212B REAP key format.

212B REAP has keys like 'model.layers.0...' (pure LLM)
262B REAP has keys like 'model.language_model.layers.0...' (VL)
Both use Qwen3_5MoeForConditionalGeneration architecture.

We add 'model.' -> 'language_model.model.' as fallback prefix mapping.
"""
import sys

QWEN35_PY = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"

# The mapper is inherited from qwen3_vl.py, so we need to override it in qwen3_5.py
QWEN3VL_PY = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_vl.py"

with open(QWEN3VL_PY) as f:
    content = f.read()

if "REAP_212B_MAPPER" in content:
    print("patch_212b_mapper: already applied")
    sys.exit(0)

# Extend the existing mapper with a fallback for 'model.' prefix
old = '''    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )'''

new = '''    # REAP_212B_MAPPER: Added 'model.' fallback for 212B (pure LLM keys)
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.": "language_model.model.",
        }
    )'''

if old not in content:
    print("ERROR: mapper pattern not found in qwen3_vl.py")
    sys.exit(1)

content = content.replace(old, new)
with open(QWEN3VL_PY, "w") as f:
    f.write(content)

print("patch_212b_mapper: applied (added model. -> language_model.model. fallback)")
