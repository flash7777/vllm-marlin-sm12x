#!/usr/bin/env python3
"""Patch: Register Qwen3_5MoeForCausalLM in vLLM model registry.

The 212B REAP model has pure LLM weight keys (model.layers.N...)
but config says Qwen3_5MoeForConditionalGeneration (VL model expecting
model.language_model.layers.N...). We register the CausalLM variant.
"""
import sys

REGISTRY = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py"

with open(REGISTRY) as f:
    content = f.read()

if "Qwen3_5MoeForCausalLM" in content:
    print("patch_212b_arch: already applied")
    sys.exit(0)

# Add registration after the existing Qwen3_5MoeForConditionalGeneration entry
old = '''    "Qwen3_5MoeForConditionalGeneration": (
        "qwen3_5",
        "Qwen3_5MoeForConditionalGeneration",
    ),'''

new = '''    "Qwen3_5MoeForConditionalGeneration": (
        "qwen3_5",
        "Qwen3_5MoeForConditionalGeneration",
    ),
    "Qwen3_5MoeForCausalLM": (
        "qwen3_5",
        "Qwen3_5MoeForCausalLM",
    ),'''

if old not in content:
    print("ERROR: registry pattern not found")
    sys.exit(1)

content = content.replace(old, new)
with open(REGISTRY, "w") as f:
    f.write(content)

print("patch_212b_arch: Qwen3_5MoeForCausalLM registered")
