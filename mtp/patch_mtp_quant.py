#!/usr/bin/env python3
"""Patch vLLM to pass quant_config to MTP drafter model.

Bug: For MTP method, speculative.py sets self.model and self.quantization
but NOT self.draft_model_config. This means the drafter VllmConfig has
quant_config=None → MoE experts load as Unquantized BF16 instead of INT4.

Fix: Add draft_model_config = target_model_config for MTP method,
same as ngram does.
"""

import sys

SPEC_FILE = "/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py"

with open(SPEC_FILE) as f:
    src = f.read()

# Find the MTP block and add draft_model_config
OLD = '''                self.model = self.target_model_config.model
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization'''

NEW = '''                self.model = self.target_model_config.model
                # MTP uses the same model — inherit full config for quant
                self.draft_model_config = self.target_model_config
                self.draft_parallel_config = self.target_parallel_config
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization'''

if OLD in src:
    src = src.replace(OLD, NEW)
    with open(SPEC_FILE, 'w') as f:
        f.write(src)
    print("OK: Patched speculative.py — MTP draft_model_config inherits from target")
elif "self.draft_model_config = self.target_model_config" in src.split("mtp")[1].split("ngram")[0]:
    print("Already patched")
else:
    print("WARNING: Pattern not found — speculative.py may have changed")
    sys.exit(1)
