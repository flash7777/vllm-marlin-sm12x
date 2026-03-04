#!/usr/bin/env python3
"""Patch: Force MTP layers to load as BF16 (ignore base model quantization).

Problem: When the base model is GPTQ-quantized (e.g., INT4 AutoRound),
vLLM 0.16 constructs MTP layers with GPTQ quant_config. FusedMoE then
creates GPTQ params (qweight/qzeros/scales) instead of BF16 params
(w13_weight/w2_weight), causing KeyError when loading BF16 MTP weights.

Fix: Temporarily set quant_config=None on the existing vllm_config during
MTP model construction, then restore it. This avoids creating a new
VllmConfig (which triggers __post_init__ and layer registry conflicts).

Activation: Set environment variable VLLM_MTP_FORCE_BF16=1 to enable.
"""

import os
import sys

EAGLE_FILE = "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py"

# Match any version of _get_model (original or previously patched)
# Strategy: find the function and replace it entirely

NEW_GET_MODEL = '''\
    def _get_model(self) -> nn.Module:
        """
        Default method to call get_model(). Can be overridden by subclasses which
        need to customize model loading.
        """
        from vllm.compilation.backends import set_model_tag

        # MTP BF16 patch: temporarily strip quant_config so MTP layers are
        # constructed as unquantized BF16. This is needed when the base model
        # is GPTQ-quantized but MTP weights in the checkpoint are BF16.
        _force_bf16 = os.environ.get("VLLM_MTP_FORCE_BF16", "0") == "1"
        _saved_qc = None
        if _force_bf16 and self.vllm_config.quant_config is not None:
            _saved_qc = self.vllm_config.quant_config
            object.__setattr__(self.vllm_config, 'quant_config', None)
            import logging
            logging.getLogger(__name__).info(
                "MTP BF16: temporarily disabled quant_config for MTP model")

        try:
            with set_model_tag("eagle_head"):
                model = get_model(
                    vllm_config=self.vllm_config,
                    model_config=self.speculative_config.draft_model_config,
                    load_config=self.speculative_config.draft_load_config,
                )
        finally:
            if _saved_qc is not None:
                object.__setattr__(self.vllm_config, 'quant_config', _saved_qc)
                import logging
                logging.getLogger(__name__).info(
                    "MTP BF16: restored quant_config after MTP model loading")

        return model'''


def main():
    if not os.path.exists(EAGLE_FILE):
        print(f"SKIP: {EAGLE_FILE} not found")
        sys.exit(0)

    with open(EAGLE_FILE) as f:
        content = f.read()

    # Ensure 'import os' is at the top
    if "\nimport os\n" not in content:
        content = content.replace(
            "\nimport torch\n",
            "\nimport os\nimport torch\n",
        )

    # Find and replace _get_model method
    # Look for the method start and end
    import re
    # Match from "    def _get_model(self)" to the next method at same indent level
    pattern = r'(    def _get_model\(self\) -> nn\.Module:.*?)(    def load_model)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        if "VLLM_MTP_FORCE_BF16" in content and "finally:" in content:
            print("SKIP: MTP BF16 v3 patch already applied")
            return
        print(f"ERROR: Could not find _get_model method in {EAGLE_FILE}")
        sys.exit(1)

    old_method = match.group(1)
    content = content.replace(old_method, NEW_GET_MODEL + "\n\n")

    with open(EAGLE_FILE, "w") as f:
        f.write(content)

    print(f"OK: MTP BF16 v3 patch applied to {EAGLE_FILE}")
    print("   Set VLLM_MTP_FORCE_BF16=1 to force BF16 MTP weights")
    print("   Temporarily patches vllm_config.quant_config during MTP loading")


if __name__ == "__main__":
    main()
