#!/usr/bin/env python3
"""Patch: Control MTP weight format independently from base model quantization.

Problem: When the base model is GPTQ-quantized (e.g., INT4 AutoRound),
vLLM constructs MTP layers with GPTQ quant_config. This fails when:
- MTP weights are BF16 (KeyError on w13_weight vs w13_qweight)
- MTP weights are INT4 but quant_config doesn't cover mtp.* prefix

Fix: Check VLLM_MTP_FORCE env var at MTP model construction time:
- VLLM_MTP_FORCE=BF16  → strip quant_config → BF16 FusedMoE
- VLLM_MTP_FORCE=INT4  → keep quant_config → GPTQ FusedMoE (+ patch_mtp_quant.py)
- unset                → default vLLM behavior (inherits base quant_config)

Also supports legacy VLLM_MTP_FORCE_BF16=1 for backwards compatibility.
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
        import logging

        # VLLM_MTP_FORCE: control MTP weight format
        #   BF16 → strip quant_config (MTP weights are unquantized BF16)
        #   INT4 → keep quant_config (MTP weights are GPTQ INT4)
        #   unset → default behavior
        _mtp_force = os.environ.get("VLLM_MTP_FORCE", "").upper()
        # Legacy compat
        if not _mtp_force and os.environ.get("VLLM_MTP_FORCE_BF16", "0") == "1":
            _mtp_force = "BF16"

        _saved_qc = None
        if _mtp_force == "BF16" and self.vllm_config.quant_config is not None:
            _saved_qc = self.vllm_config.quant_config
            object.__setattr__(self.vllm_config, 'quant_config', None)
            logging.getLogger(__name__).info(
                "MTP: VLLM_MTP_FORCE=BF16 — disabled quant_config for MTP model")
        elif _mtp_force == "INT4":
            logging.getLogger(__name__).info(
                "MTP: VLLM_MTP_FORCE=INT4 — keeping quant_config for MTP model")
        elif _mtp_force:
            logging.getLogger(__name__).warning(
                f"MTP: unknown VLLM_MTP_FORCE={_mtp_force!r}, ignoring")

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
                logging.getLogger(__name__).info(
                    "MTP: restored quant_config after MTP model loading")

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
        if "VLLM_MTP_FORCE" in content and "finally:" in content:
            print("SKIP: MTP FORCE patch already applied")
            return
        print(f"ERROR: Could not find _get_model method in {EAGLE_FILE}")
        sys.exit(1)

    old_method = match.group(1)
    content = content.replace(old_method, NEW_GET_MODEL + "\n\n")

    with open(EAGLE_FILE, "w") as f:
        f.write(content)

    print(f"OK: MTP FORCE patch applied to {EAGLE_FILE}")
    print("   VLLM_MTP_FORCE=BF16  → MTP loads as unquantized BF16")
    print("   VLLM_MTP_FORCE=INT4  → MTP loads as GPTQ INT4 (needs patch_mtp_quant.py)")
    print("   unset                → default vLLM behavior")


if __name__ == "__main__":
    main()
