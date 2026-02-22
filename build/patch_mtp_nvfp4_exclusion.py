#!/usr/bin/env python3
"""
patch_mtp_nvfp4_exclusion.py — Fix MTP layer exclusion for NVFP4

Bug: NVFP4 checkpoint exclude list has 'mtp.layers.0*' to skip MTP decoder
layers, but ALL MTP weights (including mtp.fc, mtp.norm, etc.) are stored as
BF16 in the checkpoint. The pattern 'mtp.layers.0*' does NOT match 'mtp.fc',
so the fc layer gets initialized as FP4 quantized while the checkpoint has
BF16 weights — causing a shape mismatch assertion failure.

Fix: In is_layer_excluded(), if any exclude pattern starts with 'mtp.' and
the current prefix also starts with 'mtp.', exclude the entire MTP module.

Source: Avarok-Cybersecurity/dgx-vllm (fix_mtp_nvfp4_exclusion.py)
Adapted for vllm-next2 install paths.

Run AFTER vLLM installation (post-install patch).
"""

import os
import sys
import glob


def find_modelopt():
    """Find modelopt.py in installed vLLM."""
    candidates = [
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py",
        "/usr/local/lib/python3.*/dist-packages/vllm/model_executor/layers/quantization/modelopt.py",
    ]
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def patch_modelopt(target):
    with open(target, "r") as f:
        content = f.read()

    # Check if already patched
    if 'prefix.startswith("mtp.")' in content:
        print(f"Already patched: {target}")
        return True

    # Find the is_layer_excluded method's fnmatch block
    old = """        # modelopt exclude modules are not simple strings, they are wildcards
        for wildcard_pattern in self.exclude_modules:
            if fnmatch(prefix, wildcard_pattern):
                return True

        return False"""

    new = """        # modelopt exclude modules are not simple strings, they are wildcards
        for wildcard_pattern in self.exclude_modules:
            if fnmatch(prefix, wildcard_pattern):
                return True

        # All MTP weights in NVFP4 checkpoints are BF16 (unquantized).
        # The exclude list only has 'mtp.layers.0*' which misses 'mtp.fc'.
        # If any exclude pattern targets mtp.*, exclude all mtp.* layers.
        if prefix.startswith("mtp."):
            for wildcard_pattern in self.exclude_modules:
                if wildcard_pattern.startswith("mtp."):
                    return True

        return False"""

    if old not in content:
        # Try alternative pattern (whitespace differences)
        import re
        pattern = re.compile(
            r'(# modelopt exclude modules.*?for wildcard_pattern in self\.exclude_modules:\s*\n'
            r'\s*if fnmatch\(prefix, wildcard_pattern\):\s*\n'
            r'\s*return True\s*\n)'
            r'(\s*return False)',
            re.DOTALL
        )
        match = pattern.search(content)
        if match:
            mtp_block = """
        # All MTP weights in NVFP4 checkpoints are BF16 (unquantized).
        if prefix.startswith("mtp."):
            for wildcard_pattern in self.exclude_modules:
                if wildcard_pattern.startswith("mtp."):
                    return True

"""
            content = content[:match.end(1)] + mtp_block + content[match.start(2):]
        else:
            print(f"ERROR: Could not find target pattern in {target}")
            print("  The fnmatch/exclude_modules block was not found.")
            print("  modelopt.py may have changed — manual patching needed.")
            return False
    else:
        content = content.replace(old, new)

    with open(target, "w") as f:
        f.write(content)
    print(f"Patched {target}: ALL MTP layers now excluded from NVFP4")
    return True


def main():
    print("=" * 70)
    print("MTP NVFP4 Exclusion Patch")
    print("=" * 70)

    target = find_modelopt()
    if not target:
        print("ERROR: modelopt.py not found")
        print("  Expected in /usr/local/lib/python3.*/dist-packages/vllm/")
        sys.exit(1)

    print(f"Target: {target}")
    success = patch_modelopt(target)

    if success:
        print()
        print("MTP layers (mtp.fc, mtp.norm, mtp.layers.*) will now be")
        print("excluded from NVFP4 quantization, keeping them in BF16.")
        print()
        print("Usage: --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'")
    else:
        sys.exit(1)

    print("=" * 70)


if __name__ == "__main__":
    main()
