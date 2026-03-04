#!/usr/bin/env python3
"""Extract MTP (Multi-Token Prediction) weights from BF16 model to INT4 AutoRound model.

vLLM 0.16 supports native MTP for GLM-4.7-Flash. The MTP layer is an extra decoder
layer (layer N, where N = num_hidden_layers) that acts as a draft predictor.

Problem: INT4 AutoRound quantization drops MTP weights (only layers 0..N-1 are quantized).
Solution: Copy the BF16 MTP weights into the INT4 model directory as a separate safetensors file.

Note: The BF16 MTP weights use individual expert format (experts.0.gate_proj.weight),
but vLLM 0.16's FusedMoE loader expects stacked format (experts.w13_weight, experts.w2_weight).
When using --speculative-config '{"method":"mtp"}', vLLM loads the MTP weights from the
main model, so this extraction ensures the INT4 model has the MTP layer available.

Usage:
    python3 extract_mtp_weights.py \\
        --bf16-model /data/tensordata/GLM-4.7-Flash \\
        --int4-model /data/tensordata/GLM-4.7-Flash-int4-AutoRound

Requirements:
    pip install safetensors torch
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors")
    sys.exit(1)


def find_mtp_config(config_path: Path) -> tuple[int, int]:
    """Find MTP layer range from model config. Returns (num_hidden, num_mtp)."""
    with open(config_path / "config.json") as f:
        config = json.load(f)

    num_hidden = config.get("num_hidden_layers")
    num_mtp = config.get("num_nextn_predict_layers", 0)

    if not num_mtp:
        print(f"ERROR: Model has no MTP layers (num_nextn_predict_layers={num_mtp})")
        sys.exit(1)

    print(f"  num_hidden_layers: {num_hidden}")
    print(f"  num_nextn_predict_layers: {num_mtp}")
    print(f"  MTP layer range: {num_hidden}..{num_hidden + num_mtp - 1}")
    return num_hidden, num_mtp


def extract_mtp_weights(bf16_path: Path, int4_path: Path, dry_run: bool = False):
    """Extract MTP weights from BF16 model and add to INT4 model."""

    print(f"\n=== MTP Weight Extraction ===")
    print(f"  BF16 source: {bf16_path}")
    print(f"  INT4 target: {int4_path}")

    # Find MTP layer range
    num_hidden, num_mtp = find_mtp_config(bf16_path)
    mtp_prefixes = [f"model.layers.{num_hidden + i}." for i in range(num_mtp)]

    # Scan BF16 safetensors files for MTP weights
    bf16_files = sorted(bf16_path.glob("model*.safetensors"))
    if not bf16_files:
        print("ERROR: No safetensors files found in BF16 model directory")
        sys.exit(1)

    print(f"\n  Scanning {len(bf16_files)} BF16 safetensors files...")
    mtp_tensors = {}

    for sf_path in bf16_files:
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(key.startswith(prefix) for prefix in mtp_prefixes):
                    mtp_tensors[key] = f.get_tensor(key)

    if not mtp_tensors:
        print(f"ERROR: No MTP weights found with prefix '{mtp_prefix}'")
        sys.exit(1)

    total_params = sum(t.numel() for t in mtp_tensors.values())
    total_bytes = sum(t.numel() * t.element_size() for t in mtp_tensors.values())
    print(f"  Found {len(mtp_tensors)} MTP tensors ({total_params:,} params, {total_bytes / 1e9:.2f} GB)")

    if dry_run:
        print("\n  [DRY RUN] Would save to:")
        print(f"    {int4_path}/model-mtp-00001-of-00001.safetensors")
        for key in sorted(mtp_tensors.keys())[:5]:
            t = mtp_tensors[key]
            print(f"    {key}: {list(t.shape)} {t.dtype}")
        if len(mtp_tensors) > 5:
            print(f"    ... and {len(mtp_tensors) - 5} more")
        return

    # Save MTP weights as separate safetensors file
    mtp_file = "model-mtp-00001-of-00001.safetensors"
    mtp_path = int4_path / mtp_file
    print(f"\n  Saving MTP weights to: {mtp_path}")
    save_file(mtp_tensors, str(mtp_path))
    print(f"  Saved: {os.path.getsize(mtp_path) / 1e9:.2f} GB")

    # Update model.safetensors.index.json
    index_path = int4_path / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"WARNING: {index_path} not found, skipping index update")
        return

    with open(index_path) as f:
        index = json.load(f)

    # Add MTP weight mappings
    added = 0
    for key in sorted(mtp_tensors.keys()):
        if key not in index["weight_map"]:
            index["weight_map"][key] = mtp_file
            added += 1

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"  Updated index: added {added} MTP weight mappings")

    # Patch config.json if num_nextn_predict_layers is missing
    int4_config_path = int4_path / "config.json"
    with open(int4_config_path) as f:
        int4_config = json.load(f)

    if not int4_config.get("num_nextn_predict_layers"):
        int4_config["num_nextn_predict_layers"] = num_mtp
        with open(int4_config_path, "w") as f:
            json.dump(int4_config, f, indent=2, ensure_ascii=False)
        print(f"  Patched config.json: num_nextn_predict_layers={num_mtp}")

    print(f"\n  Done! MTP weights are now available in the INT4 model.")
    print(f"\n  To serve with MTP on vLLM 0.16:")
    print(f"    --speculative-config '{{\"method\":\"mtp\",\"num_speculative_tokens\":1}}'")


def main():
    parser = argparse.ArgumentParser(description="Extract MTP weights from BF16 to INT4 model")
    parser.add_argument("--bf16-model", type=str, required=True,
                        help="Path to BF16 model with MTP weights")
    parser.add_argument("--int4-model", type=str, required=True,
                        help="Path to INT4 AutoRound model (target)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing files")
    args = parser.parse_args()

    bf16_path = Path(args.bf16_model)
    int4_path = Path(args.int4_model)

    if not bf16_path.exists():
        print(f"ERROR: BF16 model not found: {bf16_path}")
        sys.exit(1)
    if not int4_path.exists():
        print(f"ERROR: INT4 model not found: {int4_path}")
        sys.exit(1)

    extract_mtp_weights(bf16_path, int4_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
