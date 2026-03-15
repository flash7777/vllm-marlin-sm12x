#!/usr/bin/env python3
"""Extract MTP weights from BF16 Qwen3.5-397B into a separate safetensors file.

The BF16 model has 1553 MTP keys (512 experts × 3 proj + attention/norms)
totaling ~13 GB BF16. These are missing from the Intel AutoRound INT4 model.

Usage:
    python3 extract_mtp_weights.py \
        --source /data/tensordata/Qwen3.5-397B-A17B \
        --output /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/mtp_weights.safetensors

    # Verify:
    python3 extract_mtp_weights.py \
        --source /data/tensordata/Qwen3.5-397B-A17B \
        --info-only

After extraction, add to model config.json:
    "num_nextn_predict_layers": 1

Then start vllm with:
    --num-speculative-tokens 1

Requires: safetensors, torch (for tensor loading)
Source model: Qwen/Qwen3.5-397B-A17B (BF16, ~750 GB, on RTX/Spiegel 2)
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def get_mtp_info(model_dir):
    """Analyze MTP weights in model."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)

    wm = idx["weight_map"]
    mtp_keys = {k: v for k, v in wm.items() if "mtp" in k.lower()}

    # Group by shard
    shard_keys = defaultdict(list)
    for k, v in mtp_keys.items():
        shard_keys[v].append(k)

    # Categorize
    expert_keys = [k for k in mtp_keys if "experts" in k]
    attn_keys = [k for k in mtp_keys if "self_attn" in k]
    other_keys = [k for k in mtp_keys if k not in expert_keys and k not in attn_keys]

    return {
        "total_keys": len(wm),
        "mtp_keys": len(mtp_keys),
        "expert_keys": len(expert_keys),
        "attn_keys": len(attn_keys),
        "other_keys": len(other_keys),
        "other_key_names": other_keys,
        "shards": {s: len(keys) for s, keys in sorted(shard_keys.items())},
        "shard_keys": shard_keys,
        "all_mtp_keys": sorted(mtp_keys.keys()),
    }


def extract_mtp(model_dir, output_path):
    """Extract MTP weights into a single safetensors file."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    info = get_mtp_info(model_dir)
    print(f"Extracting {info['mtp_keys']} MTP keys from {len(info['shards'])} shards...")

    tensors = {}
    for shard, keys in sorted(info["shard_keys"].items()):
        shard_path = os.path.join(model_dir, shard)
        print(f"  {shard}: {len(keys)} keys...", end=" ", flush=True)
        with safe_open(shard_path, framework="pt") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
        print("OK")

    print(f"\nSaving {len(tensors)} tensors to {output_path}...")
    save_file(tensors, output_path)

    size_gb = os.path.getsize(output_path) / 1e9
    print(f"Done: {output_path} ({size_gb:.2f} GB)")
    return size_gb


def main():
    parser = argparse.ArgumentParser(description="Extract MTP weights from BF16 Qwen3.5-397B")
    parser.add_argument("--source", required=True, help="Path to BF16 model directory")
    parser.add_argument("--output", help="Output safetensors file path")
    parser.add_argument("--info-only", action="store_true", help="Only show info, don't extract")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: {args.source} not found")
        sys.exit(1)

    info = get_mtp_info(args.source)

    print(f"Model: {args.source}")
    print(f"Total keys: {info['total_keys']}")
    print(f"MTP keys: {info['mtp_keys']}")
    print(f"  Experts: {info['expert_keys']}")
    print(f"  Attention: {info['attn_keys']}")
    print(f"  Other: {info['other_keys']}")
    for name in info["other_key_names"]:
        print(f"    {name}")
    print(f"Shards containing MTP:")
    for shard, count in info["shards"].items():
        shard_path = os.path.join(args.source, shard)
        size = os.path.getsize(shard_path) / 1e9 if os.path.exists(shard_path) else 0
        print(f"  {shard}: {count} keys ({size:.2f} GB)")

    if args.info_only:
        return

    if not args.output:
        print("\nError: --output required for extraction")
        sys.exit(1)

    extract_mtp(args.source, args.output)

    print(f"\nNext steps:")
    print(f"  1. Copy to INT4 model dir:")
    print(f"     cp {args.output} /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/")
    print(f"  2. Add to config.json: \"num_nextn_predict_layers\": 1")
    print(f"  3. Start with: --num-speculative-tokens 1")


if __name__ == "__main__":
    main()
