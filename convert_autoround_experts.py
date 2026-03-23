#!/usr/bin/env python3
"""Convert AutoRound per-expert fused GPTQ weights to vLLM-compatible format.

AutoRound format:
    model.language_model.layers.X.mlp.experts.gate_up_proj.{id}.{qweight|qzeros|scales}
    model.language_model.layers.X.mlp.experts.down_proj.{id}.{qweight|qzeros|scales}

vLLM standard format:
    model.layers.X.mlp.experts.{id}.gate_proj.{qweight|qzeros|scales}
    model.layers.X.mlp.experts.{id}.up_proj.{qweight|qzeros|scales}
    model.layers.X.mlp.experts.{id}.down_proj.{qweight|qzeros|scales}

Transformations:
    1. Strip "model.language_model." -> "model."
    2. Remap experts.down_proj.{id}.X -> experts.{id}.down_proj.X
    3. Split experts.gate_up_proj.{id}.X into experts.{id}.gate_proj.X + experts.{id}.up_proj.X
       For GPTQ auto_gptq format: qweight=[in/pack, out], split on dim 1 (output dim)

Usage:
    python3 convert_autoround_experts.py --input /path/to/model --output /path/to/converted
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert_key_and_tensor(name, tensor):
    """Convert a single key+tensor. Returns list of (new_name, new_tensor) pairs."""
    # Keep VL prefix as-is (vLLM's hf_to_vllm_mapper handles it)
    # Only remap expert key order within the existing prefix

    # Match expert pattern: experts.{proj}.{id}.{suffix}
    m = re.match(
        r"(.*\.mlp\.experts)\.(gate_up_proj|down_proj)\.(\d+)\.(.*)",
        name,
    )
    if not m:
        return [(name, tensor)]

    pfx, proj, eid, sfx = m.groups()

    if proj == "down_proj":
        return [(f"{pfx}.{eid}.down_proj.{sfx}", tensor)]
    else:
        # gate_up_proj: split into gate_proj + up_proj
        # GPTQ auto_gptq format:
        #   qweight: [in_features/pack_factor, out_features] - split dim 1
        #   scales:  [num_groups, out_features]               - split dim 1
        #   qzeros:  [num_groups, out_features/pack_factor]   - split dim 1
        # In all cases, gate and up are concatenated on the LAST dimension
        split_dim = tensor.ndim - 1
        half = tensor.shape[split_dim] // 2
        gate = tensor.narrow(split_dim, 0, half).contiguous()
        up = tensor.narrow(split_dim, half, half).contiguous()
        return [
            (f"{pfx}.{eid}.gate_proj.{sfx}", gate),
            (f"{pfx}.{eid}.up_proj.{sfx}", up),
        ]


def main():
    parser = argparse.ArgumentParser(description="Convert AutoRound expert weights")
    parser.add_argument("--input", required=True, help="Input model directory")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--dry-run", action="store_true", help="Only show conversions")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Load index
    index_path = input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    # Collect shard files that need conversion
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Shards: {len(shard_files)}")

    if args.dry_run:
        # Show sample conversions
        count = 0
        for name in sorted(index["weight_map"].keys()):
            results = convert_key_and_tensor(name, None)
            for new_name, _ in results:
                if new_name != name:
                    print(f"  {name}")
                    print(f"  -> {new_name}")
                    count += 1
                    if count >= 10:
                        print(f"  ... and more")
                        return
        return

    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build new weight map
    new_weight_map = {}
    total_converted = 0

    for shard_idx, shard_file in enumerate(shard_files):
        shard_path = input_dir / shard_file
        print(f"[{shard_idx+1}/{len(shard_files)}] Processing {shard_file}...")

        tensors = load_file(str(shard_path))
        new_tensors = {}

        for name, tensor in tensors.items():
            results = convert_key_and_tensor(name, tensor)
            for new_name, new_tensor in results:
                new_tensors[new_name] = new_tensor
                new_weight_map[new_name] = shard_file
                if new_name != name:
                    total_converted += 1

        # Save converted shard
        save_file(new_tensors, str(output_dir / shard_file))

    # Update index
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy non-safetensor files
    for fname in os.listdir(input_dir):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        src = input_dir / fname
        dst = output_dir / fname
        if src.is_file() and not dst.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Copied {fname}")

    # Keep config.json as-is (VL architecture preserved)

    print(f"\nDone! Converted {total_converted} weight keys.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
