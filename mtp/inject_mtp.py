#!/usr/bin/env python3
"""Inject INT4 MTP weights into an INT4 AutoRound model for vLLM.

vLLM's Qwen3_5MultiTokenPredictor loads MTP weights from the main model
directory with prefix 'mtp.*'. This script:

1. Copies the INT4 MTP safetensors file into the model directory
2. Updates model.safetensors.index.json with MTP weight mappings
3. Patches config.json with mtp_num_hidden_layers if missing

Usage:
    python3 inject_mtp.py \\
        --mtp-weights /data/tensordata/mtp_weights_397b_int4/mtp_int4.safetensors \\
        --int4-model /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound \\
        --dry-run

Requirements:
    pip install safetensors (torch NOT required)
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from safetensors import safe_open


def inject_mtp_weights(mtp_path: str, model_dir: str, dry_run: bool = False):
    model_path = Path(model_dir)
    mtp_file = Path(mtp_path)

    if not mtp_file.exists():
        print(f"ERROR: MTP weights not found: {mtp_file}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model directory not found: {model_path}")
        sys.exit(1)

    print(f"=== MTP Weight Injection ===")
    print(f"  MTP source:  {mtp_file}")
    print(f"  INT4 target: {model_path}")

    # Verify MTP weights have mtp.* prefix
    with safe_open(str(mtp_file), framework="numpy") as f:
        keys = list(f.keys())
    mtp_keys = [k for k in keys if k.startswith("mtp.")]
    print(f"  MTP tensors: {len(keys)} total, {len(mtp_keys)} with mtp.* prefix")
    if not mtp_keys:
        print("ERROR: No tensors with 'mtp.' prefix found")
        sys.exit(1)

    # Target filename in model dir
    target_name = "model-mtp-00001-of-00001.safetensors"
    target_path = model_path / target_name

    if dry_run:
        print(f"\n  [DRY RUN] Would:")
        print(f"    1. Copy {mtp_file.name} -> {target_path}")
        print(f"    2. Add {len(keys)} entries to model.safetensors.index.json")
        print(f"    3. Set mtp_num_hidden_layers=1 in config.json if missing")
        for k in sorted(keys)[:10]:
            print(f"       {k}")
        if len(keys) > 10:
            print(f"       ... and {len(keys) - 10} more")
        return

    # 1. Copy MTP weights file + fix key format
    #    AutoRound produces: experts.{proj}.{id}.qweight
    #    vLLM expects:       experts.{id}.{proj}.qweight
    print(f"\n  Loading + fixing key format...")
    pattern = re.compile(r"(.*\.experts)\.(gate_proj|up_proj|down_proj)\.(\d+)\.(.*)")
    from safetensors.torch import save_file as torch_save_file
    tensors = {}
    fixes = 0
    with safe_open(str(mtp_file), framework="pt") as f:
        for key in f.keys():
            m = pattern.match(key)
            if m:
                prefix, proj, idx, suffix = m.groups()
                new_key = f"{prefix}.{idx}.{proj}.{suffix}"
                tensors[new_key] = f.get_tensor(key)
                fixes += 1
            else:
                tensors[key] = f.get_tensor(key)
    keys = list(tensors.keys())  # update keys list for index
    torch_save_file(tensors, str(target_path))
    print(f"  Saved: {os.path.getsize(target_path) / 1e9:.2f} GB ({fixes} keys reformatted)")

    # 2. Update model.safetensors.index.json
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)

        added = 0
        for key in sorted(keys):
            if key not in index["weight_map"]:
                index["weight_map"][key] = target_name
                added += 1

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print(f"  Updated index: +{added} MTP weight mappings")
    else:
        print(f"  WARNING: {index_path} not found, skipping index update")

    # 3. Patch config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        # Handle nested text_config or flat config
        text_cfg = config.get("text_config", config)
        if not text_cfg.get("mtp_num_hidden_layers"):
            text_cfg["mtp_num_hidden_layers"] = 1
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"  Patched config: mtp_num_hidden_layers=1")
        else:
            print(f"  Config already has mtp_num_hidden_layers={text_cfg['mtp_num_hidden_layers']}")

    print(f"\n  Done! MTP weights injected.")
    print(f"\n  vLLM serve with MTP:")
    print(f'    --speculative-config \'{{"method":"mtp","num_speculative_tokens":1}}\'')


def main():
    parser = argparse.ArgumentParser(
        description="Inject INT4 MTP weights into AutoRound model")
    parser.add_argument("--mtp-weights", type=str, required=True,
                        help="Path to INT4 MTP weights (.safetensors)")
    parser.add_argument("--int4-model", type=str, required=True,
                        help="Path to INT4 AutoRound model directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing")
    args = parser.parse_args()

    inject_mtp_weights(args.mtp_weights, args.int4_model, args.dry_run)


if __name__ == "__main__":
    main()
