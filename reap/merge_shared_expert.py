#!/usr/bin/env python3
"""Merge BF16 shared_expert weights into REAP-262B-RTX INT4 model.

Replaces quantized shared_expert tensors (qweight/qzeros/scales) with
original BF16 weight tensors from the REAP-262B BF16 source model.
Updates quantize_config.json and model.safetensors.index.json accordingly.
"""

import json
import os
import shutil
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import save_file

SRC_INT4 = "/data/tensordata/Qwen3.5-REAP-262B-A17B-int4-AutoRound-RTX"
BF16_WEIGHTS = "/data/tensordata/REAP-262B-shared-expert.safetensors"
OUTPUT = "/data/tensordata/Qwen3.5-REAP-262B-A17B-int4-AutoRound-RTX-fixed"

os.makedirs(OUTPUT, exist_ok=True)

# Load BF16 shared expert weights
print("Loading BF16 shared expert weights...", flush=True)
bf16_tensors = {}
with safe_open(BF16_WEIGHTS, framework="pt") as f:
    for name in f.keys():
        bf16_tensors[name] = f.get_tensor(name)
        print(f"  {name}: {list(bf16_tensors[name].shape)} {bf16_tensors[name].dtype}")
print(f"Loaded {len(bf16_tensors)} BF16 tensors\n")

# Load index
with open(os.path.join(SRC_INT4, "model.safetensors.index.json")) as f:
    idx = json.load(f)
wm = idx["weight_map"]

# Find which shards have shared_expert quantized tensors
se_by_shard = defaultdict(list)
for tensor_name, shard_file in wm.items():
    if ".shared_expert." in tensor_name and "shared_expert_gate" not in tensor_name:
        se_by_shard[shard_file].append(tensor_name)

all_shards = sorted(set(wm.values()))
new_weight_map = {}

for i, shard_file in enumerate(all_shards):
    src_path = os.path.join(SRC_INT4, shard_file)
    dst_path = os.path.join(OUTPUT, shard_file)

    if shard_file not in se_by_shard:
        # No shared_expert tensors — copy as-is
        print(f"[{i+1}/{len(all_shards)}] {shard_file} — no changes, copying", flush=True)
        shutil.copy2(src_path, dst_path)
        for k, v in wm.items():
            if v == shard_file:
                new_weight_map[k] = shard_file
        continue

    print(f"[{i+1}/{len(all_shards)}] {shard_file} — replacing {len(se_by_shard[shard_file])} quantized tensors", flush=True)

    # Load all tensors from this shard
    new_tensors = {}
    with safe_open(src_path, framework="pt") as f:
        for name in f.keys():
            if name in se_by_shard[shard_file]:
                # Skip quantized shared_expert tensors
                continue
            new_tensors[name] = f.get_tensor(name)

    # Find which layers' shared_expert are in this shard
    layers_in_shard = set()
    for t in se_by_shard[shard_file]:
        # Extract layer number from tensor name
        parts = t.split(".")
        layer_idx = parts[3]  # model.language_model.layers.X.mlp...
        layers_in_shard.add(int(layer_idx))

    # Add BF16 weights for these layers
    for layer_idx in sorted(layers_in_shard):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            bf16_name = f"model.language_model.layers.{layer_idx}.mlp.shared_expert.{proj}.weight"
            if bf16_name in bf16_tensors:
                new_tensors[bf16_name] = bf16_tensors[bf16_name]
                print(f"  + {bf16_name}")
            else:
                print(f"  WARNING: {bf16_name} not found in BF16 weights!")

    # Save modified shard
    save_file(new_tensors, dst_path)

    # Update weight map: remove old quantized keys, add new weight keys
    for k, v in wm.items():
        if v == shard_file:
            if k in se_by_shard[shard_file]:
                continue  # Drop quantized tensor from map
            new_weight_map[k] = shard_file

    for layer_idx in sorted(layers_in_shard):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            bf16_name = f"model.language_model.layers.{layer_idx}.mlp.shared_expert.{proj}.weight"
            new_weight_map[bf16_name] = shard_file

# Save new index
new_idx = {"metadata": idx.get("metadata", {}), "weight_map": dict(sorted(new_weight_map.items()))}
with open(os.path.join(OUTPUT, "model.safetensors.index.json"), "w") as f:
    json.dump(new_idx, f, indent=2)
print(f"\nSaved new index with {len(new_weight_map)} tensors")

# Copy and patch quantize_config.json
with open(os.path.join(SRC_INT4, "quantize_config.json")) as f:
    qcfg = json.load(f)

# Add shared_expert MLPs to extra_config
if "extra_config" not in qcfg:
    qcfg["extra_config"] = {}
for layer_idx in range(60):
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        key = f"model.language_model.layers.{layer_idx}.mlp.shared_expert.{proj}"
        qcfg["extra_config"][key] = {"bits": 16, "data_type": "float"}

with open(os.path.join(OUTPUT, "quantize_config.json"), "w") as f:
    json.dump(qcfg, f, indent=2)
print("Saved patched quantize_config.json")

# Copy config.json and other metadata
for fname in os.listdir(SRC_INT4):
    src = os.path.join(SRC_INT4, fname)
    dst = os.path.join(OUTPUT, fname)
    if os.path.isfile(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"Copied {fname}")

# Also patch config.json quantization_config
cfg_path = os.path.join(OUTPUT, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
if "quantization_config" in cfg:
    if "extra_config" not in cfg["quantization_config"]:
        cfg["quantization_config"]["extra_config"] = {}
    for layer_idx in range(60):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            key = f"model.language_model.layers.{layer_idx}.mlp.shared_expert.{proj}"
            cfg["quantization_config"]["extra_config"][key] = {"bits": 16, "data_type": "float"}
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Patched config.json quantization_config")

print("\nDone!")
