#!/usr/bin/env python3
"""Create expert mapping: original 397B expert indices -> REAP-262B pruned indices.

Both models use stacked expert tensors:
  layers.X.mlp.experts.down_proj  shape [num_experts, hidden, intermediate]

We extract uint16 fingerprints from down_proj and match.
"""

import json
import os
import re
import numpy as np
import ml_dtypes  # registers bfloat16 with numpy
from collections import defaultdict
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO_ORIG = "Qwen/Qwen3.5-397B-A17B"
REPO_REAP = "OpenMOSE/Qwen3.5-REAP-262B-A17B"
WORK_DIR = "/data/tensordata/expert_mapping_work"
OUTPUT = "/data/tensordata/expert_mapping.json"

os.makedirs(WORK_DIR, exist_ok=True)


def get_fingerprints(repo, label):
    """Extract fingerprints from stacked expert down_proj tensors."""
    idx_path = hf_hub_download(repo, "model.safetensors.index.json",
                               local_dir=os.path.join(WORK_DIR, label))
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    # Find stacked down_proj tensors
    down_tensors = {}
    for k, v in wm.items():
        m = re.match(r".*layers\.(\d+)\.mlp\.experts\.down_proj$", k)
        if m:
            down_tensors[k] = (int(m.group(1)), v)

    fingerprints = {}
    shards_needed = defaultdict(list)
    for k, (layer, shard) in down_tensors.items():
        shards_needed[shard].append((layer, k))

    for i, (shard, items) in enumerate(sorted(shards_needed.items())):
        print(f"  [{label}] [{i+1}/{len(shards_needed)}] {shard}", flush=True)
        path = hf_hub_download(repo, shard, local_dir=os.path.join(WORK_DIR, label))

        with safe_open(path, framework="numpy") as f:
            for layer, tensor_name in items:
                t = f.get_tensor(tensor_name)
                num_experts = t.shape[0]
                for expert_idx in range(num_experts):
                    raw = t[expert_idx].view(np.uint16).flatten()[:4]
                    fp = tuple(raw.tolist())
                    fingerprints[(layer, expert_idx)] = fp

        os.remove(path)
        print(f"    Layer {items[0][0]}: {num_experts} experts", flush=True)

    return fingerprints


print("=== Phase 1: Extract REAP-262B fingerprints ===", flush=True)
reap_fps = get_fingerprints(REPO_REAP, "reap")
print(f"REAP fingerprints: {len(reap_fps)}\n", flush=True)

print("=== Phase 2: Extract 397B original fingerprints ===", flush=True)
orig_fps = get_fingerprints(REPO_ORIG, "orig")
print(f"Original fingerprints: {len(orig_fps)}\n", flush=True)

# Build reverse lookup: layer -> {fingerprint -> orig_expert_idx}
print("=== Phase 3: Match fingerprints ===", flush=True)
orig_by_layer_fp = defaultdict(dict)
for (layer, expert), fp in orig_fps.items():
    orig_by_layer_fp[layer][fp] = expert

# Build mapping: layer -> {pruned_idx: original_idx}
mapping = {}
unmatched = 0
for (layer, reap_idx), fp in sorted(reap_fps.items()):
    layer_str = str(layer)
    if layer_str not in mapping:
        mapping[layer_str] = {}

    if fp in orig_by_layer_fp[layer]:
        orig_idx = orig_by_layer_fp[layer][fp]
        mapping[layer_str][str(reap_idx)] = orig_idx
    else:
        print(f"  WARNING: No match for layer {layer} REAP expert {reap_idx}")
        unmatched += 1

total = len(reap_fps)
matched = total - unmatched
print(f"\nMatched: {matched}/{total} experts ({100*matched/total:.1f}%)")

# Per-layer stats
print("\nPer-layer expert counts:")
for layer in sorted(mapping.keys(), key=int):
    n_reap = len(mapping[layer])
    pruned = 512 - n_reap
    print(f"  Layer {int(layer):2d}: {n_reap} kept, {pruned} pruned")

# Save mapping
with open(OUTPUT, "w") as f:
    json.dump(mapping, f, indent=2)
print(f"\nSaved mapping to {OUTPUT}")

# Save which originals were pruned per layer
pruned_map = {}
for layer_str, m in mapping.items():
    kept_orig = set(m.values())
    pruned_map[layer_str] = sorted(set(range(512)) - kept_orig)

pruned_path = OUTPUT.replace(".json", "_pruned.json")
with open(pruned_path, "w") as f:
    json.dump(pruned_map, f, indent=2)
print(f"Saved pruned indices to {pruned_path}")
