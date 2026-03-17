#!/usr/bin/env python3
"""Download REAP-262B shards one by one, extract shared_expert tensors, delete shard.
Uses safe_open to read individual tensors without mmap-ing the entire shard."""

import json
import os
from collections import defaultdict
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "OpenMOSE/Qwen3.5-REAP-262B-A17B"
BASE_DIR = "/data2/tensordata/REAP-262B-BF16"
OUTPUT = "/data2/tensordata/REAP-262B-shared-expert.safetensors"

# Load index
with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
    idx = json.load(f)

# Find shared_expert tensors grouped by shard
by_shard = defaultdict(list)
for tensor_name, shard_file in idx["weight_map"].items():
    if "shared_expert." in tensor_name:
        by_shard[shard_file].append(tensor_name)

print(f"Need to process {len(by_shard)} shards for {sum(len(v) for v in by_shard.values())} tensors")

all_tensors = {}
for i, (shard_file, tensor_names) in enumerate(sorted(by_shard.items())):
    print(f"[{i+1}/{len(by_shard)}] Downloading {shard_file} ...", flush=True)
    path = hf_hub_download(REPO, shard_file, local_dir=BASE_DIR)

    print(f"  Extracting: {', '.join(t.split('.')[-2] for t in tensor_names)}", flush=True)
    with safe_open(path, framework="pt") as f:
        for name in tensor_names:
            tensor = f.get_tensor(name)
            all_tensors[name] = tensor.clone()
            print(f"    {name}: {list(tensor.shape)} {tensor.dtype}")

    # Delete shard to free space
    os.remove(path)
    print(f"  Deleted {shard_file}", flush=True)

print(f"\nSaving {len(all_tensors)} tensors to {OUTPUT} ...", flush=True)
save_file(all_tensors, OUTPUT)
size_gb = os.path.getsize(OUTPUT) / (1024**3)
print(f"Done! Output: {size_gb:.1f} GB")
