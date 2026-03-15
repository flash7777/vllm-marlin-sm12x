#!/usr/bin/env python3
"""Convert MTP weights from AutoRound format to main-model-compatible format.

Fixes:
1. gate_up_proj -> separate gate_proj + up_proj (split dim=-1)
2. Remove g_idx tensors (main model doesn't have them)
3. Fix key format: experts.gate_up_proj.{id} -> experts.{id}.gate_proj/up_proj

Usage:
    python3 fix_mtp_to_main_format.py /data/tensordata/.../model-mtp-00001-of-00001.safetensors
"""

import os
import re
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

path = sys.argv[1]
print(f"Converting MTP weights: {path}")

# Load all tensors
tensors = {}
with safe_open(path, framework="pt") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

print(f"  Input: {len(tensors)} tensors")

output = {}
splits = 0
dropped_gidx = 0

for key, tensor in tensors.items():
    # Drop g_idx
    if '.g_idx' in key:
        dropped_gidx += 1
        continue

    # Split gate_up_proj into gate_proj + up_proj
    # Pattern: mtp.layers.0.mlp.experts.gate_up_proj.{id}.{suffix}
    # Or:      mtp.layers.0.mlp.experts.{id}.gate_up_proj.{suffix}
    m1 = re.match(r"(.*\.experts)\.gate_up_proj\.(\d+)\.(qweight|scales|qzeros)", key)
    m2 = re.match(r"(.*\.experts)\.(\d+)\.gate_up_proj\.(qweight|scales|qzeros)", key)
    m = m1 or m2

    if m:
        prefix, eid, suffix = m.groups()
        half = tensor.shape[-1] // 2
        gate = tensor.narrow(-1, 0, half).contiguous()
        up = tensor.narrow(-1, half, half).contiguous()
        output[f"{prefix}.{eid}.gate_proj.{suffix}"] = gate
        output[f"{prefix}.{eid}.up_proj.{suffix}"] = up
        splits += 1
        continue

    # Fix down_proj key format if needed: experts.{id}.down_proj stays as-is
    output[key] = tensor

print(f"  Splits: {splits} gate_up_proj -> gate_proj + up_proj")
print(f"  Dropped: {dropped_gidx} g_idx tensors")
print(f"  Output: {len(output)} tensors")

# Verify: check a sample
sample_keys = [k for k in sorted(output.keys()) if 'experts' in k][:6]
print(f"  Sample keys:")
for k in sample_keys:
    print(f"    {k}: {output[k].shape}")

# Save (backup + overwrite)
backup = path + ".pre-fix"
if not os.path.exists(backup):
    os.rename(path, backup)
    print(f"  Backup: {backup}")

save_file(output, path)
print(f"  Saved: {os.path.getsize(path) / 1e9:.2f} GB")
