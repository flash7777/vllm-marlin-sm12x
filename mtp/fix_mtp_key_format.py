#!/usr/bin/env python3
"""Fix MTP INT4 key format: experts.{proj}.{id} -> experts.{id}.{proj}

AutoRound quantizes with format: mtp.layers.0.mlp.experts.down_proj.0.qweight
vLLM expects:                    mtp.layers.0.mlp.experts.0.down_proj.qweight

Usage:
    python3 fix_mtp_key_format.py /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-00001-of-00001.safetensors
"""

import re
import sys
from safetensors import safe_open
from safetensors.torch import save_file

path = sys.argv[1]

print(f"Fixing key format: {path}")

# Load
tensors = {}
with safe_open(path, framework="pt") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# Rename: experts.{proj}.{id}.{suffix} -> experts.{id}.{proj}.{suffix}
# Pattern: mtp.layers.0.mlp.experts.down_proj.42.qweight
#       -> mtp.layers.0.mlp.experts.42.down_proj.qweight
pattern = re.compile(r"(.*\.experts)\.(gate_proj|up_proj|down_proj)\.(\d+)\.(.*)")

renamed = {}
fixes = 0
for key, tensor in tensors.items():
    m = pattern.match(key)
    if m:
        prefix, proj, idx, suffix = m.groups()
        new_key = f"{prefix}.{idx}.{proj}.{suffix}"
        renamed[new_key] = tensor
        fixes += 1
    else:
        renamed[key] = tensor

print(f"  {fixes} keys umbenannt")
print(f"  Beispiel: {list(tensors.keys())[2]} -> {list(renamed.keys())[2]}")

# Save (überschreibt)
save_file(renamed, path)
print(f"  Gespeichert: {path}")
