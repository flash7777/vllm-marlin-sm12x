#!/usr/bin/env python3
"""Enable RIY expert pruning on MTP drafter layers.

By default, RIY excludes drafter/MTP layers from expert pruning
(_is_drafter check). This patch removes that exclusion so MTP
experts get the same pruning mask as Main-Model layer 0.

Usage: piped into container
  cat patch_mtp_riy_enable.py | podman exec -i CONTAINER python3 -
"""

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py"

with open(path) as f:
    src = f.read()

old = '''_is_drafter = "mtp" in prefix.lower() or "drafter" in prefix.lower()
        if _riy_profile and _os.path.exists(_riy_profile) and _layer_idx >= 0 and not _is_drafter:'''

new = '''_is_drafter = "mtp" in prefix.lower() or "drafter" in prefix.lower()
        if _riy_profile and _os.path.exists(_riy_profile) and _layer_idx >= 0:'''

if old in src:
    src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    print("PATCHED: MTP drafter now uses RIY expert mask")
elif "not _is_drafter" not in src and "_is_drafter" in src:
    print("ALREADY PATCHED")
else:
    print("PATTERN NOT FOUND")
