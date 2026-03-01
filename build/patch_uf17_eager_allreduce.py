#!/usr/bin/env python3
"""
Patch UF17: EAGER_ALLREDUCE — run NCCL AllReduce outside CUDA Graphs.

Adds vllm::all_reduce to piecewise splitting_ops when VLLM_UF_EAGER_ALLREDUCE=1.
AllReduce runs eager (18us) instead of inside graph replay (46us),
saving ~2.66ms/token on multi-node TP=2.

Result: 107.8 -> 196 tok/s (+82%) on GB10 TP=2 RoCE.
"""

import glob
import sys

SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"

# ============================================================
# 1. Patch compilation.py: add vllm::all_reduce to splitting_ops
# ============================================================
compilation_py = f"{SITE_PACKAGES}/vllm/config/compilation.py"

with open(compilation_py) as f:
    src = f.read()

if "VLLM_UF_EAGER_ALLREDUCE" in src:
    print("UF17: compilation.py already patched, skipping")
else:
    # Ensure 'import os' is present
    if "\nimport os\n" not in src:
        # Try common import anchors
        for anchor in ["\nimport copy\n", "\nimport enum\n"]:
            if anchor in src:
                src = src.replace(anchor, anchor.rstrip("\n") + "\nimport os\n", 1)
                print(f"UF17: Added 'import os' after '{anchor.strip()}'")
                break
        else:
            print("ERROR: Could not find import anchor for 'import os'")
            sys.exit(1)

    # Insert UF17 block after the unified_kv_cache_update append
    marker = 'self.splitting_ops.append("vllm::unified_kv_cache_update")'
    if marker not in src:
        print(f"ERROR: Could not find marker in compilation.py: {marker}")
        sys.exit(1)

    uf17_block = '''self.splitting_ops.append("vllm::unified_kv_cache_update")

                # UF17: EAGER_ALLREDUCE — run NCCL AllReduce outside
                # CUDA Graphs as a piecewise split point. This causes
                # AllReduce to run eager (18us) instead of inside graph
                # replay (46us), saving ~2.66ms/token on multi-node TP=2.
                if os.environ.get("VLLM_UF_EAGER_ALLREDUCE", "0") == "1":
                    self.splitting_ops.append("vllm::all_reduce")
                    logger.info(
                        "UF17: Added vllm::all_reduce to splitting_ops "
                        "(eager NCCL AllReduce between CUDA Graph segments)"
                    )'''

    src = src.replace(marker, uf17_block)

    with open(compilation_py, "w") as f:
        f.write(src)
    print("UF17: Patched compilation.py — EAGER_ALLREDUCE support added")

print("UF17: Done")
