#!/bin/bash
# Setup UF17v2 patches on running ng-tp2-head and ng-tp2-worker containers.
# Run AFTER start.ng_tp2.head and start.ng_tp2.worker
set -euo pipefail

echo "=== Applying UF17v2 patches to TP=2 containers ==="

PATCH_SCRIPT="/home/flash/vllm-marlin-sm12x/build/patch_uf17_eager_allreduce.py"
SERVE_SCRIPT="/home/flash/vllm-marlin-sm12x/build/serve_torchrun.py"

# --- HEAD (DGX) ---
echo "[HEAD] Copying files..."
podman cp "$PATCH_SCRIPT" ng-tp2-head:/tmp/patch_uf17v2.py
podman cp "$SERVE_SCRIPT" ng-tp2-head:/opt/vllm/serve_torchrun.py

echo "[HEAD] Applying UF17v2 patch..."
podman exec ng-tp2-head python3 /tmp/patch_uf17v2.py

# Safety: ensure 'import os' in compilation.py (v1 image was missing it)
podman exec ng-tp2-head python3 -c "
p='/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py'
t=open(p).read()
if '\nimport os\n' not in t:
    for a in ['\nimport enum\n','\nimport copy\n']:
        if a in t:
            t=t.replace(a,a.rstrip('\n')+'\nimport os\n',1)
            open(p,'w').write(t); print('[HEAD] Added import os'); break
"

echo "[HEAD] Verifying..."
HEAD_V2=$(podman exec ng-tp2-head grep -c "all_reduce_with_output" /usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py)
HEAD_COMP=$(podman exec ng-tp2-head grep -c "all_reduce_with_output" /usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py)
echo "[HEAD] parallel_state.py: $HEAD_V2 occurrences of all_reduce_with_output"
echo "[HEAD] compilation.py:    $HEAD_COMP occurrences of all_reduce_with_output"

# --- WORKER (PGX) ---
echo ""
echo "[WORKER] Copying files..."
# First copy to PGX host, then into container
scp -q "$PATCH_SCRIPT" flash@192.168.1.116:/tmp/patch_uf17v2.py
scp -q "$SERVE_SCRIPT" flash@192.168.1.116:/tmp/serve_torchrun.py

ssh flash@192.168.1.116 "podman cp /tmp/patch_uf17v2.py ng-tp2-worker:/tmp/patch_uf17v2.py"
ssh flash@192.168.1.116 "podman cp /tmp/serve_torchrun.py ng-tp2-worker:/opt/vllm/serve_torchrun.py"

echo "[WORKER] Applying UF17v2 patch..."
ssh flash@192.168.1.116 "podman exec ng-tp2-worker python3 /tmp/patch_uf17v2.py"

# Safety: ensure 'import os' in compilation.py
ssh flash@192.168.1.116 "podman exec ng-tp2-worker python3 -c \"
p='/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py'
t=open(p).read()
if '\nimport os\n' not in t:
    for a in ['\nimport enum\n','\nimport copy\n']:
        if a in t:
            t=t.replace(a,a.rstrip('\n')+'\nimport os\n',1)
            open(p,'w').write(t); print('[WORKER] Added import os'); break
\""

echo "[WORKER] Verifying..."
WORKER_V2=$(ssh flash@192.168.1.116 "podman exec ng-tp2-worker grep -c 'all_reduce_with_output' /usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py")
WORKER_COMP=$(ssh flash@192.168.1.116 "podman exec ng-tp2-worker grep -c 'all_reduce_with_output' /usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py")
echo "[WORKER] parallel_state.py: $WORKER_V2 occurrences of all_reduce_with_output"
echo "[WORKER] compilation.py:    $WORKER_COMP occurrences of all_reduce_with_output"

echo ""
echo "=== UF17v2 setup complete ==="
echo "Next: ./start.ng_tp2.serve.<config>"
