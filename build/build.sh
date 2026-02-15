#!/bin/bash
# Build vllm-next Container (vLLM latest + CUTLASS 4.3.5 + GLM-4.7 Patches)
# Base: nvcr.io/nvidia/vllm:26.01-py3
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================"
echo "Building vllm-next on $(uname -m) / $(hostname)"
echo "============================================"

podman build -f Dockerfile.vllm -t vllm-next . 2>&1 | tee build-vllm.log

echo ""
echo "============================================"
echo "Done: vllm-next image built"
echo "============================================"
