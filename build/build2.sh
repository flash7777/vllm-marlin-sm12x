#!/bin/bash
# Build vllm-next2 Container (vLLM latest + CUTLASS 4.3.5 + Branchless E2M1)
# Base: nvcr.io/nvidia/vllm:26.01-py3
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================"
echo "Building vllm-next2 on $(uname -m) / $(hostname)"
echo "============================================"

podman build -f Dockerfile.next2 -t vllm-next2 . 2>&1 | tee build-next2.log

echo ""
echo "============================================"
echo "Done: vllm-next2 image built"
echo "============================================"
