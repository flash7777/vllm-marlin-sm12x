#!/bin/bash
# Build vllm-ng16 Container (vLLM 0.16 + torchrun + UF17 EAGER_ALLREDUCE)
# Base: nvcr.io/nvidia/vllm:26.01-py3
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================"
echo "Building vllm-ng16 on $(uname -m) / $(hostname)"
echo "  Base:    nvcr.io/nvidia/vllm:26.02-py3"
echo "  vLLM:    0.16.0rc2 (commit 882682ab8)"
echo "  Features: Marlin SM12x, CUTLASS 4.3.5,"
echo "            torchrun serve, UF17 EAGER_ALLREDUCE,"
echo "            native MTP (GLM, Qwen3.5, Qwen3-Next)"
echo "============================================"

podman build -f Dockerfile.ng16 -t vllm-ng16 . 2>&1 | tee build-ng16.log

echo ""
echo "============================================"
echo "Done: vllm-ng16 image built"
echo "============================================"
