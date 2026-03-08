#!/bin/bash
# Build vllm-ng17 Container (vLLM 0.17.0 + torchrun + UF17 EAGER_ALLREDUCE)
# Base: nvcr.io/nvidia/vllm:26.02-py3
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================"
echo "Building vllm-ng17 on $(uname -m) / $(hostname)"
echo "  Base:    nvcr.io/nvidia/vllm:26.02-py3"
echo "  vLLM:    0.17.0 (tag v0.17.0)"
echo "  Features: Marlin SM12x, CUTLASS 4.4.0,"
echo "            torchrun serve, UF17 EAGER_ALLREDUCE,"
echo "            native Qwen3.5 + MTP + EAGLE3 CUDA Graphs,"
echo "            FlashAttention 4, Model Runner V2"
echo "============================================"

podman build -f Dockerfile.ng17 -t vllm-ng17 . 2>&1 | tee build-ng17.log

echo ""
echo "============================================"
echo "Done: vllm-ng17 image built"
echo "============================================"
