#!/bin/bash
# Build vllm-ng Container (nextgen: next2 + torchrun + UF17 EAGER_ALLREDUCE)
# Base: nvcr.io/nvidia/vllm:26.01-py3
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================"
echo "Building vllm-ng on $(uname -m) / $(hostname)"
echo "  Features: Marlin SM12x, CUTLASS 4.3.5,"
echo "            torchrun serve, UF17 EAGER_ALLREDUCE,"
echo "            UF19 RAW_IBVERBS AllReduce"
echo "============================================"

podman build -f Dockerfile.ng -t vllm-ng . 2>&1 | tee build-ng.log

echo ""
echo "============================================"
echo "Done: vllm-ng image built"
echo ""
echo "Usage (single-node):"
echo "  podman run ... vllm-ng vllm serve ..."
echo ""
echo "Usage (multi-node TP=2 with UF17):"
echo "  # Set VLLM_UF_EAGER_ALLREDUCE=1 in container env"
echo "  # Use serve_torchrun.py:"
echo "  podman exec <container> torchrun \\"
echo "    --nnodes=2 --nproc-per-node=1 \\"
echo "    /opt/vllm/serve_torchrun.py --model ... "
echo ""
echo "Usage (multi-node TP=2 with UF19 ibverbs):"
echo "  # Add to container env:"
echo "  #   VLLM_UF_EAGER_ALLREDUCE=1"
echo "  #   VLLM_UF_UF19_RDMA=1"
echo "  #   VLLM_UF19_PEER_IP=192.168.0.116"
echo "  # Container needs RDMA uverbs devices (same as NCCL RoCE):"
echo "  #   --device /dev/infiniband/uverbs0"
echo "  #   --device /dev/infiniband/rdma_cm"
echo "============================================"
