#!/bin/bash
# Start FMAAQ quantization on RTX (Spiegel 2)
# Usage: ./start_fmaaq.sh [--no-teacher]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRA_ARGS="$@"

podman run -d \
  --replace \
  --name autoround-fmaaq \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -v /data/tensordata:/data/tensordata \
  -v "${SCRIPT_DIR}":/opt/fmaaq:ro \
  --shm-size=64g \
  -e PYTHONPATH=/opt/fmaaq \
  localhost/autoround:latest \
  python3 /opt/fmaaq/fmaaq_quantize.py \
    --model-path /data/tensordata/Qwen3.5-REAP-262B-A17B \
    --teacher-path /data/tensordata/Qwen3.5-397B-A17B \
    --output-dir /data/tensordata/Qwen3.5-REAP-262B-A17B-int4-FMAAQ \
    --bits 4 \
    --group-size 128 \
    --sym \
    --iters 50 \
    --seqlen 512 \
    --nsamples 128 \
    --batch-size 1 \
    --gradient-accumulate-steps 8 \
    --low-gpu-mem-usage \
    ${EXTRA_ARGS}

echo "Container 'autoround-fmaaq' started. Follow logs with:"
echo "  podman logs -f autoround-fmaaq"
