#!/bin/bash
# Qwen3-Coder-30B INT4 AutoRound — W4A8 (FP8 activations, FP8 MMA k=32)
MODEL=/data/tensordata/Qwen3-Coder-30B-A3B-Instruct-int4-AutoRound
NAME=vllm-qwen3-i4ar-w4a8

podman stop $NAME 2>/dev/null; podman rm $NAME 2>/dev/null

podman run -d \
  --name $NAME \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -p 8011:8000 \
  -v /data/tensordata:/data/tensordata \
  --shm-size=16g \
  -e VLLM_MLA_DISABLE=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_MARLIN_INPUT_DTYPE=fp8 \
  vllm-next \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen3-coder-30b-int4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --trust-remote-code \
    --quantization auto_round

echo "Container $NAME started on port 8011 (Qwen3-Coder W4A8 FP8 activations)"
echo "Logs: podman logs -f $NAME"
