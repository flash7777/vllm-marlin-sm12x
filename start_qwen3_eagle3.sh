#!/bin/bash
# Qwen3-Coder-30B INT4 AutoRound — W4A16 + EAGLE3 NST=3 (310 tok/s on RTX PRO 6000)
MODEL=/data/tensordata/Qwen3-Coder-30B-A3B-Instruct-int4-AutoRound
DRAFTER=/data/tensordata/SGLang-EAGLE3-Qwen3-Coder-30B-A3B
NAME=vllm-qwen3-i4ar-eagle3

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
  vllm-next \
  vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen3-coder-int4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --trust-remote-code \
    --quantization auto_round \
    --speculative-config='{"model":"'"$DRAFTER"'","num_speculative_tokens":3,"method":"eagle3"}'

echo "Container $NAME started on port 8011 (Qwen3-Coder W4A16 + EAGLE3)"
echo "Logs: podman logs -f $NAME"
