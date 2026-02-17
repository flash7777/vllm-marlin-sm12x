#!/bin/bash
# Qwen3-Coder-30B INT4 AutoRound — W4A16 + EAGLE3
# NST=1 optimal on DGX Spark (bandwidth-limited, 273 GB/s)
# NST=3 optimal on RTX PRO 6000 (1800 GB/s, 310 tok/s)
#
# DGX Spark: --gpu-memory-utilization 0.33 --kv-cache-memory-bytes 10G
# Spiegel 2: --gpu-memory-utilization 0.90
MODEL=/data/tensordata/Qwen3-Coder-30B-A3B-Instruct-int4-AutoRound
DRAFTER=/data/tensordata/SGLang-EAGLE3-Qwen3-Coder-30B-A3B
NAME=vllm-qwen3-i4ar-eagle3

# Detect platform
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"dgx"* ]] || [[ "$(uname -m)" == "aarch64" ]]; then
  NST=1
  GPU_MEM="--gpu-memory-utilization 0.05 --kv-cache-memory-bytes 10G"
  echo "Platform: DGX Spark (NST=$NST, unified memory workaround)"
else
  NST=3
  GPU_MEM="--gpu-memory-utilization 0.90"
  echo "Platform: Spiegel 2 (NST=$NST)"
fi

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
    $GPU_MEM \
    --max-model-len 32768 \
    --trust-remote-code \
    --quantization auto_round \
    --speculative-config='{"model":"'"$DRAFTER"'","num_speculative_tokens":'"$NST"',"method":"eagle3"}'

echo "Container $NAME started on port 8011 (Qwen3-Coder W4A16 + EAGLE3 NST=$NST)"
echo "Logs: podman logs -f $NAME"
