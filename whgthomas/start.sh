#!/bin/bash
# Qwen3-Coder-Next INT4 AutoRound (W4A8-FP8) on DGX Spark
# Model: Intel/Qwen3-Coder-Next-int4-AutoRound
#
# Download:
#   hf download Intel/Qwen3-Coder-Next-int4-AutoRound \
#     --local-dir /data/tensordata/Qwen3-Coder-Next-int4-AutoRound

MODEL_PATH="/data/tensordata/Qwen3-Coder-Next-int4-AutoRound"
NAME="vllm-qwen3-coder-next-int4"

podman stop $NAME 2>/dev/null; podman rm $NAME 2>/dev/null

podman run -d \
  --name $NAME \
  --network host --ipc=host \
  --device nvidia.com/gpu=all \
  --security-opt label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -v /data/tensordata:/data/tensordata \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_MLA_DISABLE=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_MARLIN_INPUT_DTYPE=fp8 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  vllm-next vllm \
  serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.05 \
    --kv-cache-memory-bytes 30G \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --quantization auto_round \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --served-model-name qwen3-coder-next \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
#   --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
