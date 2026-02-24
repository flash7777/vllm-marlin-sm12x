#!/bin/bash

podman run -it --rm \
  --name vllm-int4 \
  --network host --gpus all --ipc=host \
  --device nvidia.com/gpu=all \
  --security-opt label=disable \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v /home/edison/Downloads/vllm/models:/models \
  -e VLLM_MLA_DISABLE=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_MARLIN_INPUT_DTYPE=fp8 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e HF_HUB_OFFLINE=1 \
  vllm-next vllm \
  serve /models/Qwen3-Coder-Next-int4-AutoRound \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --quantization auto_round \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --served-model-name qwen3-coder-next \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
#   --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
