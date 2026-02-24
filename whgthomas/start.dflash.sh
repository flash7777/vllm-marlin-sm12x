#!/bin/bash
# Qwen3-Coder-Next + DFlash on DGX Spark
# Target: Qwen3-Coder-Next (BF16 oder NVFP4)
# Drafter: z-lab/Qwen3-Coder-Next-DFlash (noch nicht released, Platzhalter)
# Image: localhost/sglang-next (DFlash PR #16818, sgl-kernel 0.3.21)
#
# Download (sobald verfuegbar):
#   hf download z-lab/Qwen3-Coder-Next-DFlash \
#     --local-dir /data/tensordata/Qwen3-Coder-Next-DFlash

TARGET_PATH="/data/tensordata/Qwen3-Coder-Next"
DRAFTER_PATH="/data/tensordata/Qwen3-Coder-Next-DFlash"
NAME="sglang-qwen3-coder-next-dflash"

podman stop $NAME 2>/dev/null; podman rm $NAME 2>/dev/null

podman run -d \
  --name $NAME \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --network host \
  -v /data/tensordata:/data/tensordata \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=false \
  localhost/sglang-next \
  python3 -m sglang.launch_server \
    --model-path "$TARGET_PATH" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFTER_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen3-coder-next \
    --attention-backend flashinfer \
    --trust-remote-code \
    --mem-fraction-static 0.20 \
    --max-total-tokens 32768

echo "Container $NAME gestartet"
echo "Logs: podman logs -f $NAME"
