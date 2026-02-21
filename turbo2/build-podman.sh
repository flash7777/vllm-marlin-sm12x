#!/usr/bin/env bash
set -Eeuo pipefail

# Build vllm-turbo image using podman
# Based on Avarok dgx-vllm v22

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="vllm-turbo"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=== Building ${IMAGE_NAME}:${IMAGE_TAG} ==="
echo "Base: Avarok dgx-vllm v22 (vLLM v0.16.0rc2, PyTorch 2.10+cu130)"
echo "Target: NVIDIA GB10 SM121 NVFP4"
echo "Build time: 30-60 minutes"
echo ""

START_TIME=$(date +%s)

podman build \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -t "${IMAGE_NAME}:v22" \
  "${SCRIPT_DIR}" \
  --progress=plain \
  2>&1 | tee "${SCRIPT_DIR}/build.log"

BUILD_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
BUILD_TIME=$(( (END_TIME - START_TIME) / 60 ))

if [ $BUILD_EXIT_CODE -eq 0 ]; then
  echo ""
  echo "Build successful! (${BUILD_TIME}m)"
  podman images | grep "${IMAGE_NAME}"
else
  echo "Build failed! Check build.log"
  exit 1
fi
