#!/usr/bin/env bash
set -Eeuo pipefail

# Pull pre-built vllm-turbo image from Docker Hub
# Much faster than building from source (~30-60 min saved)

IMAGE="docker.io/avarok/dgx-vllm-nvfp4-kernel:v22"
LOCAL_NAME="vllm-turbo:latest"

echo "=== Pulling pre-built image ==="
echo "Source: ${IMAGE}"
echo ""

podman pull "${IMAGE}"
podman tag "${IMAGE}" "${LOCAL_NAME}"
podman tag "${IMAGE}" "vllm-turbo:v22"

echo ""
echo "Tagged as:"
echo "  - vllm-turbo:latest"
echo "  - vllm-turbo:v22"
podman images | grep vllm-turbo
