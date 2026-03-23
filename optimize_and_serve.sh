#!/bin/bash
# All-in-One: System optimieren + Ray Cluster + vLLM Serve
# Ausfuehren ALS USER (nicht sudo) — sudo-Teile sind optional markiert
#
# Usage:
#   bash optimize_and_serve.sh [--mtp] [--no-mtp]
#   bash optimize_and_serve.sh --optimize-only  # nur System optimieren, kein Serve
#
# Voraussetzung: sudo bash setup_uma_swap.sh wurde einmalig ausgefuehrt
# Empfehlung: Claude Code und Desktop-Session vorher beenden!

set -euo pipefail

# === Config ===
HEAD_IP="192.168.0.117"
WORKER_IP="192.168.0.116"
ETH_IF="enp1s0f0np0"
IMAGE="localhost/vllm-ng17e:latest"
MODEL_PATH="/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound"
MODEL_NAME="qwen3.5-397b-vl"
PORT=8011

# MTP default: aus (sicherer)
USE_MTP=false
OPTIMIZE_ONLY=false

for arg in "$@"; do
  case $arg in
    --mtp) USE_MTP=true ;;
    --no-mtp) USE_MTP=false ;;
    --optimize-only) OPTIMIZE_ONLY=true ;;
  esac
done

if $USE_MTP; then
  KV_CACHE="3G"
  MAX_MODEL_LEN=65536
  MAX_SEQS=2
  SPEC_CONFIG='--speculative-config {"method":"mtp","num_speculative_tokens":1}'
  echo "=== Qwen3.5-397B TP=2 + MTP INT4 ==="
else
  KV_CACHE="8G"
  MAX_MODEL_LEN=262144
  MAX_SEQS=3
  SPEC_CONFIG=""
  echo "=== Qwen3.5-397B TP=2 (256K, kein MTP) ==="
fi

# === Phase 1: System optimieren ===
echo ""
echo "--- Phase 1: System Memory optimieren ---"

# Alte Container aufräumen
echo "Aufraemen..."
for c in ng17e-head ng17e-worker eugr-head eugr-worker ng17-tp2-head ng17-tp2-worker; do
  podman stop "$c" 2>/dev/null || true
  podman rm "$c" 2>/dev/null || true
done
ssh flash@192.168.1.116 "for c in ng17e-worker eugr-worker ng17-tp2-worker; do podman stop \$c 2>/dev/null || true; podman rm \$c 2>/dev/null || true; done" 2>/dev/null || true

# Drop Page Cache (braucht kein sudo wenn /proc/sys/vm/drop_caches beschreibbar)
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "  drop_caches: braucht sudo"
ssh flash@192.168.1.116 "echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true" 2>/dev/null

# Desktop-Prozesse killen (Audio, Desktop-Integration)
pkill -f wireplumber 2>/dev/null || true
pkill -f pipewire 2>/dev/null || true
pkill -f snapd-desktop-integration 2>/dev/null || true
pkill -f pulseaudio 2>/dev/null || true
ssh flash@192.168.1.116 "pkill -f wireplumber 2>/dev/null; pkill -f pipewire 2>/dev/null; pkill -f snapd-desktop-integration 2>/dev/null; pkill -f pulseaudio 2>/dev/null" 2>/dev/null || true

echo "DGX: $(free -h | awk '/Speicher/{print $4}') frei"
ssh flash@192.168.1.116 "echo 'PGX: $(free -h | awk '/Speicher/{print \$4}') frei'" 2>/dev/null || true

if $OPTIMIZE_ONLY; then
  echo "Done (--optimize-only)"
  exit 0
fi

# === Phase 2: Container erstellen ===
echo ""
echo "--- Phase 2: Container erstellen ---"

COMMON_ENVS="\
  -e VLLM_UF_EAGER_ALLREDUCE=1 \
  -e NCCL_SOCKET_IFNAME=$ETH_IF \
  -e GLOO_SOCKET_IFNAME=$ETH_IF \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1"

podman run -d --name ng17e-head \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --network host --ipc host \
  -v /data/tensordata:/data/tensordata \
  -v /dev/infiniband:/dev/infiniband \
  -v /home/flash/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_HOST_IP=$HEAD_IP \
  -e RAY_NODE_IP_ADDRESS=$HEAD_IP \
  $COMMON_ENVS \
  "$IMAGE" sleep infinity

ssh flash@192.168.1.116 "podman run -d --name ng17e-worker \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --network host --ipc host \
  -v /data/tensordata:/data/tensordata \
  -v /dev/infiniband:/dev/infiniband \
  -v /home/flash/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_HOST_IP=$WORKER_IP \
  -e RAY_NODE_IP_ADDRESS=$WORKER_IP \
  $COMMON_ENVS \
  $IMAGE sleep infinity"
sleep 3
echo "Container erstellt"

# === Phase 3: Ray Cluster ===
echo ""
echo "--- Phase 3: Ray Cluster ---"

ssh flash@192.168.1.116 "podman exec -d ng17e-worker bash -c '
  ray start --block --object-store-memory 1073741824 --num-cpus 2 \
    --disable-usage-stats --address=\"$HEAD_IP:6379\" \
    --node-ip-address $WORKER_IP \
  > /tmp/ray-worker.log 2>&1
'"

podman exec -d ng17e-head bash -c "
  ray start --head --port 6379 --object-store-memory 1073741824 --num-cpus 2 \
    --node-ip-address $HEAD_IP --include-dashboard=false --disable-usage-stats
"
sleep 5

for i in $(seq 1 30); do
  NODES=$(podman exec ng17e-head bash -c "ray status 2>/dev/null | grep -c 'node_' || echo 0" | tail -1)
  if [ "$NODES" -ge 2 ]; then
    echo "Ray: $NODES nodes ready"
    break
  fi
  sleep 2
done

# === Phase 4: vLLM Serve ===
echo ""
echo "--- Phase 4: vLLM Serve ---"
echo "  Model:     $MODEL_NAME"
echo "  Context:   $MAX_MODEL_LEN"
echo "  KV-Cache:  $KV_CACHE"
echo "  MTP:       $USE_MTP"
echo ""

podman exec ng17e-head \
  vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.05 \
    --kv-cache-memory-bytes "$KV_CACHE" \
    --kv-cache-dtype fp8 \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens 8192 \
    --load-format fastsafetensors \
    --enable-prefix-caching \
    ${SPEC_CONFIG:+$SPEC_CONFIG} \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
  2>&1 | tee /tmp/ng17e-serve.log
