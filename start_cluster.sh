#!/bin/bash
# Cluster Start: Container + Ray + vLLM Serve
# Ausfuehren als User (flash), NACH optimize_node.sh auf beiden Nodes
#
# Usage:
#   bash start_cluster.sh              # Standard: 256K, 8G KV, kein MTP
#   bash start_cluster.sh --mtp        # MTP INT4: 128K, 8G KV
#   bash start_cluster.sh --mtp-eager  # MTP INT4 ohne CUDA Graphs (Debug)
#   bash start_cluster.sh --riy <profile.json>  # RIY Expert-Pruning
#   bash start_cluster.sh --riy <p> --mtp       # RIY + MTP
#   bash start_cluster.sh --riy-monitor         # Stats + HTTP Server aktivieren
#   bash start_cluster.sh --no-riy              # Ohne RIY (vllm-ng17e Image)
#   bash start_cluster.sh --stop       # Alles stoppen

set -euo pipefail

# === Config ===
HEAD_IP="192.168.0.117"
WORKER_IP="192.168.0.116"
ETH_IF="enp1s0f0np0"
IMAGE="localhost/vllm-ng17e-riy:latest"
MODEL_PATH="/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound"
MODEL_NAME="qwen3.5-397b-vl"
PORT=8011

# Parse args
USE_MTP=false
USE_EAGER=false
STOP_ONLY=false
RIY_PROFILE=""
RIY_MONITOR=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --mtp) USE_MTP=true; shift ;;
    --mtp-eager) USE_MTP=true; USE_EAGER=true; shift ;;
    --riy) RIY_PROFILE="$2"; shift 2 ;;
    --riy-monitor) RIY_MONITOR=true; shift ;;
    --no-riy) IMAGE="localhost/vllm-ng17e:latest"; shift ;;
    --stop) STOP_ONLY=true; shift ;;
    *) echo "Unbekannt: $1"; exit 1 ;;
  esac
done

# RIY: Profil prüfen
if [ -n "$RIY_PROFILE" ]; then
  [ -f "$RIY_PROFILE" ] || { echo "ERROR: RIY-Profil $RIY_PROFILE nicht gefunden"; exit 1; }
fi

# === Stop ===
stop_all() {
  echo "Stoppe Cluster..."
  podman exec ng17e-head bash -c "ray stop --force 2>/dev/null; pkill -f 'vllm serve' 2>/dev/null" 2>/dev/null || true
  ssh flash@192.168.1.116 "podman exec ng17e-worker bash -c 'ray stop --force 2>/dev/null' 2>/dev/null" 2>/dev/null || true
  for c in ng17e-head; do
    podman stop "$c" 2>/dev/null || true
    podman rm "$c" 2>/dev/null || true
  done
  ssh flash@192.168.1.116 "podman stop ng17e-worker 2>/dev/null || true; podman rm ng17e-worker 2>/dev/null || true" 2>/dev/null || true
  echo "Gestoppt."
}

if $STOP_ONLY; then
  stop_all
  exit 0
fi

# === Mode Config ===
EAGER=""
if $USE_MTP; then
  KV_CACHE="8G"; MAX_LEN=131072; MAX_SEQS=2
  SPEC="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1}"
  if $USE_EAGER; then
    EAGER="--compilation-config {\"cudagraph_mode\":\"none\"}"
    echo "=== Cluster Start: 397B TP=2 + MTP BF16 (no CUDA Graphs) ==="
  else
    # Minimale CUDA Graphs: nur batch=1, reduziert Triton JIT-Compile-Zeit
    EAGER="--compilation-config {\"cudagraph_capture_sizes\":[1],\"cudagraph_num_of_warmups\":0}"
    echo "=== Cluster Start: 397B TP=2 + MTP BF16 (CUDA Graphs batch=1) ==="
  fi
else
  KV_CACHE="8G"; MAX_LEN=262144; MAX_SEQS=3
  SPEC=""
  echo "=== Cluster Start: 397B TP=2 (256K) ==="
fi
if [ -n "$RIY_PROFILE" ]; then
  echo "  RIY-Profil: $RIY_PROFILE"
  echo "  Image:      $IMAGE"
fi

# === Prüfe Voraussetzungen ===
echo ""
echo "--- Pruefe Nodes ---"
ping -c 1 -W 2 $WORKER_IP > /dev/null 2>&1 || { echo "ERROR: PGX ($WORKER_IP) nicht erreichbar"; exit 1; }
echo "  PGX: OK"
podman image exists "$IMAGE" 2>/dev/null || { echo "ERROR: Image $IMAGE nicht auf DGX"; exit 1; }
ssh flash@192.168.1.116 "podman image exists $IMAGE" 2>/dev/null || { echo "ERROR: Image $IMAGE nicht auf PGX"; exit 1; }
echo "  Image: OK"
[ -d "$MODEL_PATH" ] || { echo "ERROR: Model $MODEL_PATH nicht gefunden"; exit 1; }
echo "  Model: OK"

# === Container ===
echo ""
echo "--- Container ---"
stop_all 2>/dev/null

ENVS="-e VLLM_UF_EAGER_ALLREDUCE=1 \
  -e NCCL_SOCKET_IFNAME=$ETH_IF -e GLOO_SOCKET_IFNAME=$ETH_IF \
  -e NCCL_IB_DISABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1"

# RIY Monitor: Stats + HTTP Server nur wenn explizit aktiviert
if $RIY_MONITOR; then
  ENVS="$ENVS -e VLLM_RIY_MONITOR=1"
fi

# MTP: Weight-Format steuern
# BF16: quant_config wird gestripped, BF16 MTP-Weights geladen
# fastsafetensors + viele kleine BF16 Expert-Tensoren → NCCL Timeout → load-format auto
MTP_MODE="BF16"
if $USE_MTP; then
  ENVS="$ENVS -e VLLM_MTP_FORCE=$MTP_MODE"
  if [ "$MTP_MODE" = "BF16" ]; then
    LOAD_FORMAT="auto"
  fi
fi

podman run -d --name ng17e-head \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --network host --ipc host \
  -v /data/tensordata:/data/tensordata \
  -v /dev/infiniband:/dev/infiniband \
  -v /home/flash/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_HOST_IP=$HEAD_IP -e RAY_NODE_IP_ADDRESS=$HEAD_IP \
  $ENVS \
  "$IMAGE" sleep infinity > /dev/null

ssh flash@192.168.1.116 "podman run -d --name ng17e-worker \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --network host --ipc host \
  -v /data/tensordata:/data/tensordata \
  -v /dev/infiniband:/dev/infiniband \
  -v /home/flash/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_HOST_IP=$WORKER_IP -e RAY_NODE_IP_ADDRESS=$WORKER_IP \
  $ENVS \
  $IMAGE sleep infinity" > /dev/null
sleep 2
echo "  Head + Worker erstellt"

# === RIY Monitor Guard (beide Nodes) — deaktiviert Stats+HTTP im Hot-Path ===
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "  Patching RIY monitor guard..."
cat "$SCRIPT_DIR/mtp/patch_riy_monitor_guard.py" | podman exec -i ng17e-head python3 - 2>&1
cat "$SCRIPT_DIR/mtp/patch_riy_monitor_guard.py" | ssh flash@192.168.1.116 "podman exec -i ng17e-worker python3 -" 2>&1

# === MTP Runtime-Patches (beide Nodes) ===
if $USE_MTP; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  # patch_drafter_load_format.py — immer (fastsafetensors OOM Workaround)
  echo "  Patching drafter load_format..."
  cat "$SCRIPT_DIR/mtp/patch_drafter_load_format.py" | podman exec -i ng17e-head python3 - 2>&1
  cat "$SCRIPT_DIR/mtp/patch_drafter_load_format.py" | ssh flash@192.168.1.116 "podman exec -i ng17e-worker python3 -" 2>&1
  # patch_mtp_riy_enable.py — RIY Expert-Pruning auch auf MTP-Layer anwenden
  echo "  Patching MTP RIY enable..."
  cat "$SCRIPT_DIR/mtp/patch_mtp_riy_enable.py" | podman exec -i ng17e-head python3 - 2>&1
  cat "$SCRIPT_DIR/mtp/patch_mtp_riy_enable.py" | ssh flash@192.168.1.116 "podman exec -i ng17e-worker python3 -" 2>&1
  # patch_mtp_quant.py — NUR bei INT4 (forciert quantized=True, widerspricht BF16)
  if [ "$MTP_MODE" = "INT4" ]; then
    echo "  Patching MTP quant_config (INT4)..."
    cat "$SCRIPT_DIR/mtp/patch_mtp_quant.py" | podman exec -i ng17e-head python3 - 2>&1
    cat "$SCRIPT_DIR/mtp/patch_mtp_quant.py" | ssh flash@192.168.1.116 "podman exec -i ng17e-worker python3 -" 2>&1
  fi
fi

# === Ray Cluster ===
echo ""
echo "--- Ray Cluster ---"

ssh flash@192.168.1.116 "podman exec -d ng17e-worker bash -c '
  ray start --block --object-store-memory 1073741824 --num-cpus 2 \
    --disable-usage-stats --address=\"$HEAD_IP:6379\" \
    --node-ip-address $WORKER_IP > /tmp/ray-worker.log 2>&1
'"

podman exec -d ng17e-head bash -c "
  ray start --head --port 6379 --object-store-memory 1073741824 --num-cpus 2 \
    --node-ip-address $HEAD_IP --include-dashboard=false --disable-usage-stats
"

for i in $(seq 1 30); do
  NODES=$(podman exec ng17e-head bash -c "ray status 2>/dev/null | grep -c 'node_' || echo 0" | tail -1)
  if [ "$NODES" -ge 2 ]; then
    echo "  Ray: $NODES nodes"
    break
  fi
  [ "$i" -eq 30 ] && { echo "ERROR: Ray Cluster Timeout"; exit 1; }
  sleep 2
done

# === vLLM Serve ===
echo ""
echo "--- vLLM Serve ---"
echo "  Model:   $MODEL_NAME"
echo "  Context: $MAX_LEN"
echo "  KV:      $KV_CACHE"
echo "  MTP:     $USE_MTP (eager: $USE_EAGER)"
echo "  RIY:     ${RIY_PROFILE:-none}"
echo "  Port:    $PORT"
echo ""

podman exec ng17e-head \
  vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.05 \
    --kv-cache-memory-bytes "$KV_CACHE" \
    --kv-cache-dtype fp8 \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens 8192 \
    --load-format ${LOAD_FORMAT:-fastsafetensors} \
    --enable-prefix-caching \
    ${SPEC:+$SPEC} \
    ${EAGER:+$EAGER} \
    ${RIY_PROFILE:+--riy-expert-profile "$RIY_PROFILE"} \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
  2>&1 | tee /tmp/ng17e-serve.log
