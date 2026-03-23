#!/bin/bash
# nsys profiling of decode phase — launch/start approach
# Runs inside uf19-tp2-head container, worker must already be running
set -euo pipefail

NSYS=/opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys
MASTER_ADDR=192.168.0.117
MASTER_PORT=29500
MODEL_PATH=/data/tensordata/Qwen3-Coder-30B-A3B-Instruct-int4-AutoRound
EAGLE3_PATH=/data/tensordata/SGLang-EAGLE3-Qwen3-Coder-30B-A3B
PORT=8011

echo "=== Phase 1: Launch server under nsys (no collection yet) ==="
$NSYS launch \
  --session-new=vllm-decode \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --sample=none \
  -- \
  torchrun \
    --nnodes=2 --nproc-per-node=1 --node-rank=0 \
    --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
    /opt/vllm/serve_torchrun.py \
      --model $MODEL_PATH \
      --served-model-name qwen3-coder \
      --port $PORT \
      --max-model-len 16384 \
      --gpu-memory-utilization 0.05 \
      --kv-cache-memory-bytes 10G \
      --quantization auto_round \
      --speculative-model $EAGLE3_PATH \
      --num-speculative-tokens 1 \
      --trust-remote-code \
  &

NSYS_PID=$!
echo "nsys PID: $NSYS_PID"

echo "=== Phase 2: Wait for server ready ==="
for i in $(seq 1 600); do
  if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 1
done

# Warmup request
echo "=== Phase 3: Warmup request ==="
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-coder","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}' \
  > /dev/null 2>&1
sleep 2

echo "=== Phase 4: Start nsys collection ==="
$NSYS start \
  --session=vllm-decode \
  --output=/tmp/vllm_decode \
  --force-overwrite=true \
  --duration=15

echo "Collection started, sending decode request..."

# Send a request that generates ~100 tokens (should take ~1s at 118 tok/s)
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-coder","messages":[{"role":"user","content":"Write a short story about a cat who learns to fly. Be creative and detailed."}],"max_tokens":200,"temperature":0.7,"seed":42}' \
  > /tmp/nsys_response.json 2>&1 &

CURL_PID=$!

# Wait for curl to finish or nsys duration to expire
wait $CURL_PID 2>/dev/null || true
echo "Request complete"

# Wait for nsys collection to finish
sleep 5

echo "=== Phase 5: Stop and generate report ==="
$NSYS stop --session=vllm-decode 2>/dev/null || true
sleep 2

# Generate stats
$NSYS stats --report cuda_gpu_kern_sum --format csv /tmp/vllm_decode.nsys-rep > /tmp/vllm_decode_kernels.csv 2>/dev/null || true
$NSYS stats --report nvtx_sum --format csv /tmp/vllm_decode.nsys-rep > /tmp/vllm_decode_nvtx.csv 2>/dev/null || true
$NSYS stats --report cuda_gpu_kern_sum /tmp/vllm_decode.nsys-rep 2>/dev/null || true

echo "=== Phase 6: Shutdown ==="
$NSYS shutdown --session=vllm-decode 2>/dev/null || true

echo "Done. Files: /tmp/vllm_decode.nsys-rep, /tmp/vllm_decode_kernels.csv"
