#!/bin/bash
# Run NCCL tuning benchmarks with various configs
# Usage: ./run_nccl_tuning.sh
set -euo pipefail

MASTER_ADDR="192.168.0.117"
MASTER_PORT_BASE=29510
HEAD="ng-tp2-head"
WORKER="ng-tp2-worker"
PGX="flash@192.168.1.116"
SCRIPT="/tmp/bench_nccl_tuning.py"

# Common env for both nodes
COMMON_ENV="-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_DEBUG=WARN"

run_test() {
    local label="$1"
    shift
    local extra_env="$@"
    local port=$((MASTER_PORT_BASE++))

    echo ""
    echo "=== Test: $label ==="

    # Start worker
    ssh $PGX "podman exec -d $COMMON_ENV $extra_env $WORKER \
      bash -c 'torchrun --nnodes=2 --nproc-per-node=1 --node-rank=1 \
        --master-addr=$MASTER_ADDR --master-port=$port $SCRIPT'" 2>/dev/null

    sleep 1

    # Run head (foreground, capture output)
    podman exec $COMMON_ENV $extra_env $HEAD \
      bash -c "torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 \
        --master-addr=$MASTER_ADDR --master-port=$port $SCRIPT" 2>&1 | grep -E '(µs/call|===|NCCL)'

    sleep 1
}

# Test 1: Default (baseline)
run_test "Default (8ch auto)" ""

# Test 2: Force LL protocol
run_test "Force LL proto" "-e NCCL_PROTO=LL"

# Test 3: 1 channel + LL
run_test "1ch + LL" "-e NCCL_MAX_NCHANNELS=1 -e NCCL_PROTO=LL"

# Test 4: 2 channels + LL
run_test "2ch + LL" "-e NCCL_MAX_NCHANNELS=2 -e NCCL_PROTO=LL"

# Test 5: Force Ring algo + LL
run_test "Ring + LL" "-e NCCL_ALGO=Ring -e NCCL_PROTO=LL"

# Test 6: Fewer threads (reduce kernel overhead)
run_test "256 threads + LL" "-e NCCL_NTHREADS=256 -e NCCL_PROTO=LL"

# Test 7: 64 threads + LL (minimal)
run_test "64 threads + LL" "-e NCCL_NTHREADS=64 -e NCCL_PROTO=LL"

# Test 8: Small buffer size
run_test "Small buf (256K)" "-e NCCL_BUFFSIZE=262144 -e NCCL_PROTO=LL"

# Test 9: Very small buffer
run_test "Tiny buf (32K)" "-e NCCL_BUFFSIZE=32768 -e NCCL_PROTO=LL"

# Test 10: CTA policy efficiency
run_test "CTA efficiency" "-e NCCL_CTA_POLICY=EFFICIENCY"

echo ""
echo "=== Done ==="
