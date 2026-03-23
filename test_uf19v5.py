#!/usr/bin/env python3
"""
Test UF19v5 sys-scope AllReduce: correctness + latency.
Run on 2 ranks (head=rank0, worker=rank1).

Usage:
  Head:   python3 /tmp/test_uf19v5.py --rank 0 --peer-ip 192.168.0.116
  Worker: python3 /tmp/test_uf19v5.py --rank 1 --peer-ip 192.168.0.117
"""
import argparse
import sys
import time
import os

sys.path.insert(0, "/opt/vllm")
os.environ["UF19V5_LIB_PATH"] = "/opt/vllm/libuf19v5_sys.so"

import torch
from uf19v5_sys import UF19v5Communicator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--peer-ip", type=str, required=True)
    parser.add_argument("--ib-dev", type=str, default="rocep1s0f0")
    parser.add_argument("--numel", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print(f"[Rank {args.rank}] Init UF19v5 (peer={args.peer_ip}, dev={args.ib_dev}, numel={args.numel})")

    comm = UF19v5Communicator(
        rank=args.rank, world_size=2,
        peer_ip=args.peer_ip, ib_dev=args.ib_dev,
        gid_idx=-1, numel=args.numel)

    # Activate immediately (no warmup/graph capture phase)
    comm.activate()
    print(f"[Rank {args.rank}] Activated")

    # --- Correctness Test ---
    print(f"\n[Rank {args.rank}] === Correctness Test ===")
    for step in range(10):
        # Each rank creates unique data: rank0=[1,2,3,...]*step, rank1=[10,20,30,...]*step
        if args.rank == 0:
            inp = torch.arange(1, args.numel + 1, dtype=torch.bfloat16, device=device) * (step + 1)
        else:
            inp = torch.arange(1, args.numel + 1, dtype=torch.bfloat16, device=device) * (step + 1) * 10

        out = torch.zeros(args.numel, dtype=torch.bfloat16, device=device)

        ret = comm.all_reduce(inp, out)
        torch.cuda.synchronize()

        if ret != 0:
            print(f"  step {step}: all_reduce returned {ret} FAIL")
            continue

        # Expected: rank0_val + rank1_val
        if args.rank == 0:
            expected = torch.arange(1, args.numel + 1, dtype=torch.bfloat16, device=device) * (step + 1) * 11
        else:
            expected = torch.arange(1, args.numel + 1, dtype=torch.bfloat16, device=device) * (step + 1) * 11

        maxdiff = (out.float() - expected.float()).abs().max().item()
        # bf16 tolerance: relative error up to ~1%
        threshold = expected.float().abs().max().item() * 0.02 + 0.1
        ok = maxdiff < threshold
        print(f"  step {step}: maxdiff={maxdiff:.4f} (thr={threshold:.2f}) {'OK' if ok else 'FAIL'}")
        if not ok:
            # Show first few mismatches
            diff = (out.float() - expected.float()).abs()
            bad_idx = (diff > threshold).nonzero(as_tuple=True)[0][:5]
            for i in bad_idx:
                print(f"    [{i.item()}] got={out[i].item():.4f} exp={expected[i].item():.4f}")

    # --- Latency Benchmark ---
    print(f"\n[Rank {args.rank}] === Latency Benchmark ===")
    inp = torch.ones(args.numel, dtype=torch.bfloat16, device=device) * (args.rank + 1)
    out = torch.zeros(args.numel, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(args.warmup):
        comm.all_reduce(inp, out)
    torch.cuda.synchronize()

    # Timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        comm.all_reduce(inp, out)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    us_per = elapsed / args.iters * 1e6
    print(f"  {args.iters} iters: {elapsed*1e3:.1f} ms total, {us_per:.1f} µs/iter")
    print(f"  vs NCCL AllReduce: ~18 µs")
    print(f"  Speedup: {18.0/us_per:.1f}x")

    print(f"\n[Rank {args.rank}] Done")

if __name__ == "__main__":
    main()
