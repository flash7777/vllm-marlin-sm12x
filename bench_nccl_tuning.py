#!/usr/bin/env python3
"""Benchmark NCCL AllReduce with various tuning parameters.
Run via torchrun on 2 nodes. Only tests 4K elements (8KB) — the dominant case.
"""
import torch
import torch.distributed as dist
import os, time

def bench(label, tensor, n_warmup=200, n_iter=3000):
    for _ in range(n_warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    us = elapsed * 1e6 / n_iter
    if dist.get_rank() == 0:
        print(f"  {label}: {us:.1f} µs/call  ({elapsed*1000:.1f} ms / {n_iter})")

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    if rank == 0:
        print("=== NCCL AllReduce Tuning Benchmark (4K elem / 8KB BF16) ===")
        print(f"NCCL env vars:")
        for var in sorted(os.environ):
            if 'NCCL' in var:
                print(f"  {var}={os.environ[var]}")
        print()

    numel = 4096
    t = torch.randn(numel, dtype=torch.bfloat16, device="cuda")

    bench("AllReduce 4K", t)

    # Also test a few other sizes
    for label, n in [("8K elem (16KB)", 8192), ("2K elem (4KB)", 2048),
                     ("1K elem (2KB)", 1024), ("512 elem (1KB)", 512)]:
        t2 = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        bench(label, t2)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
