#!/usr/bin/env python3
"""Benchmark NCCL AllReduce latency with various configs.
Run on both ranks simultaneously via torchrun."""
import torch
import torch.distributed as dist
import os, time

def bench(label, tensor, n_warmup=200, n_iter=2000):
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
        print(f"{label}: {us:.1f} µs/call  ({elapsed*1000:.1f} ms / {n_iter})")

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # Test sizes matching real vLLM AllReduce sizes
    sizes = [
        ("4K elem (8 KB)", 4096),
        ("32K elem (64 KB)", 32768),
        ("49K elem (98 KB)", 49152),
    ]

    for label, numel in sizes:
        t = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
        bench(label, t)
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
