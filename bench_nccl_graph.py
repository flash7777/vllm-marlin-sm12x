#!/usr/bin/env python3
"""Benchmark NCCL AllReduce: regular vs CUDA Graph captured.
Run via torchrun on 2 nodes.
"""
import torch
import torch.distributed as dist
import os, time

def bench(label, fn, n_warmup=200, n_iter=5000):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    us = elapsed * 1e6 / n_iter
    if dist.get_rank() == 0:
        print(f"  {label}: {us:.1f} µs/call")

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    if rank == 0:
        print("=== NCCL AllReduce: Regular vs CUDA Graph ===\n")

    sizes = [
        ("4K (8KB)", 4096),
        ("2K (4KB)", 2048),
        ("1K (2KB)", 1024),
    ]

    for label, numel in sizes:
        t = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
        t_graph = t.clone()

        # Regular AllReduce
        bench(f"Regular {label}", lambda: dist.all_reduce(t))

        # CUDA Graph captured AllReduce
        # Warmup for graph capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(20):
                dist.all_reduce(t_graph)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            dist.all_reduce(t_graph)
        torch.cuda.synchronize()

        bench(f"Graph  {label}", lambda: g.replay())

        if rank == 0:
            print()

        dist.barrier()

    # Also test: multiple AllReduces in a single graph (amortize overhead)
    if rank == 0:
        print("=== Batched AllReduces in Graph ===")

    for n_ops in [1, 5, 10, 20, 50]:
        numel = 4096
        tensors = [torch.randn(numel, dtype=torch.bfloat16, device="cuda") for _ in range(n_ops)]

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(20):
                for t in tensors:
                    dist.all_reduce(t)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture graph with n_ops AllReduces
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for t in tensors:
                dist.all_reduce(t)
        torch.cuda.synchronize()

        def replay():
            g.replay()

        # Benchmark
        n_warmup = 100
        n_iter = 2000
        for _ in range(n_warmup):
            replay()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            replay()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_us = elapsed * 1e6 / n_iter
        per_op = total_us / n_ops

        if rank == 0:
            print(f"  {n_ops:2d} AllReduces/graph: {total_us:.1f} µs total, {per_op:.1f} µs/op")

        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
