#!/usr/bin/env python3
"""Probe NCCL 2.29 Device API + GIN support on a 2-rank communicator.
Run: torchrun --nnodes=2 --nproc-per-node=1 --node-rank=N \
     --master-addr=192.168.0.117 --master-port=29500 probe_nccl_device_api.py
"""
import torch
import torch.distributed as dist
import ctypes, os, struct

def probe():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    print(f"[Rank {rank}] torch.distributed initialized (world={world})")

    # Get the underlying NCCL communicator from PyTorch
    # PyTorch stores it in the ProcessGroupNCCL backend
    pg = dist.distributed_c10d._get_default_group()
    backend = pg._get_backend(torch.device("cuda"))
    print(f"[Rank {rank}] Backend: {type(backend).__name__}")

    # Load libnccl
    nccl = ctypes.CDLL("libnccl.so.2")

    # Define ncclCommProperties_t struct (from nccl.h)
    # struct ncclCommProperties {
    #   size_t size; unsigned int magic; unsigned int version;
    #   int rank; int nRanks; int cudaDev; int nvmlDev;
    #   bool deviceApiSupport; bool multimemSupport; uint8_t ginType;
    # };
    class NcclCommProperties(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_size_t),
            ("magic", ctypes.c_uint),
            ("version", ctypes.c_uint),
            ("rank", ctypes.c_int),
            ("nRanks", ctypes.c_int),
            ("cudaDev", ctypes.c_int),
            ("nvmlDev", ctypes.c_int),
            ("deviceApiSupport", ctypes.c_bool),
            ("multimemSupport", ctypes.c_bool),
            ("ginType", ctypes.c_uint8),
        ]

    # Force a collective to ensure NCCL comm is initialized
    t = torch.zeros(1, device="cuda")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    print(f"[Rank {rank}] AllReduce test passed")

    # Try to extract ncclComm_t - this is tricky since PyTorch doesn't expose it directly
    # Instead, let's use the C probe approach with proper multi-rank

    # Alternative: use ncclPutSignal/ncclWaitSignal through the host API
    # which works on the existing torch.distributed comm

    # For now, let's benchmark the host-side RMA API
    # ncclPutSignal, ncclSignal, ncclWaitSignal

    # Test: Register a window
    # We can't easily get ncclComm_t from PyTorch, but we can benchmark alternatives

    # === Benchmark: AllReduce vs manual split ===
    import time

    # Small tensor matching vLLM AllReduce sizes
    sizes = [
        ("4K elem (8 KB)", 4096),
        ("32K elem (64 KB)", 32768),
        ("49K elem (98 KB)", 49152),
    ]

    for label, numel in sizes:
        t = torch.randn(numel, dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(200):
            dist.all_reduce(t)
        torch.cuda.synchronize()

        # Benchmark
        n_iter = 2000
        start = time.perf_counter()
        for _ in range(n_iter):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        us = elapsed * 1e6 / n_iter

        if rank == 0:
            print(f"AllReduce {label}: {us:.1f} µs/call")

        dist.barrier()

    # === Now test: send+recv (manual AllReduce) ===
    # For 2-rank AllReduce: each rank sends its half, receives peer's half, reduces
    # Split approach: reduce-scatter + allgather
    for label, numel in sizes:
        t_in = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
        t_out = torch.empty_like(t_in)
        t_recv = torch.empty_like(t_in)
        peer = 1 - rank

        # Warmup
        for _ in range(200):
            # Simultaneous send+recv (full data swap)
            if rank == 0:
                dist.send(t_in, dst=1)
                dist.recv(t_recv, src=1)
            else:
                dist.recv(t_recv, src=0)
                dist.send(t_in, dst=0)
            t_out = t_in + t_recv
        torch.cuda.synchronize()

        # Benchmark
        n_iter = 2000
        start = time.perf_counter()
        for _ in range(n_iter):
            if rank == 0:
                dist.send(t_in, dst=1)
                dist.recv(t_recv, src=1)
            else:
                dist.recv(t_recv, src=0)
                dist.send(t_in, dst=0)
            t_out = t_in + t_recv
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        us = elapsed * 1e6 / n_iter

        if rank == 0:
            print(f"Send+Recv+Add {label}: {us:.1f} µs/call")

        dist.barrier()

    # === Now test: batch_isend_irecv (concurrent send+recv) ===
    for label, numel in sizes:
        t_in = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
        t_out = torch.empty_like(t_in)
        t_recv = torch.empty_like(t_in)
        peer = 1 - rank

        # Warmup
        for _ in range(200):
            ops = []
            ops.append(dist.P2POp(dist.isend, t_in, peer))
            ops.append(dist.P2POp(dist.irecv, t_recv, peer))
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
            t_out = t_in + t_recv
        torch.cuda.synchronize()

        # Benchmark
        n_iter = 2000
        start = time.perf_counter()
        for _ in range(n_iter):
            ops = []
            ops.append(dist.P2POp(dist.isend, t_in, peer))
            ops.append(dist.P2POp(dist.irecv, t_recv, peer))
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
            t_out = t_in + t_recv
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        us = elapsed * 1e6 / n_iter

        if rank == 0:
            print(f"Batch P2P+Add {label}: {us:.1f} µs/call")

        dist.barrier()

    # === Test: reduce_scatter + all_gather (NCCL native) ===
    for label, numel in sizes:
        if numel % 2 != 0:
            continue
        half = numel // 2
        t_in = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
        t_half = torch.empty(half, dtype=torch.bfloat16, device="cuda")
        t_out = torch.empty(numel, dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(200):
            dist.reduce_scatter_tensor(t_half, t_in)
            dist.all_gather_into_tensor(t_out, t_half)
        torch.cuda.synchronize()

        # Benchmark
        n_iter = 2000
        start = time.perf_counter()
        for _ in range(n_iter):
            dist.reduce_scatter_tensor(t_half, t_in)
            dist.all_gather_into_tensor(t_out, t_half)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        us = elapsed * 1e6 / n_iter

        if rank == 0:
            print(f"RS+AG {label}: {us:.1f} µs/call")

        dist.barrier()

    dist.destroy_process_group()
    print(f"[Rank {rank}] Done.")

if __name__ == "__main__":
    probe()
