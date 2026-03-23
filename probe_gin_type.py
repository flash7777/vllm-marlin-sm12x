#!/usr/bin/env python3
"""Probe GIN type on 2-rank NCCL comm via ctypes.
Run: torchrun --nnodes=2 --nproc-per-node=1 ... probe_gin_type.py
"""
import torch
import torch.distributed as dist
import ctypes, os

def probe():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # Force NCCL comm creation
    t = torch.zeros(1, device="cuda")
    dist.all_reduce(t)
    torch.cuda.synchronize()

    if rank == 0:
        print("AllReduce works. Check NCCL_DEBUG=INFO output for transport details.")
        print("Look for 'Channel', 'transport', 'NET/', 'GIN' in the log.")

    dist.destroy_process_group()

if __name__ == "__main__":
    probe()
