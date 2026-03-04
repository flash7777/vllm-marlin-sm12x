"""
UF19v4: CUDA-Graph-Compatible 2-Rank AllReduce via ibverbs RDMA.

Python wrapper around libuf19_rdma.so for vLLM integration.
Bypasses NCCL for TP=2 AllReduce on RoCE networks.

Architecture:
  GPU kernels (captured in CUDA graph):
    wait_send_done → prepare_send → poll_recv → add_recv
  CPU proxy thread (background):
    Polls send_flag → RDMA WRITE → polls CQ → sets send_done

Usage:
    comm = UF19Communicator(rank=0, world_size=2,
                            peer_ip="192.168.0.116",
                            ib_dev="rocep1s0f0",
                            gid_idx=-1, numel=2048)
    comm.all_reduce(input_tensor, output_tensor)
"""

import ctypes
import os
import torch


class UF19Communicator:
    """Custom 2-rank AllReduce via raw ibverbs RDMA (v4: graph-capturable)."""

    _lib = None
    _initialized = False

    def __init__(self, rank: int, world_size: int,
                 peer_ip: str, ib_dev: str = "rocep1s0f0",
                 gid_idx: int = -1, numel: int = 2048):
        assert world_size == 2, "UF19 only supports TP=2"

        lib_path = os.environ.get("UF19_LIB_PATH", "/opt/vllm/libuf19_rdma.so")
        self._lib = ctypes.CDLL(lib_path)

        # Set argument types
        self._lib.uf19_init.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_int
        ]
        self._lib.uf19_init.restype = ctypes.c_int

        self._lib.uf19_step.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        ]
        self._lib.uf19_step.restype = ctypes.c_int

        self._lib.uf19_activate.argtypes = []
        self._lib.uf19_activate.restype = None

        self._lib.uf19_cleanup.argtypes = []
        self._lib.uf19_cleanup.restype = None

        # Init ibverbs + start proxy thread (blocking: TCP handshake with peer)
        ret = self._lib.uf19_init(
            rank, world_size,
            peer_ip.encode(), ib_dev.encode(),
            gid_idx, numel)
        if ret != 0:
            raise RuntimeError(f"uf19_init failed with code {ret}")
        self._initialized = True

    def all_reduce(self, input_: torch.Tensor, output: torch.Tensor) -> int:
        """Replace NCCL AllReduce. Only launches GPU kernels (graph-capturable).
        Returns: 0=success, -2=skip/inactive (use NCCL), other=fatal."""
        ret = self._lib.uf19_step(
            ctypes.c_void_p(input_.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            input_.numel())
        return ret

    def activate(self):
        """Enable UF19 RDMA. Call after warmup/CUDA graph capture is complete."""
        self._lib.uf19_activate()

    def __del__(self):
        if self._initialized and self._lib is not None:
            self._lib.uf19_cleanup()
            self._initialized = False
