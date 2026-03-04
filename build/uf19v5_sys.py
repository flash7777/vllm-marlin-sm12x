"""
UF19v5: System-Scope Minimal Proxy AllReduce.

Python wrapper around libuf19v5_sys.so for vLLM integration.
Replaces NCCL AllReduce with system-scope PTX + ibverbs RDMA proxy.
Target: ~6.5 µs/call (vs NCCL 18 µs).

Architecture:
  GPU kernel (1 launch, sys-scope PTX):
    st.release.sys input → send_buf → proxy RDMA → peer recv_buf
    ld.acquire.sys recv_buf → reduce + output
  CPU proxy (background): polls send_flag → ibv_post_send → CQ → send_done

Usage:
    comm = UF19v5Communicator(rank=0, world_size=2,
                              peer_ip="192.168.0.116",
                              ib_dev="rocep1s0f0",
                              gid_idx=-1, numel=4096)
    comm.all_reduce(input_tensor, output_tensor)
"""

import ctypes
import os
import torch


class UF19v5Communicator:
    """Custom 2-rank AllReduce via sys-scope PTX + ibverbs RDMA (v5)."""

    _lib = None
    _initialized = False

    def __init__(self, rank: int, world_size: int,
                 peer_ip: str, ib_dev: str = "rocep1s0f0",
                 gid_idx: int = -1, numel: int = 4096):
        assert world_size == 2, "UF19v5 only supports TP=2"

        lib_path = os.environ.get("UF19V5_LIB_PATH", "/opt/vllm/libuf19v5_sys.so")
        self._lib = ctypes.CDLL(lib_path)

        self._lib.uf19v5_init.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_int
        ]
        self._lib.uf19v5_init.restype = ctypes.c_int

        self._lib.uf19v5_step.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        ]
        self._lib.uf19v5_step.restype = ctypes.c_int

        self._lib.uf19v5_activate.argtypes = []
        self._lib.uf19v5_activate.restype = None

        self._lib.uf19v5_cleanup.argtypes = []
        self._lib.uf19v5_cleanup.restype = None

        ret = self._lib.uf19v5_init(
            rank, world_size,
            peer_ip.encode(), ib_dev.encode(),
            gid_idx, numel)
        if ret != 0:
            raise RuntimeError(f"uf19v5_init failed with code {ret}")
        self._initialized = True

    def all_reduce(self, input_: torch.Tensor, output: torch.Tensor) -> int:
        """Replace NCCL AllReduce. Launches 1 GPU kernel on current stream.
        Returns: 0=success, -2=skip/inactive, other=fatal."""
        stream = torch.cuda.current_stream()
        ret = self._lib.uf19v5_step(
            ctypes.c_void_p(input_.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            input_.numel(),
            stream.cuda_stream)
        return ret

    def activate(self):
        """Enable v5 AllReduce. Call after warmup is complete."""
        self._lib.uf19v5_activate()

    def __del__(self):
        if self._initialized and self._lib is not None:
            self._lib.uf19v5_cleanup()
            self._initialized = False
