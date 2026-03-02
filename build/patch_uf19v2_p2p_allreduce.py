#!/usr/bin/env python3
"""
Patch UF19v2: Simplified 2-rank AllReduce via ncclSend/ncclRecv.

Replaces NCCL's generic Ring AllReduce (3 sequential steps for 2 ranks)
with a single bidirectional exchange + local add.

NCCL AllReduce for TP=2:  ReduceScatter → AllGather → Barrier  =  ~19µs
UF19v2 P2P:               ncclSend + ncclRecv + torch.add       =  ~10µs (expected)

Uses NCCL transport (handles GPU coherence on GB10 unified memory correctly).
No ibverbs, no C/CUDA code, no .so needed. Pure Python patch.

Requires: UF17v3 patch already applied (all_reduce_with_output must exist).

Env vars at runtime:
  VLLM_UF_EAGER_ALLREDUCE=1    (UF17, must be set)
  VLLM_UF_UF19V2_P2P=1         (this patch)
"""

import sys

SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"

# ============================================================
# 1. Patch parallel_state.py: UF19v2 P2P dispatch
# ============================================================
parallel_state_py = f"{SITE_PACKAGES}/vllm/distributed/parallel_state.py"

with open(parallel_state_py) as f:
    src = f.read()

# Verify UF17v3 is applied
if "all_reduce_with_output" not in src:
    print("ERROR: UF17v3 patch not found. Apply patch_uf17_eager_allreduce.py first.")
    sys.exit(1)

if "uf19v2" in src:
    print("UF19v2: parallel_state.py already patched, skipping")
else:
    # Find the NCCL dispatch block — could be plain UF17v3 or UF19-patched
    # We look for the NCCL fallback that's always present
    old_nccl_block = """    # v3: Direct NCCL AllReduce with separate sendbuf/recvbuf.
    # NCCL writes directly to output — no temp alloc, no copy_.
    comm = group.device_communicator
    pynccl = getattr(comm, 'pynccl_comm', None)
    if pynccl is not None and not pynccl.disabled:
        pynccl.all_reduce(input_, out_tensor=output)
    else:
        # Fallback: copy + in-place torch.distributed all_reduce
        output.copy_(input_)
        import torch.distributed as _dist
        _dist.all_reduce(output, group=comm.device_group)"""

    if old_nccl_block not in src:
        print("ERROR: Could not find NCCL dispatch block in all_reduce_with_output")
        print("       Expected the v3 NCCL AllReduce block")
        sys.exit(1)

    new_nccl_block = """    # UF19v2+UF17v3: Simplified 2-rank AllReduce via ncclSend/ncclRecv.
    # Replaces NCCL Ring (3 steps) with 1 bidirectional exchange + local add.
    # Uses NCCL transport (handles GPU coherence on GB10 correctly).
    import os as _os
    comm = group.device_communicator
    pynccl = getattr(comm, 'pynccl_comm', None)

    if (pynccl is not None and not pynccl.disabled
            and _os.environ.get("VLLM_UF_UF19V2_P2P", "0") == "1"
            and group.world_size == 2):
        # Lazy init: allocate persistent recv buffer on first call
        if not hasattr(group, '_uf19v2_recv_buf'):
            group._uf19v2_recv_buf = None
            group._uf19v2_failed = False
            try:
                import torch
                group._uf19v2_recv_buf = torch.empty_like(input_)
                import logging
                logging.getLogger("vllm").info(
                    "UF19v2: P2P AllReduce initialized (rank=%d, numel=%d)",
                    group.rank_in_group, input_.numel())
            except Exception as e:
                import logging
                logging.getLogger("vllm").warning(
                    "UF19v2: init failed (%s), falling back to NCCL", e)
                group._uf19v2_failed = True

        if group._uf19v2_recv_buf is not None and not group._uf19v2_failed:
            try:
                peer = 1 - group.rank_in_group  # 0→1, 1→0
                recv_buf = group._uf19v2_recv_buf

                # Resize recv buffer if tensor size changed
                if recv_buf.numel() != input_.numel() or recv_buf.dtype != input_.dtype:
                    import torch
                    group._uf19v2_recv_buf = torch.empty_like(input_)
                    recv_buf = group._uf19v2_recv_buf

                # Bidirectional exchange: both ranks send+recv simultaneously
                pynccl.nccl.ncclGroupStart()
                pynccl.send(input_, dst=peer)
                pynccl.recv(recv_buf, src=peer)
                pynccl.nccl.ncclGroupEnd()

                # Local add: output = input + received
                import torch
                torch.add(input_, recv_buf, out=output)
                return
            except Exception as e:
                import logging
                logging.getLogger("vllm").warning(
                    "UF19v2: P2P failed (%s), disabling permanently", e)
                group._uf19v2_failed = True

    # Fallback: NCCL AllReduce (Ring protocol)
    if pynccl is not None and not pynccl.disabled:
        pynccl.all_reduce(input_, out_tensor=output)
    else:
        # Fallback: copy + in-place torch.distributed all_reduce
        output.copy_(input_)
        import torch.distributed as _dist
        _dist.all_reduce(output, group=comm.device_group)"""

    src = src.replace(old_nccl_block, new_nccl_block)

    with open(parallel_state_py, "w") as f:
        f.write(src)
    print("UF19v2: Patched parallel_state.py — P2P ncclSend/ncclRecv dispatch added")

print("UF19v2: Done")
