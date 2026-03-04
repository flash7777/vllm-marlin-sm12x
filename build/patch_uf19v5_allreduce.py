#!/usr/bin/env python3
"""
Patch UF19v5: System-Scope Minimal Proxy AllReduce.

Modifies the UF17v3 all_reduce_with_output() in parallel_state.py
to dispatch through UF19v5Communicator when VLLM_UF_UF19V5=1.

v5 vs v4: Single GPU kernel with st.release.sys / ld.acquire.sys PTX,
           cudaHostAlloc pinned buffers, ~6.5 µs vs NCCL 18 µs.

Requires: UF17v3 patch (all_reduce_with_output must exist).
Requires: libuf19v5_sys.so at UF19V5_LIB_PATH.

Runtime env vars:
  VLLM_UF_EAGER_ALLREDUCE=1     (UF17, must be set)
  VLLM_UF_UF19V5=1              (this patch)
  VLLM_UF19_PEER_IP=192.168.0.116
  VLLM_UF19_IB_DEV=rocep1s0f0
  VLLM_UF19_GID_IDX=-1
  UF19V5_LIB_PATH=/opt/vllm/libuf19v5_sys.so
"""

import sys

SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"

# ============================================================
# 1. Patch parallel_state.py: UF19v5 dispatch
# ============================================================
parallel_state_py = f"{SITE_PACKAGES}/vllm/distributed/parallel_state.py"

with open(parallel_state_py) as f:
    src = f.read()

if "all_reduce_with_output" not in src:
    print("ERROR: UF17v3 patch not found. Apply patch_uf17_eager_allreduce.py first.")
    sys.exit(1)

if "uf19v5" in src:
    print("UF19v5: parallel_state.py already patched, skipping")
else:
    old_body = """    # v3: Direct NCCL AllReduce with separate sendbuf/recvbuf.
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

    # Also try matching v4 patched version (UF19+UF17v3 block)
    if old_body not in src:
        # v4 might have wrapped the NCCL block — look for just the NCCL part
        alt_body = """    comm = group.device_communicator
    pynccl = getattr(comm, 'pynccl_comm', None)
    if pynccl is not None and not pynccl.disabled:
        pynccl.all_reduce(input_, out_tensor=output)
    else:
        # Fallback: copy + in-place torch.distributed all_reduce
        output.copy_(input_)
        import torch.distributed as _dist
        _dist.all_reduce(output, group=comm.device_group)"""
        if alt_body not in src:
            print("ERROR: Could not find UF17v3 NCCL dispatch block")
            sys.exit(1)
        old_body = alt_body

    new_body = """    # UF19v5: System-scope PTX + ibverbs RDMA AllReduce (TP=2).
    # Falls back to NCCL if not enabled or init fails.
    import os as _os
    if _os.environ.get("VLLM_UF_UF19V5", "0") == "1":
        if not hasattr(group, '_uf19v5_comm'):
            group._uf19v5_comm = None
            group._uf19v5_failed = False
            try:
                import sys as _sys
                _sys.path.insert(0, "/opt/vllm")
                from uf19v5_sys import UF19v5Communicator
                rank = group.rank_in_group
                peer_ip = _os.environ.get("VLLM_UF19_PEER_IP", "")
                if not peer_ip:
                    raise ValueError("VLLM_UF19_PEER_IP not set")
                ib_dev = _os.environ.get("VLLM_UF19_IB_DEV", "rocep1s0f0")
                gid_idx = int(_os.environ.get("VLLM_UF19_GID_IDX", "-1"))
                group._uf19v5_comm = UF19v5Communicator(
                    rank, group.world_size, peer_ip, ib_dev,
                    gid_idx, input_.numel())
                import logging
                logging.getLogger("vllm").info(
                    "UF19v5: sys-scope AllReduce initialized (rank=%d, peer=%s)",
                    rank, peer_ip)
            except Exception as e:
                import logging
                logging.getLogger("vllm").warning(
                    "UF19v5: init failed (%s), falling back to NCCL", e)
                group._uf19v5_comm = None
                group._uf19v5_failed = True

        if group._uf19v5_comm is not None and not group._uf19v5_failed:
            ret = group._uf19v5_comm.all_reduce(input_, output)
            if ret == 0:
                return
            if ret == -2:
                pass  # skip (warmup/inactive) → fall through to NCCL
            else:
                import logging
                logging.getLogger("vllm").warning(
                    "UF19v5: all_reduce failed (ret=%d), disabling", ret)
                group._uf19v5_failed = True

    # NCCL fallback
    comm = group.device_communicator
    pynccl = getattr(comm, 'pynccl_comm', None)
    if pynccl is not None and not pynccl.disabled:
        pynccl.all_reduce(input_, out_tensor=output)
    else:
        output.copy_(input_)
        import torch.distributed as _dist
        _dist.all_reduce(output, group=comm.device_group)"""

    src = src.replace(old_body, new_body)

    with open(parallel_state_py, "w") as f:
        f.write(src)
    print("UF19v5: Patched parallel_state.py — sys-scope dispatch added")

# ============================================================
# 2. Install uf19v5_sys.py to /opt/vllm/
# ============================================================
import shutil
import os

src_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uf19v5_sys.py")
dst_py = "/opt/vllm/uf19v5_sys.py"

if os.path.exists(src_py):
    shutil.copy2(src_py, dst_py)
    print(f"UF19v5: Installed {dst_py}")
elif os.path.exists("/tmp/uf19v5_sys.py"):
    shutil.copy2("/tmp/uf19v5_sys.py", dst_py)
    print(f"UF19v5: Installed {dst_py} (from /tmp)")
else:
    print(f"WARNING: uf19v5_sys.py not found at {src_py}")

# ============================================================
# 3. Patch serve_torchrun.py: activate UF19v5 after warmup
# ============================================================
serve_py = "/opt/vllm/serve_torchrun.py"

try:
    with open(serve_py) as f:
        serve_src = f.read()

    if "uf19v5_activate" in serve_src:
        print("UF19v5: serve_torchrun.py already patched, skipping")
    else:
        old_ready = '    print(f"[Rank {rank}/{world_size}] LLM engine ready.")'
        new_ready = '''    print(f"[Rank {rank}/{world_size}] LLM engine ready.")

    # UF19v5: Activate sys-scope AllReduce after warmup.
    import os as _os
    if _os.environ.get("VLLM_UF_UF19V5", "0") == "1":
        from vllm.distributed.parallel_state import get_tp_group
        tp = get_tp_group()
        if hasattr(tp, '_uf19v5_comm') and tp._uf19v5_comm is not None:
            tp._uf19v5_comm.activate()
            print(f"[Rank {rank}/{world_size}] UF19v5 sys-scope AllReduce ACTIVATED")'''

        if old_ready not in serve_src:
            print("WARNING: Could not find injection point in serve_torchrun.py")
        else:
            serve_src = serve_src.replace(old_ready, new_ready)
            with open(serve_py, "w") as f:
                f.write(serve_src)
            print("UF19v5: Patched serve_torchrun.py — activation after warmup")
except FileNotFoundError:
    print("WARNING: serve_torchrun.py not found, skipping activation patch")

print("UF19v5: Done")
