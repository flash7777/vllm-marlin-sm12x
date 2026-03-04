#!/usr/bin/env python3
"""
Patch UF19v4: CUDA-Graph-Compatible AllReduce via ibverbs RDMA.

Modifies the UF17v3 all_reduce_with_output() function in parallel_state.py
to dispatch through UF19Communicator when VLLM_UF_UF19_RDMA=1.

v4 Architecture:
  GPU kernels (graph-capturable): wait_send_done → prepare_send → poll_recv → add_recv
  CPU proxy thread (background): polls send_flag → RDMA WRITE → CQ → send_done

Requires: UF17v3 patch already applied (all_reduce_with_output must exist).
Requires: libuf19_rdma.so compiled and available at UF19_LIB_PATH.

Env vars at runtime:
  VLLM_UF_EAGER_ALLREDUCE=1  (UF17, must be set)
  VLLM_UF_UF19_RDMA=1        (this patch)
  VLLM_UF19_PEER_IP=192.168.0.116   (peer's RDMA IP)
  VLLM_UF19_IB_DEV=rocep1s0f0       (RDMA device, default: rocep1s0f0)
  VLLM_UF19_GID_IDX=-1              (GID index, -1=auto-detect RoCEv2)
  UF19_LIB_PATH=/opt/vllm/libuf19_rdma.so  (default)
"""

import sys

SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"

# ============================================================
# 1. Patch parallel_state.py: UF19 dispatch in all_reduce_with_output
# ============================================================
parallel_state_py = f"{SITE_PACKAGES}/vllm/distributed/parallel_state.py"

with open(parallel_state_py) as f:
    src = f.read()

# Verify UF17v3 is applied
if "all_reduce_with_output" not in src:
    print("ERROR: UF17v3 patch not found. Apply patch_uf17_eager_allreduce.py first.")
    sys.exit(1)

if "uf19_rdma" in src:
    print("UF19: parallel_state.py already patched, skipping")
else:
    # Find the existing UF17v3 body of all_reduce_with_output
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

    if old_body not in src:
        print("ERROR: Could not find UF17v3 body in all_reduce_with_output")
        print("       Expected the v3 NCCL dispatch block")
        sys.exit(1)

    new_body = """    # UF19+UF17v3: Custom ibverbs RDMA AllReduce (TP=2 only).
    # Falls back to NCCL if UF19 not enabled or init fails.
    import os as _os
    if _os.environ.get("VLLM_UF_UF19_RDMA", "0") == "1":
        # Lazy init: create UF19Communicator on first call
        if not hasattr(group, '_uf19_comm'):
            group._uf19_comm = None  # sentinel during init
            group._uf19_failed = False
            try:
                import sys as _sys
                _sys.path.insert(0, "/opt/vllm")
                from uf19_rdma import UF19Communicator
                rank = group.rank_in_group
                peer_ip = _os.environ.get("VLLM_UF19_PEER_IP", "")
                if not peer_ip:
                    raise ValueError("VLLM_UF19_PEER_IP not set")
                ib_dev = _os.environ.get("VLLM_UF19_IB_DEV", "rocep1s0f0")
                gid_idx = int(_os.environ.get("VLLM_UF19_GID_IDX", "-1"))
                group._uf19_comm = UF19Communicator(
                    rank, group.world_size, peer_ip, ib_dev,
                    gid_idx, input_.numel())
                import logging
                logging.getLogger("vllm").info(
                    "UF19: ibverbs AllReduce initialized (rank=%d, peer=%s)",
                    rank, peer_ip)
            except Exception as e:
                import logging
                logging.getLogger("vllm").warning(
                    "UF19: init failed (%s), falling back to NCCL", e)
                group._uf19_comm = None
                group._uf19_failed = True

        if group._uf19_comm is not None and not group._uf19_failed:
            ret = group._uf19_comm.all_reduce(input_, output)
            if ret == 0:
                return  # success
            if ret == -2:
                pass  # skip (warmup/capture or too large) → fall through to NCCL
            else:
                # Fatal error (timeout, QP error) → disable UF19 permanently
                import logging
                logging.getLogger("vllm").warning(
                    "UF19: all_reduce failed (ret=%d), disabling UF19 permanently", ret)
                group._uf19_failed = True

    # v3: Direct NCCL AllReduce with separate sendbuf/recvbuf.
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

    src = src.replace(old_body, new_body)

    with open(parallel_state_py, "w") as f:
        f.write(src)
    print("UF19: Patched parallel_state.py — ibverbs RDMA dispatch added")

# ============================================================
# 2. Install uf19_rdma.py to /opt/vllm/
# ============================================================
import shutil
import os

src_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uf19_rdma.py")
dst_py = "/opt/vllm/uf19_rdma.py"

if os.path.exists(src_py):
    shutil.copy2(src_py, dst_py)
    print(f"UF19: Installed {dst_py}")
elif os.path.exists("/tmp/uf19_rdma.py"):
    shutil.copy2("/tmp/uf19_rdma.py", dst_py)
    print(f"UF19: Installed {dst_py} (from /tmp)")
else:
    print(f"WARNING: uf19_rdma.py not found at {src_py}")

# ============================================================
# 3. Patch serve_torchrun.py: activate UF19 after warmup/capture
# ============================================================
serve_py = "/opt/vllm/serve_torchrun.py"

try:
    with open(serve_py) as f:
        serve_src = f.read()

    if "uf19_activate" in serve_src:
        print("UF19: serve_torchrun.py already patched, skipping")
    else:
        old_ready = '    print(f"[Rank {rank}/{world_size}] LLM engine ready.")'
        new_ready = '''    print(f"[Rank {rank}/{world_size}] LLM engine ready.")

    # UF19: Activate RDMA AllReduce now that warmup+capture is done.
    # During warmup/capture, UF19 returns -2 (skip) to avoid deadlocks
    # with torch.distributed barriers. Now inference is lockstep-safe.
    import os as _os
    if _os.environ.get("VLLM_UF_UF19_RDMA", "0") == "1":
        from vllm.distributed.parallel_state import get_tp_group
        tp = get_tp_group()
        if hasattr(tp, '_uf19_comm') and tp._uf19_comm is not None:
            tp._uf19_comm.activate()
            print(f"[Rank {rank}/{world_size}] UF19 RDMA AllReduce ACTIVATED")'''

        if old_ready not in serve_src:
            print("WARNING: Could not find injection point in serve_torchrun.py")
        else:
            serve_src = serve_src.replace(old_ready, new_ready)
            with open(serve_py, "w") as f:
                f.write(serve_src)
            print("UF19: Patched serve_torchrun.py — activation after warmup")
except FileNotFoundError:
    print("WARNING: serve_torchrun.py not found, skipping activation patch")

print("UF19: Done")
