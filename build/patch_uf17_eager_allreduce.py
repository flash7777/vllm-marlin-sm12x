#!/usr/bin/env python3
"""
Patch UF17v3: EAGER_ALLREDUCE — run NCCL AllReduce outside CUDA Graphs.

Uses all_reduce_with_output (mutates_args=["output"]) as a splitting op
so AllReduce runs eagerly between piecewise CUDA graph segments.

v3 improvement over v2: Calls pynccl_comm.all_reduce(input_, out_tensor=output)
directly, using NCCL's native sendbuf/recvbuf separation. NCCL writes the
reduced result directly into the deterministic output buffer. No temp tensor
allocation, no extra copy_ kernel.

v2 had: _all_reduce_out_place(input_) → temp alloc + ncclAllReduce → output.copy_(temp)
v3 has: ncclAllReduce(input_.ptr, output.ptr) — zero-copy, one NCCL call.

Saves 97× temp alloc + 97× copy_ per token on Qwen3-Coder TP=2.
"""

import sys

SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"

# ============================================================
# 1. Patch parallel_state.py: register all_reduce_with_output
# ============================================================
parallel_state_py = f"{SITE_PACKAGES}/vllm/distributed/parallel_state.py"

with open(parallel_state_py) as f:
    src = f.read()

if "all_reduce_with_output" in src:
    # Check if it's v2 (has output.copy_) and upgrade to v3
    if "output.copy_(result)" in src or "output.copy_(result )" in src:
        # v2 → v3 upgrade: replace the function body
        old_body = """    result = group._all_reduce_out_place(input_)
    output.copy_(result)"""
        new_body = """    # v3: Direct NCCL AllReduce with separate sendbuf/recvbuf.
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
        if old_body in src:
            src = src.replace(old_body, new_body)
            src = src.replace(
                "# UF17v2: In-place AllReduce",
                "# UF17v3: Zero-copy AllReduce",
            )
            with open(parallel_state_py, "w") as f:
                f.write(src)
            print("UF17v3: Upgraded parallel_state.py v2 → v3 (zero-copy)")
        else:
            print("UF17: parallel_state.py has all_reduce_with_output but "
                  "unexpected body format, skipping")
    else:
        print("UF17v3: parallel_state.py already patched (v3), skipping")
else:
    # Fresh install: add the all_reduce_with_output op
    # Insert AFTER the existing all_reduce registration block
    marker = '''direct_register_custom_op(
    op_name="all_reduce",
    op_func=all_reduce,
    fake_impl=all_reduce_fake,
)'''

    if marker not in src:
        # Try with leading spaces
        marker = 'direct_register_custom_op(\n    op_name="all_reduce",'
        # Find the full block
        idx = src.find(marker)
        if idx == -1:
            print("ERROR: Could not find all_reduce registration in parallel_state.py")
            sys.exit(1)
        # Find the closing paren
        end_idx = src.find("\n)", idx) + 2
        marker = src[idx:end_idx]

    new_block = marker + '''


# UF17v3: Zero-copy AllReduce for piecewise CUDA graph compatibility.
# The output buffer is pre-allocated inside the graph segment (deterministic
# pool address). NCCL writes the reduced result directly into output via
# separate sendbuf/recvbuf — no temp allocation, no copy_.
def all_reduce_with_output(
    input_: torch.Tensor,
    output: torch.Tensor,
    group_name: str,
) -> None:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
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
        _dist.all_reduce(output, group=comm.device_group)


def all_reduce_with_output_fake(
    input_: torch.Tensor,
    output: torch.Tensor,
    group_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="all_reduce_with_output",
    op_func=all_reduce_with_output,
    mutates_args=["output"],
    fake_impl=all_reduce_with_output_fake,
)'''

    src = src.replace(marker, new_block)

    # 1b. Modify GroupCoordinator.all_reduce to use all_reduce_with_output
    # when VLLM_UF_EAGER_ALLREDUCE=1
    old_dispatch = '''if self.use_custom_op_call:
            return torch.ops.vllm.all_reduce(input_,
                                             group_name=self.unique_name)
        else:
            return self._all_reduce_out_place(input_)'''

    if old_dispatch not in src:
        # Try alternate formatting
        old_dispatch = 'if self.use_custom_op_call:\n            return torch.ops.vllm.all_reduce(input_,\n                                             group_name=self.unique_name)\n        else:\n            return self._all_reduce_out_place(input_)'

    if old_dispatch not in src:
        print("WARNING: Could not find all_reduce dispatch block, trying flexible match")
        # Try to find it with grep-like approach
        import re
        pattern = r'if self\.use_custom_op_call:\s+return torch\.ops\.vllm\.all_reduce\(input_,\s+group_name=self\.unique_name\)\s+else:\s+return self\._all_reduce_out_place\(input_\)'
        match = re.search(pattern, src)
        if match:
            old_dispatch = match.group(0)
        else:
            print("ERROR: Could not find all_reduce dispatch block in parallel_state.py")
            sys.exit(1)

    new_dispatch = '''if self.use_custom_op_call:
            # UF17v3: Use in-place variant for piecewise CUDA graph compat.
            # Pre-allocate output in the graph pool (deterministic address),
            # then NCCL AllReduce writes result directly via sendbuf/recvbuf.
            import os as _os
            if _os.environ.get("VLLM_UF_EAGER_ALLREDUCE", "0") == "1":
                output = torch.empty_like(input_)
                torch.ops.vllm.all_reduce_with_output(
                    input_, output, group_name=self.unique_name)
                return output
            return torch.ops.vllm.all_reduce(input_,
                                             group_name=self.unique_name)
        else:
            return self._all_reduce_out_place(input_)'''

    src = src.replace(old_dispatch, new_dispatch)

    with open(parallel_state_py, "w") as f:
        f.write(src)
    print("UF17v3: Patched parallel_state.py — all_reduce_with_output registered + dispatch added")


# ============================================================
# 2. Patch compilation.py: use all_reduce_with_output as splitting op
# ============================================================
compilation_py = f"{SITE_PACKAGES}/vllm/config/compilation.py"

with open(compilation_py) as f:
    src = f.read()

if "vllm::all_reduce_with_output" in src:
    # Update v2 comments to v3 if present
    if "UF17v2" in src:
        src = src.replace("UF17v2", "UF17v3")
        with open(compilation_py, "w") as f:
            f.write(src)
        print("UF17v3: Updated compilation.py comments v2 → v3")
    else:
        print("UF17v3: compilation.py already patched, skipping")
elif "VLLM_UF_EAGER_ALLREDUCE" in src:
    # v1 -> v3 upgrade: change splitting op name from all_reduce to all_reduce_with_output
    # Also ensure 'import os' is present (v1 image may be missing it)
    if "\nimport os\n" not in src:
        for anchor in ["\nimport copy\n", "\nimport enum\n"]:
            if anchor in src:
                src = src.replace(anchor, anchor.rstrip("\n") + "\nimport os\n", 1)
                print(f"UF17v3: Added 'import os' after '{anchor.strip()}'")
                break
    src = src.replace(
        'self.splitting_ops.append("vllm::all_reduce")',
        'self.splitting_ops.append("vllm::all_reduce_with_output")',
    )
    src = src.replace(
        "UF17: Added vllm::all_reduce to splitting_ops ",
        "UF17v3: Added vllm::all_reduce_with_output to splitting_ops ",
    )
    with open(compilation_py, "w") as f:
        f.write(src)
    print("UF17v3: Upgraded compilation.py from v1 → v3 (all_reduce_with_output)")
else:
    # Ensure 'import os' is present
    if "\nimport os\n" not in src:
        for anchor in ["\nimport copy\n", "\nimport enum\n"]:
            if anchor in src:
                src = src.replace(anchor, anchor.rstrip("\n") + "\nimport os\n", 1)
                print(f"UF17v3: Added 'import os' after '{anchor.strip()}'")
                break
        else:
            print("ERROR: Could not find import anchor for 'import os'")
            sys.exit(1)

    # Insert UF17v3 block after the unified_kv_cache_update append
    marker = 'self.splitting_ops.append("vllm::unified_kv_cache_update")'
    if marker not in src:
        print(f"ERROR: Could not find marker in compilation.py: {marker}")
        sys.exit(1)

    uf17_block = '''self.splitting_ops.append("vllm::unified_kv_cache_update")

                # UF17v3: EAGER_ALLREDUCE — run NCCL AllReduce outside
                # CUDA Graphs via piecewise split. Uses the in-place
                # all_reduce_with_output op (mutates_args=["output"])
                # with zero-copy NCCL sendbuf/recvbuf separation.
                if os.environ.get("VLLM_UF_EAGER_ALLREDUCE", "0") == "1":
                    self.splitting_ops.append("vllm::all_reduce_with_output")
                    logger.info(
                        "UF17v3: Added vllm::all_reduce_with_output to "
                        "splitting_ops (eager NCCL AllReduce between "
                        "CUDA Graph segments, zero-copy)"
                    )'''

    src = src.replace(marker, uf17_block)

    with open(compilation_py, "w") as f:
        f.write(src)
    print("UF17v3: Patched compilation.py — EAGER_ALLREDUCE with zero-copy op")

print("UF17v3: Done")
