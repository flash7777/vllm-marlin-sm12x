#!/usr/bin/env python3
"""
Patch UF17v2: EAGER_ALLREDUCE — run NCCL AllReduce outside CUDA Graphs.

V2 fix: Uses all_reduce_with_output (in-place, mutates_args=["output"])
instead of all_reduce (out-of-place). This ensures the AllReduce result
lands at a deterministic buffer address that the next CUDA graph segment
can read from during replay.

Root cause of v1 bug: The out-of-place all_reduce allocates a NEW tensor
on every call. During CUDA graph capture, the next segment records this
address. On replay, eager AllReduce allocates at a DIFFERENT address,
but the next segment still reads from the captured (stale) address.

Fix: Pre-allocate the output buffer inside the graph segment (deterministic
pool address), then pass it to all_reduce_with_output which writes in-place.

Result: Correct output + eager AllReduce performance.
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
    print("UF17v2: parallel_state.py already patched, skipping")
else:
    # 1a. Add the all_reduce_with_output op function + fake impl + registration
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


# UF17v2: In-place AllReduce for piecewise CUDA graph compatibility.
# The output buffer is pre-allocated inside the graph segment (deterministic
# pool address), so the next graph segment can read from it during replay.
def all_reduce_with_output(
    input_: torch.Tensor,
    output: torch.Tensor,
    group_name: str,
) -> None:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    result = group._all_reduce_out_place(input_)
    output.copy_(result)


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
            # UF17v2: Use in-place variant for piecewise CUDA graph compat.
            # Pre-allocate output in the graph pool (deterministic address),
            # then AllReduce writes result in-place via mutates_args.
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
    print("UF17v2: Patched parallel_state.py — all_reduce_with_output registered + dispatch added")


# ============================================================
# 2. Patch compilation.py: use all_reduce_with_output as splitting op
# ============================================================
compilation_py = f"{SITE_PACKAGES}/vllm/config/compilation.py"

with open(compilation_py) as f:
    src = f.read()

if "vllm::all_reduce_with_output" in src:
    print("UF17v2: compilation.py already patched (v2), skipping")
elif "VLLM_UF_EAGER_ALLREDUCE" in src:
    # v1 -> v2 upgrade: change splitting op name from all_reduce to all_reduce_with_output
    # Also ensure 'import os' is present (v1 image may be missing it)
    if "\nimport os\n" not in src:
        for anchor in ["\nimport copy\n", "\nimport enum\n"]:
            if anchor in src:
                src = src.replace(anchor, anchor.rstrip("\n") + "\nimport os\n", 1)
                print(f"UF17v2: Added 'import os' after '{anchor.strip()}'")
                break
    src = src.replace(
        'self.splitting_ops.append("vllm::all_reduce")',
        'self.splitting_ops.append("vllm::all_reduce_with_output")',
    )
    src = src.replace(
        "UF17: Added vllm::all_reduce to splitting_ops ",
        "UF17v2: Added vllm::all_reduce_with_output to splitting_ops ",
    )
    with open(compilation_py, "w") as f:
        f.write(src)
    print("UF17v2: Upgraded compilation.py from v1 → v2 (all_reduce_with_output)")
else:
    # Ensure 'import os' is present
    if "\nimport os\n" not in src:
        for anchor in ["\nimport copy\n", "\nimport enum\n"]:
            if anchor in src:
                src = src.replace(anchor, anchor.rstrip("\n") + "\nimport os\n", 1)
                print(f"UF17v2: Added 'import os' after '{anchor.strip()}'")
                break
        else:
            print("ERROR: Could not find import anchor for 'import os'")
            sys.exit(1)

    # Insert UF17v2 block after the unified_kv_cache_update append
    marker = 'self.splitting_ops.append("vllm::unified_kv_cache_update")'
    if marker not in src:
        print(f"ERROR: Could not find marker in compilation.py: {marker}")
        sys.exit(1)

    uf17_block = '''self.splitting_ops.append("vllm::unified_kv_cache_update")

                # UF17v2: EAGER_ALLREDUCE — run NCCL AllReduce outside
                # CUDA Graphs via piecewise split. Uses the in-place
                # all_reduce_with_output op (mutates_args=["output"])
                # so the output buffer has a deterministic pool address
                # that the next graph segment can read during replay.
                if os.environ.get("VLLM_UF_EAGER_ALLREDUCE", "0") == "1":
                    self.splitting_ops.append("vllm::all_reduce_with_output")
                    logger.info(
                        "UF17v2: Added vllm::all_reduce_with_output to "
                        "splitting_ops (eager NCCL AllReduce between "
                        "CUDA Graph segments, in-place buffer)"
                    )'''

    src = src.replace(marker, uf17_block)

    with open(compilation_py, "w") as f:
        f.write(src)
    print("UF17v2: Patched compilation.py — EAGER_ALLREDUCE with in-place op")

print("UF17v2: Done")
