#!/usr/bin/env python3
"""Patch: Fix torch.compile for Qwen3-Next (torch.Size across split boundaries).

Problem: In Qwen3NextSparseMoeBlock.forward(), `orig_shape = hidden_states.shape`
creates a `torch.Size` object. This Size crosses a compile split boundary at
`vllm::all_reduce_with_output`, causing AOT autograd to fail with:
  AssertionError: out_spec != out_desc_spec

Similarly in Qwen3NextGatedDeltaNet.forward(), `z_shape_og = z.shape` could cross
the `vllm::gdn_attention_core` boundary.

Fix: Replace `.shape` with individual `.size()` calls (SymInts) or static dimensions
that don't create torch.Size objects crossing boundaries.
"""

import os
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"


def apply():
    if not os.path.exists(TARGET):
        print(f"SKIP: {TARGET} not found")
        sys.exit(0)

    with open(TARGET) as f:
        content = f.read()

    if "compile fix v2" in content:
        print("SKIP: v2 patch already applied")
        return

    changes = 0

    # Fix 1: MoE block - orig_shape crosses all_reduce split
    old_moe = """\
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)"""

    new_moe = """\
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        # compile fix v2: use .size() instead of .shape to avoid torch.Size
        # crossing the all_reduce split boundary in AOT autograd
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)
        hidden_states = hidden_states.view(-1, hidden_dim)"""

    if old_moe in content:
        content = content.replace(old_moe, new_moe)
        changes += 1
        print("  [1] Fixed MoE orig_shape")
    else:
        print("  [1] MoE block: pattern not found (may be already patched)")

    # Also fix the return: view(orig_shape) -> view(num_tokens, hidden_dim)
    old_return = "        return final_hidden_states.view(orig_shape)"
    new_return = "        return final_hidden_states.view(num_tokens, hidden_dim)"
    if old_return in content:
        content = content.replace(old_return, new_return)
        changes += 1
        print("  [2] Fixed MoE return view")
    else:
        print("  [2] MoE return: pattern not found (may be already patched)")

    if changes == 0:
        print("WARNING: No changes made")
        sys.exit(1)

    with open(TARGET, "w") as f:
        f.write(content)

    print(f"OK: {changes} torch.compile fixes applied to {TARGET}")


if __name__ == "__main__":
    apply()
