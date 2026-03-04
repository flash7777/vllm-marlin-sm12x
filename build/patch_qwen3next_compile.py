#!/usr/bin/env python3
"""Patch: Fix torch.compile for Qwen3-Next (GatedDeltaNet z.shape issue).

Problem: Qwen3NextGatedDeltaNet.forward() saves `z_shape_og = z.shape` which
creates a `torch.Size` object that leaks into the AOT autograd output spec when
the graph is split at `vllm::gdn_attention_core`. AOT autograd expects all outputs
to be tensors/SymInts, causing:
  AssertionError: out_spec != out_desc_spec (TreeSpec has Size instead of plain output)

Fix: Replace `z_shape_og = z.shape` with `num_tokens` (already available) and
known static dimensions (num_v_heads, head_v_dim). This avoids the torch.Size
object entirely.
"""

import os
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"

OLD = """\
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)"""

NEW = """\
        # torch.compile fix: avoid z.shape (torch.Size leaks into AOT graph)
        # Use num_tokens + static dims instead
        _nv = self.num_v_heads // self.tp_size
        _hd = self.head_v_dim
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.size(-1))
        z = z.reshape(-1, z.size(-1))
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(num_tokens, _nv, _hd)"""


def main():
    if not os.path.exists(TARGET):
        print(f"SKIP: {TARGET} not found")
        sys.exit(0)

    with open(TARGET) as f:
        content = f.read()

    if "torch.compile fix" in content:
        print("SKIP: patch already applied")
        return

    if OLD not in content:
        print(f"ERROR: Could not find target code block in {TARGET}")
        print("Expected:")
        print(OLD)
        sys.exit(1)

    content = content.replace(OLD, NEW)

    with open(TARGET, "w") as f:
        f.write(content)

    print(f"OK: torch.compile fix applied to {TARGET}")
    print("   Replaced z_shape_og = z.shape with static dimension reshape")


if __name__ == "__main__":
    main()
