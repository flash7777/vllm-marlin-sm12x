#!/usr/bin/env python3
"""Patch qwen3_next.py + qwen3_5.py: Fix torch.compile AOT autograd crash.

Problem: torch.Size objects crossing AOT autograd split boundaries cause
  AssertionError: out_spec != out_desc_spec
  TreeSpec(Size, ...) vs TreeSpec(tuple, ...)

Fix: Replace `.shape` captures with explicit int extraction so torch.Size
never appears as a graph output.

Affected patterns:
  1. qwen3_next.py SparseMoeBlock: orig_shape = hidden_states.shape
  2. qwen3_next.py GatedDeltaNet: z_shape_og = z.shape
  3. qwen3_5.py GatedDeltaNet: z_shape_og = z.shape
  4. qwen3_next.py Attention: orig_shape = q_gate.shape[:-1]
"""

import sys

FILES = {
    "qwen3_next": "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py",
    "qwen3_5": "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py",
}

patches_applied = 0

# =============================================================
# 1. qwen3_next.py: SparseMoeBlock.forward — orig_shape
# =============================================================
with open(FILES["qwen3_next"], "r") as f:
    code = f.read()

old_moe = """\
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)"""

new_moe = """\
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)"""

old_moe_return = """\
        return final_hidden_states.view(orig_shape)"""

new_moe_return = """\
        return final_hidden_states.view(num_tokens, hidden_dim)"""

if old_moe in code and old_moe_return in code:
    code = code.replace(old_moe, new_moe)
    code = code.replace(old_moe_return, new_moe_return)
    patches_applied += 1
    print("[patch_compile] 1/4 qwen3_next SparseMoeBlock: orig_shape → (num_tokens, hidden_dim)")
elif "orig_shape = hidden_states.shape" not in code:
    print("[patch_compile] 1/4 qwen3_next SparseMoeBlock: already patched, skipping")
else:
    print("[patch_compile] 1/4 ERROR: SparseMoeBlock pattern mismatch!")
    sys.exit(1)

# =============================================================
# 2. qwen3_next.py: GatedDeltaNet — z_shape_og
# =============================================================
old_gdn = """\
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)"""

new_gdn = """\
        z_num, z_heads, z_dim = z.size(0), z.size(1), z.size(2)
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.size(-1))
        z = z.reshape(-1, z.size(-1))
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_num, z_heads, z_dim)"""

if old_gdn in code:
    code = code.replace(old_gdn, new_gdn)
    patches_applied += 1
    print("[patch_compile] 2/4 qwen3_next GatedDeltaNet: z_shape_og → (z_num, z_heads, z_dim)")
elif "z_shape_og" not in code:
    print("[patch_compile] 2/4 qwen3_next GatedDeltaNet: already patched, skipping")
else:
    print("[patch_compile] 2/4 ERROR: GatedDeltaNet pattern mismatch!")
    sys.exit(1)

# =============================================================
# 4. qwen3_next.py: Attention — orig_shape = q_gate.shape[:-1]
# =============================================================
old_attn = """\
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)"""

new_attn = """\
            qg_dim0 = q_gate.size(0)
            q_gate = q_gate.view(qg_dim0, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(qg_dim0, -1)
            gate = gate.reshape(qg_dim0, -1)"""

if old_attn in code:
    code = code.replace(old_attn, new_attn)
    patches_applied += 1
    print("[patch_compile] 3/4 qwen3_next Attention: orig_shape → qg_dim0")
elif "orig_shape = q_gate.shape" not in code:
    print("[patch_compile] 3/4 qwen3_next Attention: already patched, skipping")
else:
    print("[patch_compile] 3/4 ERROR: Attention pattern mismatch!")
    sys.exit(1)

with open(FILES["qwen3_next"], "w") as f:
    f.write(code)

# =============================================================
# 3. qwen3_5.py: GatedDeltaNet — z_shape_og (same pattern)
# =============================================================
with open(FILES["qwen3_5"], "r") as f:
    code5 = f.read()

# Same old/new patterns as qwen3_next
if old_gdn in code5:
    code5 = code5.replace(old_gdn, new_gdn)
    patches_applied += 1
    print("[patch_compile] 4/4 qwen3_5 GatedDeltaNet: z_shape_og → (z_num, z_heads, z_dim)")
elif "z_shape_og" not in code5:
    print("[patch_compile] 4/4 qwen3_5 GatedDeltaNet: already patched, skipping")
else:
    print("[patch_compile] 4/4 ERROR: qwen3_5 GatedDeltaNet pattern mismatch!")
    sys.exit(1)

with open(FILES["qwen3_5"], "w") as f:
    f.write(code5)

print(f"\n[patch_compile] Done. {patches_applied} patches applied.")
