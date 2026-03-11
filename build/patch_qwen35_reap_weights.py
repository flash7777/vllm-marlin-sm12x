#!/usr/bin/env python3
"""
Patch qwen3_5.py load_weights for AutoRound GPTQ models (e.g. REAP-262B).

Per-expert fused GPTQ weight loading:
   REAP-262B stores experts as experts.gate_up_proj.N.qweight (per-expert, fused)
   but vLLM expects stacked format. Fix: extract expert_id, route to weight_loader.

Note: AutoRound auto_gptq packing stores qzeros=0x77777777 (=zp-1=7), but the
actual zero_point IS 8. Marlin uint4b8 (bias=8) is correct. No zp correction needed.
"""

import re
import sys

SITE = "/usr/local/lib/python3.12/dist-packages"
TARGET = f"{SITE}/vllm/model_executor/models/qwen3_5.py"

# Read original
with open(TARGET) as f:
    src = f.read()

# Check if already patched
if "per_expert_gptq" in src or "REAP" in src:
    print("patch_qwen35_reap_weights: already applied")
    sys.exit(0)

# The original fused expert handling block (inside the expert_params_mapping loop):
OLD = '''\
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )'''

NEW = '''\
                    if is_fused_expert:
                        # Check if this is per-expert GPTQ (e.g. REAP models):
                        # Pattern: experts.gate_up_proj.N.qweight (per_expert_gptq)
                        per_expert_match = re.search(
                            r'experts\.(gate_up_proj|down_proj)\.(\d+)\.',
                            name
                        )
                        if per_expert_match:
                            # Per-expert GPTQ: extract expert_id, fix param name
                            expert_id = int(per_expert_match.group(2))
                            proj_type = per_expert_match.group(1)
                            # Map GPTQ per-expert name to vLLM param name
                            # e.g. "experts.w13_weight.0.qweight" -> "experts.w13_qweight"
                            # e.g. "experts.w2_weight.0.scales"  -> "experts.w2_scales"
                            # e.g. "experts.w13_weight.0.qzeros" -> "experts.w13_qzeros"
                            base_name = re.sub(
                                r'(experts\.w\d+)_weight\.\d+\.(\w+)',
                                r'\\1_\\2',
                                name_mapped
                            )
                            if is_pp_missing_parameter(base_name, self):
                                continue
                            if base_name not in params_dict:
                                continue
                            param = params_dict[base_name]
                            weight_loader = param.weight_loader
                            if proj_type == "gate_up_proj":
                                # gate_up_proj is fused: chunk into gate (w1) and up (w3)
                                # GPTQ layout: qweight [K/8, N], scales [groups, N], qzeros [groups, N/8]
                                # gate+up are concatenated along N (output dim = last dim)
                                chunks = loaded_weight.chunk(2, dim=-1)
                                success_w1 = weight_loader(
                                    param, chunks[0], base_name,
                                    shard_id="w1", expert_id=expert_id,
                                    return_success=True,
                                )
                                success_w3 = weight_loader(
                                    param, chunks[1], base_name,
                                    shard_id="w3", expert_id=expert_id,
                                    return_success=True,
                                )
                                success = (success_w1 or False) and (success_w3 or False)
                            else:
                                # down_proj
                                success = weight_loader(
                                    param, loaded_weight, base_name,
                                    shard_id="w2", expert_id=expert_id,
                                    return_success=True,
                                ) or False
                        else:
                            # Original stacked fused path (all experts in one tensor)
                            if "experts.gate_up_proj" in name:
                                loaded_weight = loaded_weight.chunk(2, dim=-2)
                                success_w1 = self.load_fused_expert_weights(
                                    name_mapped,
                                    params_dict,
                                    loaded_weight[0],
                                    "w1",
                                    num_experts,
                                )
                                success_w3 = self.load_fused_expert_weights(
                                    name_mapped,
                                    params_dict,
                                    loaded_weight[1],
                                    "w3",
                                    num_experts,
                                )
                                success = success_w1 and success_w3
                            else:
                                # down_proj
                                success = self.load_fused_expert_weights(
                                    name_mapped,
                                    params_dict,
                                    loaded_weight,
                                    shard_id,
                                    num_experts,
                                )'''

if OLD not in src:
    print("ERROR: Could not find target code block in qwen3_5.py")
    print("The code may have changed. Manual review needed.")
    sys.exit(1)

src = src.replace(OLD, NEW)

# Add 'import re' if not already imported
if '\nimport re\n' not in src:
    src = src.replace('\nimport typing\n', '\nimport re\nimport typing\n')

# --- Patch 2: AutoRound zero-point correction ---
# Add helper function and wrap the weights iterator in load_weights
# to correct qweight nibbles from zp=7 (AutoRound) to zp=8 (Marlin uint4b8)


with open(TARGET, 'w') as f:
    f.write(src)

print("patch_qwen35_reap_weights: applied successfully")
print("  - Per-expert GPTQ fused weight loading for REAP-style models")
