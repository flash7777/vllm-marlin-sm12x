#!/usr/bin/env python3
"""
Patch qwen3_5.py load_weights to handle per-expert GPTQ fused weights.

Problem: REAP-262B (and similar AutoRound MoE models) stores experts as:
  experts.gate_up_proj.N.qweight   (per-expert, GPTQ quantized, gate+up fused)
  experts.gate_up_proj.N.qzeros
  experts.gate_up_proj.N.scales
  experts.down_proj.N.qweight
  experts.down_proj.N.qzeros
  experts.down_proj.N.scales

But vLLM's qwen3_5.py load_fused_expert_weights() expects ALL experts stacked:
  experts.gate_up_proj  -> tensor [num_experts, ...]

The stacked loading path does params_dict[name] where name still contains ".N.qweight",
causing KeyError: 'layers.0.mlp.experts.w2_weight.0.qweight'

Fix: Detect per-expert GPTQ pattern, extract expert_id, strip ".N" from param name,
and use the standard FusedMoE weight_loader with expert_id parameter.
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
                            # Strip ".N" from mapped name to get base param
                            # e.g. "experts.w13_weight.0.qweight" -> "experts.w13_weight.qweight"
                            base_name = re.sub(
                                r'(experts\.w\d+_weight)\.\d+\.',
                                r'\1.',
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

with open(TARGET, 'w') as f:
    f.write(src)

print("patch_qwen35_reap_weights: applied successfully")
print("  - Per-expert GPTQ fused weight loading for REAP-style models")
