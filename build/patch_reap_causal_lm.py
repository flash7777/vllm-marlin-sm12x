#!/usr/bin/env python3
"""Patch: Add REAP fused per-expert support to Qwen3_5MoeForCausalLM.

REAP models have fused expert keys:
  experts.gate_up_proj.{expert_id}.{qweight|scales|qzeros}
  experts.down_proj.{expert_id}.{qweight|scales|qzeros}

Standard vLLM expects:
  experts.{expert_id}.gate_proj.{qweight|scales|qzeros}
  experts.{expert_id}.up_proj.{qweight|scales|qzeros}
  experts.{expert_id}.down_proj.{qweight|scales|qzeros}

This patch overrides load_weights on Qwen3_5MoeForCausalLM to handle
the fused format by chunking gate_up_proj along dim=-1 into w1+w3.
"""
import sys

QWEN35_PY = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"

with open(QWEN35_PY) as f:
    content = f.read()

if "REAP_FUSED_EXPERT" in content:
    print("patch_reap_causal_lm: already applied")
    sys.exit(0)

old = '''class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()'''

new = '''class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    # REAP_FUSED_EXPERT: Override load_weights to handle fused per-expert format
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        import re
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("mtp."):
                continue

            # REAP fused per-expert: experts.gate_up_proj.N.{suffix}
            reap_match = re.search(
                r"experts\.(gate_up_proj|down_proj)\.(\d+)\.", name
            )
            if reap_match and "mlp.experts" in name:
                proj_type = reap_match.group(1)
                expert_id = int(reap_match.group(2))
                suffix = name[reap_match.end():]  # qweight, scales, qzeros, g_idx

                # Map to vLLM param names
                layer_prefix = name[:name.index("mlp.experts")] + "mlp.experts."
                if proj_type == "gate_up_proj":
                    param_name_base = layer_prefix + "w13_" + suffix
                    if is_pp_missing_parameter(param_name_base, self):
                        continue
                    if param_name_base not in params_dict:
                        continue
                    param = params_dict[param_name_base]
                    weight_loader = param.weight_loader
                    # Chunk gate_up_proj into w1 (gate) and w3 (up)
                    chunks = loaded_weight.chunk(2, dim=-1)
                    weight_loader(
                        param, chunks[0], param_name_base,
                        shard_id="w1", expert_id=expert_id,
                    )
                    weight_loader(
                        param, chunks[1], param_name_base,
                        shard_id="w3", expert_id=expert_id,
                    )
                else:  # down_proj
                    param_name_base = layer_prefix + "w2_" + suffix
                    if is_pp_missing_parameter(param_name_base, self):
                        continue
                    if param_name_base not in params_dict:
                        continue
                    param = params_dict[param_name_base]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param, loaded_weight, param_name_base,
                        shard_id="w2", expert_id=expert_id,
                    )
                loaded_params.add(name)
                continue

            # Standard stacked params (qkv_proj, shared_expert gate_up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Standard expert params (separate per-expert format)
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param, loaded_weight, name,
                        shard_id=shard_id, expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params'''

if old not in content:
    print("ERROR: Qwen3_5MoeForCausalLM pattern not found")
    sys.exit(1)

content = content.replace(old, new)
with open(QWEN35_PY, "w") as f:
    f.write(content)

print("patch_reap_causal_lm: applied (REAP fused expert support in CausalLM)")
