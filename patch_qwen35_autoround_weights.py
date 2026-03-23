#!/usr/bin/env python3
"""Patch vLLM qwen3_5.py to handle AutoRound per-expert fused GPTQ weight format.

AutoRound quantizes Qwen3.5 MoE models with per-expert fused keys:
    experts.gate_up_proj.{expert_id}.{qweight|qzeros|scales}
    experts.down_proj.{expert_id}.{qweight|qzeros|scales}
  with model.language_model. prefix (VL architecture).

This patch adds a new code block in load_weights() that:
  1. Detects per-expert fused keys (experts.{proj}.{id}.{suffix})
  2. Strips model.language_model. prefix
  3. For gate_up_proj: splits on dim -1 (output dim) into gate + up,
     calls weight_loader with shard_id w1/w3 directly
  4. For down_proj: calls weight_loader with shard_id w2 directly

Usage: python3 patch_qwen35_autoround_weights.py
"""

QWEN35_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"

with open(QWEN35_PATH, "r") as f:
    content = f.read()

# Idempotency check
if "_per_expert_fused_pat" in content:
    print("patch_qwen35_autoround_weights: already patched, skipping")
    exit(0)

original = content

# 1. Add 'import re' after 'import typing' if not already present
if "import re\n" not in content:
    content = content.replace("import typing\n", "import typing\nimport re\n", 1)

# 2. Insert the per-expert fused detection block inside load_weights()
#    Right after "for name, loaded_weight in weights:" and the rotary/mtp/scale skips,
#    but BEFORE the stacked_params_mapping loop that detects fused experts.
#
#    We insert before: "for param_name, weight_name, shard_id in stacked_params_mapping:"

OLD_STACKED_LOOP = """\
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping"""

NEW_BLOCK = """\
            # --- AutoRound per-expert fused GPTQ support ---
            # Detect: experts.gate_up_proj.{id}.suffix / experts.down_proj.{id}.suffix
            if not hasattr(self, '_per_expert_fused_pat'):
                self._per_expert_fused_pat = re.compile(
                    r"(.*\\.mlp\\.experts)\\.(gate_up_proj|down_proj)\\.(\\d+)\\.(.*)"
                )
            # Strip VL prefix (model.language_model. -> model.)
            if "model.language_model." in name:
                name = name.replace("model.language_model.", "model.")
            _pef_m = self._per_expert_fused_pat.match(name)
            if _pef_m:
                _pef_pfx, _pef_proj, _pef_eid_str, _pef_sfx = _pef_m.groups()
                _pef_expert_id = int(_pef_eid_str)
                if _pef_proj == "gate_up_proj":
                    _pef_param_name = f"{_pef_pfx}.w13_weight.{_pef_sfx}"
                    if is_pp_missing_parameter(_pef_param_name, self):
                        continue
                    if _pef_param_name not in params_dict:
                        continue
                    # Split output dim: qweight=[in/pack, 2*inter], scales=[g, 2*inter], qzeros=[g, 2*inter/pack]
                    _pef_half = loaded_weight.shape[-1] // 2
                    _pef_gate = loaded_weight.narrow(-1, 0, _pef_half).contiguous()
                    _pef_up = loaded_weight.narrow(-1, _pef_half, _pef_half).contiguous()
                    _pef_param = params_dict[_pef_param_name]
                    _pef_wl = _pef_param.weight_loader
                    _pef_wl(_pef_param, _pef_gate, _pef_param_name, "w1", _pef_expert_id)
                    _pef_wl(_pef_param, _pef_up, _pef_param_name, "w3", _pef_expert_id)
                    loaded_params.add(_pef_param_name)
                else:  # down_proj
                    _pef_param_name = f"{_pef_pfx}.w2_weight.{_pef_sfx}"
                    if is_pp_missing_parameter(_pef_param_name, self):
                        continue
                    if _pef_param_name not in params_dict:
                        continue
                    _pef_param = params_dict[_pef_param_name]
                    _pef_wl = _pef_param.weight_loader
                    _pef_wl(_pef_param, loaded_weight, _pef_param_name, "w2", _pef_expert_id)
                    loaded_params.add(_pef_param_name)
                continue
            # --- End AutoRound per-expert fused ---

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping"""

content = content.replace(OLD_STACKED_LOOP, NEW_BLOCK, 1)

if content == original:
    print("patch_qwen35_autoround_weights: pattern not found, no changes made")
    exit(1)
else:
    with open(QWEN35_PATH, "w") as f:
        f.write(content)
    print("patch_qwen35_autoround_weights: patched qwen3_5.py")
    print("  - model.language_model. prefix stripping")
    print("  - per-expert fused gate_up_proj split (dim -1) -> w1/w3")
    print("  - per-expert fused down_proj -> w2")
    print("  - direct weight_loader call (bypasses stacked/unfused paths)")
