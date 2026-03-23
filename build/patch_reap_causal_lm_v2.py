#!/usr/bin/env python3
"""Patch: Extend Qwen3_5MoeForCausalLM with IsHybrid + REAP fused expert support.

This is the v2 patch that:
1. Adds IsHybrid mixin (required for GatedDeltaNet/Mamba state management)
2. Adds get_mamba_state_* classmethods (copied from Qwen3_5ForConditionalGeneration)
3. Adds AutoWeightsLoader + _reap_transform_weights (handles fused gate_up_proj)
"""
import sys

QWEN35_PY = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"

with open(QWEN35_PY) as f:
    content = f.read()

if "REAP_V2_PATCH" in content:
    print("patch_reap_causal_lm_v2: already applied")
    sys.exit(0)

old = '''class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()'''

new = '''class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts, IsHybrid):
    # REAP_V2_PATCH: Extended with IsHybrid + REAP fused expert loading
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    # --- IsHybrid: Mamba/GatedDeltaNet state management ---
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return torch.float32

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        config = vllm_config.model_config.hf_config
        mamba_d_ssm = getattr(config, "mamba_d_ssm", 0)
        if mamba_d_ssm == 0:
            return {}
        return {
            "mamba_state": (1, mamba_d_ssm, config.mamba_d_state),
            "mamba_conv_state": (1, mamba_d_ssm + 2 * config.mamba_d_inner_s, config.mamba_d_conv),
        }

    @classmethod
    def get_mamba_state_copy_func(cls, vllm_config):
        import functools
        def copy_func(dst, src, offset, size):
            dst[offset:offset+size] = src[:size]
        return functools.partial(copy_func)

    # --- REAP: Transform fused expert keys for AutoWeightsLoader ---
    def _reap_transform_weights(self, weights):
        """Transform REAP fused expert keys to separate per-expert format.

        REAP: experts.gate_up_proj.{id}.{suffix} -> experts.{id}.gate_proj.{suffix} + experts.{id}.up_proj.{suffix}
        REAP: experts.down_proj.{id}.{suffix} -> experts.{id}.down_proj.{suffix}
        """
        import re as _re
        for name, tensor in weights:
            match = _re.search(r'experts\\.(gate_up_proj|down_proj)\\.(\\d+)\\.', name)
            if match and 'mlp.experts' in name:
                proj_type = match.group(1)
                expert_id = match.group(2)
                suffix = name[match.end():]
                base = name[:match.start()]
                if proj_type == 'gate_up_proj':
                    chunks = tensor.chunk(2, dim=-1)
                    yield (f'{base}experts.{expert_id}.gate_proj.{suffix}', chunks[0])
                    yield (f'{base}experts.{expert_id}.up_proj.{suffix}', chunks[1])
                else:
                    yield (f'{base}experts.{expert_id}.down_proj.{suffix}', tensor)
            else:
                yield (name, tensor)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(self._reap_transform_weights(weights))'''

if old not in content:
    print("ERROR: Qwen3_5MoeForCausalLM pattern not found in qwen3_5.py")
    print("Checking what's there...")
    import re
    m = re.search(r'class Qwen3_5MoeForCausalLM.*?(?=\n########|\nclass )', content, re.DOTALL)
    if m:
        print(f"Found at offset {m.start()}: {repr(m.group()[:200])}")
    sys.exit(1)

content = content.replace(old, new)
with open(QWEN35_PY, "w") as f:
    f.write(content)

print("patch_reap_causal_lm_v2: applied (IsHybrid + REAP transform + AutoWeightsLoader)")
