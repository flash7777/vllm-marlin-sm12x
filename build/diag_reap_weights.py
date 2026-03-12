#!/usr/bin/env python3
"""
Diagnostic: Load REAP-262B on TP=1 and inspect loaded expert weights.
Compare checkpoint values with what ends up in vLLM buffers after Marlin repack.
"""
import os
import sys
import torch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from safetensors import safe_open
from vllm import LLM

MODEL = "/data/tensordata/Qwen3.5-REAP-262B-A17B-int4-AutoRound"

print("=== Loading model (TP=1, enforce_eager) ===")
llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    enforce_eager=True,
    gpu_memory_utilization=0.05,
    kv_cache_memory_bytes=1 * 1024**3,  # 1GB
    max_model_len=256,
    trust_remote_code=True,
    disable_log_stats=True,
)

print("\n=== Inspecting loaded weights ===")
model = llm.llm_engine.model_executor.driver_worker.model_runner.model

# Get layer 5 MoE (not layer 0 which has mostly zeros)
layer = model.language_model.layers[5]
experts = layer.mlp.experts

print(f"\nLayer 5 expert params:")
for name, param in experts.named_parameters():
    if 'w13' in name or 'w2' in name:
        data = param.data
        print(f"  {name}: shape={list(data.shape)}, dtype={data.dtype}, "
              f"min={data.min().item():.6f}, max={data.max().item():.6f}, "
              f"nonzero={data.count_nonzero().item()}/{data.numel()}")

# Quick generation test
print("\n=== Generation test ===")
from vllm import SamplingParams
sp = SamplingParams(max_tokens=30, temperature=0)
outputs = llm.generate(["Hello! What is 2+2?"], sp)
for o in outputs:
    print(f"Output: {repr(o.outputs[0].text[:200])}")
    print(f"Token IDs: {list(o.outputs[0].token_ids[:15])}")

# Now compare with checkpoint for one expert
print("\n=== Checkpoint comparison (layer 5, expert 3, scales) ===")
import glob
files = sorted(glob.glob(os.path.join(MODEL, "*.safetensors")))
target_key = "model.language_model.layers.5.mlp.experts.gate_up_proj.3.scales"

for sf_file in files:
    with safe_open(sf_file, framework="pt", device="cpu") as f:
        if target_key in f.keys():
            ckpt_scales = f.get_tensor(target_key)
            print(f"Checkpoint {target_key}: shape={list(ckpt_scales.shape)}")
            print(f"  Full: min={ckpt_scales.min():.6f}, max={ckpt_scales.max():.6f}")
            # gate (w1) half
            w1_scales = ckpt_scales[:, :1024]
            w3_scales = ckpt_scales[:, 1024:]
            print(f"  w1 (gate) half [:, :1024]: min={w1_scales.min():.6f}, max={w1_scales.max():.6f}")
            print(f"  w3 (up) half [:, 1024:]: min={w3_scales.min():.6f}, max={w3_scales.max():.6f}")

            # Now check what's in the vLLM buffer for this expert
            w13_scales = experts.w13_scales.data[3]  # expert 3
            print(f"\nvLLM w13_scales[expert=3]: shape={list(w13_scales.shape)}")
            # w13 is [2*intermediate, groups] or similar after Marlin repack
            # Just show stats
            print(f"  min={w13_scales.min():.6f}, max={w13_scales.max():.6f}")
            print(f"  nonzero={w13_scales.count_nonzero().item()}/{w13_scales.numel()}")
            break

print("\nDone.")
