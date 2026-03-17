#!/usr/bin/env python3
"""RIY: Mask-aware AutoRound quantization.

Loads the full 397B model, zeros out masked experts (from RIY profile),
then runs standard AutoRound. The calibration sees the masked routing,
so scales/zeros are optimized for the actual inference pattern.

Usage:
    python3 riy_quantize.py \
        --model-path /data/tensordata/Qwen3.5-397B-A17B \
        --riy-profile /data/tensordata/riy_reap262b.json \
        --output-dir /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound-Masked
"""

import argparse
import json
import logging
import re
import torch

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("riy_quantize")


def apply_riy_mask(model, profile_path):
    """Zero out weights of pruned experts in the model."""
    with open(profile_path) as f:
        profile = json.load(f)

    pruned_set = set(tuple(x) for x in profile["pruned_experts"])
    logger.info(f"RIY profile: {len(pruned_set)} experts to mask ({profile['workload']})")

    # Find expert modules and zero them out
    zeroed = 0
    for name, param in model.named_parameters():
        # Match patterns like:
        #   model.layers.X.mlp.experts.gate_up_proj  [num_experts, ...]
        #   model.layers.X.mlp.experts.down_proj     [num_experts, ...]
        m = re.match(r"(.*)layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)", name)
        if not m:
            continue

        layer = int(m.group(2))
        # Zero out the pruned expert slices in this stacked tensor
        for expert_id in range(param.shape[0]):
            if (layer, expert_id) in pruned_set:
                param.data[expert_id].zero_()
                zeroed += 1

    logger.info(f"RIY: Zeroed {zeroed} expert slices (2 tensors × {len(pruned_set)} experts)")
    return pruned_set


def build_layer_config(model, pruned_set):
    """Build layer_config: shared experts BF16, detect correct prefix."""
    layer_config = {}

    for n, m in model.named_modules():
        if "layers.0.mlp.shared_expert_gate" in n:
            prefix = n.split("layers.0.mlp.shared_expert_gate")[0]
            logger.info(f"Detected layer prefix: '{prefix}'")

            num_layers = sum(1 for nn, _ in model.named_modules()
                           if nn.endswith(".mlp.shared_expert_gate"))
            logger.info(f"Detected {num_layers} layers")

            for i in range(num_layers):
                layer_config[f"{prefix}layers.{i}.mlp.shared_expert_gate"] = {
                    "bits": 16, "data_type": "float"
                }
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    layer_config[f"{prefix}layers.{i}.mlp.shared_expert.{proj}"] = {
                        "bits": 16, "data_type": "float"
                    }
            break

    logger.info(f"layer_config: {len(layer_config)} entries (shared experts BF16)")
    return layer_config


def main():
    parser = argparse.ArgumentParser(description="RIY: Mask-aware AutoRound")
    parser.add_argument("--model-path", required=True, help="Path to full BF16 model (397B)")
    parser.add_argument("--riy-profile", required=True, help="RIY profile JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action="store_true", default=True)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulate-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--minmax-lr", type=float, default=0.005)
    parser.add_argument("--low-gpu-mem-usage", action="store_true", default=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from auto_round import AutoRound

    # Load model
    logger.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        processor = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Apply RIY mask — zero out pruned experts
    pruned_set = apply_riy_mask(model, args.riy_profile)

    # Build layer_config for shared experts BF16
    layer_config = build_layer_config(model, pruned_set)

    # Run AutoRound
    logger.info("Starting mask-aware AutoRound quantization...")
    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        iters=args.iters,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        lr=args.lr,
        minmax_lr=args.minmax_lr,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        layer_config=layer_config,
        processor=processor,
    )

    model, quant_config = autoround.quantize()

    # Save with RIY profile metadata
    logger.info(f"Saving to {args.output_dir}")
    autoround.save_quantized(args.output_dir, format="auto_round")

    # Copy RIY profile into output dir
    import shutil
    shutil.copy2(args.riy_profile, f"{args.output_dir}/riy_profile.json")
    logger.info("Done!")


if __name__ == "__main__":
    main()
