#!/usr/bin/env python3
"""FMAAQ: Full-Model-Assisted AutoRound Quantization.

Quantizes REAP-262B with the unpruned 397B as teacher reference.

Usage:
    python3 fmaaq_quantize.py \
        --model-path /data/tensordata/Qwen3.5-REAP-262B-A17B \
        --teacher-path /data/tensordata/Qwen3.5-397B-A17B \
        --output-dir /data/tensordata/Qwen3.5-REAP-262B-A17B-int4-FMAAQ
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("fmaaq_quantize")


def main():
    parser = argparse.ArgumentParser(description="FMAAQ: AutoRound with teacher reference")
    parser.add_argument("--model-path", required=True, help="Path to student (REAP-262B BF16)")
    parser.add_argument("--teacher-path", required=True, help="Path to teacher (397B BF16)")
    parser.add_argument("--output-dir", required=True, help="Output directory for quantized model")
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
    parser.add_argument("--no-teacher", action="store_true", help="Disable teacher (standard AutoRound)")
    args = parser.parse_args()

    # Apply FMAAQ patch before importing AutoRound
    if not args.no_teacher:
        import fmaaq_patch
        fmaaq_patch.patch_autoround(args.teacher_path)
        logger.info(f"FMAAQ mode: teacher={args.teacher_path}")
    else:
        logger.info("Standard AutoRound mode (no teacher)")

    from auto_round import AutoRound
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

    # Build layer_config: shared experts + gates stay BF16
    # Detect correct prefix by checking model's named modules
    logger.info(f"Loading student model: {args.model_path}")
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

    # Find the correct module prefix for shared_expert
    layer_config = {}
    for n, m in model.named_modules():
        if "layers.0.mlp.shared_expert_gate" in n:
            # Extract prefix: e.g. "model.language_model." or "model."
            prefix = n.split("layers.0.mlp.shared_expert_gate")[0]
            logger.info(f"Detected layer prefix: '{prefix}'")
            num_layers = sum(1 for nn, _ in model.named_modules()
                           if nn.endswith(".mlp.shared_expert_gate"))
            logger.info(f"Detected {num_layers} layers with shared_expert_gate")
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

    logger.info("Starting quantization...")
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

    model, layer_config = autoround.quantize()

    logger.info(f"Saving to {args.output_dir}")
    autoround.save_quantized(args.output_dir, format="auto_round")

    logger.info("Done!")


if __name__ == "__main__":
    main()
