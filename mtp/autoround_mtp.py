#!/usr/bin/env python3
"""AutoRound INT4 quantization of isolated MTP weights from Qwen3.5-397B.

Strategy:
  The MTP layer is architecturally identical to a regular Qwen3.5 MoE decoder
  layer (full_attention + 512 experts). We build a fake 1-layer
  Qwen3_5MoeForCausalLM, rename the MTP weights to match, and run AutoRound.

  Calibration quality is limited (no real activations from the 60-layer trunk),
  but AutoRound still optimizes rounding directions vs. pure RTN.

Usage (inside autoround container, CPU):
    podman run --rm -it \\
        -v /data/tensordata:/data/tensordata \\
        -v /root/vllm-marlin-sm12x/mtp:/workspace/mtp \\
        localhost/autoround:latest \\
        python3 /workspace/mtp/autoround_mtp.py \\
            --mtp-weights /data/tensordata/mtp_weights_397b.safetensors \\
            --tokenizer /data/tensordata/Qwen3.5-397B-A17B \\
            --output /data/tensordata/mtp_weights_397b_int4 \\
            --bits 4 --group-size 128 --iters 200

Requirements (in autoround container):
    auto-round, torch, transformers, safetensors
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def build_fake_model_dir(mtp_weights_path: str, tokenizer_path: str,
                         work_dir: str) -> str:
    """Create a fake 1-layer Qwen3_5MoeForCausalLM model directory.

    Renames mtp.layers.0.* -> model.layers.0.* and adds dummy
    embed_tokens + lm_head so AutoRound can do forward passes.
    """
    fake_dir = os.path.join(work_dir, "fake_model")
    os.makedirs(fake_dir, exist_ok=True)

    print(f"Loading MTP weights from {mtp_weights_path} ...")
    mtp_tensors = load_file(mtp_weights_path, device="cpu")
    print(f"  {len(mtp_tensors)} tensors loaded")

    # Rename: mtp.layers.0.* -> model.layers.0.*
    #         mtp.norm.weight -> model.norm.weight
    #         mtp.fc.weight   -> lm_head.weight  (abuse as lm_head for the fake model)
    renamed = {}
    extra_mtp = {}  # MTP-specific weights we keep aside (not quantized)

    for key, tensor in mtp_tensors.items():
        if key.startswith("mtp.layers."):
            new_key = key.replace("mtp.layers.", "model.layers.", 1)
            renamed[new_key] = tensor
        elif key == "mtp.norm.weight":
            renamed["model.norm.weight"] = tensor
        elif key in ("mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight",
                      "mtp.pre_fc_norm_hidden.weight"):
            extra_mtp[key] = tensor
        else:
            print(f"  WARNING: unmapped key {key}, keeping as extra")
            extra_mtp[key] = tensor

    # Create dummy embed_tokens (needed for forward pass)
    hidden_size = 4096
    vocab_size = 248320
    print(f"  Creating dummy embed_tokens ({vocab_size}x{hidden_size}) ...")
    renamed["model.embed_tokens.weight"] = torch.randn(
        vocab_size, hidden_size, dtype=torch.bfloat16) * 0.02

    # Create dummy lm_head from mtp.fc if available, else random
    if "mtp.fc.weight" in extra_mtp:
        # fc is [hidden_size, hidden_size*2] -> not compatible with lm_head [vocab_size, hidden_size]
        # Use random lm_head instead
        pass
    print(f"  Creating dummy lm_head ({vocab_size}x{hidden_size}) ...")
    renamed["lm_head.weight"] = torch.randn(
        vocab_size, hidden_size, dtype=torch.bfloat16) * 0.02

    # Save renamed weights
    weights_path = os.path.join(fake_dir, "model.safetensors")
    print(f"  Saving {len(renamed)} renamed tensors to {weights_path} ...")
    save_file(renamed, weights_path)

    # Save extra MTP weights separately (not quantized)
    if extra_mtp:
        extra_path = os.path.join(fake_dir, "extra_mtp_weights.safetensors")
        print(f"  Saving {len(extra_mtp)} extra MTP tensors to {extra_path}")
        save_file(extra_mtp, extra_path)

    # Create config.json for 1-layer Qwen3_5MoeForCausalLM (text-only)
    config = {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe_text",
        "hidden_size": hidden_size,
        "num_hidden_layers": 1,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": vocab_size,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attn_output_gate": True,
        "hidden_act": "silu",
        "layer_types": ["full_attention"],
        "full_attention_interval": 4,
        "partial_rotary_factor": 0.25,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    }
    config_path = os.path.join(fake_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {config_path}")

    # Copy tokenizer files
    tokenizer_dir = Path(tokenizer_path)
    for tok_file in ["tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "merges.txt",
                     "vocab.json", "added_tokens.json"]:
        src = tokenizer_dir / tok_file
        if src.exists():
            shutil.copy2(str(src), fake_dir)
            print(f"  Copied {tok_file}")

    return fake_dir


def rename_back(output_dir: str, extra_mtp_path: str | None):
    """Rename quantized weights back to mtp.* prefix and merge extra weights."""
    print("\nRenaming quantized weights back to mtp.* prefix ...")

    # Find all safetensors in output
    st_files = sorted(Path(output_dir).glob("*.safetensors"))
    all_tensors = {}
    for sf in st_files:
        tensors = load_file(str(sf), device="cpu")
        all_tensors.update(tensors)

    renamed = {}
    skipped = []
    for key, tensor in all_tensors.items():
        if key.startswith("model.layers."):
            new_key = key.replace("model.layers.", "mtp.layers.", 1)
            renamed[new_key] = tensor
        elif key == "model.norm.weight":
            renamed["mtp.norm.weight"] = tensor
        elif key in ("model.embed_tokens.weight", "lm_head.weight"):
            skipped.append(key)  # dummy weights, discard
        else:
            # GPTQ metadata etc
            if key.startswith("model.layers."):
                new_key = key.replace("model.layers.", "mtp.layers.", 1)
                renamed[new_key] = tensor
            else:
                skipped.append(key)

    print(f"  Renamed {len(renamed)} tensors, skipped {len(skipped)}: {skipped}")

    # Merge extra MTP weights (fc, norms)
    if extra_mtp_path and os.path.exists(extra_mtp_path):
        extra = load_file(extra_mtp_path, device="cpu")
        print(f"  Merging {len(extra)} extra MTP tensors (fc, norms)")
        renamed.update(extra)

    # Save final MTP INT4 weights
    final_path = os.path.join(output_dir, "mtp_int4.safetensors")
    save_file(renamed, final_path)
    print(f"  Saved final MTP INT4 weights: {final_path}")
    print(f"  Total tensors: {len(renamed)}")
    print(f"  Size: {os.path.getsize(final_path) / 1e9:.2f} GB")

    return final_path


def run_autoround(fake_model_dir: str, output_dir: str,
                  bits: int = 4, group_size: int = 128,
                  iters: int = 200, seqlen: int = 512,
                  nsamples: int = 128, batch_size: int = 4):
    """Run AutoRound on the fake 1-layer model."""
    from auto_round import AutoRound

    print(f"\n=== Running AutoRound ===")
    print(f"  bits={bits}, group_size={group_size}, iters={iters}")
    print(f"  seqlen={seqlen}, nsamples={nsamples}, batch_size={batch_size}")
    print(f"  model_dir={fake_model_dir}")
    print(f"  output_dir={output_dir}")

    # Only quantize the decoder layer, not embed_tokens/lm_head
    layer_config = {
        "model.embed_tokens": {"bits": 16},  # skip quantization
        "lm_head": {"bits": 16},             # skip quantization
        "model.norm": {"bits": 16},          # skip quantization (RMSNorm)
    }

    autoround = AutoRound(
        model=fake_model_dir,
        tokenizer=None,  # loaded from model dir
        scheme=f"W{bits}A16",
        layer_config=layer_config,
        dataset="NeelNanda/pile-10k",
        iters=iters,
        seqlen=seqlen,
        nsamples=nsamples,
        batch_size=batch_size,
        device_map="cpu",
        low_cpu_mem_usage=True,
        seed=42,
    )

    autoround.quantize()
    autoround.save_quantized(output_dir, format="auto_gptq")

    print(f"\n  AutoRound complete. Output in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="AutoRound INT4 quantization of isolated MTP weights")
    parser.add_argument("--mtp-weights", type=str, required=True,
                        help="Path to BF16 MTP weights (.safetensors)")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer (original model dir)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for INT4 MTP weights")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=200,
                        help="AutoRound optimization iterations (0=RTN)")
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Working directory for temp files (default: /tmp)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="autoround_mtp_")
    print(f"Work directory: {work_dir}")

    # Step 1: Build fake 1-layer model
    fake_dir = build_fake_model_dir(args.mtp_weights, args.tokenizer, work_dir)

    # Step 2: Run AutoRound
    autoround_output = os.path.join(work_dir, "autoround_output")
    run_autoround(
        fake_dir, autoround_output,
        bits=args.bits, group_size=args.group_size,
        iters=args.iters, seqlen=args.seqlen,
        nsamples=args.nsamples, batch_size=args.batch_size,
    )

    # Step 3: Rename back to mtp.* and merge extra weights
    extra_mtp_path = os.path.join(fake_dir, "extra_mtp_weights.safetensors")
    final_path = rename_back(autoround_output, extra_mtp_path)

    # Copy final result to output dir
    import shutil
    final_dest = os.path.join(args.output, "mtp_int4.safetensors")
    shutil.copy2(final_path, final_dest)

    # Copy quantize_config.json if generated
    qconfig = os.path.join(autoround_output, "quantize_config.json")
    if os.path.exists(qconfig):
        shutil.copy2(qconfig, os.path.join(args.output, "quantize_config.json"))

    print(f"\n=== Done ===")
    print(f"  INT4 MTP weights: {final_dest}")
    print(f"  Size: {os.path.getsize(final_dest) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
