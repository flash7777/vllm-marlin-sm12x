#!/usr/bin/env python3
"""Pre-pack MTP INT4 GPTQ weights into Marlin format.

Converts GPTQ-packed MTP weights (qweight + scales + qzeros + g_idx)
into Marlin's optimized layout offline. This avoids the OOM-causing
repacking at model load time on memory-constrained UMA systems.

The output file replaces model-mtp-00001-of-00001.safetensors in the
model directory. vLLM loads the pre-packed weights directly.

Usage (in container with GPU):
    python3 prepack_mtp_marlin.py \
        --input /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-00001-of-00001.safetensors \
        --output /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-marlin-00001-of-00001.safetensors \
        --bits 4 --group-size 128

Requires: torch (CUDA), vllm (for ops.gptq_marlin_moe_repack), safetensors
"""

import argparse
import os
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def prepack_expert_weights(qweight, scales, qzeros, g_idx, bits=4, group_size=128):
    """Repack a single expert's GPTQ weights to Marlin format."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales,
        marlin_sort_g_idx,
    )

    pack_factor = 32 // bits  # 8 for 4-bit
    num_experts = qweight.shape[0]
    k = qweight.shape[1] * pack_factor  # unpacked input dim
    n = qweight.shape[2]  # output dim

    # Sort g_idx (required for act_order, no-op if sequential)
    if g_idx is not None and g_idx.numel() > 0:
        g_idx_sort = torch.empty_like(g_idx)
        sorted_g_idx = torch.empty_like(g_idx)
        for e in range(num_experts):
            g_idx_sort[e] = torch.argsort(g_idx[e]).to(torch.int32)
            sorted_g_idx[e] = g_idx[e][g_idx_sort[e]]
    else:
        g_idx_sort = torch.empty((num_experts, 0), dtype=torch.int32, device=qweight.device)
        sorted_g_idx = g_idx_sort.clone()

    # Repack qweight
    marlin_qweight = ops.gptq_marlin_moe_repack(
        qweight, g_idx_sort, k, n, bits, is_a_8bit=False,
    )

    # Permute scales
    marlin_scales = marlin_moe_permute_scales(
        s=scales, size_k=k, size_n=n, group_size=group_size,
    )

    return marlin_qweight, marlin_scales, sorted_g_idx, g_idx_sort


def main():
    parser = argparse.ArgumentParser(description="Pre-pack MTP GPTQ weights to Marlin format")
    parser.add_argument("--input", required=True, help="Input GPTQ safetensors")
    parser.add_argument("--output", required=True, help="Output Marlin safetensors")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    print(f"Pre-packing MTP weights: {args.input}")
    print(f"  bits={args.bits}, group_size={args.group_size}")

    # Load all tensors
    tensors = {}
    with safe_open(args.input, framework="pt", device="cuda") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    print(f"  Loaded {len(tensors)} tensors")

    # Group expert weights by layer
    # Format 1: mtp.layers.0.mlp.experts.{id}.{proj}.suffix (fixed down_proj)
    # Format 2: mtp.layers.0.mlp.experts.{proj}.{id}.suffix (unfixed gate_up_proj)
    import re
    pat1 = re.compile(r"(mtp\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj|gate_up_proj)\.(qweight|scales|qzeros|g_idx)")
    pat2 = re.compile(r"(mtp\.layers\.\d+\.mlp\.experts)\.(gate_proj|up_proj|down_proj|gate_up_proj)\.(\d+)\.(qweight|scales|qzeros|g_idx)")

    # Collect all experts per projection
    projections = {}  # (layer_prefix, proj) -> {id -> {suffix -> tensor}}
    passthrough = {}  # Non-expert tensors (norms, attention, fc)

    for key, tensor in tensors.items():
        m1 = pat1.match(key)
        m2 = pat2.match(key) if not m1 else None
        if m1:
            prefix, eid, proj, suffix = m1.groups()
        elif m2:
            prefix, proj, eid, suffix = m2.groups()
        else:
            passthrough[key] = tensor
            continue
        k = (prefix, proj)
        if k not in projections:
            projections[k] = {}
        if int(eid) not in projections[k]:
            projections[k][int(eid)] = {}
        projections[k][int(eid)][suffix] = tensor

    print(f"  Expert projections: {len(projections)}")
    print(f"  Passthrough tensors: {len(passthrough)}")

    if not projections:
        print("ERROR: No expert weights found")
        sys.exit(1)

    # Stack experts per projection and repack
    output_tensors = dict(passthrough)  # Start with non-expert tensors

    for (prefix, proj), experts in sorted(projections.items()):
        num_experts = max(experts.keys()) + 1
        print(f"  Repacking {prefix}.*.{proj} ({num_experts} experts)...", end=" ", flush=True)

        # Stack into [num_experts, K_packed, N]
        qweight_list = [experts[e]["qweight"] for e in range(num_experts)]
        scales_list = [experts[e]["scales"] for e in range(num_experts)]

        qweight = torch.stack(qweight_list)
        scales = torch.stack(scales_list)

        # g_idx and qzeros may or may not exist
        has_g_idx = "g_idx" in experts[0]
        has_qzeros = "qzeros" in experts[0]

        if has_g_idx:
            g_idx = torch.stack([experts[e]["g_idx"] for e in range(num_experts)])
        else:
            g_idx = None

        # Repack
        marlin_qw, marlin_sc, sorted_gidx, gidx_sort = prepack_expert_weights(
            qweight, scales, None, g_idx, args.bits, args.group_size
        )

        # Save as stacked tensors with Marlin suffix
        # Use same naming but with _marlin suffix to distinguish
        for eid in range(num_experts):
            output_tensors[f"{prefix}.{eid}.{proj}.qweight"] = marlin_qw[eid].cpu()
            output_tensors[f"{prefix}.{eid}.{proj}.scales"] = marlin_sc[eid].cpu()
            if has_g_idx:
                output_tensors[f"{prefix}.{eid}.{proj}.g_idx"] = sorted_gidx[eid].cpu()

        print("OK")

    print(f"\n  Saving {len(output_tensors)} tensors to {args.output}")
    save_file(output_tensors, args.output)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"  Done: {size_gb:.2f} GB")
    print(f"\n  Replace in model dir:")
    print(f"    mv {args.output} {os.path.dirname(args.input)}/model-mtp-00001-of-00001.safetensors")


if __name__ == "__main__":
    main()
