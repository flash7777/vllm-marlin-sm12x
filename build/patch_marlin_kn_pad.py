#!/usr/bin/env python3
"""Patch: Marlin K+N dimension padding for SM121.

On SM121, Marlin is the ONLY available INT4 kernel. When layer dimensions
are not divisible by min_thread_k=128 or min_thread_n=64, there's no
fallback. This patch:
  1. Relaxes shape checks in marlin_utils.py (warn instead of raise)
  2. Adds actual zero-padding in MarlinLinearKernel (marlin.py)

Affected layers in Qwen3.5 GatedDeltaNet at TP=2:
  - in_proj_ba: K=576 (not % 128), N=2152 (not % 64)
"""
import sys

MARLIN_UTILS = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils.py"
MARLIN_KERNEL = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/marlin.py"


def patch_marlin_utils():
    """Relax K, N, group_size checks in verify_marlin_supports_shape."""
    with open(MARLIN_UTILS) as f:
        content = f.read()

    if "MARLIN_KN_PAD" in content:
        print("patch_marlin_kn_pad: marlin_utils.py already patched")
        return

    # 1. Replace K check
    old_k = """    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"Weight input_size_per_partition = "
            f"{input_size_per_partition} is not divisible "
            f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )"""

    new_k = """    # MARLIN_KN_PAD: Relax K check — padding will be applied during weight loading
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        import logging as _mkn
        _mkn.getLogger(__name__).warning(
            f"Marlin KN-pad: K={input_size_per_partition} not divisible by "
            f"{GPTQ_MARLIN_MIN_THREAD_K}, will pad during weight loading")"""

    if old_k not in content:
        old_k_v2 = """    # MARLIN_K_RELAX: Allow K not divisible by min_thread_k.
    # On SM121 no other INT4 kernel exists. Return False to signal
    # that Marlin cannot handle this layer (vLLM will use dequant fallback).
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        import logging as _mk_log
        _mk_log.getLogger(__name__).warning(
            f"Marlin: K={input_size_per_partition} not divisible by "
            f"{GPTQ_MARLIN_MIN_THREAD_K}, skipping Marlin for this layer")
        return"""
        if old_k_v2 in content:
            content = content.replace(old_k_v2, new_k)
            print("  Replaced MARLIN_K_RELAX with MARLIN_KN_PAD (K)")
        else:
            print("ERROR: K check pattern not found")
            sys.exit(1)
    else:
        content = content.replace(old_k, new_k)
        print("  Patched K check")

    # 2. Replace N check
    old_n = """    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"Weight output_size_per_partition = "
            f"{output_size_per_partition} is not divisible by "
            f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )"""

    new_n = """    # MARLIN_KN_PAD: Relax N check — padding will be applied during weight loading
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        import logging as _mkn2
        _mkn2.getLogger(__name__).warning(
            f"Marlin KN-pad: N={output_size_per_partition} not divisible by "
            f"{GPTQ_MARLIN_MIN_THREAD_N}, will pad during weight loading")"""

    if old_n not in content:
        print("WARNING: N check pattern not found (may already be patched)")
    else:
        content = content.replace(old_n, new_n)
        print("  Patched N check")

    # 3. Replace group_size check too (576 % 128 != 0)
    old_g = """    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )"""

    new_g = """    # MARLIN_KN_PAD: Relax group_size check for padded layers
    if group_size < input_size and input_size_per_partition % group_size != 0:
        import logging as _mkn3
        _mkn3.getLogger(__name__).warning(
            f"Marlin KN-pad: K={input_size_per_partition} not divisible by "
            f"group_size={group_size}, will pad during weight loading")"""

    if old_g in content:
        content = content.replace(old_g, new_g)
        print("  Patched group_size check")

    with open(MARLIN_UTILS, "w") as f:
        f.write(content)
    print("  marlin_utils.py done")


def patch_marlin_kernel():
    """Add actual K/N zero-padding in MarlinLinearKernel."""
    with open(MARLIN_KERNEL) as f:
        content = f.read()

    if "MARLIN_KN_PAD" in content:
        print("patch_marlin_kn_pad: marlin.py already patched")
        return

    # --- Patch 1: Add padding in process_weights_after_loading ---
    # Insert padding logic right after workspace allocation and before transform functions

    old_process = """        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace_new(device)

        # Default names since marlin requires empty parameters for these,"""

    new_process = """        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace_new(device)

        # MARLIN_KN_PAD: Compute padded dimensions
        import math
        orig_k = c.partition_weight_shape[0]
        orig_n = c.partition_weight_shape[1]
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            GPTQ_MARLIN_MIN_THREAD_K, GPTQ_MARLIN_MIN_THREAD_N)
        gs = c.group_size if c.group_size > 0 else GPTQ_MARLIN_MIN_THREAD_K
        k_align = math.lcm(GPTQ_MARLIN_MIN_THREAD_K, gs)
        pad_k = math.ceil(orig_k / k_align) * k_align
        pad_n = math.ceil(orig_n / GPTQ_MARLIN_MIN_THREAD_N) * GPTQ_MARLIN_MIN_THREAD_N
        self._marlin_orig_k = orig_k
        self._marlin_orig_n = orig_n
        need_pad = (pad_k != orig_k or pad_n != orig_n)
        if need_pad:
            import logging as _mkn_log
            _mkn_log.getLogger(__name__).warning(
                f"Marlin KN-pad: padding K={orig_k}->{pad_k}, N={orig_n}->{pad_n}")
            num_bits = c.weight_type.size_bits
            pack_factor = 32 // num_bits  # 8 for INT4
            # Pad qweight: [K/pack_factor, N] -> [pad_K/pack_factor, pad_N]
            w_q = getattr(layer, self.w_q_name)
            q_data = w_q.data
            new_q = torch.zeros(pad_k // pack_factor, pad_n,
                                dtype=q_data.dtype, device=device)
            new_q[:q_data.shape[0], :q_data.shape[1]] = q_data
            w_q.data = new_q
            # Pad scales: [num_groups, N] -> [pad_K/gs, pad_N]
            w_s = getattr(layer, self.w_s_name)
            s_data = w_s.data
            new_num_groups = pad_k // gs if c.group_size > 0 else 1
            new_s = torch.zeros(new_num_groups, pad_n,
                                dtype=s_data.dtype, device=device)
            new_s[:s_data.shape[0], :s_data.shape[1]] = s_data
            w_s.data = new_s
            # Pad zero_points if present
            if c.zero_points and hasattr(layer, self.w_zp_name):
                w_zp = getattr(layer, self.w_zp_name)
                zp_data = w_zp.data
                new_zp = torch.zeros(new_num_groups, pad_n // pack_factor,
                                     dtype=zp_data.dtype, device=device)
                new_zp[:zp_data.shape[0], :zp_data.shape[1]] = zp_data
                w_zp.data = new_zp
            # Update config shapes to padded values
            c.partition_weight_shape = (pad_k, pad_n)

        # Default names since marlin requires empty parameters for these,"""

    if old_process not in content:
        print("ERROR: process_weights_after_loading pattern not found")
        sys.exit(1)

    content = content.replace(old_process, new_process)
    print("  Patched process_weights_after_loading (padding)")

    # --- Patch 2: Pad input and slice output in apply_weights ---
    old_apply = """        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=self.is_k_full,
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )"""

    new_apply = """        # MARLIN_KN_PAD: Pad input along K if needed, slice output along N
        orig_k = getattr(self, '_marlin_orig_k', c.partition_weight_shape[0])
        orig_n = getattr(self, '_marlin_orig_n', c.partition_weight_shape[1])
        padded_k = c.partition_weight_shape[0]
        padded_n = c.partition_weight_shape[1]
        if padded_k != orig_k:
            # Pad input: [..., orig_k] -> [..., padded_k]
            pad_size = padded_k - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_size))
        result = apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=padded_k,
            output_size_per_partition=padded_n,
            is_k_full=self.is_k_full,
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )
        if padded_n != orig_n:
            result = result[..., :orig_n]
        return result"""

    if old_apply not in content:
        print("ERROR: apply_weights pattern not found")
        sys.exit(1)

    content = content.replace(old_apply, new_apply)
    print("  Patched apply_weights (pad input, slice output)")

    with open(MARLIN_KERNEL, "w") as f:
        f.write(content)
    print("  marlin.py done")


def patch():
    patch_marlin_utils()
    patch_marlin_kernel()
    print("patch_marlin_kn_pad: all patches applied")


if __name__ == "__main__":
    patch()
