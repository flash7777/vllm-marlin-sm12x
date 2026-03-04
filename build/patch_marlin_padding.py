#!/usr/bin/env python3
"""Patch: Marlin N-dimension padding for output_size_per_partition < 64.

Problem: Marlin GPTQ kernel requires output_size_per_partition % 64 == 0.
Models like Qwen3-Coder-Next have layers with output_size_per_partition=32
at TP=2, causing ValueError.

Fix: Zero-pad weights to next multiple of 64 during process_weights_after_loading,
run Marlin kernel with padded N, then slice output back to actual N.

IMPORTANT: Padding must happen BEFORE gptq_marlin_repack and marlin_permute_scales,
because those C++ ops themselves check size_n % 64 == 0.

Zero-padding is mathematically correct: Y = X @ W where padded W columns are zero
produces zero output columns that get sliced away.

Files modified:
  - marlin_utils.py: relax check_marlin_supports_shape for N < MIN_THREAD_N
  - marlin.py: pad weights BEFORE repack/permute, slice output in apply_weights
"""

import os
import sys

MARLIN_PY = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/kernels/mixed_precision/marlin.py"
MARLIN_UTILS_PY = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils.py"


def patch_marlin_utils():
    """Relax shape check to allow N < GPTQ_MARLIN_MIN_THREAD_N."""
    with open(MARLIN_UTILS_PY) as f:
        content = f.read()

    if "MARLIN_PADDING" in content:
        print("SKIP: marlin_utils.py already patched")
        return

    # Replace the N-dimension check in verify_marlin_supports_shape
    old = '''\
    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"Weight output_size_per_partition = "
            f"{output_size_per_partition} is not divisible by "
            f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )'''

    new = '''\
    # Validate output_size_per_partition
    # MARLIN_PADDING: Allow N < min_thread_n — weights will be zero-padded
    # to the next multiple of GPTQ_MARLIN_MIN_THREAD_N during weight loading.
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        import logging as _mp_log
        _padded_n = ((output_size_per_partition + GPTQ_MARLIN_MIN_THREAD_N - 1)
                     // GPTQ_MARLIN_MIN_THREAD_N * GPTQ_MARLIN_MIN_THREAD_N)
        if _padded_n <= 256:  # Only pad small layers, not grossly misaligned ones
            _mp_log.getLogger(__name__).info(
                f"Marlin N-padding: {output_size_per_partition} -> {_padded_n}")
        else:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
                "Consider reducing tensor_parallel_size or running "
                "with --quantization gptq."
            )'''

    if old not in content:
        print(f"ERROR: Could not find expected code in {MARLIN_UTILS_PY}")
        sys.exit(1)

    content = content.replace(old, new)

    with open(MARLIN_UTILS_PY, "w") as f:
        f.write(content)
    print(f"OK: marlin_utils.py patched (N-padding shape check relaxed)")


def patch_marlin_kernel():
    """Add zero-padding to MarlinLinearKernel weights and output slicing.

    Key insight: padding must happen BEFORE gptq_marlin_repack / marlin_permute_scales
    because those C++ ops check size_n % 64 == 0 internally.

    The raw qweight has shape [K/pack_factor, N] where packing is along K (dim 0),
    so dim 1 is plain N — pad it to padded_n directly (no pack_factor division).
    """
    with open(MARLIN_PY) as f:
        content = f.read()

    if "MARLIN_PADDING" in content:
        print("SKIP: marlin.py already patched")
        return

    # 1. Add padding helper at top of file
    old_imports = "from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig"
    new_imports = """from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# MARLIN_PADDING: round up to next multiple of 64
_MARLIN_PAD_N = 64

def _pad_dim(t: torch.Tensor, dim: int, target: int) -> torch.Tensor:
    \"\"\"Zero-pad tensor along given dim to target size.\"\"\"
    if t.shape[dim] >= target:
        return t
    pad_size = target - t.shape[dim]
    pad_shape = list(t.shape)
    pad_shape[dim] = pad_size
    padding = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, padding], dim=dim)"""

    if old_imports not in content:
        print(f"ERROR: Could not find imports in {MARLIN_PY}")
        sys.exit(1)

    content = content.replace(old_imports, new_imports)

    # 2. Add padding calculation BEFORE transform functions, and modify transforms
    #    to pad raw weights before repack/permute.
    #
    # Match: from default names through zero_points handling + transforms + bias
    old_block = '''\
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.gptq_marlin_repack(
                x.data.contiguous(),
                perm=layer.g_idx_sort_indices,
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                num_bits=c.weight_type.size_bits,
                is_a_8bit=is_a_8bit,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                group_size=c.group_size,
                is_a_8bit=is_a_8bit,
            )

            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            if c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                layer.register_parameter(
                    "input_global_scale",
                    torch.nn.Parameter(input_global_scale, requires_grad=False),
                )
            else:
                layer.input_global_scale = None
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    unpack_cols(
                        x.t(),
                        c.weight_type.size_bits,
                        grouped_k,
                        c.partition_weight_shape[1],
                    ),
                    size_k=grouped_k,
                    size_n=c.partition_weight_shape[1],
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data = marlin_permute_bias(layer.bias)'''

    new_block = '''\
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        # MARLIN_PADDING: Calculate padding BEFORE transforms.
        # gptq_marlin_repack and marlin_permute_scales require size_n % 64 == 0,
        # so we must pad raw weights before calling those functions.
        actual_n = c.partition_weight_shape[1]
        padded_n = ((actual_n + _MARLIN_PAD_N - 1) // _MARLIN_PAD_N) * _MARLIN_PAD_N
        self._marlin_actual_n = actual_n
        self._marlin_padded_n = padded_n
        _needs_pad = padded_n != actual_n
        if _needs_pad:
            import logging
            logging.getLogger(__name__).info(
                f"Marlin N-padding: {actual_n} -> {padded_n}")
        # Use padded_n for all repack/permute calls
        _size_n = padded_n

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            # MARLIN_PADDING: pad raw qweight before repack.
            # qweight shape is [K/pack_factor, N] — packing is along K (dim 0),
            # dim 1 is plain N. So pad dim=1 to padded_n directly.
            if _needs_pad:
                x.data = _pad_dim(x.data, dim=1, target=padded_n)
            x.data = ops.gptq_marlin_repack(
                x.data.contiguous(),
                perm=layer.g_idx_sort_indices,
                size_k=c.partition_weight_shape[0],
                size_n=_size_n,
                num_bits=c.weight_type.size_bits,
                is_a_8bit=is_a_8bit,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            # MARLIN_PADDING: pad scales before permute
            if _needs_pad:
                x.data = _pad_dim(x.data, dim=1, target=padded_n)
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=c.partition_weight_shape[0],
                size_n=_size_n,
                group_size=c.group_size,
                is_a_8bit=is_a_8bit,
            )

            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            if c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                layer.register_parameter(
                    "input_global_scale",
                    torch.nn.Parameter(input_global_scale, requires_grad=False),
                )
            else:
                layer.input_global_scale = None
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            # MARLIN_PADDING: unpack with actual_n, then pad before marlin_zero_points
            def _transform_zp(x):
                zp = unpack_cols(
                    x.t(),
                    c.weight_type.size_bits,
                    grouped_k,
                    actual_n,
                )
                if _needs_pad:
                    zp = _pad_dim(zp, dim=1, target=padded_n)
                return marlin_zero_points(
                    zp,
                    size_k=grouped_k,
                    size_n=_size_n,
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                )
            self._transform_param(layer, self.w_zp_name, _transform_zp)
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data = marlin_permute_bias(layer.bias)'''

    if old_block not in content:
        print(f"ERROR: Could not find process_weights block in {MARLIN_PY}")
        sys.exit(1)

    content = content.replace(old_block, new_block)

    # 3. Modify apply_weights to use padded N and slice output
    old_apply = '''\
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin

        return apply_gptq_marlin_linear(
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
        )'''

    new_apply = '''\
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin

        # MARLIN_PADDING: use padded N for kernel, slice output to actual N
        actual_n = getattr(self, '_marlin_actual_n', c.partition_weight_shape[1])
        padded_n = getattr(self, '_marlin_padded_n', c.partition_weight_shape[1])

        output = apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=padded_n,
            is_k_full=self.is_k_full,
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )

        # Slice back to actual output size if padding was applied
        if padded_n != actual_n:
            output = output[..., :actual_n]

        return output'''

    if old_apply not in content:
        print(f"ERROR: Could not find apply_weights in {MARLIN_PY}")
        sys.exit(1)

    content = content.replace(old_apply, new_apply)

    with open(MARLIN_PY, "w") as f:
        f.write(content)
    print(f"OK: marlin.py patched (pre-repack padding + output slicing)")


def main():
    for f in [MARLIN_PY, MARLIN_UTILS_PY]:
        if not os.path.exists(f):
            print(f"SKIP: {f} not found")
            sys.exit(0)

    patch_marlin_utils()
    patch_marlin_kernel()
    print("\nMarlin N-padding patch complete.")
    print("Layers with output_size_per_partition < 64 will be zero-padded to 64.")


if __name__ == "__main__":
    main()
