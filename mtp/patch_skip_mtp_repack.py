#!/usr/bin/env python3
"""Patch vLLM to skip Marlin repacking for pre-packed MTP weights.

Detects pre-packed Marlin weights by shape change:
  GPTQ:   [num_experts, K/pack, N]  (e.g. [512, 128, 4096])
  Marlin:  [num_experts, K/16, N*16/pack] (e.g. [512, 64, 8192])

If w2_qweight shape[1] < K/pack_factor, weights are already Marlin.

Usage: piped into container
  cat patch_skip_mtp_repack.py | podman exec -i CONTAINER python3 -
"""

MARKER = '[MTP-MARLIN-SKIP]'

path = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gptq_marlin.py'

with open(path) as f:
    src = f.read()

if MARKER in src:
    print('gptq_marlin.py: already patched')
else:
    old = '        # Repack weights\n        marlin_w13_qweight = ops.gptq_marlin_moe_repack('
    new = f'''        # {MARKER} Skip repacking if weights are already in Marlin format
        # Pre-packed Marlin weights have different shape than GPTQ
        _k_packed = layer.w2_qweight.shape[1]
        _k_expected = layer.intermediate_size_per_partition // self.quant_config.pack_factor
        if _k_packed != _k_expected:
            import sys
            sys.stderr.write(f'{MARKER} Skipping repack: w2 shape={{layer.w2_qweight.shape}} (k_packed={{_k_packed}} != k_expected={{_k_expected}}) — already Marlin\\n')
            sys.stderr.flush()
            # Alias for modular kernel
            layer.w13_weight = layer.w13_qweight
            layer.w2_weight = layer.w2_qweight
            # Scales already permuted in prepack — skip everything
            return

        # Repack weights
        marlin_w13_qweight = ops.gptq_marlin_moe_repack('''

    if old in src:
        src = src.replace(old, new)
        with open(path, 'w') as f:
            f.write(src)
        print('gptq_marlin.py: patched (skip repack for pre-packed Marlin)')
    else:
        print('gptq_marlin.py: pattern not found')
