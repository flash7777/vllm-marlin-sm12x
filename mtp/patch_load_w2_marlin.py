#!/usr/bin/env python3
"""Patch FusedMoE._load_w2 and _load_w13 to skip TP-sharding for pre-packed Marlin weights.

Problem: Pre-packed Marlin weights have permuted shape that doesn't match the
expected GPTQ shape. The narrow() call for TP sharding fails with size mismatch.

Fix: If loaded_weight.shape[shard_dim] already equals the target shard_size,
skip the narrow — weight is already the correct size for this TP rank.

Usage: cat patch_load_w2_marlin.py | podman exec -i CONTAINER python3 -
"""

MARKER = '[MTP-W2-SKIP]'

path = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py'

with open(path) as f:
    src = f.read()

if MARKER in src:
    print('layer.py: already patched')
else:
    patched = 0

    # Patch both narrow blocks in _load_w13 and _load_w2
    # They have the same pattern:
    #   if not load_full and loaded_weight.ndim > 0:
    #       loaded_weight = loaded_weight.narrow(
    #           shard_dim, shard_size * tp_rank, shard_size
    #       )

    old = '''        if not load_full and loaded_weight.ndim > 0:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )'''

    new = f'''        if not load_full and loaded_weight.ndim > 0:
            # {MARKER} Skip narrow if weight already matches target size (pre-packed Marlin)
            if loaded_weight.shape[shard_dim] != shard_size:
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )'''

    count = src.count(old)
    if count >= 2:
        src = src.replace(old, new)
        with open(path, 'w') as f:
            f.write(src)
        print(f'layer.py: patched ({count} narrow blocks in _load_w13 + _load_w2)')
    elif count == 1:
        src = src.replace(old, new)
        with open(path, 'w') as f:
            f.write(src)
        print(f'layer.py: patched (1 narrow block)')
    else:
        print('layer.py: pattern not found')
