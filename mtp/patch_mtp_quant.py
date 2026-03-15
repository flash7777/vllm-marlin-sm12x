#!/usr/bin/env python3
"""Patch vLLM for INT4 MTP support + debug output.

Problem: INC quantization config uses block_name_to_quantize which gets
mapped by apply_vllm_mapper. After mapping, only 'language_model.model.layers'
remains. MTP layers have prefix 'mtp.*' which doesn't match.

Fix: Patch INC get_layer_config to treat 'mtp.' prefix as quantized,
regardless of block_name_to_quantize.

Debug markers in log: [MTP-QUANT]

Usage: piped into container
  cat patch_mtp_quant.py | podman exec -i CONTAINER python3 -
"""

import sys

MARKER = '[MTP-QUANT]'
patched = 0

# === 1. INC: force quantized=True for mtp.* layers ===
inc = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/inc.py'
try:
    with open(inc) as f:
        src = f.read()
    if MARKER not in src:
        old = '        # 2. Determine whether layer should be quantized'
        new = f'''        # 2. Determine whether layer should be quantized
        # MTP INT4 patch: treat mtp.* layers as quantized
        if layer_name.startswith('mtp.') or layer_name.startswith('model.mtp.'):
            import sys as _s
            _s.stderr.write(f'{MARKER} layer={{layer.__class__.__name__}} name={{layer_name}} FORCED quantized=True\\n')
            _s.stderr.flush()
            return get_config(layer_name, True)'''
        if old in src:
            src = src.replace(old, new)
            with open(inc, 'w') as f:
                f.write(src)
            print('inc.py: patched (mtp.* forced quantized)')
            patched += 1
        else:
            print('inc.py: pattern not found')
    else:
        print('inc.py: already patched')
except Exception as e:
    print(f'inc.py: ERROR {e}')

# === 2. eagle.py: debug quant_config at drafter load ===
eagle = '/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py'
try:
    with open(eagle) as f:
        src = f.read()
    if MARKER not in src:
        old = '        # MTP BF16 patch:'
        new = f'''        import sys as _s, os as _o
        _qc = self.vllm_config.quant_config
        _s.stderr.write(f'{MARKER} quant_config={{_qc}} type={{type(_qc).__name__}}\\n')
        _s.stderr.write(f'{MARKER} FORCE_BF16={{_o.environ.get("VLLM_MTP_FORCE_BF16","0")}}\\n')
        _s.stderr.flush()
        # MTP BF16 patch:'''
        if old in src:
            src = src.replace(old, new)
            with open(eagle, 'w') as f:
                f.write(src)
            print('eagle.py: patched (debug)')
            patched += 1
        else:
            print('eagle.py: pattern not found')
    else:
        print('eagle.py: already patched')
except Exception as e:
    print(f'eagle.py: ERROR {e}')

print(f'Total: {patched} files patched')
