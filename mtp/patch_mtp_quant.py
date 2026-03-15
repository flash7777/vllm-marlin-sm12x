#!/usr/bin/env python3
"""Patch vLLM for INT4 MTP support + debug output.

Patches 3 files:
1. speculative.py — inherit draft_model_config from target (quant_config flow)
2. eagle.py — debug print quant_config at drafter load time
3. inc.py — debug print bits/quantized for MTP layers

Debug markers in log: [MTP-QUANT]
  GOOD:  [MTP-QUANT] quant_config=INCConfig type=INCConfig
  GOOD:  [MTP-QUANT] bits=4 quantized=True
  BAD:   [MTP-QUANT] quant_config=None type=NoneType
  BAD:   [MTP-QUANT] bits=16 quantized=False

Usage: piped into container
  cat patch_mtp_quant.py | podman exec -i CONTAINER python3 -
"""

import sys

MARKER = '[MTP-QUANT]'
patched = 0

# === 1. speculative.py: draft_model_config for MTP ===
spec = '/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py'
try:
    with open(spec) as f:
        lines = f.readlines()
    done = False
    for i, l in enumerate(lines):
        if 'self.model = self.target_model_config.model' in l and 350 < i < 400:
            if 'draft_model_config' in (lines[i+1] if i+1 < len(lines) else ''):
                print(f'speculative.py: already patched')
                done = True
                break
            lines.insert(i+1, '                self.draft_model_config = self.target_model_config\n')
            lines.insert(i+2, '                self.draft_parallel_config = self.target_parallel_config\n')
            with open(spec, 'w') as f:
                f.writelines(lines)
            print(f'speculative.py: patched')
            patched += 1
            done = True
            break
    if not done:
        print(f'speculative.py: pattern not found')
except Exception as e:
    print(f'speculative.py: ERROR {e}')

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
            print('eagle.py: patched')
            patched += 1
        else:
            print('eagle.py: pattern not found')
    else:
        print('eagle.py: already patched')
except Exception as e:
    print(f'eagle.py: ERROR {e}')

# === 3. inc.py: debug bits for MTP layers ===
inc = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/inc.py'
try:
    with open(inc) as f:
        src = f.read()
    if MARKER not in src:
        old = '    def get_layer_config(self, layer, layer_name: str):'
        new = f'''    def get_layer_config(self, layer, layer_name: str):
        if 'mtp' in layer_name.lower() or (layer_name.startswith('model.layers.0') and 'experts' in layer_name):
            import sys as _s
            _q = any(layer_name.startswith(n) for n in (self.block_name_to_quantize or []))
            _s.stderr.write(f'{MARKER} layer={{layer.__class__.__name__}} name={{layer_name}} quantized={{_q}} blocks={{self.block_name_to_quantize}}\\n')
            _s.stderr.flush()'''
        if old in src:
            src = src.replace(old, new)
            with open(inc, 'w') as f:
                f.write(src)
            print('inc.py: patched')
            patched += 1
        else:
            print('inc.py: pattern not found')
    else:
        print('inc.py: already patched')
except Exception as e:
    print(f'inc.py: ERROR {e}')

print(f'Total: {patched} files patched')
