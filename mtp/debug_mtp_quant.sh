#!/bin/bash
# MTP Debug: Patcht laufende Container mit Debug-Output + MTP quant fix
# Ausfuehren NACH start_cluster.sh Container erstellt hat, VOR vllm serve
#
# Usage:
#   bash debug_mtp_quant.sh
#
# Markante Ausgaben im Log (grep danach):
#   [MTP-QUANT] quant_config=...    → was der Drafter als quant_config bekommt
#   [MTP-QUANT] layer=... bits=...  → welche bits/quantized fuer jede MTP-Layer
#   [MTP-QUANT] FORCE_BF16=...     → ob BF16-Patch aktiv ist
#
# Erwartete GUTE Ausgabe:
#   [MTP-QUANT] quant_config=INCConfig(...) type=INCConfig
#   [MTP-QUANT] layer=SharedFusedMoE bits=4 quantized=True
#
# SCHLECHTE Ausgabe (Unquantized):
#   [MTP-QUANT] quant_config=None type=NoneType
#   [MTP-QUANT] layer=SharedFusedMoE bits=16 quantized=False

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== MTP Debug Patches ==="

# Patch-Code als heredoc
PATCH_CODE=$(cat << 'PYEOF'
import sys, os

# 1. Debug in eagle.py _get_model (zeigt quant_config beim Drafter-Load)
eagle = '/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py'
with open(eagle) as f: src = f.read()
marker = '[MTP-QUANT]'
if marker not in src:
    old = '        # MTP BF16 patch:'
    new = """        import sys as _sys
        _qc = self.vllm_config.quant_config
        _bf16 = os.environ.get("VLLM_MTP_FORCE_BF16", "0")
        print(f'[MTP-QUANT] quant_config={_qc} type={type(_qc).__name__}', file=_sys.stderr, flush=True)
        print(f'[MTP-QUANT] FORCE_BF16={_bf16}', file=_sys.stderr, flush=True)
        # MTP BF16 patch:"""
    if old in src:
        src = src.replace(old, new)
        with open(eagle, 'w') as f: f.write(src)
        print('  eagle.py: OK')
    else:
        print('  eagle.py: pattern not found')
else:
    print('  eagle.py: already patched')

# 2. Debug in INC get_layer_config (zeigt bits fuer jede Layer)
inc = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/inc.py'
with open(inc) as f: src = f.read()
if marker not in src:
    old = '    def get_layer_config(self, layer, layer_name: str):'
    new = """    def get_layer_config(self, layer, layer_name: str):
        if 'mtp' in layer_name.lower() or 'model.layers.0' in layer_name:
            import sys as _sys
            _q = any(layer_name.startswith(n) for n in (self.block_name_to_quantize or []))
            print(f'[MTP-QUANT] layer={layer.__class__.__name__} name={layer_name} blocks={self.block_name_to_quantize} quantized={_q}', file=_sys.stderr, flush=True)"""
    if old in src:
        src = src.replace(old, new)
        with open(inc, 'w') as f: f.write(src)
        print('  inc.py: OK')
    else:
        print('  inc.py: pattern not found')
else:
    print('  inc.py: already patched')

# 3. MTP quant config patch (speculative.py)
spec = '/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py'
with open(spec) as f: lines = f.readlines()
patched = False
for i, l in enumerate(lines):
    if 'self.model = self.target_model_config.model' in l and 350 < i < 400:
        if 'draft_model_config' in lines[i+1]:
            print('  speculative.py: already patched')
            patched = True
            break
        lines.insert(i+1, '                self.draft_model_config = self.target_model_config\n')
        lines.insert(i+2, '                self.draft_parallel_config = self.target_parallel_config\n')
        with open(spec, 'w') as f: f.writelines(lines)
        print('  speculative.py: OK')
        patched = True
        break
if not patched:
    print('  speculative.py: pattern not found')

print('Done')
PYEOF
)

# Auf Head anwenden
echo "DGX Head:"
echo "$PATCH_CODE" | podman exec -i ng17e-head python3 - 2>&1

# Auf Worker anwenden
echo "PGX Worker:"
echo "$PATCH_CODE" | ssh flash@192.168.1.116 "podman exec -i ng17e-worker python3 -" 2>&1

echo ""
echo "=== Fertig ==="
echo "Jetzt vllm serve starten. Im Log suchen nach:"
echo "  grep 'MTP-QUANT' /tmp/ng17e-serve.log"
echo ""
echo "GUT:    [MTP-QUANT] quant_config=INCConfig type=INCConfig"
echo "GUT:    [MTP-QUANT] layer=SharedFusedMoE bits=4 quantized=True"
echo "SCHLECHT: [MTP-QUANT] quant_config=None type=NoneType"
echo "SCHLECHT: [MTP-QUANT] layer=SharedFusedMoE bits=16 quantized=False"
