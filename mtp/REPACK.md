# MTP Marlin Pre-Packing

## Problem

vLLM repackt GPTQ-Weights beim Laden in Marlin-Format. Bei 512 MoE-Experts braucht das ~3-5 GB temporären Peak-Memory — auf DGX Spark UMA (119 GB, davon 100 GB Modell) zu viel → OOM.

## Lösung

Weights **offline** vorpacken. Dann überspringt vLLM das Repacking beim Laden.

## Pipeline

### 1. Voraussetzung

MTP INT4 GPTQ Weights müssen bereits extrahiert und injiziert sein:
```
/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-00001-of-00001.safetensors
```

### 2. Pre-Pack (einmalig, braucht GPU + ~7 GB freien VRAM)

```bash
# Frischen Container starten (kein Modell geladen!)
podman run -d --name repack \
  --device nvidia.com/gpu=all --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -v /data/tensordata:/data/tensordata \
  -v /home/flash/vllm-marlin-sm12x:/workspace:z \
  localhost/vllm-ng17e:latest sleep infinity

# Pre-Pack ausführen
podman exec repack python3 /workspace/mtp/prepack_mtp_marlin.py \
  --input /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-00001-of-00001.safetensors \
  --output /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-marlin.safetensors \
  --bits 4 --group-size 128

# Container aufräumen
podman stop repack && podman rm repack
```

### 3. Ersetzen

```bash
# Backup der GPTQ-Version
mv /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-00001-of-00001.safetensors \
   /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-gptq-backup.safetensors

# Marlin-Version aktivieren
mv /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-marlin.safetensors \
   /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-00001-of-00001.safetensors
```

### 4. Index aktualisieren

```bash
python3 -c "
import json
from safetensors import safe_open

idx_path = '/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model.safetensors.index.json'
mtp_path = '/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-00001-of-00001.safetensors'

with open(idx_path) as f:
    idx = json.load(f)

# Alte MTP-Keys entfernen, neue einsetzen
idx['weight_map'] = {k: v for k, v in idx['weight_map'].items() if not k.startswith('mtp.')}
with safe_open(mtp_path, framework='numpy') as f:
    for key in f.keys():
        idx['weight_map'][key] = 'model-mtp-00001-of-00001.safetensors'

with open(idx_path, 'w') as f:
    json.dump(idx, f, indent=2)
print(f'Index: {len(idx[\"weight_map\"])} keys')
"
```

### 5. PGX synchronisieren

```bash
rsync -avP \
  /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-mtp-00001-of-00001.safetensors \
  /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model.safetensors.index.json \
  flash@192.168.0.116:/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/
```

### 6. Starten

```bash
bash start_cluster.sh --mtp-eager   # oder --mtp
```

`start_cluster.sh` wendet automatisch zwei Patches an:
- `patch_mtp_quant.py` — INC erkennt MTP-Experts als INT4
- `patch_skip_mtp_repack.py` — Überspringt Marlin-Repacking (Weights schon vorgepackt)

## Shape-Erkennung

Der Skip-Patch erkennt vorgepackte Weights an der geänderten Shape:

| | GPTQ | Marlin (vorgepackt) |
|---|---|---|
| qweight | `[K/pack, N]` z.B. `[128, 4096]` | `[K/16, N*16/pack]` z.B. `[64, 8192]` |
| scales | `[groups, N]` z.B. `[8, 4096]` | `[groups, N]` (gleich, Werte permutiert) |

## Dateien

| Script | Zweck |
|--------|-------|
| `prepack_mtp_marlin.py` | Offline GPTQ→Marlin Konvertierung |
| `patch_skip_mtp_repack.py` | vLLM Patch: Repacking überspringen |
| `patch_mtp_quant.py` | vLLM Patch: INC erkennt MTP als INT4 |
| `fix_mtp_key_format.py` | Key-Umbenennung experts.proj.id → experts.id.proj |

## Erwartete Log-Ausgaben

```
[MTP-QUANT] layer=SharedFusedMoE name=mtp.layers.0.mlp.experts FORCED quantized=True (MoE only)
[MTP-MARLIN-SKIP] Skipping repack: w2 shape=... — already Marlin
```
