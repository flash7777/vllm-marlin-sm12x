# Intel AutoRound Safetensors Duplicate Key Fix

## Problem

`Intel/Qwen3.5-397B-A17B-int4-AutoRound` hat **2688 duplizierte Weight-Keys** über zwei Safetensor-Shards (model-00039-of-00040 und model-00040-of-00040). Betrifft Layer 59, Experts 213-511.

**Symptom**: `fastsafetensors` Loader crasht mit:
```
Exception: FilesBufferOnDevice: key model.language_model.layers.59.mlp.experts.213.down_proj.qweight must be unique among files
```

Der Standard-Loader (`--load-format auto`) ignoriert die Duplikate (lädt den letzten Shard), aber `fastsafetensors` verlangt unique Keys.

## Root Cause

AutoRound hat beim Sharding die letzten Experts (Layer 59, ab Expert 213) sowohl in Shard 39 als auch Shard 40 geschrieben. Der `model.safetensors.index.json` zeigt korrekt auf Shard 40, aber Shard 39 enthält dieselben Keys als Duplikate.

## Fix

Entferne die duplizierten Keys aus Shard 39 (behalte sie in Shard 40 wo sie laut Index hingehören):

```bash
python3 fix_duplicate_keys.py /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound
```

Das Script erstellt ein Backup (`*.bak`) und schreibt den bereinigten Shard.

**Danach**: Bereinigte Datei auch auf PGX kopieren:
```bash
rsync -avP /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/model-00039-of-00040.safetensors \
  flash@192.168.0.116:/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/
```

## Verifizierung

```bash
python3 fix_duplicate_keys.py /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound --verify-only
```
