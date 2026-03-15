# MTP (Multi-Token Prediction) für Qwen3.5-397B INT4 AutoRound

## Status

- **BF16 Original** (`Qwen/Qwen3.5-397B-A17B`): Hat 1553 MTP-Weights (Shards 91-94)
- **INT4 AutoRound** (`Intel/Qwen3.5-397B-A17B-int4-AutoRound`): MTP fehlt
- **Config `num_nextn_predict_layers`**: Steht auf `None` in allen Qwen3.5 configs, aber Weights existieren

## MTP-Architektur (Qwen3.5-397B)

| Komponente | Größe |
|---|---|
| Expert Weights | 512 × 3 proj × 4096 × 1024 × BF16 = 12.9 GB |
| Attention (q/k/v/o) | ~0.13 GB |
| Norms, FC, Embed | ~0.1 GB |
| **Total BF16** | **~13.1 GB** |
| **Pro Node (TP=2)** | **~6.6 GB** |

## Memory-Budget (Dual DGX Spark)

| Komponente | Pro Node |
|---|---|
| GPU Total | 128.5 GB (119 GB nutzbar) |
| Modell INT4 | ~100 GB |
| MTP BF16 | ~6.6 GB |
| KV-Cache | 3-4 GB (reduziert) |
| Rest | ~2-3 GB |

→ **MTP passt**, aber KV-Cache muss von 8G auf 3-4G reduziert werden.
→ Concurrent Capacity: ~1× 256K oder ~2× 64K

## Schritt 1: MTP-Weights extrahieren (auf RTX/Spiegel 2)

```bash
# Auf Spiegel 2 (dort liegt das BF16 Modell):
ssh -p 2020 root@10.249.0.99

python3 /path/to/extract_mtp_weights.py \
    --source /data/tensordata/Qwen3.5-397B-A17B \
    --output /data/tensordata/mtp_weights_397b.safetensors

# Oder nur Info:
python3 extract_mtp_weights.py \
    --source /data/tensordata/Qwen3.5-397B-A17B \
    --info-only
```

## Schritt 2: MTP-Weights nach DGX kopieren

```bash
# Von Spiegel 2 nach DGX:
scp -P 2020 root@10.249.0.99:/data/tensordata/mtp_weights_397b.safetensors \
    /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/

# Nach PGX:
rsync -avP /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/mtp_weights_397b.safetensors \
    flash@192.168.0.116:/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/
```

## Schritt 3: Config anpassen

```bash
# In config.json des INT4 Modells, unter text_config:
python3 -c "
import json
config_path = '/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound/config.json'
with open(config_path) as f:
    c = json.load(f)
c['text_config']['num_nextn_predict_layers'] = 1
with open(config_path, 'w') as f:
    json.dump(c, f, indent=2)
print('Added num_nextn_predict_layers=1')
"
```

## Schritt 4: Starten mit MTP

```bash
vllm serve /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound \
    ...
    --num-speculative-tokens 1 \
    --kv-cache-memory-bytes 3G \   # reduziert wegen MTP-Memory
    ...
```

## Erwarteter Speedup

- GB10 ist bandwidth-limited (273 GB/s)
- MTP mit NST=1 war bei Qwen3-Coder auf GB10 optimal
- **Erwartung: +20-40%** (28 → 35-40 tok/s)
- Hängt von Acceptance Rate ab (>70% nötig für Speedup)

## Alternative: MTP-Weights quantisieren

INT4 MTP-Weights (statt BF16) würden nur ~3.3 GB brauchen (1.6 GB/Node).
Dann bleibt KV-Cache bei 8G. Braucht AutoRound Quantisierung der MTP-Weights.

## Relevante Patches (in ng17e Image)

- `patch_mtp_bf16_weights.py`: Lädt BF16 MTP-Weights in INT4 Modell
- `patch_mtp_nvfp4_exclusion.py`: Ermöglicht MTP mit NVFP4 Modellen
