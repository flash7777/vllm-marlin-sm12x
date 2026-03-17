# REAP-262B Tooling

Tools zum Konvertieren von `OpenMOSE/Qwen3.5-REAP-262B-A17B` (gepruned aus `Qwen/Qwen3.5-397B-A17B`) in ein vLLM-ladbares INT4-Format mit originalen BF16 Shared-Expert-Weights.

## Problem

Die veröffentlichten INT4-AutoRound-Quants der REAP-262B haben die Shared-Expert-MLPs mitquantisiert. vLLM erwartet diese aber in FP16 (wie bei den funktionierenden 122B/397B Quants). Zusätzlich fehlen MTP-Layer und ein Expert-Mapping zum Original.

## Scripts

### `extract_shared_expert.py`

Extrahiert die BF16 Shared-Expert-Weights (gate_proj, up_proj, down_proj) aus dem BF16-Original (`OpenMOSE/Qwen3.5-REAP-262B-A17B`). Lädt Shard für Shard, extrahiert 3 Tensoren pro Layer, löscht den Shard.

- **Input**: HuggingFace Repo `OpenMOSE/Qwen3.5-REAP-262B-A17B`
- **Output**: `REAP-262B-shared-expert.safetensors` (~1.4 GB, 180 Tensoren)
- **Laufzeit**: ~1h (60 × 8 GB Shards)
- **Voraussetzung**: `safetensors`, `huggingface-hub`, `torch` (oder numpy + ml_dtypes)

### `merge_shared_expert.py`

Ersetzt die INT4-quantisierten Shared-Expert-Tensoren (qweight/qzeros/scales) in der REAP-262B-RTX durch die originalen BF16-Weights. Patcht `quantize_config.json` und `config.json`.

- **Input**: `Qwen3.5-REAP-262B-A17B-int4-AutoRound-RTX/` + `REAP-262B-shared-expert.safetensors`
- **Output**: `Qwen3.5-REAP-262B-A17B-int4-AutoRound-RTX-fixed/` (vLLM-ladbar)
- **Laufzeit**: ~30 min (26 Shards rewriten)
- **Voraussetzung**: `safetensors`, `torch`

### `expert_mapping.py`

Erstellt ein per-Layer Mapping von REAP Expert-Indices auf Original-397B Expert-Indices. Vergleicht uint16-Fingerprints der gestackten `experts.down_proj` Tensoren beider Modelle.

- **Input**: HuggingFace Repos `OpenMOSE/Qwen3.5-REAP-262B-A17B` + `Qwen/Qwen3.5-397B-A17B`
- **Output**: `expert_mapping.json` (pruned_idx → orig_idx pro Layer), `expert_mapping_pruned.json` (entfernte orig_indices pro Layer)
- **Laufzeit**: ~2h (60 + 60 Shards à 8 GB)
- **Voraussetzung**: `safetensors`, `huggingface-hub`, `numpy`, `ml_dtypes`

Das Mapping wird benötigt für:
- MTP-Layer-Transplantation (397B MTP Experts auf 333 prunen)
- Quantisierung mit Original-Referenz (AutoRound gegen ungepruntes Modell kalibrieren)
- Verifikation der REAP-Pruning-Ergebnisse

## Workflow

```
1. extract_shared_expert.py     → REAP-262B-shared-expert.safetensors
2. merge_shared_expert.py       → Qwen3.5-REAP-262B-A17B-int4-AutoRound-RTX-fixed/
3. expert_mapping.py            → expert_mapping.json
4. (TODO) prune_mtp.py          → MTP-Layer mit 333 Experts für REAP-262B
```

## Tensor-Formate

Beide Modelle (397B und REAP-262B) nutzen gestackte Expert-Tensoren:
- `experts.down_proj` → `[num_experts, 4096, 1024]` BF16
- `experts.gate_up_proj` → `[num_experts, 2048, 4096]` BF16 (fused gate+up)

REAP pruned per-Layer unterschiedlich viele Experts (global 512 → 333) und re-indiziert die verbleibenden.

## Infrastruktur

Die Scripts laufen auf brandis.eu im `tensortools` Container (python:3.12-slim + safetensors + numpy + ml_dtypes). Setup unter `/data/tensortools/` (Dockerfile, build.sh, start.sh). Ergebnisse werden nach DGX transferiert.
