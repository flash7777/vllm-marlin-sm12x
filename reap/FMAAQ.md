# FMAAQ — Full-Model-Assisted AutoRound Quantization

Quantisierung eines REAP-geprunten MoE-Modells mit dem ungeprunten Original als Referenz.

## Motivation

Standard-AutoRound optimiert per-Layer:
- Input → INT4 Layer → Output vs Input → BF16 Layer → Output
- Minimiert den Quantisierungsfehler pro Layer

Bei einem geprunten Modell fehlen aber Experts. Der Layer-Output weicht schon **vor** der Quantisierung vom Original ab. Standard-AutoRound kompensiert nur den Quantisierungsfehler, nicht den Pruning-Verlust.

FMAAQ optimiert stattdessen:
- Input → INT4 REAP Layer → Output vs Input → **BF16 Original** Layer → Output
- Kompensiert **Pruning + Quantisierung gleichzeitig**

## Kernidee: Direkter Layer-Output-Vergleich

Die Layer-Outputs beider Modelle haben **identische Dimensionen** (hidden_size=4096), unabhängig von der Expert-Anzahl. Intern haben die MoE-Layers unterschiedlich viele Experts (512 vs 333), aber der Output nach dem MoE-Block ist derselbe Tensor-Raum. Deshalb:

- Kein Re-Expansion nötig
- Kein Expert-Mapping nötig für die Kalibrierung (nur für MTP-Transplantation)
- Beide Modelle laufen unverändert, nur der Referenz-Output wird ausgetauscht

## Voraussetzungen

1. **Ungepruntes BF16-Original** (`Qwen/Qwen3.5-397B-A17B`): Dient als Referenz für die Layer-Outputs.

2. **Gepruntes BF16-Modell** (`OpenMOSE/Qwen3.5-REAP-262B-A17B`): Das Modell das quantisiert wird.

3. **Expert-Mapping** (`expert_mapping.json`): Nicht für die Kalibrierung selbst nötig, aber für MTP-Transplantation und Verifikation.

## Ansatz

### AutoRound-Modifikation

AutoRound's Kernloop pro Layer (vereinfacht):

```python
# Standard AutoRound:
ref_output = bf16_layer(input)          # eigener BF16-Layer als Referenz
quant_output = int4_layer(input)
loss = mse(quant_output, ref_output)

# FMAAQ:
ref_output = original_397b_layer(input)  # Teacher-Layer als Referenz
quant_output = int4_layer(input)
loss = mse(quant_output, ref_output)
```

Der Input zu beiden Layers muss identisch sein. Bei Layer 0 ist das trivial (Embedding-Output). Bei tieferen Layers divergieren die Hidden States — der Input muss vom Teacher stammen, nicht vom Student.

### Variante A: Online Teacher (empfohlen)

1. Kalibrationsdaten durch den 397B-Teacher schicken
2. Pro Layer: Input und Output speichern
3. AutoRound nutzt den Teacher-Input als Layer-Input und den Teacher-Output als Referenz
4. Optimiert Scales/Zeros des 262B INT4-Layers

**Vorteil**: Sauberster Vergleich, Teacher-Hidden-States als Ground Truth.
**Aufwand**: 397B BF16 muss einmal vollständig durchlaufen (Layer für Layer streambar).

### Variante B: Cached Teacher

1. 397B einmal durchlaufen, alle Layer-Inputs und -Outputs auf Disk cachen
2. AutoRound kalibriert offline gegen die gecachten Referenzen

**Vorteil**: Teacher nur einmal laden, Student-Kalibrierung iteriert schnell.
**Nachteil**: Storage für 60 Layer-Outputs × Kalibrationsdaten.

### Variante C: Hybrid

1. Ersten N Layers: Student-Input verwenden (divergiert noch wenig)
2. Ab Layer N: Teacher-Input verwenden (Divergenz zu groß)

**Vorteil**: Spart Teacher-Inference für frühe Layers.
**Risiko**: Wo ist der Cutoff? Empirisch bestimmen.

## Intel AutoRound: Anpassungspunkte

AutoRound (github.com/intel/auto-round) hat keine eingebaute Unterstützung für externe Referenzmodelle. Der Code ist aber übersichtlich (Python). Nötige Änderungen:

1. **`autoround.py`**: Im Kalibrierungsloop den Referenz-Output austauschen
2. **Teacher-Loading**: 397B Layer-für-Layer streamen (nicht komplett in RAM)
3. **Input-Alignment**: Teacher-Hidden-States als Input für den Student-Layer verwenden

Geschätzter Aufwand: ~50-100 Zeilen Code-Änderung, kein Fork nötig (Wrapper/Monkey-Patch).

## Erwarteter Qualitätsgewinn

| Methode | Kompensiert | Aufwand |
|---|---|---|
| Standard AutoRound (iters=50) | Nur Quantisierungsfehler | 1× Modell im RAM |
| FMAAQ Variante A | Pruning + Quantisierung | 2× Modell (streambar) |
| FMAAQ Variante B | Pruning + Quantisierung | 1× Modell + Disk-Cache |

Der Gewinn hängt davon ab, wie stark der Pruning-Verlust ist. Bei REAP-262B (35% gepruned) ist der Effekt vermutlich messbar. Benchmark: FMAAQ vs Standard-AutoRound auf MMLU, HumanEval, etc.

## Shared Experts als Anker

Die Shared-Expert-MLPs bleiben in BF16 (via `extra_config`). Da der Shared Expert bei jedem Token aktiv ist, stabilisiert er den Layer-Output. FMAAQ kann die Quantisierung der Routed Experts aggressiver optimieren, weil der Shared Expert als unveränderter Referenzpunkt dient.

## Hardware-Anforderungen

- **Variante A (Online)**: 397B BF16 Layer-für-Layer streamen (~15 GB pro Layer-Paar) + 262B im RAM (~260 GB BF16). Realistisch auf 2× RTX PRO 6000 (192 GB) wenn Layer-weise gestreamt wird.
- **Variante B (Cached)**: 262B im RAM + ~50-100 GB Disk-Cache für Teacher-Outputs.

## Abhängigkeiten

- `REAP-262B-shared-expert.safetensors` (→ `extract_shared_expert.py`)
- AutoRound >= 0.12.0
- `expert_mapping.json` (→ `expert_mapping.py`) — für MTP, nicht für Kalibrierung
