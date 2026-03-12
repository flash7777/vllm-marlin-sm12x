# Qwen3.5 AutoRound INT4: Erkenntnisse & Patches

## Zero-Point: KEIN Problem (KORRIGIERT)

AutoRound auto_gptq packt qzeros als `zp - 1`:
- Tatsächliches zp = 8 (standard für sym=True)
- Gepackter Wert = 7 → 0x77777777

| Modell | quant_method | qzeros (gepackt) | Tatsächliches zp | Marlin uint4b8 |
|--------|-------------|-------------------|-----------------|----------------|
| 122B GPTQ (standard) | gptq | 0x88888888 | 8 | korrekt |
| 262B REAP AutoRound | auto-round | 0x77777777 | 8 (=7+1) | korrekt |

**auto_gptq subtrahiert 1 vom Zero-Point vor dem Packen.** Marlin mit `uint4b8` (bias=8)
ist korrekt für beide Formate. KEINE Korrektur nötig.

## Aktuelle Root Cause: UNBEKANNT

Output ist immer Token 0 ("!"). Logits sind offenbar uniform ~0.
Mögliche Ursachen:
- Per-expert fused weight chunk Dimension/Reihenfolge falsch
- GatedDeltaNet linear attention Implementierung inkompatibel mit INT4
- Fehlende Patches (in_proj_ba ColumnParallel→Replicated)
- MoE Routing/Expert-Indexing Problem

## REAP Per-Expert Fused Weight Loading

### Problem
REAP-262B speichert MoE-Experten als:
```
experts.gate_up_proj.N.qweight   [K/8, 2*intermediate]  (gate+up fused)
experts.down_proj.N.qweight      [intermediate/8, hidden]
```

vLLM erwartet entweder:
- Stacked: `experts.gate_up_proj` → `[num_experts, ...]`
- Separate: `experts.N.gate_proj` + `experts.N.up_proj`

REAP ist weder noch → **KeyError** beim Laden.

### Fix: Per-Expert GPTQ Pattern Detection
```python
per_expert_match = re.search(r'experts\.(gate_up_proj|down_proj)\.(\d+)\.', name)
```

Wenn Match:
1. `expert_id` extrahieren
2. `.N` aus gemapptem Param-Name strippen
3. `gate_up_proj` chunken (dim=-1, entlang N/Output-Dimension)
4. `weight_loader()` direkt mit `shard_id` und `expert_id` aufrufen

### Chunk-Dimension
GPTQ-Layout: `qweight [K/8, N]`, `scales [groups, N]`, `qzeros [groups, N/8]`
Gate+Up sind entlang **N (dim=-1)** konkateniert, daher `chunk(2, dim=-1)`.

## INC vs GPTQ Dispatch

vLLM erkennt `quant_method: "auto-round"` als **INC** (Intel Neural Compressor).
INC dispatcht intern zu GPTQMarlin via `apply_gptq_quant_layer()`.
Ergebnis: identisch mit direktem GPTQ-Marlin, aber `--quantization gptq_marlin` wird überschrieben.

`--quantization gptq_marlin` auf der Kommandozeile wird IGNORIERT wenn die Config `quant_method: "auto-round"` enthält.

## Modell-Details

| Eigenschaft | Wert |
|-------------|------|
| Architektur | Qwen3_5MoeForConditionalGeneration |
| Parameter | 262B total, 17B aktiv |
| Experten | 333, top-k=10 |
| Layers | 60 |
| hidden_size | 4096 |
| moe_intermediate_size | 1024 |
| Attention | Hybrid: linear_attention (GatedDeltaNet) + full_attention |
| Quantisierung | INT4 sym, group_size=128, bits=4 |
| INT4 Größe | ~131 GB |

### Weight Shapes (GPTQ gepackt)
- gate_up_proj.N.qweight: `[512, 2048]` (512=4096/8, 2048=2*1024)
- gate_up_proj.N.scales: `[32, 2048]` (32=4096/128 Gruppen)
- gate_up_proj.N.qzeros: `[32, 256]` (256=2048/8)
- down_proj.N.qweight: `[128, 4096]` (128=1024/8)
- down_proj.N.scales: `[8, 4096]` (8=1024/128)
- down_proj.N.qzeros: `[8, 512]` (512=4096/8)

## Patches (build/)

| Patch | Datei | Zweck |
|-------|-------|-------|
| patch_qwen35_reap_weights.py | qwen3_5.py | Per-expert fused GPTQ loading + AutoRound zp7→zp8 Korrektur |
| patch_marlin_padding.py | gptq_marlin.py | N=32→64 Padding für in_proj_ba bei TP=2 |
| patch_qwen35_compile.py | qwen3_5.py | torch.Size AOT autograd fix |

## Performance

| Modus | tok/s | Anmerkung |
|-------|-------|-----------|
| tbd | tbd | Erster erfolgreicher Run pending |

## Referenzen

- AutoRound: https://github.com/intel/auto-round
- REAP-262B: https://huggingface.co/REAP-262B-A17B-int4-AutoRound (?)
- auto_gptq zp-Konvention: zp=7 vs Standard-GPTQ zp=8
- Marlin uint4b8: 4-bit unsigned, bias=8, Wertebereich [-8, 7]
