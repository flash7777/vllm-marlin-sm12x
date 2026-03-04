# Token Time Map — Qwen3-Coder INT4 TP=2

Tabellarische Aufschluesselung der Token-Generierungszeit.
Jeder Schritt ist mit Messmethode und Konfidenz markiert.

## Konfiguration

| Parameter | Wert |
|-----------|------|
| Modell | Qwen3-Coder-30B-A3B (MoE 128E, top-8, 48 Layer) |
| Quantisierung | INT4 W4A16 (Marlin, AutoRound) |
| Speculation | EAGLE3 NST=1 (1 Draft-Token) |
| TP | 2 (DGX Spark + PGX ThinkStation, RoCE 200 Gbps) |
| GPU | 2× GB10 SM121, je 273 GB/s LPDDR5x |
| NCCL | 2.29.2, ConnectX-7 RoCE, UF17 Eager AllReduce |
| Batch | 1 (single request, decode phase) |
| Context | ~0 (steady-state generation, kein langer Kontext) |

## Gemessener Gesamtwert

```
118 tok/s  →  Token Time = 8.47 ms
```

Quelle: bench.py (long, 400 Tokens, seed=42, n=5), 2026-03-02

## Legende

- **[M]** = Gemessen (Messwerkzeug angegeben)
- **[M/nsys]** = Aus nsys-Profiling abgeleitet (proportionale Zuordnung)
- **[E]** = Geschaetzt / Berechnet (Modell oder Ableitung)
- **[S]** = Hardware-Spezifikation

## Token Time Budget

| # | Phase | pro Aufruf | × Aufrufe | Gesamt | % | Typ | Quelle |
|---|-------|-----------|-----------|--------|---|-----|--------|
| 1 | **AllReduce (NCCL, in CUDA Graph)** | 19.4 µs | 97 | **1.88 ms** | **22.2%** | [M] | nsys NVTX median |
| 1a | ↳ Proxy-Roundtrip (GPU↔CPU↔NIC) | ~10 µs | 97 | 0.97 ms | 11.5% | [E] | Proxy pollt GPU-Flag, postet ibverbs, signalisiert zurueck |
| 1b | ↳ Ring-Overhead (2. Phase) | ~2.5 µs | 97 | 0.24 ms | 2.9% | [M] | UF19v2 spart ~1.5µs/Call |
| 1c | ↳ RoCE RTT | ~3 µs | 97 | 0.29 ms | 3.4% | [M] | ib_write_lat: 3.2µs |
| 1d | ↳ GPU Reduce + Kernel-Launch | ~2 µs | 97 | 0.19 ms | 2.3% | [E] | 4 KiB Add + CUDA Launch |
| 1e | ↳ Datentransfer (4 KiB) | ~0.2 µs | 97 | 0.02 ms | 0.2% | [M] | 4 KiB @ 25 GB/s |
| 2 | **MoE GEMMs (Marlin)** | — | 96 | **2.55 ms** | **30.1%** | [M/nsys] | marlin_moe_wna16 |
| 2a | ↳ Gate+Up GEMM (8 Exp) | 39.1 µs | 48 | 1.88 ms | 22.2% | [M/nsys] | 8×2048×1536 INT4, med |
| 2b | ↳ Down GEMM (8 Exp) | 59.1 µs | 48 | 2.84 ms | — | [M/nsys] | 8×768×2048 INT4, med |
| 2* | ↳ *nsys-proportional* | — | 96 | *2.55 ms* | — | | *Budget-Zuordnung* |
| 3 | **Attention (FlashAttn)** | 25 µs | 48+2 | **1.32 ms** | **15.6%** | [M/nsys] | flash_fwd_splitkv |
| 3a | ↳ pro Layer (Target) | ~25 µs | 48 | 1.20 ms | 14.2% | [M/nsys] | Ctx≈0, med 52.8µs/2tok |
| 3b | ↳ EAGLE3 Attention | ~25 µs | 2 | 0.05 ms | 0.6% | [M/nsys] | Draft verify |
| 4 | **Non-MoE GEMMs (Marlin)** | — | ~150 | **1.44 ms** | **17.0%** | [M/nsys] | Attn-Proj + EAGLE |
| 4a | ↳ QKV-Projektion | 17.8 µs | 48 | 0.85 ms | 10.1% | [M/nsys] | marlin v1, med |
| 4b | ↳ O-Projektion | 10.0 µs | 48 | 0.48 ms | 5.7% | [M/nsys] | marlin v3, med |
| 4c | ↳ EAGLE3 Projektionen | ~13 µs | ~6 | 0.08 ms | 0.9% | [M/nsys] | marlin v4+v5 |
| 5 | **MoE Routing** | — | 48 | **0.28 ms** | **3.3%** | [M/nsys] | |
| 5a | ↳ topkGating | 3.4 µs | 48 | 0.16 ms | 1.9% | [M/nsys] | Top-8 aus 128 Experts |
| 5b | ↳ align_block_size | 3.1 µs | 48 | 0.15 ms | 1.8% | [M/nsys] | Token-Alignment |
| 5c | ↳ count_and_sort | 1.2 µs | 48 | 0.06 ms | 0.7% | [M/nsys] | Expert-Token-Zuordnung |
| 6 | **RMSNorm + Aktivierungen** | — | 96+48 | **0.31 ms** | **3.7%** | [M/nsys] | |
| 6a | ↳ RMSNorm (Triton fused) | 2.7 µs | 96 | 0.26 ms | 3.1% | [M/nsys] | 2 pro Layer, med |
| 6b | ↳ SiLU (act_and_mul) | 4.8 µs | 48 | 0.23 ms | 2.7% | [M/nsys] | 1 pro Layer, med |
| 6* | ↳ *nsys-proportional* | — | — | *0.31 ms* | — | | *Budget-Zuordnung* |
| 7 | **KV-Cache Write** | 2.5 µs | 48 | **0.12 ms** | **1.4%** | [M/nsys] | reshape_and_cache |
| 8 | **CPU Scheduling + Framework** | — | — | **~0.46 ms** | **~5.4%** | [E] | Scheduler+Input+Sampling |
| 9 | **Sonstiges** | — | — | **~0.11 ms** | **~1.3%** | [E] | Embedding, LM-Head, Sampling |
| | **Summe** | | | **8.47 ms** | **100%** | [M] | bench.py |

### Methodik der nsys-Werte

nsys profile waehrend CUDA-Graph-Capture (identische Kernel wie Inference).
AllReduce per NVTX (19.4µs median, 4487 Aufrufe) direkt gemessen.
Restliche GPU-Kernel proportional auf 6.59ms (= 8.47 - 1.88 AllReduce) verteilt:

```
nsys GPU-Kernel-Zeiten (aggregiert, ~46 Warmup-Passes):
  MoE Marlin GEMM:     104.7 ms  (38.7% der Compute-Zeit)
  Flash Attention:       54.3 ms  (20.1%)
  Non-MoE Marlin GEMM:  59.3 ms  (21.9%)
  MoE Routing:           11.4 ms  ( 4.2%)
  RMSNorm+SiLU+Fused:   24.0 ms  ( 8.9%)
  KV-Cache Write:         3.3 ms  ( 1.2%)
  EAGLE/CUTLASS misc:    13.6 ms  ( 5.0%)
                        --------
  Compute Total:        270.6 ms  (100%)
```

Proportionale Zuordnung auf 6.59ms verfuegbare GPU-Zeit:
6.59 × Anteil = Wert pro Token-Step.
CPU-Overhead (0.61ms) = Token-Zeit - GPU-Kernelzeit = 8.47 - 7.86 = 0.61ms.

Hinweis: Piecewise CUDA Graphs kompilieren den FX-Graph in Segmente,
aber der CUDA-Graph-Capture erfasst ALLES (Compute + NCCL AllReduce)
in EINEN grossen Graph. Replay = 1× cudaGraphLaunch(), KEIN Python-Loop.
Python-Overhead entsteht nur zwischen Token-Steps (Scheduler, Sampling,
EAGLE3-Orchestrierung), NICHT innerhalb der 97 AllReduce-Iterationen.

## nsys Kernel-Details

### Top GPU-Kernels (nsys profile, CUDA-Graph-Capture-Phase)

| # | Kernel | Instances | Avg (µs) | Med (µs) | Total (ms) | % GPU |
|---|--------|-----------|----------|----------|------------|-------|
| 1 | ncclAllReduce_Sum_bf16_RING_LL | 2353 | 89.0 | 65.4 | 209.4 | 31.7% |
| 2 | marlin_moe_wna16 (Variante 1) | 1536 | 41.6 | 39.1 | 63.9 | 9.7% |
| 3 | flash_fwd_splitkv_kernel | 1050 | 51.7 | 52.8 | 54.3 | 8.2% |
| 4 | marlin_moe_wna16 (Variante 2) | 564 | 59.3 | 59.1 | 33.4 | 5.1% |
| 5 | marlin (non-MoE, Variante 1) | 960 | 19.3 | 17.8 | 18.5 | 2.8% |
| 6 | marlin (non-MoE, Variante 2) | 563 | 30.4 | 28.2 | 17.1 | 2.6% |
| 7 | marlin_moe_wna16 (Variante 3) | 96 | 76.6 | 78.7 | 7.4 | 1.1% |
| 8 | marlin (non-MoE, Variante 3) | 563 | 12.1 | 10.0 | 6.8 | 1.0% |
| 9 | reduce_kernel (RMSNorm) | 1098 | 6.2 | 5.7 | 6.8 | 1.0% |
| 10 | act_and_mul_kernel (SiLU) | 1098 | 5.1 | 4.8 | 5.6 | 0.9% |
| 11 | marlin (non-MoE, Variante 4) | 384 | 14.6 | 13.7 | 5.6 | 0.8% |
| 12 | marlin (non-MoE, Variante 5) | 480 | 10.7 | 12.4 | 5.2 | 0.8% |
| 13 | topkGating (MoE) | 1098 | 3.8 | 3.4 | 4.1 | 0.6% |
| 14 | moe_align_block_size | 1098 | 3.3 | 3.1 | 3.6 | 0.5% |
| 15 | triton_red_fused_rmsnorm | 1211 | 3.0 | 2.7 | 3.6 | 0.5% |
| 16 | triton_red_fused_marlin_rmsnorm | 1097 | 3.3 | 2.8 | 3.6 | 0.5% |
| 17 | reshape_and_cache_flash | 1142 | 2.9 | 2.5 | 3.3 | 0.5% |
| 18 | count_and_sort_expert_tokens | 1098 | 1.4 | 1.2 | 1.5 | 0.2% |

Anmerkungen:
- NCCL AllReduce GPU-Kernelzeit (65-89µs) ist HOEHER als NVTX/CUDA-Events (19µs)
  weil nsys Tracing-Overhead die GPU-Kernel aufblaeht
- NVTX Median (19.4µs) bestaetigt unabhaengige CUDA-Events-Messung (19.0µs)
- Relative Proportionen der Compute-Kernels sind zuverlaessig
- Absolute GPU-Kernelzeiten um ~2-3× durch nsys-Overhead inflated
- AllReduce ist IN den CUDA Graph captured (nicht eager zwischen Segmenten)
- Piecewise = Compilation-Konzept; Execution = 1× cudaGraphLaunch()
- NCCL-Overhead (~17µs) setzt sich zusammen aus: Proxy-Roundtrip GPU↔CPU↔NIC (~10µs),
  Ring 2. Phase (~2.5µs), RoCE RTT (~3µs), GPU Kernel-Launch+Add (~2µs)
- Proxy-Thread ist Hauptquelle: GB10 hat kein GPUDirect → NIC kann nicht direkt
  aus GPU-Speicher lesen → CPU-Proxy vermittelt mit ~10µs Overhead
- UF19v2 ncclSend/Recv eliminiert Ring-Phase → nur ~1.5µs/Call Verbesserung

### NVTX Zusammenfassung

| Range | Instances | Avg (µs) | Med (µs) |
|-------|-----------|----------|----------|
| NCCL:ncclAllReduce | 4487 | 20.0 | **19.4** |
| NCCL:ncclAllGather | 1 | 33.2 | 33.2 |

## Detailaufschluesselung pro Transformer-Layer (×48)

Ein Decode-Step fuer Batch=1 (2 Tokens: 1 real + 1 draft wegen EAGLE3 NST=1).
AllReduce numel=4096 (2 × hidden_size=2048), 8 KiB BF16.

| Schritt | Operation | Gewichte | Zeit | Typ | Anmerkung |
|---------|-----------|----------|------|-----|-----------|
| 1 | RMSNorm (Attn) | — | ~3 µs | [M/nsys] | triton_red_fused, median 2.7µs |
| 2 | QKV-Projektion | 3 × h×h INT4 | ~18 µs | [M/nsys] | marlin non-MoE v1, median 17.8µs |
| 3 | Rotary Embedding | — | <1 µs | [E] | In-place, kein Memory-Read |
| 4 | FlashAttention | KV-Cache | ~25 µs | [M/nsys] | flash_fwd_splitkv, median 52.8µs / 2 Tokens |
| 5 | Output-Projektion | h×h INT4 | ~10 µs | [M/nsys] | marlin non-MoE v3, median 10.0µs |
| 6 | **AllReduce #1** | — | **19 µs** | **[M]** | nsys NVTX median 19.4µs |
| 7 | Residual Add | — | <1 µs | [E] | Elementweise |
| 8 | RMSNorm (FFN) | — | ~3 µs | [M/nsys] | triton_red_fused_marlin, median 2.8µs |
| 9 | MoE Gate + TopK | h×E INT4 | ~6 µs | [M/nsys] | topkGating 3.4 + align 3.1 + sort 1.2µs |
| 10 | Gate+Up GEMM (8 Exp) | 8 × h×N INT4 | ~39 µs | [M/nsys] | marlin_moe_wna16 v1, median 39.1µs |
| 11 | SiLU Aktivierung | — | ~5 µs | [M/nsys] | act_and_mul_kernel, median 4.8µs |
| 12 | Down GEMM (8 Exp) | 8 × N×h INT4 | ~59 µs | [M/nsys] | marlin_moe_wna16 v2, median 59.1µs |
| 13 | Unpermute + Combine | — | ~2 µs | [E] | Weighted sum der Expert-Outputs |
| 14 | **AllReduce #2** | — | **19 µs** | **[M]** | nsys NVTX median 19.4µs |
| 15 | KV-Cache Write | — | ~3 µs | [M/nsys] | reshape_and_cache_flash, median 2.5µs |
| 16 | Residual Add | — | <1 µs | [E] | Elementweise |
| | **Summe pro Layer** | | **~212 µs** | | |
| | **× 48 Layer** | | **~10.2 ms** | | Inkl. nsys-Overhead auf Kernel-Zeiten |

**Wichtig**: Die Summe pro Layer (212µs) ist HOEHER als die reale Zeit (~130µs),
weil nsys ~2-3× Overhead auf GPU-Kernel-Zeiten addiert.
Die reale Layer-Zeit ergibt sich aus: (8.47ms - 1.88ms AllReduce - 0.61ms CPU) / 48 = **125µs/Layer**.

## EAGLE3 Overhead

EAGLE3 Kernels sind in den nsys-Daten nicht separat markiert.
Geschaetzte Zuordnung basierend auf Kernel-Varianten und Instance-Counts:

| Phase | Zeit | Typ | Anmerkung |
|-------|------|-----|-----------|
| EAGLE Marlin GEMMs | ~0.4 ms | [M/nsys] | non-MoE v4+v5 (384+480 inst, EAGLE Proj) |
| EAGLE Attention | ~0.1 ms | [M/nsys] | Anteil flash_fwd_splitkv |
| EAGLE cutlass/nvjet | ~0.3 ms | [M/nsys] | nvjet_sm121 Varianten (LM-Head) |
| Sampling + Accept/Reject | ~0.2 ms | [E] | Logit-Vergleich |
| **EAGLE3 Total** | **~1.0 ms** | [M/nsys]+[E] | Inkl. in MoE/Attention-Budgets oben |

## AllReduce: Gemessene Varianten

| Methode | pro Aufruf | × 97 | tok/s | Typ | Datum |
|---------|-----------|------|-------|-----|-------|
| NCCL AllReduce (Ring) | 19.0 µs | 1.84 ms | 116.6 | [M] CUDA Events | 2026-03-02 |
| NCCL AllReduce (Ring) | 19.4 µs | 1.88 ms | — | [M] nsys NVTX | 2026-03-02 |
| ncclSend/Recv P2P (UF19v2) | ~17 µs | ~1.65 ms | 118.6 | [M] bench.py | 2026-03-02 |
| NCCL in CUDA Graph | 813-1032 µs | 78.9-100 ms | — | [M] CUDA Events | 2025 |
| Raw RDMA Write (ibverbs) | 3.2 µs | 0.31 ms | — | [M] ib_write_lat | 2025 |
| Raw ibverbs + CPU-Add | 12.1 µs | 1.17 ms | — | [M] CUDA Events | 2025 |

## Bandwidth-Modell (Plausibilitaetspruefung)

```
Aktive Parameter:    3B (MoE top-8 von 128)
INT4 Gewichte:       3B × 0.5 B           = 1.50 GB
Scales (group=128):  3B / 128 × 2 B       = 0.05 GB
Total Weight-Read:                         ≈ 1.55 GB

Aggregierte Bandbreite TP=2:  2 × 273 GB/s = 546 GB/s
Reine Lesezeit:               1.55 / 546   = 2.84 ms

Gemessene Token-Zeit:                        8.47 ms
Bandbreiten-Effizienz:        2.84 / 8.47  = 33.5%
```

nsys zeigt: GEMM-Kernelzeit (MoE+non-MoE) = 3.99 ms > 2.84 ms BW-Modell.
Differenz: Marlin Dequant-Compute + Kernel-Launch-Overhead + L2-Cache-Effekte.

Overhead-Quellen fuer die fehlenden 66.5%:
- AllReduce: 1.88 ms (22.2%) — [M] nsys NVTX
- Attention: 1.32 ms (15.6%) — [M/nsys]
- GEMM Overhead vs BW-Modell: 1.15 ms (13.6%) — [M/nsys]
- MoE Routing + Norms + SiLU: 0.59 ms (7.0%) — [M/nsys]
- CPU/Framework: 0.61 ms (7.2%) — [E]
- KV-Cache + Sonstiges: 0.08 ms (0.9%) — [M/nsys]

## Kontextskalierung (gemessen)

| Kontext | tok/s | Token-Zeit | Degradation | Typ |
|---------|-------|-----------|-------------|-----|
| 0 | 118 | 8.47 ms | — | [M] bench.py |
| 512 | ~115 | ~8.70 ms | -2.6% | [M] context matrix |
| 2K | ~111 | ~9.01 ms | -6.0% | [M] context matrix |
| 8K | ~102 | ~9.80 ms | -13.6% | [M] context matrix |
| 16K | ~92 | ~10.87 ms | -22.1% | [M] context matrix |

Treiber: KV-Cache Attention (FlashAttn Compute steigt linear mit Kontext).

## TODO: Verbleibende Messungen

| Was | Messmethode | Prioritaet | Status |
|-----|------------|------------|--------|
| GEMM-Zeiten pro Layer | nsys | HOCH | DONE (proportional aus nsys) |
| FlashAttention pro Layer | nsys | HOCH | DONE (proportional aus nsys) |
| MoE Routing + Permutation | nsys | MITTEL | DONE (proportional aus nsys) |
| EAGLE3 Draft-Forward separat | nsys mit NVTX | MITTEL | Teilweise (nicht separierbar) |
| CPU Scheduling Overhead | CPU profiler | NIEDRIG | Residual: 0.61ms |
| Piecewise Split Kosten | nsys Graph-Segmente | ERLEDIGT | Kein Python-Overhead (alles in CUDA Graph) |
| Per-Layer CUDA Events | Custom CUDA Events in Code | OPTIONAL | Wuerde nsys-Overhead eliminieren |

## nsys Profiling Setup

```
nsys profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --sample=none \
  --delay=180 --duration=15 \
  --stats=true \
  torchrun ... (TP=2, NCCL baseline, Qwen3-Coder INT4 + EAGLE3 NST=1)

Capture-Fenster: CUDA-Graph-Capture-Phase (~15s)
→ Kernels identisch zu Inference-Decode
→ NVTX fuer AllReduce direkt gemessen
→ GPU-Kernel-Zeiten ~2-3× inflated durch nsys
→ Proportionen zuverlaessig fuer Budget-Zuordnung
```
