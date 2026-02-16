# Benchmark-Ergebnisse: Marlin INT4 AutoRound auf SM120/SM121

Zwei Benchmark-Tools:
- **bench.py**: End-to-end mit echten Prompts + Math-Accuracy (deterministic, seed=42, temperature=0)
- **[llama-benchy](https://github.com/eugr/llama-benchy)**: Prefill (pp) vs Decode (tg) isoliert bei verschiedenen Kontextlängen

Image: `vllm-next` (vLLM 26.01 base, CUTLASS 4.3.5, SM120a/SM121a).

## Qwen3-Coder-30B-A3B — DGX Spark (GB10, SM121, 273 GB/s)

| Quant | Spec | GEMM Kernel | tok/s (med) | tok/s (long) | Math (50) |
|---|---|---|---:|---:|---:|
| **INT4 W4A16** | **EAGLE3 NST=3** | **Marlin FP16 mma.sync** | **69.1** | **94.6** | **78%** |
| INT4 W4A16 | — | Marlin FP16 mma.sync | 86.4 | 86.9 | 78% |
| INT4 W4A8 | — | Marlin FP8 mma.sync k=32 | 86.0 | 86.7 | 78% |
| NVFP4 | EAGLE3 | FlashInfer CUTLASS | — | 68.1 | 72% |
| NVFP4 | — | FlashInfer CUTLASS | — | 65.0 | 74% |
| FP8 dynamic | EAGLE3 | Triton MoE | — | 51.0 | ~80% |
| FP8 dynamic | — | Triton MoE | — | 50.5 | ~80% |
| BF16 | EAGLE3 | — | — | 28.5 | ~80% |
| BF16 | — | — | — | 30.6 | ~80% |

## Qwen3-Coder-30B-A3B — Spiegel 2 (RTX PRO 6000, SM120, 1800 GB/s)

| Quant | Spec | GEMM Kernel | tok/s (med) | tok/s (long) | Math (50) |
|---|---|---|---:|---:|---:|
| **INT4 W4A16** | **EAGLE3 NST=3** | **Marlin FP16 mma.sync** | **178.0** | **298.2** | **78%** |
| INT4 W4A16 | — | Marlin FP16 mma.sync | 210.5 | 211.7 | 78% |
| INT4 W4A8 | — | Marlin FP8 mma.sync k=32 | 210.5 | 211.8 | 78% |
| NVFP4 | EAGLE3 | FlashInfer CUTLASS | — | 183.4 | — |
| NVFP4 | — | FlashInfer CUTLASS | 157.9 | — | — |
| FP8 dynamic | EAGLE3 | Triton MoE | — | 166.5 | — |
| FP8 dynamic | — | Triton MoE | 135.7 | — | — |
| BF16 | EAGLE3 | — | — | 147.4 | — |
| BF16 | — | — | 140.9 | — | — |

## llama-benchy: Prefill vs Decode — Spiegel 2 (RTX PRO 6000, SM120)

Qwen3-Coder-30B INT4 W4A16 + EAGLE3 NST=3, llama-benchy 0.3.1, runs=2.

| Prompt (pp) | Prefill (tok/s) | Decode tg=32 (tok/s) | Decode tg=128 (tok/s) | TTFT (ms) |
|---:|---:|---:|---:|---:|
| 512 | 16,181 ± 1,330 | 137.5 ± 3.6 | 133.9 ± 0.3 | 30 |
| 2,048 | 23,333 ± 74 | 112.2 ± 2.2 | 108.5 ± 0.0 | 76 |
| 8,192 | 22,632 ± 44 | 73.4 ± 0.3 | 72.7 ± 0.3 | 305 |

- Prefill: ~23K tok/s, nicht der Bottleneck
- Decode degradiert mit Kontextlänge: 137→73 tok/s (512→8K pp) — KV-Cache Attention wird teurer
- TTFT: 30ms (512 tok) bis 305ms (8K tok)

## Analyse

### INT4 AutoRound ist schnellste Quantisierung auf DGX Spark

- INT4 Vanilla: **86.9 tok/s** → +34% vs NVFP4 (65.0), +72% vs FP8 (50.5), +184% vs BF16 (30.6)
- INT4 + EAGLE3: **94.6 tok/s** → +39% vs NVFP4+EAGLE3 (68.1), +85% vs FP8+EAGLE3 (51.0)

### INT4 + EAGLE3 = Bestwert auf Spiegel 2

- **298.2 tok/s** (long) — neuer Bestwert für Qwen3-Coder auf RTX PRO 6000
- +41% vs INT4 Vanilla (211.7), +63% vs NVFP4+EAGLE3 (183.4)

### W4A8 = kein Speedup bei Batch=1

W4A8 (FP8 MMA k=32) bringt keinen Speedup vs W4A16 (FP16 MMA k=16) bei Batch=1.
Grund: Memory-bandwidth-bound. Weights sind in beiden Fällen INT4 gepackt (gleiche Bandwidth).
W4A8 bringt erst bei hohem Batch-Size Compute-Speedup (2× via FP8 k=32).

### Warum INT4 schneller als NVFP4

- Marlin INT4: Dense GEMM, ein Kernel pro Linear-Layer, minimaler Overhead
- NVFP4: MoE-Routing + Grouped GEMM + Expert-Dispatch → mehr Kernel-Launches
- INT4 halbiert Weight-Reads vs FP8 → ~2× bei Weight-Bandwidth-Bound

### Math-Accuracy

- INT4 W4A16: 78% (Qwen3-Coder), 88% (GLM) — akzeptabel
- BF16/FP8: ~80% (Qwen3-Coder), 94% (GLM) — Baseline
- NVFP4: 72-74% — niedrigste Accuracy

## Env-Vars

| Variable | Wert | Beschreibung |
|---|---|---|
| `VLLM_MARLIN_INPUT_DTYPE` | `fp8` | Aktiviert W4A8 (INT4→FP8 Dequant + FP8 MMA k=32) |
| `VLLM_MLA_DISABLE` | `1` | MLA deaktivieren (Qwen3/GLM MoE) |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` | FlashInfer Version-Mismatch umgehen |
