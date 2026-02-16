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
| **INT4 W4A16** | **EAGLE3 NST=3** | **Marlin FP16 mma.sync** | **179.7** | **303.1** | **78%** |
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

| Prompt (pp) | Prefill (tok/s) | Decode tg=32 (tok/s) | Decode tg=128 (tok/s) | Decode tg=512 (tok/s) | TTFT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 (zero) | 69 | 143.1 ± 1.1 | 142.6 ± 0.5 | 141.6 ± 0.1 | 15 |
| 512 | 16,181 ± 1,330 | 137.5 ± 3.6 | 133.9 ± 0.3 | — | 30 |
| 2,048 | 23,333 ± 74 | 112.2 ± 2.2 | 108.5 ± 0.0 | — | 76 |
| 8,192 | 22,632 ± 44 | 73.4 ± 0.3 | 72.7 ± 0.3 | — | 305 |
| 16,384 | 17,103 ± 86 | 47.5 ± 0.2 | 47.0 ± 0.3 | — | 850 |

- **Zero-Context Decode: 143 tok/s** — reine Marlin+EAGLE3 Geschwindigkeit ohne Attention-Last
- Decode degradiert mit Kontextlänge: 143→73→47 tok/s (0→8K→16K pp) — KV-Cache Attention dominiert
- Prefill: ~23K tok/s, nicht der Bottleneck
- TTFT: 15ms (zero) bis 305ms (8K tok)

## bench.py Context-Scaling — Spiegel 2 (RTX PRO 6000, SM120)

Qwen3-Coder-30B INT4 W4A16 + EAGLE3 NST=3, `bench.py --context`, tg=128, runs=2.

| Context | bench.py (tok/s) | llama-benchy (tok/s) | Verhältnis |
|---:|---:|---:|---:|
| 0 | 207.6 | 143 | 1.45× |
| 512 | 203.8 | 134 | 1.52× |
| 2,048 | 171.0 | 109 | 1.57× |
| 8,192 | 82.8 | 73 | 1.13× |
| 16,384 | 53.7 | 47 | 1.14× |

- Bei kurzem Kontext: bench.py ~1.5× höher — EAGLE3 profitiert von Thinking-Tokens (vorhersagbar, hohe Akzeptanz)
- Bei langem Kontext: Beide konvergieren (~1.08×) — Attention dominiert, EAGLE3-Vorteil schwindet
- llama-benchy misst reine Decode-Rate (Streaming), bench.py misst completion_tokens/wall_time (inkl. Thinking)

## Analyse

### INT4 AutoRound ist schnellste Quantisierung auf DGX Spark

- INT4 Vanilla: **86.9 tok/s** → +34% vs NVFP4 (65.0), +72% vs FP8 (50.5), +184% vs BF16 (30.6)
- INT4 + EAGLE3: **94.6 tok/s** → +39% vs NVFP4+EAGLE3 (68.1), +85% vs FP8+EAGLE3 (51.0)

### INT4 + EAGLE3 = Bestwert auf Spiegel 2

- **303.1 tok/s** (long) — neuer Bestwert für Qwen3-Coder auf RTX PRO 6000
- +43% vs INT4 Vanilla (211.7), +65% vs NVFP4+EAGLE3 (183.4)

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
