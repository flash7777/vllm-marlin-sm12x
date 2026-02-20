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

## Multi-Node: DGX Spark + PGX ThinkStation (2× GB10, SM121, RoCE)

Zwei identische GB10-Knoten ueber ConnectX-7 QSFP56 200 Gbps mit RoCE (RDMA over Converged Ethernet).
NCCL Transport: `NET/IB : Using [0]rocep1s0f0:1/RoCE` (~5-10μs Latenz vs ~200-500μs TCP Socket).

### Qwen3-Coder-30B-A3B INT4 AutoRound — Multi-Node

| Modus | Socket (tok/s) | RoCE (tok/s) | Speedup | Anmerkung |
|---|---:|---:|---:|---|
| PP=2 | 70.4 | 71.3 | +1% | Pipeline-Bubble ist Bottleneck, nicht Netzwerk |
| TP=2 | 45.3 | **91.6** | **+102%** | Schneller als Single-Node vanilla (86.9)! |
| EP=2 | 35.0 | **85.5** | **+144%** | |
| TP=2 + EAGLE3 NST=1 | 60.0 | **94.8** | +58% | Crasht ab 8K Context |

### MiniMax-M2.5 INT4 AutoRound — Multi-Node

| Modus | Socket (tok/s) | RoCE (tok/s) | Speedup |
|---|---:|---:|---:|
| PP=2 | 29.0 | 29.0 | 0% |
| TP=2 | 22.4 | **41.7** | **+86%** |
| EP=2 | 25.9 | **38.7** | +49% |

### Context-Matrix: Qwen3-Coder TP=2 RoCE (tok/s)

| | ctx=0 | ctx=512 | ctx=2K | ctx=8K | ctx=16K |
|------|---:|---:|---:|---:|---:|
| short | 91.6 | 91.5 | 88.1 | 81.3 | 73.5 |
| medium | 91.6 | 91.5 | 88.1 | 81.3 | 73.5 |
| long | 91.6 | 91.5 | 88.1 | 81.3 | 73.5 |

### Context-Matrix: MiniMax-M2.5 TP=2 RoCE (tok/s)

| | ctx=0 | ctx=512 | ctx=2K | ctx=8K | ctx=16K |
|------|---:|---:|---:|---:|---:|
| short | 39.2 | 38.0 | 36.2 | 30.3 | 25.2 |
| medium | 41.7 | 40.2 | 38.5 | 32.3 | 26.2 |
| long | 41.7 | 40.2 | 38.5 | 32.3 | 26.2 |

### RoCE Container-Flags

```bash
--device /dev/infiniband/uverbs0
--device /dev/infiniband/rdma_cm
-e NCCL_IB_DISABLE=0
-e NCCL_IB_HCA=rocep1s0f0
```

Details: [BENCHMARK_MULTINODE.md](BENCHMARK_MULTINODE.md)

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

Qwen3-Coder-30B INT4 W4A16 + EAGLE3 NST=3, `bench.py --context`, runs=2.
Matrix: short (20 tok) / medium (150 tok) / long (400 tok) × Kontextlänge.

| Context | short (tok/s) | medium (tok/s) | long (tok/s) |
|---:|---:|---:|---:|
| 0 | 106 | 180 | 309 |
| 512 | 96 | 171 | 261 |
| 2,048 | 80 | 142 | 220 |
| 8,192 | 54 | 80 | 80 |
| 16,384 | 36 | 53 | 53 |

## bench.py vs llama-benchy Vergleich — Spiegel 2 (RTX PRO 6000, SM120)

Gleiche Config: INT4 W4A16 + EAGLE3 NST=3. Nächstliegende Output-Längen verglichen.

| Context | bench.py short (20) | benchy tg=32 | Ratio | bench.py med (150) | benchy tg=128 | Ratio | bench.py long (400) | benchy tg=512 | Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 106 | 143 | 0.74× | 180 | 143 | 1.26× | 309 | 142 | 2.18× |
| 512 | 96 | 138 | 0.70× | 171 | 134 | 1.28× | 261 | — | — |
| 2,048 | 80 | 112 | 0.71× | 142 | 109 | 1.30× | 220 | — | — |
| 8,192 | 54 | 73 | 0.74× | 80 | 73 | 1.10× | 80 | — | — |
| 16,384 | 36 | 48 | 0.75× | 53 | 47 | 1.13× | 53 | — | — |

- **Short < benchy (~0.74×)**: Nur 3 echte Tokens generiert (Thinking dominiert), TTFT-Overhead relativ groß
- **Medium > benchy (~1.25×)**: Thinking-Tokens erhöhen completion_tokens/wall_time, EAGLE3-Akzeptanz bei Thinking hoch
- **Long >> benchy (2.18×!)**: 400 Output-Tokens = viel Thinking → maximaler EAGLE3-Vorteil bei Thinking-Patterns
- **Konvergenz bei 8K+**: Beide Tools zeigen ~80 tok/s medium, ~73 tok/s benchy — Attention dominiert, EAGLE3/Thinking irrelevant
- **llama-benchy** misst reine Decode-Rate (Streaming, kein Thinking), **bench.py** misst completion_tokens/wall_time (inkl. Thinking)

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
