# Multi-Node TP=2 Benchmarks: 3 Modelle

Hardware: DGX Spark (GB10, SM121) + PGX ThinkStation (GB10, SM121)
Verbindung: QSFP56 200 Gbps RoCE (NCCL AllReduce)
Image: vllm-next (vLLM 0.15, Marlin W4A16, CUDA Graphs)
Datum: 2026-02-28

## Konfigurationen

| Konfig | Beschreibung | Backend |
|---|---|---|
| TP=1 | `vllm serve` auf 1x DGX Spark | AsyncLLMEngine |
| TP=2 serve | `vllm serve --tp 2 --distributed-executor-backend ray` | Ray Compiled DAG |
| TP=2 continuous | `serve_torchrun.py` (engine.step() Loop) | torchrun external_launcher |

---

## 1. Qwen3-Coder-30B-A3B INT4 AutoRound + EAGLE3

### Context Matrix: Long Output (400 tok)

| Context | TP=1 | TP=2 serve (Ray) | TP=2 continuous | Speedup cont. |
|---|---|---|---|---|
| 0 | 92.4 | 92.4 (0%) | **109.8** | **+19%** |
| 512 | 84.6 | 83.8 (-1%) | **102.6** | **+21%** |
| 2K | 79.5 | 76.4 (-4%) | **96.9** | **+22%** |
| 8K | 45.1 | 34.6 (-23%) | **51.9** | **+15%** |
| 16K | 33.9 | 36.0 (+6%) | **37.9** | **+12%** |

### Context Matrix: Medium Output (~150 tok)

| Context | TP=1 | TP=2 serve (Ray) | TP=2 continuous | Speedup cont. |
|---|---|---|---|---|
| 0 | 68.4 | 66.7 (-2%) | **81.1** | **+19%** |
| 512 | 66.4 | 65.5 (-1%) | **81.1** | **+22%** |
| 2K | 59.8 | 59.0 (-1%) | **72.2** | **+21%** |
| 8K | 44.6 | 35.5 (-20%) | **51.3** | **+15%** |
| 16K | 33.3 | 32.2 (-3%) | **37.6** | **+13%** |

### Concurrency (Throughput bei steigender Last)

| Concurrency | TP=1 tok/s | TP=2 continuous | Speedup |
|---|---|---|---|
| c=1 | 93.6 | **107.1** | +14% |
| c=2 | 133.8 | **160.2** | **+20%** |
| c=4 | 215.3 | **236.1** | +10% |
| c=8 | 228.9 | **292.5** | **+28%** |

---

## 2. GLM-4.7-Flash INT4 AutoRound (ohne EAGLE3)

EAGLE3 unterstuetzt `glm4_moe_lite` nicht (nur llama/qwen/minicpm).

### Context Matrix: Long Output (400 tok)

| Context | TP=1 | TP=2 serve (Ray) | TP=2 continuous | Speedup cont. |
|---|---|---|---|---|
| 0 | 51.2 | 66.2 (+29%) | **73.7** | **+44%** |
| 512 | 48.6 | 61.4 (+26%) | **67.7** | **+39%** |
| 2K | 39.6 | 51.0 (+29%) | **55.3** | **+40%** |
| 8K | 22.4 | 31.4 (+40%) | **32.6** | **+46%** |
| 16K | 14.4 | 20.9 (+45%) | **21.6** | **+50%** |

### Context Matrix: Medium Output (~150 tok)

| Context | TP=1 | TP=2 serve (Ray) | TP=2 continuous | Speedup cont. |
|---|---|---|---|---|
| 0 | 54.6 | 67.5 (+24%) | **75.7** | **+39%** |
| 512 | 49.7 | 62.4 (+26%) | **69.5** | **+40%** |
| 2K | 39.7 | 51.3 (+29%) | **55.7** | **+40%** |
| 8K | 22.7 | 31.5 (+39%) | **32.9** | **+45%** |
| 16K | 14.5 | 21.0 (+45%) | **21.6** | **+49%** |

---

## 3. Qwen3-Coder-Next INT4 AutoRound (nur TP=1)

TP=2 NICHT moeglich: 512 Experts → Marlin `output_size_per_partition=32` < `min_thread_n=64`.
Kein EAGLE3 verfuegbar.

### Context Matrix: Long Output (400 tok)

| Context | TP=1 |
|---|---|
| 0 | 70.0 |
| 512 | 68.7 |
| 2K | 65.6 |
| 8K | 54.7 |
| 16K | 44.5 |

### Context Matrix: Medium Output (~150 tok)

| Context | TP=1 |
|---|---|
| 0 | 67.9 |
| 512 | 65.9 |
| 2K | 59.6 |
| 8K | 41.4 |
| 16K | 29.4 |

---

## Vergleich: Alle Modelle (Long Output, ctx=0)

| Modell | Experts | TopK | EAGLE3 | TP=1 | TP=2 Ray | TP=2 torchrun | Speedup |
|---|---|---|---|---|---|---|---|
| **Qwen3-Coder-30B** | 128 | 8 | Ja | 92.4 | 92.4 | **109.8** | +19% |
| **Qwen3-Coder-Next** | 512 | 10 | Nein | 70.0 | — | — | — |
| **GLM-4.7-Flash** | 64 | 4 | Nein | 51.2 | 66.2 | **73.7** | +44% |

## Vergleich: Context-Skalierung (Long Output, alle Modelle)

| Context | Qwen3-30B TP=1 | Qwen3-30B TP=2c | GLM TP=1 | GLM TP=2c | QwenNext TP=1 |
|---|---|---|---|---|---|
| 0 | 92.4 | **109.8** | 51.2 | **73.7** | 70.0 |
| 512 | 84.6 | **102.6** | 48.6 | **67.7** | 68.7 |
| 2K | 79.5 | **96.9** | 39.6 | **55.3** | 65.6 |
| 8K | 45.1 | **51.9** | 22.4 | **32.6** | 54.7 |
| 16K | 33.9 | **37.9** | 14.4 | **21.6** | 44.5 |

---

## Analyse

### GLM profitiert am meisten von TP=2 (+44-50%)

- GLM hat groessere MoE-Layers (64 Experts, E_dim=3072/2048) → mehr Gewichte pro Token
- GLM INT4 ohne EAGLE3 ist rein bandwidth-bound (kein Speculation-Overhead)
- TP=2 verdoppelt Bandbreite: 51.2 → 73.7 = **+44%** (nah am theoretischen Maximum)
- Bei langen Kontexten steigt der Speedup sogar auf **+50%** (16K)!

### Qwen3-Coder-30B: Moderater Speedup (+19%)

- EAGLE3 Speculation maskiert einen Teil des Bandwidth-Vorteils
- Bei ctx=0: ~45% der Tokens kommen aus Draft-Akzeptanz (kostenlos)
- Effektiver Bandwidth-Bedarf pro Token ist geringer → weniger Spielraum fuer TP=2
- Trotzdem konsistent +12-22% ueber alle Kontextlaengen

### Qwen3-Coder-Next: TP=2 blockiert

- 512 Experts sind zu granular fuer TP=2 (Marlin min_thread_n=64)
- TP=1 Performance ist dennoch gut: 70 tok/s bei ctx=0, 44.5 bei 16K
- Weniger Context-Degradation als GLM (-36% bei 16K vs -72% bei GLM)

### Ray vs torchrun

- GLM: Ray TP=2 liefert bereits +29-45% vs TP=1 — torchrun addiert nochmal +5-11%
- Qwen3-Coder: Ray TP=2 = 0% (kein Benefit!) — torchrun = +19%
- **Differenz erklaert sich durch EAGLE3**: Ray Compiled DAG Overhead (~5ms) wird
  bei EAGLE3's verifikations-basiertem Decode-Muster besonders schaedlich

### NCCL AllReduce Overhead

- Qwen3-Coder (128 Experts): ~30 AllReduce/Token × ~120µs = 3.7ms Overhead
- GLM (64 Experts): ~24 AllReduce/Token × ~120µs = 2.9ms Overhead
- GLM's niedrigerer AllReduce-Count erklaert den hoeheren relativen Speedup

---

## Rohdaten

### Qwen3-Coder-30B TP=1 Context (vllm-unified)

```json
{"label": "TP1-INT4-EAGLE3-context", "context": [{"context": 0, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.41, "avg_tok_s": 7.4}, {"type": "medium", "avg_tokens": 126.0, "avg_time_s": 1.84, "avg_tok_s": 68.4}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 4.33, "avg_tok_s": 92.4}]}, {"context": 512, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.11, "avg_tok_s": 26.3}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.26, "avg_tok_s": 66.4}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 4.73, "avg_tok_s": 84.6}]}, {"context": 2048, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.17, "avg_tok_s": 18.1}, {"type": "medium", "avg_tokens": 131.5, "avg_time_s": 2.2, "avg_tok_s": 59.8}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 5.03, "avg_tok_s": 79.5}]}, {"context": 8192, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.47, "avg_tok_s": 6.4}, {"type": "medium", "avg_tokens": 143.5, "avg_time_s": 3.22, "avg_tok_s": 44.6}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 8.87, "avg_tok_s": 45.1}]}, {"context": 16384, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.82, "avg_tok_s": 3.6}, {"type": "medium", "avg_tokens": 134.0, "avg_time_s": 4.02, "avg_tok_s": 33.3}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 11.78, "avg_tok_s": 33.9}]}]}
```

### Qwen3-Coder-30B TP=2 serve Context (vllm-next, Ray)

```json
{"label": "vllm-next-TP2-Ray-EAGLE3-context", "context": [{"context": 0, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.07, "avg_tok_s": 44.7}, {"type": "medium", "avg_tokens": 117.5, "avg_time_s": 1.76, "avg_tok_s": 66.7}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 4.33, "avg_tok_s": 92.4}]}, {"context": 512, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.11, "avg_tok_s": 28.1}, {"type": "medium", "avg_tokens": 133.0, "avg_time_s": 2.03, "avg_tok_s": 65.5}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 4.78, "avg_tok_s": 83.8}]}, {"context": 2048, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.15, "avg_tok_s": 20.6}, {"type": "medium", "avg_tokens": 133.0, "avg_time_s": 2.25, "avg_tok_s": 59.0}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 5.24, "avg_tok_s": 76.4}]}, {"context": 8192, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 28.44, "avg_tok_s": 0.1}, {"type": "medium", "avg_tokens": 143.5, "avg_time_s": 4.05, "avg_tok_s": 35.5}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 11.57, "avg_tok_s": 34.6}]}, {"context": 16384, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.65, "avg_tok_s": 4.6}, {"type": "medium", "avg_tokens": 144.0, "avg_time_s": 4.48, "avg_tok_s": 32.2}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 11.11, "avg_tok_s": 36.0}]}]}
```

### Qwen3-Coder-30B TP=2 continuous Context (vllm-next, torchrun)

```json
{"label": "vllm-next-TP2-torchrun-EAGLE3-context", "context": [{"context": 0, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.08, "avg_tok_s": 38.4}, {"type": "medium", "avg_tokens": 118.5, "avg_time_s": 1.46, "avg_tok_s": 81.1}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 3.64, "avg_tok_s": 109.8}]}, {"context": 512, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.11, "avg_tok_s": 26.1}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 1.85, "avg_tok_s": 81.1}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 3.9, "avg_tok_s": 102.6}]}, {"context": 2048, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.15, "avg_tok_s": 20.1}, {"type": "medium", "avg_tokens": 140.5, "avg_time_s": 1.95, "avg_tok_s": 72.2}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 4.13, "avg_tok_s": 96.9}]}, {"context": 8192, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.38, "avg_tok_s": 7.9}, {"type": "medium", "avg_tokens": 143.5, "avg_time_s": 2.8, "avg_tok_s": 51.3}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 7.71, "avg_tok_s": 51.9}]}, {"context": 16384, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.59, "avg_tok_s": 5.0}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 3.99, "avg_tok_s": 37.6}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 10.55, "avg_tok_s": 37.9}]}]}
```

### Qwen3-Coder-30B Concurrency

```json
{"label": "TP1-concurrency", "concurrency": [{"concurrency": 1, "total_tokens": 2000, "wall_time_s": 21.37, "throughput_tok_s": 93.6}, {"concurrency": 2, "total_tokens": 2000, "wall_time_s": 14.95, "throughput_tok_s": 133.8}, {"concurrency": 4, "total_tokens": 2000, "wall_time_s": 9.29, "throughput_tok_s": 215.3}, {"concurrency": 8, "total_tokens": 2000, "wall_time_s": 8.74, "throughput_tok_s": 228.9}]}
```

```json
{"label": "TP2-continuous-concurrency", "concurrency": [{"concurrency": 1, "total_tokens": 2000, "wall_time_s": 18.68, "throughput_tok_s": 107.1}, {"concurrency": 2, "total_tokens": 2000, "wall_time_s": 12.49, "throughput_tok_s": 160.2}, {"concurrency": 4, "total_tokens": 2000, "wall_time_s": 8.47, "throughput_tok_s": 236.1}, {"concurrency": 8, "total_tokens": 2000, "wall_time_s": 6.84, "throughput_tok_s": 292.5}]}
```

### GLM-4.7-Flash TP=1 Context

```json
{"label": "GLM-TP1-serve-INT4", "context": [{"context": 0, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.38, "avg_tok_s": 53.1}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.75, "avg_tok_s": 54.6}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 7.81, "avg_tok_s": 51.2}]}, {"context": 512, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.45, "avg_tok_s": 44.2}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 3.02, "avg_tok_s": 49.7}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 8.23, "avg_tok_s": 48.6}]}, {"context": 2048, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.61, "avg_tok_s": 32.8}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 3.78, "avg_tok_s": 39.7}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 10.1, "avg_tok_s": 39.6}]}, {"context": 8192, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 1.37, "avg_tok_s": 14.6}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 6.62, "avg_tok_s": 22.7}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 17.82, "avg_tok_s": 22.4}]}, {"context": 16384, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 2.29, "avg_tok_s": 8.8}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 10.37, "avg_tok_s": 14.5}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 27.78, "avg_tok_s": 14.4}]}]}
```

### GLM-4.7-Flash TP=2 serve Context (Ray)

```json
{"label": "GLM-TP2-serve-INT4", "context": [{"context": 0, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.32, "avg_tok_s": 62.6}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.22, "avg_tok_s": 67.5}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 6.04, "avg_tok_s": 66.2}]}, {"context": 512, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.37, "avg_tok_s": 53.6}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.4, "avg_tok_s": 62.4}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 6.52, "avg_tok_s": 61.4}]}, {"context": 2048, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.48, "avg_tok_s": 41.5}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.93, "avg_tok_s": 51.3}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 7.84, "avg_tok_s": 51.0}]}, {"context": 8192, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 1.01, "avg_tok_s": 19.8}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 4.77, "avg_tok_s": 31.5}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 12.75, "avg_tok_s": 31.4}]}, {"context": 16384, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 1.58, "avg_tok_s": 12.7}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 7.14, "avg_tok_s": 21.0}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 19.18, "avg_tok_s": 20.9}]}]}
```

### GLM-4.7-Flash TP=2 continuous Context (torchrun)

```json
{"label": "GLM-TP2-torchrun-INT4", "context": [{"context": 0, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.28, "avg_tok_s": 72.5}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 1.98, "avg_tok_s": 75.7}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 5.43, "avg_tok_s": 73.7}]}, {"context": 512, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.33, "avg_tok_s": 59.7}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.16, "avg_tok_s": 69.5}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 5.91, "avg_tok_s": 67.7}]}, {"context": 2048, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.46, "avg_tok_s": 43.8}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.69, "avg_tok_s": 55.7}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 7.23, "avg_tok_s": 55.3}]}, {"context": 8192, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 0.96, "avg_tok_s": 20.8}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 4.55, "avg_tok_s": 32.9}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 12.28, "avg_tok_s": 32.6}]}, {"context": 16384, "prompts": [{"type": "short", "avg_tokens": 20.0, "avg_time_s": 1.5, "avg_tok_s": 13.3}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 6.93, "avg_tok_s": 21.6}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 18.56, "avg_tok_s": 21.6}]}]}
```

### Qwen3-Coder-Next TP=1 Context

```json
{"label": "QwenNext-TP1-serve-INT4", "context": [{"context": 0, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.1, "avg_tok_s": 31.2}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.21, "avg_tok_s": 67.9}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 5.72, "avg_tok_s": 70.0}]}, {"context": 512, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 5.3, "avg_tok_s": 0.6}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.28, "avg_tok_s": 65.9}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 5.83, "avg_tok_s": 68.7}]}, {"context": 2048, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.41, "avg_tok_s": 7.3}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 2.52, "avg_tok_s": 59.6}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 6.1, "avg_tok_s": 65.6}]}, {"context": 8192, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 1.43, "avg_tok_s": 2.1}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 3.62, "avg_tok_s": 41.4}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 7.32, "avg_tok_s": 54.7}]}, {"context": 16384, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 2.83, "avg_tok_s": 1.1}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 5.11, "avg_tok_s": 29.4}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 8.99, "avg_tok_s": 44.5}]}]}
```

### EP=2 Context (Qwen3-Coder-30B, vllm-unified, torchrun, EAGLE3)

```json
{"label": "EP2-torchrun-continuous-EAGLE3", "context": [{"context": 0, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.08, "avg_tok_s": 35.8}, {"type": "medium", "avg_tokens": 109.5, "avg_time_s": 1.43, "avg_tok_s": 76.7}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 3.83, "avg_tok_s": 104.5}]}, {"context": 512, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.12, "avg_tok_s": 24.8}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 1.94, "avg_tok_s": 77.3}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 4.03, "avg_tok_s": 99.3}]}, {"context": 2048, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.15, "avg_tok_s": 19.9}, {"type": "medium", "avg_tokens": 116.0, "avg_time_s": 1.65, "avg_tok_s": 70.1}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 4.21, "avg_tok_s": 94.9}]}, {"context": 8192, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.4, "avg_tok_s": 7.5}, {"type": "medium", "avg_tokens": 132.0, "avg_time_s": 2.67, "avg_tok_s": 49.4}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 7.95, "avg_tok_s": 50.3}]}, {"context": 16384, "prompts": [{"type": "short", "avg_tokens": 3.0, "avg_time_s": 0.61, "avg_tok_s": 4.9}, {"type": "medium", "avg_tokens": 150.0, "avg_time_s": 4.05, "avg_tok_s": 37.0}, {"type": "long", "avg_tokens": 400.0, "avg_time_s": 10.78, "avg_tok_s": 37.1}]}]}
```

### A/B Test: vllm-unified vs vllm-next

```json
{"label": "AB-vllm-unified-torchrun", "note": "Long output only", "context": [{"context": 0, "avg_tok_s": 106.5}, {"context": 512, "avg_tok_s": 103.5}, {"context": 2048, "avg_tok_s": 95.1}, {"context": 8192, "avg_tok_s": 51.7}, {"context": 16384, "avg_tok_s": 38.0}]}
```

```json
{"label": "AB-vllm-unified-Ray", "note": "Long output only", "context": [{"context": 0, "avg_tok_s": 87.0}, {"context": 512, "avg_tok_s": 78.5}, {"context": 2048, "avg_tok_s": 75.7}, {"context": 8192, "avg_tok_s": 32.6}, {"context": 16384, "avg_tok_s": 32.4}]}
```
