# Qwen3-Coder-Next INT4 AutoRound — Benchmark-Ergebnisse

Model: `Intel/Qwen3-Coder-Next-int4-AutoRound` (W4A8-FP8, ~80B MoE, 512 Experts, top-10)
Image: `vllm-next` (vLLM 26.01 base, CUTLASS 4.3.5, SM120a/SM121a)
Hardware: DGX Spark (GB10, SM121, 273 GB/s LPDDR5X)

## Vanilla (kein Speculative Decoding)

### Performance (n=5)

| | short (20 tok) | medium (150 tok) | long (400 tok) | Math (50) |
|---|---:|---:|---:|---:|
| tok/s | 34.2 | 69.5 | 70.3 | 92% |

Short nur 3 Content-Tokens (Modell denkt viel in Reasoning-Tokens).

### Context-Matrix (tok/s)

| Context | short (20) | medium (150) | long (400) |
|--------:|-----------:|-------------:|-----------:|
| 0 | 36.3 | 69.4 | 70.3 |
| 1K | 9.5 | 66.9 | 69.0 |
| 4K | 7.6 | 65.3 | 67.5 |
| 8K | 6.2 | 64.4 | 66.1 |
| 16K | 3.6 | 60.4 | 62.9 |
| 32K | 1.8 | 56.7 | 58.5 |

Context-Degradation mild: 70.3 → 58.5 tok/s (long, 0→32K) = -17%.

## MTP (qwen3_next_mtp)

**MTP ist langsamer als Vanilla auf DGX Spark.** Bandwidth-bound System, MTP-Overhead > Spekulations-Gewinn.
NST=1 besser als NST=2, aber beides langsamer als Vanilla.

### Vergleich (long, ctx=0)

| Modus | tok/s | vs Vanilla | Math |
|---|---:|---:|---:|
| **Vanilla** | **70.3** | — | **92%** |
| MTP NST=1 | 48.8 | -31% | 92% |
| MTP NST=2 | 39.3 | -44% | 92% |

### MTP NST=1 — Performance (n=5)

| | short (20 tok) | medium (150 tok) | long (400 tok) | Math (50) |
|---|---:|---:|---:|---:|
| tok/s | 8.9 | 48.3 | 48.8 | 92% |

### MTP NST=1 — Context-Matrix (tok/s)

| Context | short (20) | medium (150) | long (400) |
|--------:|-----------:|-------------:|-----------:|
| 0 | 25.6 | 48.3 | 48.8 |
| 1K | 9.3 | 45.0 | 46.9 |
| 4K | 4.0 | 37.6 | 42.7 |
| 8K | 2.2 | 31.3 | 37.5 |
| 16K | 1.1 | 22.8 | 30.2 |
| 32K | 0.5 | 14.3 | 21.3 |

### MTP NST=2 — Performance (n=5)

| | short (20 tok) | medium (150 tok) | long (400 tok) | Math (50) |
|---|---:|---:|---:|---:|
| tok/s | 0.3 | 39.0 | 39.3 | 92% |

### MTP NST=2 — Context-Matrix (tok/s)

| Context | short (20) | medium (150) | long (400) |
|--------:|-----------:|-------------:|-----------:|
| 0 | 22.7 | 39.0 | 39.1 |
| 1K | 8.9 | 36.6 | 38.2 |
| 4K | 4.0 | 32.0 | 35.0 |
| 8K | 2.2 | 27.0 | 31.4 |
| 16K | 1.1 | 20.4 | 26.0 |
| 32K | 0.5 | 13.2 | 19.1 |

## Konfiguration

### Vanilla
```bash
./start.sh   # Port 8000, --enable-prefix-caching
```

### MTP
```bash
./start.mtp.sh   # Port 8000, --no-enable-chunked-prefill
# --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
```

### Env-Vars
| Variable | Wert | Beschreibung |
|---|---|---|
| `VLLM_MARLIN_INPUT_DTYPE` | `fp8` | W4A8 (INT4→FP8 Dequant + FP8 MMA k=32) |
| `VLLM_MLA_DISABLE` | `1` | MLA deaktivieren |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` | FlashInfer Version-Mismatch umgehen |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `1` | Atomic Add fuer Marlin |
