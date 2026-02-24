# Qwen3-Coder-Next INT4 AutoRound — Benchmark-Ergebnisse

Model: `Intel/Qwen3-Coder-Next-int4-AutoRound` (W4A8-FP8, ~80B MoE, 512 Experts, top-10)
Image: `vllm-next` (vLLM 26.01 base, CUTLASS 4.3.5, SM120a/SM121a)
Hardware: DGX Spark (GB10, SM121, 273 GB/s LPDDR5X)

## Vanilla (kein Speculative Decoding)

### Performance (n=5)

| | short (20 tok) | medium (150 tok) | long (400 tok) | Math (50) |
|---|---:|---:|---:|---:|
| tok/s | — | — | — | —% |

### Context-Matrix (tok/s)

| Context | short (20) | medium (150) | long (400) |
|--------:|-----------:|-------------:|-----------:|
| 0 | — | — | — |
| 1K | — | — | — |
| 4K | — | — | — |
| 8K | — | — | — |
| 16K | — | — | — |
| 32K | — | — | — |

## MTP NST=2 (qwen3_next_mtp)

### Performance (n=5)

| | short (20 tok) | medium (150 tok) | long (400 tok) | Math (50) |
|---|---:|---:|---:|---:|
| tok/s | — | — | — | —% |

### Context-Matrix (tok/s)

| Context | short (20) | medium (150) | long (400) |
|--------:|-----------:|-------------:|-----------:|
| 0 | — | — | — |
| 1K | — | — | — |
| 4K | — | — | — |
| 8K | — | — | — |
| 16K | — | — | — |
| 32K | — | — | — |

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
