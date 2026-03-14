# Qwen3.5-397B-A17B INT4 AutoRound — Benchmark Results

## Setup
- **Model**: Intel/Qwen3.5-397B-A17B-int4-AutoRound (199 GB, 512 Experts, 60 Layers)
- **Hardware**: 2× DGX Spark GB10 (SM121, 119 GiB Unified Memory each)
- **Interconnect**: QSFP56 RoCE 200 Gbps (ConnectX-7)
- **Engine**: vLLM 0.17.1rc1.dev158 (eugr prebuilt wheels)
- **Backend**: Ray TP=2, NCCL over RoCE
- **Quantization**: Marlin INT4 (GPTQMarlinLinearMethod)
- **Attention**: FlashInfer 0.6.6
- **KV-Cache**: 3 GB FP8 (104K tokens), Prefix Caching enabled
- **Date**: 2026-03-14

## Performance (n=5)

| Output Length | tok/s |
|---|---|
| Short (20 tok) | 24.2 |
| Medium (150 tok) | 27.9 |
| Long (400 tok) | 24.8 |

**Peak: 28.6 tok/s** (400 tok, ctx=0)

## Math Accuracy

**96%** (48/50) — 2 Fehler wegen Reasoning-Truncation (max_tokens zu klein), nicht falsche Berechnung.

## Context Scaling (tok/s, n=2)

| Context | Short (20t) | Medium (150t) | Long (400t) |
|---|---|---|---|
| 0 | 24.3 | 28.1 | **28.6** |
| 512 | 17.7 | 26.6 | 27.9 |
| 2K | 10.2 | 24.5 | 26.9 |
| 8K | 1.8 | 23.2 | 25.9 |
| 16K | 6.6 | 26.8 | 27.7 |

**Beobachtungen:**
- Generation Speed (long) bleibt stabil bei 26-28 tok/s über alle Context-Größen
- Short-Prompts bei großem Context haben hohe TTFT (Prefill-Latenz)
- 8K Context Short anomal niedrig (1.8 tok/s) — vermutlich Prefill-dominiert
- 16K Context zeigt Recovery — Prefix Caching greift

## Vergleich mit Forum-Angaben

| Quelle | tok/s (single user) |
|---|---|
| **Unser Bench** | **24-28 tok/s** |
| eugr Forum | 26-30 tok/s |
| eugr Forum (concurrent) | 42-46 tok/s |

## Bekannte Limitierungen
- KV-Cache nur 3 GB (dirty start nach OOM). Sauberer Start → 0.85 util → ~15 GB KV-Cache → bessere Concurrency
- Kein MTP (Intel AutoRound hat MTP-Weights nicht mitquantisiert)
- Unified Memory Leak nach Crash → Power-Cycle nötig
