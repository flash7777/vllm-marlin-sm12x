# Multi-Node Benchmark: DGX Spark + PGX ThinkStation (2x GB10)

## Hardware

| | DGX Spark (Head) | PGX ThinkStation (Worker) |
|---|---|---|
| GPU | GB10 SM121, 128 GiB Unified | GB10 SM121, 128 GiB Unified |
| CPU | 20 Cores aarch64 | 20 Cores aarch64 |
| Interconnect | ConnectX-7 QSFP56 200 Gbps | ConnectX-7 QSFP56 200 Gbps |
| NCCL Transport | RoCE v2 via `rocep1s0f0` | RoCE v2 via `rocep1s0f0` |
| OS | Ubuntu 24.04.3, Kernel 6.14 | Ubuntu 24.04.4, Kernel 6.17 |
| Driver | nvidia-590.48.01 | nvidia-590.48.01 |

## RoCE Setup

ConnectX-7 unterstuetzt RoCE (RDMA over Converged Ethernet) ueber die QSFP56 Direct-Attach-Verbindung.
NCCL nutzt das IBext_v11 Plugin fuer Kernel-Bypass (~5-10us Latenz vs ~200-500us TCP Socket).

Container-Flags:
```bash
--device /dev/infiniband/uverbs0
--device /dev/infiniband/rdma_cm
-e NCCL_IB_DISABLE=0
-e NCCL_IB_HCA=rocep1s0f0
```

RDMA-Devices (beide Knoten identisch):
```
rocep1s0f0  -> enp1s0f0np0   uverbs0  ACTIVE  (QSFP56 200G)
rocep1s0f1  -> enp1s0f1np1   uverbs1  DOWN
roceP2p1s0f0 -> enP2p1s0f0np0 uverbs2  ACTIVE  (1GbE LAN)
roceP2p1s0f1 -> enP2p1s0f1np1 uverbs3  DOWN
```

## Modelle

| Modell | Parameter | Quant | Groesse | Experts |
|--------|-----------|-------|---------|---------|
| Qwen3-Coder-30B-A3B | 30B MoE (3B active) | INT4 AutoRound | 19 GB | 128 Experts, top-8 |
| MiniMax-M2.5 | 46B MoE | INT4 AutoRound (w4g128) | 28 GB | 256 Experts |

## Ergebnisse: Qwen3-Coder-30B INT4 AutoRound

### Decode-Throughput (ctx=0, long tok/s)

| Modus | Socket | RoCE | Speedup |
|-------|--------|------|---------|
| PP=2 (Pipeline Parallel) | 70.4 | 71.3 | +1% |
| TP=2 (Tensor Parallel) | 45.3 | **91.6** | **+102%** |
| EP=2 (Expert Parallel) | 35.0 | **85.5** | **+144%** |
| TP=2 + EAGLE3 (NST=1) | 60.0 | **94.8** | +58% |

### Context-Matrix: Qwen3-Coder TP=2 RoCE (bestes Setup)

| | ctx=0 | ctx=512 | ctx=2K | ctx=8K | ctx=16K |
|------|-------|---------|--------|--------|---------|
| short | 91.6 | 91.5 | 88.1 | 81.3 | 73.5 |
| medium | 91.6 | 91.5 | 88.1 | 81.3 | 73.5 |
| long | 91.6 | 91.5 | 88.1 | 81.3 | 73.5 |

### Context-Matrix: Qwen3-Coder PP=2 RoCE

| | ctx=0 | ctx=512 | ctx=2K | ctx=8K | ctx=16K |
|------|-------|---------|--------|--------|---------|
| short | 66.1 | 65.9 | 66.1 | 64.2 | 61.8 |
| medium | 71.3 | 71.3 | 71.0 | 70.1 | 65.3 |
| long | 71.3 | 71.3 | 71.0 | 70.1 | 65.3 |

### Context-Matrix: Qwen3-Coder EP=2 RoCE

| | ctx=0 | ctx=512 | ctx=2K | ctx=8K | ctx=16K |
|------|-------|---------|--------|--------|---------|
| short | 78.6 | 75.3 | 74.5 | 69.2 | 63.0 |
| medium | 85.5 | 84.1 | 82.7 | 75.1 | 66.4 |
| long | 85.5 | 84.1 | 82.7 | 75.1 | 66.4 |

## Ergebnisse: MiniMax-M2.5 INT4 AutoRound

### Decode-Throughput (ctx=0, long tok/s)

| Modus | Socket | RoCE | Speedup |
|-------|--------|------|---------|
| PP=2 (Pipeline Parallel) | 29.0 | 29.0 | 0% |
| TP=2 (Tensor Parallel) | 22.4 | **41.7** | **+86%** |
| EP=2 (Expert Parallel) | 25.9 | **38.7** | +49% |

### Context-Matrix: MiniMax-M2.5 PP=2 RoCE

| | ctx=0 | ctx=512 | ctx=2K | ctx=8K | ctx=16K |
|------|-------|---------|--------|--------|---------|
| short | 27.5 | 27.3 | 25.7 | 22.9 | 20.5 |
| medium | 29.0 | 28.8 | 27.1 | 24.0 | 20.8 |
| long | 29.0 | 28.8 | 27.1 | 24.0 | 20.8 |

### Context-Matrix: MiniMax-M2.5 TP=2 RoCE

| | ctx=0 | ctx=512 | ctx=2K | ctx=8K | ctx=16K |
|------|-------|---------|--------|--------|---------|
| short | 39.2 | 38.0 | 36.2 | 30.3 | 25.2 |
| medium | 41.7 | 40.2 | 38.5 | 32.3 | 26.2 |
| long | 41.7 | 40.2 | 38.5 | 32.3 | 26.2 |

### Context-Matrix: MiniMax-M2.5 EP=2 RoCE

| | ctx=0 | ctx=512 | ctx=2K | ctx=8K | ctx=16K |
|------|-------|---------|--------|--------|---------|
| short | 36.7 | 35.8 | 34.0 | 29.5 | 24.5 |
| medium | 38.7 | 37.6 | 35.8 | 30.7 | 25.2 |
| long | 38.7 | 37.6 | 35.8 | 30.7 | 25.2 |

## Erkenntnisse

1. **RoCE macht TP und EP erst brauchbar** — Socket-Latenz (~200-500us) war der Bottleneck, RoCE (~5-10us) eliminiert ihn
2. **PP profitiert kaum** — nur 1 Netzwerk-Roundtrip pro Token (Pipeline-Bubble ist das Problem, nicht Latenz)
3. **TP=2 RoCE ist optimal fuer Qwen3-Coder**: 91.6 tok/s — schneller als Single-Node EAGLE3 (~86 tok/s)
4. **EAGLE3 funktioniert nicht mit PP=2** — vLLM `NotImplementedError: Pipeline parallelism is not supported`
5. **EAGLE3 + TP=2 crasht ab 8K Context** — nur bei kleinem Context nutzbar (94.8 tok/s bei ctx=0)
6. **EP performt aehnlich wie TP mit RoCE** — bei hoher Context-Laenge sogar leicht besser
7. **ConnectX-7 QSFP56 kann RoCE** — RDMA ueber Ethernet funktioniert out-of-the-box

## Vergleich: Single-Node vs Multi-Node (Qwen3-Coder INT4)

| Setup | tok/s | vs Single-Node |
|-------|-------|---------------|
| Single-Node (DGX, vanilla) | ~86 | Baseline |
| Single-Node (DGX, EAGLE3) | ~97 | +13% |
| Multi-Node TP=2 RoCE | 91.6 | +7% |
| Multi-Node TP=2+EAGLE3 RoCE | 94.8 | +10% |
| Multi-Node PP=2 RoCE | 71.3 | -17% |

## vLLM-Konfiguration

```bash
# Gemeinsame Flags
--distributed-executor-backend ray
--gpu-memory-utilization 0.05
--kv-cache-memory-bytes 10G
--max-model-len 16384
--trust-remote-code
--quantization auto_round
--disable-log-requests

# PP=2
--tensor-parallel-size 1 --pipeline-parallel-size 2

# TP=2
--tensor-parallel-size 2

# EP=2
--tensor-parallel-size 2 --enable-expert-parallel

# EAGLE3
--speculative-config='{"model":"<path>","num_speculative_tokens":1,"method":"eagle3"}'
```

## Datum

2026-02-20
