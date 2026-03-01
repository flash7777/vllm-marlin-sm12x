# NCCL AllReduce Profiling — DGX Spark + PGX ThinkStation (TP=2, RoCE 200 Gbps)

Hardware: 2x GB10 (SM121), ConnectX-7 QSFP56 Direktverbindung, NCCL 2.29.2
Modell: Qwen3-Coder-30B-A3B INT4 AutoRound + EAGLE3 NST=1
Kernel: Marlin W4A16 (Dense + MoE), 97 AllReduces pro Token

## 1. AllReduce vs Send/Recv Latenz

CUDA Events, 1000 Iterationen, 200 Warmup. Eager (nicht im CUDA Graph).

| Methode | 4 KiB (h=2048) | 8 KiB (h=4096) | 16 KiB (h=8192) | 32 KiB (h=16384) |
|---------|---------------|----------------|-----------------|------------------|
| AllReduce (torch.dist) | **19.0 µs** | **20.3 µs** | **21.5 µs** | **28.3 µs** |
| Send/Recv (grouped) | 24.2 µs | 25.3 µs | 19.4 µs | 20.9 µs |
| Send/Recv + add | 25.5 µs | 33.6 µs | 29.3 µs | 30.1 µs |
| Local add only | 3.1 µs | 3.0 µs | 4.2 µs | 3.1 µs |

**Ergebnis**: AllReduce ist bei 4 KiB (Qwen3-Coder hidden_size=2048, BF16) 6.4 µs schneller als Send/Recv+add. NCCL AllReduce fusioniert Transport + Reduce intern — Send/Recv+add braucht 3 separate Operationen.

**Fazit**: UF16 (Send/Recv + Triton Fused Reduce+Norm) ist nicht sinnvoll. AllReduce ist bereits optimal.

## 2. AllReduce: Eager vs CUDA Graph

CUDA Events, 2000 Iterationen, 200 Warmup.

| Methode | 4 KiB (single) | 4 KiB (batch 10) | 8 KiB (single) | 16 KiB (single) |
|---------|---------------|-------------------|----------------|-----------------|
| **Eager** (UF17) | **18.9 µs** | **18.1 µs** | **20.5 µs** | **21.1 µs** |
| **CUDA Graph** | 813.0 µs | 95.8 µs | 1032.4 µs | 1024.3 µs |
| Empty graph replay | 2.5 µs | — | 2.5 µs | 2.6 µs |

**Ergebnis**: AllReduce innerhalb CUDA Graph ist **43x langsamer** (single) bzw. **5x langsamer** (batched) als eager. NCCL's interne Synchronisation lässt sich nicht effizient in CUDA Graph Replay abbilden.

**Fazit**: UF17 EAGER_ALLREDUCE (AllReduce als splitting op außerhalb der Graphs) ist der richtige Ansatz. "AllReduce zurück in CUDA Graph" wuerde von 116 auf ~62 tok/s abstuerzen.

### Hochrechnung auf 97 AllReduces/Token (4 KiB)

| Methode | pro Call | x97 | tok/s (von 116.3 Baseline) |
|---------|---------|-----|---------------------------|
| Eager (UF17) | 18.1 µs | 1.76 ms | **116.3** (aktuell) |
| CUDA Graph (batched) | 95.8 µs | 9.29 ms | ~62 (projiziert) |

## 3. TP=1 vs TP=2

| Config | short | medium | long | Math |
|--------|-------|--------|------|------|
| TP=2 UF17v3 EAGLE3 RoCE | 54.3 | 87.9 | **116.3** | 76% |
| TP=2 UF17v3 NST=2 | 0.3 | 77.2 | 115.0 | 76% |
| TP=1 single DGX EAGLE3 | 14.8 | 69.1 | **93.0** | 78% |
| TP=2 Baseline (kein UF17) | — | — | 107.8 | 76% |

### Warum TP=1 langsamer ist trotz 0 AllReduce

```
TP=1:  10.75 ms/Token (alle Gewichte auf 1 GPU, 273 GB/s)
TP=2:   8.60 ms/Token (halbe Gewichte pro GPU, 2x 273 GB/s aggregiert)

Bandbreiten-Gewinn TP=2:    ~5.4 ms (halbe Gewichte pro GPU)
AllReduce-Kosten TP=2:      ~1.8 ms (97 x 19 µs)
Sonstiger Overhead TP=2:    ~1.4 ms (EAGLE3 Sync, Scheduling)
Netto-Gewinn TP=2:          ~2.15 ms/Token → 20% schneller
```

Bei Batch=1 auf GB10 ist Memory-Bandbreite der Flaschenhals. TP=2 verdoppelt die aggregierte Bandbreite (546 GB/s), und die AllReduce-Kosten (1.8 ms) sind kleiner als der Bandbreiten-Gewinn (5.4 ms).

## 4. AllReduce Budget-Analyse

```
Qwen3-Coder-30B TP=2, 116.3 tok/s = 8.6 ms/Token

AllReduce:   97 x 19 µs = 1.8 ms  (21% der Token-Zeit)
  davon:
    NCCL Protokoll:     ~15 µs   (LL128/Tree)
    RoCE Roundtrip:     ~2-4 µs  (irreduzibel)

Compute:     ~5.4 ms  (Marlin GEMMs, Attention, MoE)
Overhead:    ~1.4 ms  (EAGLE3, Scheduling, CPU)
```

## 5. UF Task Status

| UF | Name | Ergebnis | Fazit |
|----|------|----------|-------|
| UF13 | NCCL_TUNE | Neutral | NCCL Env-Vars bringen nichts bei 4 KiB |
| UF14 | COALESCE | Nicht machbar | Layer-Dependency verhindert Batching |
| UF15 | OVERLAP_V2 | Nicht machbar | Multi-Stream bricht CUDA Graphs |
| UF16 | FUSED_REDUCE_NORM | Send/Recv langsamer | AllReduce ist bereits optimal |
| **UF17** | **EAGER_ALLREDUCE** | **+8% (107.8 → 116.3)** | **Produktiv, v3 zero-copy** |
| UF18 | NCCL_GRAPH_REG | Nicht getestet | CUDA Graph AllReduce 43x langsamer |

AllReduce-Optimierung ist ausgereizt. 19 µs/Call ueber RoCE bei 4 KiB ist nahe am theoretischen Minimum.
