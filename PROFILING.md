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

## 6. UF19: Custom AllReduce via raw ibverbs (NCCL-Bypass)

Standalone C/CUDA Benchmark, 5000 Iterationen, 500 Warmup.
Mapped pinned Memory (cudaHostAllocMapped) fuer GPU↔NIC Zero-Copy.

### 6.1 Raw RDMA Write Latenz (CPU Memory, ib_write_lat)

| Groesse | Latenz |
|---------|--------|
| 2 B | 1.74 µs |
| 64 B | 1.75 µs |
| 1 KiB | 2.59 µs |
| 4 KiB | **3.19 µs** |
| 8 KiB | 3.78 µs |
| 16 KiB | 4.83 µs |
| 32 KiB | 6.69 µs |
| 1 MiB | 82.2 µs |

Basislatenz ~1.74 µs, 200 Gbps (25 GB/s) Bandbreite ab ~4 KiB sichtbar.

### 6.2 Custom AllReduce Komponenten-Latenzen

| Komponente | Latenz |
|------------|--------|
| GPU add kernel (BF16, 2048 elem) | 2.1-2.4 µs |
| GPU→mapped write + __threadfence_system | 2.4-2.8 µs |
| RDMA post+poll (data+flag, ibv_post_send) | 4.2 µs (CPU wall-clock) |

### 6.3 Full Custom AllReduce Pipeline

```
  GPU prepare + CPU poll:      5.1 µs  (GPU writes to mapped mem + CPU polls flag)
  CPU RDMA post:               0.3 µs  (ibv_post_send, 2 chained WRs)
  GPU poll+add (incl. wait):   6.7 µs  (wait for RDMA data + BF16 add)
  ─────────────────────────────────────
  TOTAL:                      12.1 µs  (vs 19.0 µs NCCL AllReduce)
```

### 6.4 Hochrechnung auf 97 AllReduces/Token

| Methode | pro Call | x97 | tok/s | vs Baseline |
|---------|---------|-----|-------|-------------|
| NCCL AllReduce (UF17) | 19.0 µs | 1.84 ms | **116.3** | — |
| **Custom ibverbs (UF19)** | **12.1 µs** | **1.17 ms** | **126.1** | **+8.4%** |
| Raw RDMA Floor | 3.2 µs | 0.31 ms | 140+ | Theoretisch |

### 6.5 Engpaesse und Optimierungspotential

```
UF19 ohne nvidia-peermem:     12.1 µs
  └── GPU→mapped copy + fence:  5.1 µs  ← Hauptengpass
  └── RDMA + GPU poll+add:      7.0 µs  ← nahe am Floor

UF19 MIT nvidia-peermem (geschaetzt):  ~6-7 µs
  └── NIC DMA direkt aus GPU Memory (kein mapped copy)
  └── Kein CPU poll noetig
  └── Braucht sudo fuer modprobe nvidia-peermem
```

### 6.6 NCCL Net Plugin (UF19v2) — Ansatz verworfen

Versuch: NCCL Net Plugin (ncclNet_v9_t) um nur den Transport zu ersetzen.

| Methode | µs/call | Ergebnis |
|---------|---------|----------|
| NCCL Baseline | 16.4 | — |
| NCCL Net Plugin (RDMA SEND/RECV) | 20.6 | **Langsamer** |

**Erkenntnis**: Der Flaschenhals ist der NCCL Proxy-Thread (~13 µs), nicht der Transport (~3 µs). Ein Net Plugin sitzt UNTER dem Proxy und kann dessen Overhead nicht reduzieren. Zudem: Plugin nutzt RDMA SEND/RECV (two-sided), NCCL nutzt RDMA WRITE (one-sided, schneller).

### 6.7 UF19v4: Mini-Proxy Architektur (CUDA-Graph-kompatibel)

Korrekte Loesung: GPU-Kernels IN den CUDA Graph, minimaler CPU-Thread fuer ibverbs.

```
GPU Kernels (graph-capturable):
  1. wait_send_done <<<1,1>>>  — polls send_done (proxy fertig mit vorherigem RDMA)
  2. prepare_send   <<<N,256>>> — kopiert input→send_buf, atomicAdd_system(send_flag)
  3. poll_recv      <<<1,1>>>  — polls recv_flag via atomicAdd_system (NIC-DMA sichtbar)
  4. add_recv       <<<N,256>>> — addiert local + recv_buf(load_cv) → output

CPU Proxy Thread (Hintergrund, laeuft kontinuierlich):
  spin-polls send_flag → ibv_post_send(data+flag chained) → poll CQ → store send_done
```

**Benchmark** (5000 Runden, 2048 bf16 = 4 KiB):

| Methode | µs/call | vs NCCL |
|---------|---------|---------|
| NCCL Baseline (UF17 eager) | 16.4 | — |
| NCCL Net Plugin (UF19v2) | 20.6 | +26% langsamer |
| **UF19v4 Mini-Proxy** | **12.2** | **-26% schneller** |

**Hochrechnung auf 97 AllReduces/Token:**

| Methode | pro Call | x97 | tok/s | vs Baseline |
|---------|---------|-----|-------|-------------|
| NCCL (UF17 eager) | 16.4 µs | 1.59 ms | **116.3** | — |
| **UF19v4 Mini-Proxy** | **12.2 µs** | **1.18 ms** | **~120** | **+3.5%** |

### 6.8 UF Task Status (final)

| UF | Name | Ergebnis | Fazit |
|----|------|----------|-------|
| UF13 | NCCL_TUNE | Neutral | NCCL Env-Vars bringen nichts bei 4 KiB |
| UF14 | COALESCE | Nicht machbar | Layer-Dependency verhindert Batching |
| UF15 | OVERLAP_V2 | Nicht machbar | Multi-Stream bricht CUDA Graphs |
| UF16 | FUSED_REDUCE_NORM | Send/Recv langsamer | AllReduce ist bereits optimal |
| **UF17** | **EAGER_ALLREDUCE** | **+8% (107.8 → 116.3)** | **Produktiv, v3 zero-copy** |
| UF18 | NCCL_GRAPH_REG | Nicht getestet | CUDA Graph AllReduce 43x langsamer |
| UF19v1 | RAW_IBVERBS (CPU) | 12.1 µs | Prototyp, nicht graph-kompatibel |
| UF19v2 | NCCL Net Plugin | 20.6 µs | Verworfen (Proxy ist Flaschenhals, nicht Transport) |
| **UF19v4** | **Mini-Proxy** | **12.2 µs (-26% vs NCCL)** | **CUDA-Graph-kompatibel, Produktionsreif** |

NCCL-Overhead bei 4 KiB: ~13 µs Proxy-Software ueber dem Netzwerk-Floor (3.2 µs).
UF19v4 eliminiert den Proxy und ersetzt ihn durch einen minimalen CPU-Thread.
Restliche ~9 µs: GPU→mapped copy (~3 µs) + RDMA (~3 µs) + GPU poll+add (~3 µs).
