# Speed Analysis — Qwen3-Coder-30B-A3B INT4 auf DGX Spark (GB10)

## Theoretisches Bandbreiten-Limit

Qwen3-Coder-30B-A3B: 3B aktive Parameter pro Token (MoE, 128 Experts, topk=8).

```
INT4 aktive Gewichte:  3B × 0.5 Bytes       = 1.50 GB
Scales (group=128):    3B / 128 × 2 Bytes    = 0.05 GB
Total pro Token:                              ≈ 1.55 GB
```

GB10 Memory Bandwidth: 273 GB/s

```
             Bandbreite   Lese-Zeit   +AllReduce   Theorie     Gemessen   Effizienz
TP=1:        273 GB/s     5.7 ms      —            175 tok/s    93 tok/s    53%
TP=2+UF17:   546 GB/s     2.8 ms      +1.8 ms      217 tok/s   116 tok/s    53%
```

## Wohin gehen die fehlenden 47%?

| Overhead | Geschaetzt |
|----------|-----------|
| KV-Cache lesen/schreiben | 0.5–1 GB/Token bei 16K Kontext |
| EAGLE3 Draft-Modell | Eigene GEMMs + Verifikation |
| Attention Compute | Nicht rein bandwidth-bound |
| MoE Routing | Top-8 Selektion, Token-Permutation |
| Activations, Norms, Softmax | Kleine Kernel, viele Launches |
| CUDA Graph Replay Overhead | Graph-Segment-Wechsel |

53% Bandbreiten-Effizienz ist realistisch fuer ein MoE-Modell mit Speculative Decoding.

## AllReduce-Kosten bei TP=2

97 AllReduces pro Token (Qwen3-Coder, 48 Layer × 2 + Extras).

```
Eager AllReduce (UF17):     19 µs/Call × 97 = 1.8 ms/Token
Anteil an Token-Zeit:       1.8 / 8.6 = 21%
```

TP=2 lohnt sich trotzdem: 2× Bandbreite spart ~5.4 ms, AllReduce kostet nur 1.8 ms.

### AllReduce ist NICHT overlapped — fundamentale Datenabhaengigkeit

AllReduce kann nicht mit Compute ueberlappt werden. Das ist kein CUDA-Graph-Problem,
sondern eine algorithmische Abhaengigkeit im Transformer:

```
GEMM (jede GPU rechnet ihren Teil)
     ↓
AllReduce (konsolidiert Partial Sums beider GPUs)
     ↓  ← naechster Schritt braucht konsolidiertes Ergebnis
RMSNorm (braucht ALLE Elemente fuer Varianz)
     ↓
GEMM naechster Layer
     ↓
AllReduce ...
```

Jeder Schritt haengt vom Ergebnis des vorherigen ab. Es gibt keinen unabhaengigen
Compute, mit dem man AllReduce ueberlappen koennte — der naechste GEMM IST der
Consumer des AllReduce-Ergebnisses. Die 1.8 ms sind ein **unvermeidbarer Preis**
fuer TP=2, unabhaengig von CUDA Graphs oder sonstiger Infrastruktur.

```
TP=2:  2.8 ms Compute + 1.8 ms AllReduce (unvermeidbar) = 4.6 ms → 217 tok/s (theor.)
TP=1:  5.7 ms Compute + 0 ms AllReduce                  = 5.7 ms → 175 tok/s (theor.)
TP=2 Gewinn: 1.1 ms/Token durch doppelte Bandbreite trotz AllReduce-Kosten
```

## Warum TP=1 langsamer ist

```
TP=1:  10.75 ms/Token  (alle Gewichte auf 1 GPU, 273 GB/s)
TP=2:   8.60 ms/Token  (halbe Gewichte pro GPU, 2× 273 GB/s)

Bandbreiten-Gewinn TP=2:    ~5.4 ms  (halbe Gewichte pro GPU)
AllReduce-Kosten TP=2:      -1.8 ms  (97 × 19 µs ueber RoCE)
Sonstiger Overhead TP=2:    -1.4 ms  (EAGLE3 Sync, Scheduling)
Netto-Gewinn TP=2:          ~2.15 ms → 20% schneller
```

Bei Batch=1 ist Memory-Bandbreite der Flaschenhals. TP=2 verdoppelt die
aggregierte Bandbreite, und die AllReduce-Kosten sind kleiner als der Gewinn.

## Gemessene Ergebnisse

| Config | short | medium | long | Math |
|--------|-------|--------|------|------|
| TP=2 UF17v3 EAGLE3 NST=1 | 54.3 | 87.9 | **116.3** | 76% |
| TP=2 UF17v3 EAGLE3 NST=2 | 0.3 | 77.2 | 115.0 | 76% |
| TP=2 Baseline (kein UF17) | — | — | 107.8 | 76% |
| TP=1 EAGLE3 NST=1 | 14.8 | 69.1 | 93.0 | 78% |

## Warum AllReduce nicht schneller als 19 µs geht

Die 19 µs pro AllReduce bestehen fast vollstaendig aus NCCL-Protokoll-Overhead.
Die eigentliche Datenuebertragung ist vernachlaessigbar:

```
4 KiB bei 25 GB/s (RoCE 200 Gbps):   0.16 µs  (<1% der Latenz)
NCCL Protokoll-Overhead:              ~15 µs   (GPU-seitiger Kernel-Code)
GPU Reduce (Addition):                 ~1 µs
Sonstiges:                             ~3 µs
                                      ------
                                       ~19 µs
```

Das sieht man an der flachen Skalierung ueber Datengroessen:

```
 4 KiB:   19 µs
 8 KiB:   20 µs   (+1 µs fuer doppelte Daten)
16 KiB:   21 µs   (+2 µs fuer 4× Daten)
32 KiB:   28 µs   (+9 µs fuer 8× Daten)
```

### Getestete Optimierungen (alle neutral)

| Variable | Idee | Ergebnis |
|----------|------|----------|
| `NCCL_MAX_NCHANNELS=1` | Weniger Channel-Overhead | Neutral (UF13) |
| `NCCL_PROTO=LL` | Force Low-Latency Protokoll | Neutral (ist default bei 4 KiB) |
| `NCCL_ALGO=Ring` | Ring vs Tree | Neutral (identisch bei TP=2) |
| GPU Stream Priority | High-Priority fuer NCCL | Irrelevant (NCCL laeuft allein) |
| CPU Realtime-Prio | `chrt -f 99` fuer NCCL Proxy | Irrelevant (Overhead ist GPU-seitig) |
| Kompression | 4 KiB verkleinern | Sinnlos (Daten = 0.16µs, Overhead = 15µs) |
| Pipelined AllReduce | Teilergebnisse frueher senden | Sinnlos (4 KiB zu klein zum Chunken) |

### Was theoretisch helfen wuerde

Raw RDMA ohne NCCL — eigener GPU-Kernel der direkt ueber ibverbs sendet:

```
Hand-geschriebener TP=2 Reduce:
  RDMA Write (bidirektional):   ~2 µs
  Flag-Wait:                    ~2 µs
  GPU Add:                      ~1 µs
                                -----
                                ~5 µs  (statt 19 µs)

Ersparnis: 14 µs × 97 = 1.36 ms/Token → ~130 tok/s (+12%)
```

Aber: Monate Aufwand, fragil, bricht bei jedem NCCL/Driver-Update.
19 µs ist NCCL's architektureller Floor fuer cross-node Collectives.

## Naechste Hebel (theoretisch)

| Hebel | Erwartung | Machbarkeit |
|-------|-----------|-------------|
| Besseres EAGLE3 Draft-Modell | Hoehere Acceptance Rate → mehr Tokens/Step | Braucht Fine-Tuning |
| TP=4 (4× GB10) | ~280 tok/s theoretisch, ~150 tok/s realistisch | 2 weitere Knoten noetig |
| Batch>1 | Bessere Compute-Auslastung | Nur relevant bei mehreren Usern |
| INT2/INT3 Quantisierung | Weniger Bytes/Token → schneller | Qualitaetsverlust |
