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

### AllReduce ist NICHT overlapped — es sind echte Wait-Cycles

UF17 macht AllReduce zum splitting op. Der CUDA Graph wird an jeder AllReduce-Stelle
aufgebrochen. Die Ausfuehrung ist strikt sequentiell:

```
[Graph Segment 1] → STOP → [AllReduce 19µs] → STOP → [Graph Segment 2] → ...
                     ^^^^                       ^^^^
                  Tensor Cores idle           Tensor Cores idle
```

AllReduce laeuft als eigenstaendiger NCCL-Kernel. Das naechste Graph-Segment kann
erst starten wenn AllReduce fertig ist, weil es das Ergebnis als Input braucht.
Die 1.8 ms sind **echte Wartezeit** in der die Tensor Cores nichts tun.

```
Aktuell (UF17, sequentiell):
  Compute: 6.8 ms  +  AllReduce: 1.8 ms  =  8.6 ms  → 116 tok/s

Theoretisch (AllReduce overlapped mit Compute):
  Compute: 6.8 ms  (AllReduce versteckt)  =  6.8 ms  → 147 tok/s  (+27%)
```

Overlap waere der groesste verbleibende Hebel (+27%), ist aber mit CUDA Graphs
nicht machbar. Das fundamentale Dilemma:

| Modus | Overlap | Launch-Overhead | Ergebnis |
|-------|---------|-----------------|----------|
| CUDA Graphs + UF17 (splitting) | Kein Overlap | Minimal (~2.5µs Replay) | **116 tok/s** (aktuell) |
| CUDA Graphs ohne UF17 (in-graph) | Kein Overlap | Keiner, aber NCCL 43× langsamer | ~62 tok/s |
| Ohne CUDA Graphs (enforce-eager) | Overlap moeglich | Hoch (~10-20µs pro Kernel) | <50 tok/s |

Alle drei Varianten haben Nachteile. UF17 (splitting) ist der beste Kompromiss.

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

## Naechste Hebel (theoretisch)

| Hebel | Erwartung | Machbarkeit |
|-------|-----------|-------------|
| Besseres EAGLE3 Draft-Modell | Hoehere Acceptance Rate → mehr Tokens/Step | Braucht Fine-Tuning |
| TP=4 (4× GB10) | ~280 tok/s theoretisch, ~150 tok/s realistisch | 2 weitere Knoten noetig |
| Batch>1 | Bessere Compute-Auslastung | Nur relevant bei mehreren Usern |
| INT2/INT3 Quantisierung | Weniger Bytes/Token → schneller | Qualitaetsverlust |
