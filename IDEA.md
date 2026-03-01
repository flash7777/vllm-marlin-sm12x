# Zwei Speed-Durchbrüche: torchrun + UF17

## Durchbruch 1: torchrun (TP=2 Multi-Node)

### Problem

GB10 hat **1 GPU pro Knoten** und 273 GB/s Bandbreite. Bei Qwen3-Coder INT4 reicht das für ~97 tok/s (TP=1). Mehr geht nicht mit einer GPU.

### Lösung

Zwei GB10 (DGX Spark + PGX ThinkStation) per QSFP56 RoCE 200 Gbps verbinden. `serve_torchrun.py` — ein eigener Server, der PyTorch's `torchrun` Launcher nutzt statt Ray.

### Architektur

```
DGX Spark (Rank 0)                    PGX ThinkStation (Rank 1)
┌─────────────────────┐                ┌─────────────────────┐
│  HTTP API (:8011)   │                │                     │
│  ↓ Requests         │                │                     │
│  FastAPI Thread     │                │                     │
│  ↓                  │   GLOO (CPU)   │                     │
│  broadcast(request) │ ──────────────►│  receive(request)   │
│  ↓                  │                │  ↓                  │
│  engine.step()      │   NCCL (GPU)   │  engine.step()      │
│  halbe GEMMs ──────►│ ◄─AllReduce──► │◄── halbe GEMMs     │
│  ↓                  │   RoCE 200Gbps │  ↓                  │
│  Response → Client  │                │  (verworfen)        │
└─────────────────────┘                └─────────────────────┘
```

### Zwei Kommunikationsebenen

| Ebene | Protokoll | Zweck | Daten |
|---|---|---|---|
| **Control Plane** | GLOO (CPU, TCP) | Request-Verteilung | JSON (~1 KB) |
| **Data Plane** | NCCL (GPU, RoCE) | AllReduce Partial-Sums | 4 KiB BF16 × 97/Token |

### Warum torchrun statt Ray?

- Ray braucht einen eigenen Cluster-Daemon auf jedem Knoten
- Ray's Multiprocessing hatte Konflikte mit GB10 Unified Memory
- `torchrun` + `external_launcher` ist leichtgewichtiger — PyTorch bringt alles mit

### Continuous Batching Protokoll

```python
# Rank 0 (HTTP + Engine)
while True:
    requests = drain_http_queue()
    broadcast(requests)           # GLOO → alle Ranks
    engine.step()                 # NCCL sync intern

# Rank 1+ (nur Engine)
while True:
    requests = receive_broadcast() # GLOO ← Rank 0
    engine.step()                  # NCCL sync intern
```

Beide Ranks rufen `engine.step()` im **exakten Gleichschritt** auf — NCCL AllReduce inside step() synchronisiert die GPUs automatisch.

### Ergebnis

```
TP=1 (1 GPU):   97 tok/s  (Limit einer GB10)
TP=2 (2 GPUs): 108 tok/s  (nur +11%, AllReduce-Overhead frisst Gewinn)
```

Der Gewinn war nur +11% statt der erwarteten ~2×, weil **97 AllReduce-Calls pro Token** das Budget auffressen. Profiling zeigte: 4.43ms/Token für AllReduce = 48% der Token-Zeit.

---

## Durchbruch 2: UF17 EAGER_ALLREDUCE

### Problem

NCCL AllReduce **innerhalb** von CUDA Graphs ist 2.5× langsamer als raw.

```
AllReduce raw (eager):       18 µs  ← NCCL direkt aufrufen
AllReduce in CUDA Graph:     46 µs  ← NCCL hat internen Graph-Overhead
                             ━━━━
                             +28 µs Overhead × 97 Calls = 2.66ms/Token (29%!)
```

CUDA Graphs sind für **Compute-Kernels** (GEMMs, Attention) grossartig — sie eliminieren CPU-Launch-Overhead. Aber NCCL hat internen Bookkeeping-Overhead wenn es in einer Graph-Replay läuft (starre Buffer, keine dynamische Channel-Auswahl).

### Lösung

Eine einzige Zeile Code — `vllm::all_reduce` als "Splitting Op" registrieren:

```python
# compilation.py
if os.environ.get("VLLM_UF_EAGER_ALLREDUCE", "0") == "1":
    self.splitting_ops.append("vllm::all_reduce")
```

vLLM's Piecewise CUDA Graph Architektur schneidet den FX-Graph an Splitting Ops. **GEMMs, Attention, RMSNorm bleiben in CUDA Graphs** (kein Launch-Overhead), aber **AllReduce läuft eager dazwischen** (18µs statt 46µs).

```
Vorher:  [====== CUDA Graph (GEMMs + AllReduce + Norms) ======]
          AllReduce: 46µs × 97 = 4.43ms

Nachher: [= Graph =] AllReduce [= Graph =] AllReduce [= Graph =]
          18µs eager   18µs eager
          97 × 18µs + 97 × 5µs Piecewise = 2.23ms
```

### Was auf dem Tisch bleibt

```
Theoretisches Optimum (AllReduce 18µs IN Graph, kein Overhead):
  97 × 18µs + 0µs Piecewise = 1.75ms

UF17 (AllReduce 18µs eager + Piecewise-Overhead):
  97 × 18µs + 97 × ~5µs    = 2.24ms

Differenz: ~0.5ms ≈ 10% der Token-Zeit (5.1ms bei 196 tok/s)
```

~0.5ms (10%) bleiben auf dem Tisch durch Piecewise-Splits. Das wäre nur durch einen Fix in NCCL selbst holbar.

### Ergebnis

```
TP=1:              97 tok/s  (Baseline, 1 GPU)
TP=2 ohne UF17:   108 tok/s  (+11%, AllReduce-Overhead frisst Gewinn)
TP=2 mit UF17:    196 tok/s  (+102% vs TP=1, nahezu lineare Skalierung!)
```

**torchrun = Enabler** (macht TP=2 überhaupt möglich), **UF17 = Optimizer** (beseitigt den AllReduce-Overhead). Zusammen: Zwei GB10 sind doppelt so schnell wie eine einzelne — theoretisch optimale TP=2 Skalierung.

### Gemessene Basisdaten

```
NCCL 2.29.2, ConnectX-7 RoCE 200 Gbps, TP=2 (DGX + PGX)

AllReduce raw 4 KiB:   18.2 µs
CUDA Graph ×97/Call:   45.6 µs (2.5× overhead)
97× Total (Graph):   4.43 ms = 48% der 9.28ms Token-Zeit
Graph-Overhead:      2.66 ms = 29% der Token-Zeit
```

### Aktivierung

```bash
# Container-Env
VLLM_UF_EAGER_ALLREDUCE=1

# torchrun Launch
torchrun --nnodes=2 --nproc-per-node=1 \
  --master-addr=192.168.0.117 --master-port=29500 \
  /opt/vllm/serve_torchrun.py --model ... --trust-remote-code
```
