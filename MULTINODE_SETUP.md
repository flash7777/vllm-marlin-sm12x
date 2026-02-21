# Multi-Node Setup: DGX Spark + PGX ThinkStation

## Architektur

```
DGX Spark (Head)                  PGX ThinkStation (Worker)
  192.168.1.117                     192.168.1.116
  ┌──────────────┐                  ┌──────────────┐
  │  Ray Head    │◄════ RoCE ════►  │  Ray Worker  │
  │  vllm serve  │  200 Gbps QSFP56│              │
  │  Port 8011   │  ~5-10us Latenz  │              │
  └──────────────┘                  └──────────────┘
  GB10 SM121 128 GiB                GB10 SM121 128 GiB
```

## Start-Script Architektur

Jedes Multi-Node Setup hat 3 Dateien:

| Datei | Rolle | Laueft auf | Was es macht |
|-------|-------|-----------|-------------|
| `*.head` | Container starten | DGX (Head) | `podman run` mit Ray-Head-Container + IMAGE |
| `*.worker` | Container starten | PGX (Worker) | `podman run` mit Ray-Worker-Container + IMAGE |
| `*.serve` | vLLM starten | DGX (Head) | `podman exec` im Head-Container |

**Reihenfolge**: head → worker → serve

Die `.serve` Scripts brauchen kein eigenes IMAGE — sie laufen via `podman exec` im
bereits gestarteten Head-Container. Alle head/worker Scripts nutzen dasselbe Image.

## Image

```bash
IMAGE="localhost/vllm-next"    # Alle head + worker Scripts
```

Fuer NVFP4-Benchmarks: `localhost/vllm-next2` (mit branchless E2M1 Patch).

## Verfuegbare Setups

### Qwen3-Coder-30B INT4 AutoRound

| Modus | Head | Worker | Serve | EAGLE3 |
|-------|------|--------|-------|--------|
| PP=2 | `start.qwen3coder_pp2.head` | `start.qwen3coder_pp2.worker` | `start.qwen3coder_pp2.serve` | `start.qwen3coder_pp2.serve.eagle3` |
| TP=2 | (head = DGX, worker = PGX) | | `start.qwen3coder_tp2.serve` | `start.qwen3coder_tp2.serve.eagle3` |
| EP=2 | `start.qwen3coder_ep2.head` | `start.qwen3coder_ep2.worker` | `start.qwen3coder_ep2.serve` | — |

### MiniMax-M2.5 INT4 AutoRound

| Modus | Head | Worker | Serve |
|-------|------|--------|-------|
| PP=2 | `start.minimax_pp2.head` | `start.minimax_pp2.worker` | `start.minimax_pp2.serve` |
| TP=2 | (gleicher Head) | | `start.minimax_tp2.serve` |
| EP=2 | (gleicher Head) | | `start.minimax_ep2.serve` |

## Netzwerk

### QSFP56 Direktverbindung (RoCE)

```
DGX:  enp1s0f0np0  192.168.0.117/24  MTU 9000
PGX:  enp1s0f0np0  192.168.0.116/24  MTU 9000
```

- 200 Gbps (25 GB/s theoretisch)
- ConnectX-7 mit RoCE v2 (RDMA over Converged Ethernet)
- NCCL: `NET/IB : Using [0]rocep1s0f0:1/RoCE` statt `NET/Socket`

### Container-Flags (head + worker)

```bash
--device /dev/infiniband/uverbs0
--device /dev/infiniband/rdma_cm
-e NCCL_IB_DISABLE=0
-e NCCL_IB_HCA=rocep1s0f0
-e NCCL_SOCKET_IFNAME=enp1s0f0np0
```

### Ray-Cluster

Head startet Ray Dashboard (Port 8265):
```bash
ray start --head --port=6380 --dashboard-host=0.0.0.0
```

Worker verbindet sich:
```bash
ray start --address=192.168.0.117:6380
```

## Quick-Start Beispiel (Qwen3-Coder TP=2)

```bash
# 1. DGX: Head starten
./start.qwen3coder_pp2.head    # Auch fuer TP/EP nutzbar

# 2. PGX: Worker starten
ssh flash@192.168.1.116
./start.qwen3coder_pp2.worker

# 3. DGX: vLLM serve
./start.qwen3coder_tp2.serve

# 4. Test
curl http://localhost:8011/health
python3 bench.py --port 8011
```

## Wichtige Hinweise

- **EAGLE3 + PP=2**: Nicht moeglich (`NotImplementedError`)
- **EAGLE3 + TP=2**: Crasht ab 8K Context
- **Memory**: `--gpu-memory-utilization 0.05 --kv-cache-memory-bytes 10G` (Unified Memory Profiler-Bug)
- **PGX Podman Bug**: Kernel 6.17 braucht `nohup podman run` + `podman system migrate`

## Benchmarks

Siehe [BENCHMARK_MULTINODE.md](BENCHMARK_MULTINODE.md) fuer alle Ergebnisse.
