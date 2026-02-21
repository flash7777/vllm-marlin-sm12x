# vllm-turbo (Avarok dgx-vllm)

NVFP4 inference for Qwen3-Next-80B-A3B on DGX Spark GB10 (SM121).

Based on [Avarok/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm) v22.

## Key Innovation

GB10 (SM121) has FP4 tensor cores but **missing** `cvt.rn.satfinite.e2m1x2.f32` PTX instruction.
Avarok's 15-line software E2M1 conversion function fills this gap.

| Feature | Detail |
|---------|--------|
| vLLM | v0.16.0rc2 (pinned 3b30e6150) |
| PyTorch | 2.10.0+cu130 |
| Triton | 3.6.0 |
| Model | Qwen3-Next-80B-A3B (512 experts, MoE) |
| Quant | NVFP4 (E2M1 weights + FP8 block scales) |
| MoE Backend | Marlin (nicht FlashInfer CUTLASS) |
| MTP | 2 speculative tokens (63-89% acceptance) |

## Performance (Avarok Benchmarks, DGX Spark)

| Config | Throughput | Notes |
|--------|----------:|-------|
| AWQ INT4 (NVIDIA official) | ~36 tok/s | baseline |
| **NVFP4 vanilla** | **~42 tok/s** | +20% vs AWQ |
| **NVFP4 + MTP** | **~67 tok/s avg** | peak 111.9 tok/s |

## Quick Start

### Option A: Pre-built Image (schneller)

```bash
cd turbo && ./pull.sh
```

### Option B: Build from Source (30-60 min)

```bash
cd turbo && ./build-podman.sh
```

### Run

```bash
# Vanilla (single GPU, ~42 tok/s)
./start.turbo.qwen3next.vanilla

# MTP Speculative Decoding (~67 tok/s)
./start.turbo.qwen3next.mtp

# Multi-Node TP=2 (DGX + PGX, RoCE)
# 1. Head auf DGX:
./start.turbo.qwen3next.tp2.head
# 2. Worker auf PGX:
scp start.turbo.qwen3next.tp2.worker flash@192.168.1.116:
ssh flash@192.168.1.116 ./start.turbo.qwen3next.tp2.worker
# 3. Serve auf DGX:
./start.turbo.qwen3next.tp2.serve
```

### Verify

```bash
# Health check
curl http://localhost:8011/health

# Models
curl http://localhost:8011/v1/models | jq

# Chat
curl -s http://localhost:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-next","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}' \
  | jq -r '.choices[0].message.content'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Disable FlashInfer MoE (use Marlin) |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `1` | Force Marlin MoE backend |
| `VLLM_NVFP4_GEMM_BACKEND` | `marlin` | Dense GEMM backend |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Memory allocator |

## Local Model

Das Modell liegt unter `/data/tensordata/qwen3-next-80B-Thinking-NVFP4` (45 GB, NVFP4).
Start-Scripts nutzen diesen Pfad als Default, ueberschreibbar mit `MODEL=...`.

## Files

| File | Purpose |
|------|---------|
| `turbo/` | Komplettes dgx-vllm Build-System (Dockerfile, Patches, Kernels) |
| `turbo/build-podman.sh` | Podman Build-Script |
| `turbo/pull.sh` | Pre-built Image Pull |
| `start.turbo.qwen3next.vanilla` | Single-GPU ohne Speculation |
| `start.turbo.qwen3next.mtp` | Single-GPU mit MTP (2 tokens) |
| `start.turbo.qwen3next.tp2.{head,worker,serve}` | Multi-Node TP=2 mit RoCE |
