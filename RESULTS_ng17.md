# vllm-ng17 Benchmark Results

## Image

| Component | Version |
|---|---|
| Base | nvcr.io/nvidia/vllm:26.02-py3 |
| vLLM | 0.17.0 (tag v0.17.0) |
| PyTorch | 2.11.0a0 (nv26.02) |
| CUDA | 13.1 |
| transformers | TBD |
| compressed_tensors | TBD |
| NCCL | TBD |

## Hardware

- DGX Spark (SM121) + PGX ThinkStation (SM121), TP=2
- 200 Gbps RoCE (ConnectX-7, QSFP56)
- UF17 EAGER_ALLREDUCE enabled

## Key Changes vs ng16

- vLLM 0.16.0rc2 -> 0.17.0
- Native Qwen3.5 support (GDN fusion, MTP, reasoning parser)
- FlashAttention 4 backend
- Model Runner V2 (EAGLE3 CUDA Graphs, Pipeline Parallel)
- SM120 FP8 GEMM optimization

## Patches Applied

| Patch | Status | Notes |
|---|---|---|
| Marlin SM12x | TBD | |
| Patch 3 (compressed_tensors extra=ignore) | TBD | may be upstream |
| Patch 4 (MoE fallback) | TBD | may be upstream |
| Patch 7 (AutoWeightsLoader buffers) | TBD | may be upstream |
| Patch 8 (e_score_correction_bias + MTP) | TBD | may be upstream |
| Patch 11 (Anthropic streaming) | TBD | may be upstream |
| patch_streaming.py | TBD | |
| patch_mtp_nvfp4_exclusion.py | TBD | |
| patch_mtp_bf16_weights.py | TBD | |
| patch_marlin_padding.py | TBD | |
| patch_qwen3next_compile.py | TBD | may be upstream |
| patch_qwen35_compile.py | TBD | may be upstream |
| patch_qwen3next_eagle3.py | TBD | may be upstream |
| UF17 EAGER_ALLREDUCE | TBD | |
| torchrun serve | installed | |

## Benchmarks

### Qwen3.5-122B-A10B GPTQ-Int4 (TP=2)

TBD — build image first, then benchmark.

### Qwen3.5-397B-A17B INT4 AutoRound (TP=2)

TBD

### GLM-4.7-Flash INT4 + MTP (TP=2)

TBD — compare vs ng16 (105.4 tok/s).

### Qwen3-Coder-30B INT4 + EAGLE3 (TP=2)

TBD — compare vs ng16 (116.9 tok/s).
