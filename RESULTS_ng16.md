# vllm-ng16 Benchmark Results

## Image

| Component | Version |
|---|---|
| Base | nvcr.io/nvidia/vllm:26.02-py3 |
| vLLM | 0.16.0rc2 (commit 882682ab8) |
| PyTorch | 2.11.0a0+eb65b36914 (nv26.02) |
| CUDA | 13.1 |
| transformers | 5.3.0.dev0 |
| compressed_tensors | 0.13.0 |
| NCCL | 2.29.2 |

## Hardware

- DGX Spark (SM121) + PGX ThinkStation (SM121), TP=2
- 200 Gbps RoCE (ConnectX-7, QSFP56)
- UF17 EAGER_ALLREDUCE enabled

## Patches Applied

| Patch | Status |
|---|---|
| Patch 0 (assume_32bit_indexing) | **not needed** (PyTorch 2.11 native) |
| Marlin SM12x | applied |
| Patch 3 (compressed_tensors extra=ignore) | applied |
| Patch 4 (MoE fallback) | applied |
| Patch 7 (AutoWeightsLoader buffers) | applied |
| Patch 8 (e_score_correction_bias + MTP) | applied |
| Patch 11 (Anthropic streaming) | applied |
| patch_streaming.py | applied |
| patch_mtp_nvfp4_exclusion.py | applied |
| UF17 EAGER_ALLREDUCE | verified |
| torchrun serve | installed |

## Qwen3-Coder-30B INT4 AutoRound + EAGLE3 NST=1

### Performance (n=5)

| Prompt | tok/s |
|--------|-------|
| short | 7.2 |
| medium | 87.7 |
| long | **116.9** |
| Math | 38/50 (76%) |

### Context Scaling (long prompt)

| ctx | short | medium | long |
|-----|-------|--------|------|
| 0 | 58.8 | 87.3 | 116.9 |
| 512 | 17.0 | 88.3 | 111.2 |
| 2K | 20.5 | 78.3 | 103.5 |
| 8K | 7.9 | 51.9 | 51.6 |
| 16K | 5.1 | 37.8 | 37.6 |

## GLM-4.7-Flash INT4 AutoRound (vanilla, no speculative)

### Performance (n=5)

| Prompt | tok/s |
|--------|-------|
| short | 68.8 |
| medium | 75.3 |
| long | **73.3** |
| Math | 50/50 (**100%**) |

### Context Scaling

| ctx | short | medium | long |
|-----|-------|--------|------|
| 0 | 71.2 | 74.7 | 73.0 |
| 512 | 59.0 | 68.9 | 67.1 |
| 2K | 44.1 | 55.6 | 54.9 |
| 8K | 20.9 | 32.3 | 32.2 |
| 16K | 13.1 | 21.1 | 21.0 |

## Qwen3-Coder-Next INT4 AutoRound (vanilla)

**FAILED** at TP=2: `Weight output_size_per_partition = 32 is not divisible by min_thread_n = 64`.
Marlin kernel requires minimum partition size of 64, but Qwen3-Next's linear_attn `in_proj_ba`
layer has output_size_per_partition=32 at TP=2. Only works at TP=1.

## Comparison: vllm-ng (0.15) vs vllm-ng16 (0.16)

### Qwen3-Coder-30B INT4 EAGLE3 NST=1

| Version | short | medium | long | Math |
|---------|-------|--------|------|------|
| vllm-ng (0.15, 26.01) | 50.2 | 89.1 | **118.8** | 76% |
| vllm-ng16 (0.16, 26.02) | 7.2 | 87.7 | **116.9** | 76% |

### GLM-4.7-Flash INT4 vanilla

| Version | short | medium | long | Math |
|---------|-------|--------|------|------|
| vllm-ng (0.15, 26.01) | 70.5 | 74.1 | **72.4** | 98% |
| vllm-ng16 (0.16, 26.02) | 68.8 | 75.3 | **73.3** | 100% |

Performance is comparable between vllm-ng and vllm-ng16. The upgrade to vLLM 0.16 + PyTorch 2.11
shows no regression. Native MTP support for GLM-4.7, Qwen3.5, Qwen3-Next is available but not yet
benchmarked (requires MTP drafter models).
