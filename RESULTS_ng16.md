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

### GLM-4.7-Flash INT4

| Version | Spec | short | medium | long | Math |
|---------|------|-------|--------|------|------|
| vllm-ng (0.15, 26.01) | vanilla | 70.5 | 74.1 | **72.4** | 98% |
| vllm-ng16 (0.16, 26.02) | vanilla | 68.8 | 75.3 | **73.3** | 100% |
| vllm-ng16 (0.16, 26.02) | MTP NST=1 | 90.8 | 106.2 | **105.4** | 100% |

MTP gives **+44%** throughput over vanilla (105.4 vs 73.3 tok/s) with no accuracy loss.
Performance is comparable between vllm-ng and vllm-ng16 for vanilla. The upgrade to vLLM 0.16
enables native MTP support.

## GLM-4.7-Flash INT4 AutoRound + MTP NST=1

### Performance (n=5)

| Prompt | tok/s |
|--------|-------|
| short | 90.8 |
| medium | 106.2 |
| long | **105.4** |
| Math | 50/50 (**100%**) |

### Context Scaling

| ctx | short | medium | long |
|-----|-------|--------|------|
| 0 | 88.1 | 105.4 | 107.8 |
| 512 | 75.0 | 92.4 | 102.5 |
| 2K | 58.2 | 79.7 | 83.6 |
| 8K | 20.3 | 51.5 | 51.2 |
| 16K | 16.1 | 34.3 | 36.0 |

### MTP Setup

GLM-4.7-Flash has native MTP support (`num_nextn_predict_layers: 1`). The MTP layer is an extra
decoder layer (layer 47) that acts as a single-token draft predictor — no separate drafter model needed.

**Prerequisites**:
1. Extract BF16 MTP weights into INT4 model: `build/extract_mtp_weights.py`
2. Apply `build/patch_mtp_bf16_weights.py` (patches `eagle.py` to temporarily disable
   quant_config during MTP model loading)
3. Set `VLLM_MTP_FORCE_BF16=1` at runtime

**Why the patch is needed**: INT4 AutoRound drops MTP weights (only layers 0-46 quantized).
BF16 MTP weights must be loaded with `quant_config=None` so FusedMoE creates unquantized
params (`w13_weight`/`w2_weight`) instead of GPTQ params (`qweight`/`qzeros`/`scales`).
The patch temporarily sets `vllm_config.quant_config=None` during MTP model construction,
using `object.__setattr__` to bypass `VllmConfig.__post_init__` re-calculation.

## Qwen3-Coder-Next: Marlin min_thread_n=64 Analysis

### Can Marlin be patched for output_size_per_partition < 64?

**No.** The `min_thread_n=64` constraint is a software design limitation deeply embedded in the
Marlin CUDA kernel's warp-scheduling architecture:

- `thread_n_blocks = thread_n / 16` (tile_size=16)
- Multiple kernel calculations use `thread_n_blocks / 4` as integer division
- With thread_n=32: `thread_n_blocks=2`, `2/4=0` → **integer division breaks**
- Shared memory strides, warp group indexing, and register allocation all depend on this

The kernel supports only thread_n values of {64, 128, 256}. Reducing to 32 would require
a complete kernel rewrite of the warp-scheduling math, with high risk of correctness issues
and performance regression.

**Workarounds for Qwen3-Coder-Next at TP=2**:
1. TP=1 (output_size_per_partition=64, works)
2. Use `--quantization gptq` instead of Marlin (no min_thread_n constraint, ~5-10% slower)
3. Pad the problematic layer to 64 during quantization (model modification)
