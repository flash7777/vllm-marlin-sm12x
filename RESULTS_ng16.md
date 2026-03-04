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

**WORKS** at TP=2 with Marlin N-padding patch (`build/patch_marlin_padding.py`).

Qwen3-Next's `linear_attn.in_proj_ba` layer has `output_size_per_partition=32` at TP=2.
The padding patch zero-pads weights to 64 before `gptq_marlin_repack`, runs the kernel with
padded N, then slices output back to actual N. 66 layers are padded per rank.

### Performance (n=5, torch.compile + CUDA Graphs)

| Prompt | tok/s |
|--------|-------|
| short | 37.6 |
| medium | 86.2 |
| long | **87.7** |
| Math | 46/50 (**92%**) |

### vs enforce-eager

| Mode | long tok/s | Math |
|------|-----------|------|
| enforce-eager | 26.3 | 94% |
| **torch.compile** | **87.7** | 92% |

**3.3× faster** with torch.compile. Required two patches to fix `torch.Size` crossing
AOT autograd split boundaries:

1. `build/patch_qwen3next_compile.py` — Fix `z_shape_og = z.shape` in GatedDeltaNet
   (crosses `vllm::gdn_attention_core` split)
2. `build/patch_qwen3next_compile_v2.py` — Fix `orig_shape = hidden_states.shape` in
   SparseMoeBlock (crosses `vllm::all_reduce_with_output` split)

### EAGLE3 NST=1 (Aurora-Spec drafter)

| Prompt | tok/s |
|--------|-------|
| short | 34.5 |
| medium | 59.8 |
| long | **78.7** |
| Math | 46/50 (**92%**) |

EAGLE3 is **slower** than vanilla compiled (78.7 vs 87.7 tok/s). The Aurora drafter
(`togethercomputer/Aurora-Spec-Qwen3-Coder-Next-FP8`, 0.5B, 1 GB) was trained for the
FP8 model, not INT4 AutoRound. On DGX (273 GB/s, bandwidth-limited), the draft overhead
exceeds the acceptance rate benefit. Required `patch_qwen3next_eagle3.py` to add
`SupportsEagle3` interface to Qwen3NextForCausalLM.

**Note**: No MTP available for Qwen3-Next (`num_nextn_predict_layers: None`).

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

## Qwen3-Coder-Next: Marlin N-Padding Solution

### Problem

The Marlin CUDA kernel requires `output_size_per_partition % 64 == 0`. Qwen3-Next's
`linear_attn.in_proj_ba` layer has only 32 output channels at TP=2. The `min_thread_n=64`
constraint comes from the kernel's warp-scheduling: `thread_n_blocks / 4` integer division
breaks at thread_n=32.

### Solution: Zero-Padding (patch_marlin_padding.py)

Instead of modifying the CUDA kernel, we zero-pad weights at the Python level:

1. **Before repack**: Pad raw qweight `[K/pack, N]` from N=32 to N=64 along dim=1
2. **Before permute**: Pad scales `[groups, N]` from N=32 to N=64 along dim=1
3. **At inference**: Pass `output_size_per_partition=64` to the Marlin kernel
4. **After inference**: Slice output `[batch, 64]` → `[batch, 32]`

This is mathematically correct: zero-padded weight columns produce zero output columns
that are sliced away. The patch is applied at model loading time, so inference overhead
is just the extra 32 output columns per padded layer (negligible).

### torch.compile Fix (SOLVED)

Qwen3-Next crashed during `torch.compile` graph capture due to `torch.Size` objects crossing
AOT autograd split boundaries. Fixed by two patches that replace `.shape` with `.size()` calls:
- `patch_qwen3next_compile.py`: GatedDeltaNet `z_shape_og` → static dims
- `patch_qwen3next_compile_v2.py`: MoE `orig_shape` → `num_tokens, hidden_dim`

Result: **87.7 tok/s** compiled vs 26.3 tok/s eager (**3.3× improvement**).
