# Save expert mapping (original → pruned indices) in pruned model config

## Problem

When REAP prunes and re-indexes experts, the mapping from original expert indices to pruned indices is lost. The `retained_expert_indicies` are computed per layer in `prune.py` but not saved anywhere in the output model.

## Why this matters

Downstream users of pruned models need this mapping for:

1. **Quantization with original-model reference**: AutoRound/GPTQ calibration could compare quantized pruned layers against the unpruned original (same architecture, same dimensions). Without the mapping, this cross-model calibration is impossible.

2. **MTP (Multi-Token Prediction) layer transplantation**: The original model's MTP layers have experts matching the original indexing. With a mapping, MTP experts could be correctly pruned/reordered to match the pruned model.

3. **Weight verification**: Confirming that pruned experts are bit-identical to their originals (e.g., via fingerprinting) requires knowing which original index each pruned expert corresponds to.

4. **Analysis**: Understanding per-layer pruning patterns, expert importance distributions, etc.

## Current workaround

We're building the mapping by fingerprinting expert weights (comparing raw bytes between the pruned and original model shard-by-shard). This works but requires downloading both full models and is slow.

## Proposed solution

Save the per-layer mapping in the model config or as a separate JSON file. For example, in `config.json`:

```json
"expert_mapping": {
  "0": [0, 3, 7, 12, ...],
  "1": [1, 5, 8, 15, ...],
  ...
}
```

Where `expert_mapping[layer][pruned_idx] = original_idx`.

The data is already available in `prune.py` as `retained_expert_indicies` — it just needs to be persisted.

## Context

We're working with `OpenMOSE/Qwen3.5-REAP-262B-A17B` (pruned from `Qwen/Qwen3.5-397B-A17B`) and trying to:
- Create a vLLM-compatible INT4 quantization with full-precision shared experts
- Transplant MTP layers from the original 397B model
- Calibrate quantization against the unpruned original
