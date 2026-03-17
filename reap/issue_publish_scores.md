# Publish per-expert saliency scores alongside pruned models

## Problem

REAP pruning requires a full forward pass of the unpruned model over calibration data to compute per-expert saliency scores. For large models (e.g. Qwen3.5-397B-A17B at ~800 GB BF16), this requires substantial GPU resources.

Once pruned, only the final result is published (e.g. `OpenMOSE/Qwen3.5-REAP-262B-A17B` at a fixed 35% compression ratio). Anyone wanting a different pruning ratio — say 25% instead of 35%, to better fit their hardware — must rerun the expensive forward pass from scratch.

## Proposal

Publish the per-expert saliency scores as a small artifact alongside the pruned model. The data is tiny:

```
60 layers × 512 experts × 1 float32 = 30,720 values ≈ 120 KB (JSON) or 240 KB (with metadata)
```

Format example:

```json
{
  "model": "Qwen/Qwen3.5-397B-A17B",
  "calibration_dataset": "OpenMOSE/reap-calib-mix",
  "method": "reap",
  "num_samples": 256,
  "scores": {
    "0": [0.0234, 0.1872, 0.0012, ...],
    "1": [0.0891, 0.0003, 0.2341, ...],
    ...
  }
}
```

Where `scores[layer][expert_idx]` is the saliency score (higher = more important).

## Benefits

1. **Custom pruning ratios**: Users can threshold the scores to match their hardware budget — e.g. prune to 280B for 2×80GB GPUs or 320B for 2×96GB GPUs, without rerunning calibration.

2. **Pruning analysis**: Researchers can study the score distribution, identify layers with many weak experts vs. layers where all experts are important, and understand model structure.

3. **Reproducibility**: The scores fully determine the pruning outcome for any given ratio, making results independently verifiable.

4. **Incremental experimentation**: Try multiple ratios cheaply to find the quality/size sweet spot for a given hardware target.

## Implementation

The scores are already computed in `prune.py` as `saliency_data` per layer. Saving them requires adding a few lines:

```python
# After computing saliency scores, before pruning:
scores = {str(layer): saliency_data.tolist() for layer, saliency_data in ...}
with open(os.path.join(output_dir, "expert_scores.json"), "w") as f:
    json.dump({"model": model_name, "method": prune_method, "scores": scores}, f)
```

## A note on what REAP pruning actually does

We've verified (via bit-identical weight fingerprinting across all 60 layers) that REAP-pruned models contain **unmodified copies** of the original experts. There is no fine-tuning, no weight adaptation, no distillation — the pruned model is simply the original with experts deleted and indices renumbered.

This means the entire value of a REAP-pruned model is determined by **which experts were removed**. That decision is fully encoded in the saliency scores. Publishing the scores is therefore equivalent to publishing the complete pruning recipe — a 120 KB file that replaces an 800 GB forward pass.

Without the scores, users are stuck with whatever ratio the publisher chose. With the scores, anyone can reconstruct any pruning variant from the original model in minutes, on CPU, without a single GPU.

## Context

This complements the request to save expert index mappings (#issue_expert_mapping). Together, published scores + mappings would make REAP pruning fully transparent and reproducible without requiring the original model's forward pass.
