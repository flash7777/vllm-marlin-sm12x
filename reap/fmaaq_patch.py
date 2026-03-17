"""FMAAQ: Monkey-patch AutoRound to use a teacher model's block outputs as reference.

Instead of comparing the quantized student block output against the student's own BF16 output,
we compare against the teacher (unpruned 397B) block output. This compensates for both
pruning loss and quantization error simultaneously.

Usage:
    import fmaaq_patch
    fmaaq_patch.patch_autoround(teacher_path="/data/tensordata/Qwen3.5-397B-A17B")
    # Then use AutoRound normally — it will use teacher outputs as reference
"""

import os
import gc
import json
import torch
import logging
from safetensors import safe_open
from typing import Union

logger = logging.getLogger("fmaaq")

# Global state for the teacher
_teacher_state = {
    "path": None,
    "index": None,         # weight_map from model.safetensors.index.json
    "config": None,
    "model_class": None,
    "block_cache": {},      # layer_idx -> loaded teacher block on CPU
}


def _load_teacher_index(teacher_path: str):
    """Load teacher model's safetensors index and config."""
    idx_path = os.path.join(teacher_path, "model.safetensors.index.json")
    with open(idx_path, "rb") as f:
        raw = f.read()
    # Handle potential encoding issues
    _teacher_state["index"] = json.loads(raw.decode("utf-8", errors="replace"))

    cfg_path = os.path.join(teacher_path, "config.json")
    with open(cfg_path) as f:
        _teacher_state["config"] = json.load(f)

    _teacher_state["path"] = teacher_path
    logger.info(f"FMAAQ: Teacher index loaded from {teacher_path}")


def _get_teacher_block(layer_idx: int, device: torch.device) -> torch.nn.Module:
    """Load a single teacher transformer block by layer index.

    Loads the block from safetensors, materializes it on the given device,
    and returns it in eval mode.
    """
    if layer_idx in _teacher_state["block_cache"]:
        block = _teacher_state["block_cache"][layer_idx]
        block.to(device)
        return block

    teacher_path = _teacher_state["path"]
    config = _teacher_state["config"]

    # Load model class
    if _teacher_state["model_class"] is None:
        from transformers import AutoModelForCausalLM, AutoConfig
        cfg = AutoConfig.from_pretrained(teacher_path)
        # Get the model class without loading weights
        _teacher_state["model_class"] = AutoModelForCausalLM
        _teacher_state["auto_config"] = cfg

    cfg = _teacher_state["auto_config"]

    # We need to load just one transformer block. Strategy:
    # 1. Create the model on meta device
    # 2. Extract the block
    # 3. Load only the relevant weights from safetensors
    from transformers import AutoModelForCausalLM
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)

    # Find the block in the model
    # Qwen3.5 MoE: model.language_model.layers[layer_idx]
    block = None
    for name in ["model.language_model.layers", "model.layers", "transformer.h"]:
        try:
            parts = name.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            block = obj[layer_idx]
            block_prefix = f"{name}.{layer_idx}."
            break
        except (AttributeError, IndexError):
            continue

    if block is None:
        raise RuntimeError(f"Could not find transformer block {layer_idx} in teacher model")

    # Find which safetensors shards contain this block's weights
    wm = _teacher_state["index"]["weight_map"]
    block_tensors = {k: v for k, v in wm.items() if k.startswith(block_prefix)}
    shards_needed = set(block_tensors.values())

    # Load weights into the block
    state_dict = {}
    for shard in shards_needed:
        shard_path = os.path.join(teacher_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                if tensor_name.startswith(block_prefix):
                    # Strip block prefix for loading into the block module
                    local_name = tensor_name[len(block_prefix):]
                    state_dict[local_name] = f.get_tensor(tensor_name)

    # Materialize the block from meta to real tensors
    block.to_empty(device="cpu")
    missing, unexpected = block.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"FMAAQ: Teacher block {layer_idx} missing keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"FMAAQ: Teacher block {layer_idx} unexpected keys: {unexpected[:5]}...")

    block.eval()
    block.to(device)

    # Free the meta model
    del model
    gc.collect()

    logger.info(f"FMAAQ: Loaded teacher block {layer_idx} ({len(state_dict)} tensors)")
    return block


def _release_teacher_block(layer_idx: int):
    """Release a teacher block from GPU memory."""
    if layer_idx in _teacher_state["block_cache"]:
        del _teacher_state["block_cache"][layer_idx]
    gc.collect()
    torch.cuda.empty_cache()


def _patched_get_block_outputs(original_fn):
    """Create a patched version of _get_block_outputs that can use teacher blocks.

    Uses a counter to only replace the FIRST call per block (the reference output).
    Subsequent calls (quantized output, q_input output) use the student block normally.
    """

    def wrapper(self, block, input_ids, input_others, bs, device, cache_device, **kwargs):
        teacher_layer_idx = getattr(block, "_fmaaq_teacher_layer_idx", None)
        call_count = getattr(self, "_fmaaq_call_count", 0)

        if teacher_layer_idx is not None and call_count == 0:
            # First call = reference output → use teacher
            self._fmaaq_call_count = call_count + 1
            logger.info(f"FMAAQ: Using teacher block {teacher_layer_idx} for reference output")
            teacher_block = _get_teacher_block(teacher_layer_idx, device)
            try:
                result = original_fn(self, teacher_block, input_ids, input_others, bs, device, cache_device, **kwargs)
            finally:
                teacher_block.to("cpu")
                _release_teacher_block(teacher_layer_idx)
            return result
        else:
            # Subsequent calls = quantized/q_input output → use student
            self._fmaaq_call_count = call_count + 1
            return original_fn(self, block, input_ids, input_others, bs, device, cache_device, **kwargs)

    return wrapper


def _patched_quantize_block(original_fn):
    """Patch _quantize_block to tag blocks with teacher layer index and reset call counter."""

    def wrapper(self, block, input_ids, input_others, *args, **kwargs):
        block_name = getattr(block, "_global_name", "")
        import re
        m = re.search(r"layers\.(\d+)", block_name)
        if m and _teacher_state["path"] is not None:
            layer_idx = int(m.group(1))
            block._fmaaq_teacher_layer_idx = layer_idx
            logger.info(f"FMAAQ: Block {block_name} → teacher layer {layer_idx}")
        else:
            block._fmaaq_teacher_layer_idx = None

        # Reset call counter — first _get_block_outputs call will use teacher
        self._fmaaq_call_count = 0

        result = original_fn(self, block, input_ids, input_others, *args, **kwargs)

        # Clean up
        if hasattr(block, "_fmaaq_teacher_layer_idx"):
            del block._fmaaq_teacher_layer_idx
        if hasattr(self, "_fmaaq_call_count"):
            del self._fmaaq_call_count

        return result

    return wrapper


def patch_autoround(teacher_path: str):
    """Apply FMAAQ monkey-patches to AutoRound.

    Args:
        teacher_path: Path to the unpruned teacher model (e.g. Qwen3.5-397B-A17B)
    """
    _load_teacher_index(teacher_path)

    from auto_round.compressors.base import BaseCompressor

    # Save originals
    orig_get_block_outputs = BaseCompressor._get_block_outputs
    orig_quantize_block = BaseCompressor._quantize_block

    # Apply patches
    BaseCompressor._get_block_outputs = _patched_get_block_outputs(orig_get_block_outputs)
    BaseCompressor._quantize_block = _patched_quantize_block(orig_quantize_block)

    logger.info(f"FMAAQ: Patched AutoRound to use teacher from {teacher_path}")
