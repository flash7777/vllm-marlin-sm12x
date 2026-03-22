#!/usr/bin/env python3
"""Guard RIY stats + HTTP server behind VLLM_RIY_MONITOR=1.

Without VLLM_RIY_MONITOR=1:
- No scatter_add_ stats in hot path (riy_freq_view not registered)
- No on_forward() / HTTP server
- No live mask check (get_mask_tensor)
- Profile mask via logit_mask at init still works (zero overhead)

With VLLM_RIY_MONITOR=1:
- Full stats collection + HTTP server + live mask

Usage: piped into container
  cat patch_riy_monitor_guard.py | podman exec -i CONTAINER python3 -
"""

import os

LAYER_FILE = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py"
ROUTER_FILE = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/base_router.py"

patched = 0

# === 1. layer.py: skip set_riy_state unless VLLM_RIY_MONITOR=1 ===
with open(LAYER_FILE) as f:
    src = f.read()

if "VLLM_RIY_MONITOR" not in src:
    # Guard the set_riy_state call
    old = """        if _layer_idx >= 0:
            _riy = get_riy_state()
            if _riy.enabled:"""
    new = """        if _layer_idx >= 0 and _os.environ.get("VLLM_RIY_MONITOR", "0") == "1":
            _riy = get_riy_state()
            if _riy.enabled:"""
    if old in src:
        src = src.replace(old, new)
        with open(LAYER_FILE, "w") as f:
            f.write(src)
        print("layer.py: PATCHED (stats buffers guarded by VLLM_RIY_MONITOR)")
        patched += 1
    else:
        print("layer.py: pattern not found")
else:
    print("layer.py: already patched")

# === 2. base_router.py: skip on_forward + live mask unless VLLM_RIY_MONITOR=1 ===
with open(ROUTER_FILE) as f:
    src = f.read()

if "VLLM_RIY_MONITOR" not in src:
    # Guard the Python-level block (Step 3c)
    old = """        # Step 3c: RIY — mask + HTTP server (Python-level, skipped during capture)
        if not _is_capturing():
            riy = get_riy_state()
            if riy.enabled and self.layer_idx >= 0:
                riy.on_forward()
                mask_t = riy.get_mask_tensor(self.layer_idx)
                if mask_t is not None:
                    mask_t = mask_t.to(topk_ids.device)
                    topk_weights = apply_riy_mask(
                        topk_weights, topk_ids, mask_t)"""
    new = """        # Step 3c: RIY — mask + HTTP server (only with VLLM_RIY_MONITOR=1)
        if not _is_capturing() and _riy_monitor_enabled:
            riy = get_riy_state()
            if riy.enabled and self.layer_idx >= 0:
                riy.on_forward()
                mask_t = riy.get_mask_tensor(self.layer_idx)
                if mask_t is not None:
                    mask_t = mask_t.to(topk_ids.device)
                    topk_weights = apply_riy_mask(
                        topk_weights, topk_ids, mask_t)"""
    if old in src:
        # Also add the module-level flag
        import_marker = "from vllm.model_executor.layers.fused_moe.riy import"
        if import_marker in src:
            src = src.replace(
                import_marker,
                "import os as _os_riy\n"
                "_riy_monitor_enabled = _os_riy.environ.get('VLLM_RIY_MONITOR', '0') == '1'\n"
                + import_marker
            )
        src = src.replace(old, new)
        with open(ROUTER_FILE, "w") as f:
            f.write(src)
        print("base_router.py: PATCHED (on_forward + live mask guarded by VLLM_RIY_MONITOR)")
        patched += 1
    else:
        print("base_router.py: pattern not found")
else:
    print("base_router.py: already patched")

print(f"Total: {patched} files patched")
