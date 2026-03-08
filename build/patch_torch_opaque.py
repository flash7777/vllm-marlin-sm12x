#!/usr/bin/env python3
"""Patch vLLM torch_utils.py: torch._opaque_base may not exist in NVIDIA's PyTorch 2.11.0a0.

vLLM checks `is_torch_equal_or_newer("2.11.0.dev")` but NVIDIA's 2.11.0a0+nv26.02
reports True while lacking torch._opaque_base. Fix: try the import directly.
"""
import glob

targets = glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/utils/torch_utils.py")
if not targets:
    print("SKIP: torch_utils.py not found")
    exit(0)

for path in targets:
    src = open(path).read()

    old = """HAS_OPAQUE_TYPE = is_torch_equal_or_newer("2.11.0.dev")

if HAS_OPAQUE_TYPE:
    from torch._opaque_base import OpaqueBase
else:
    OpaqueBase = object  # type: ignore[misc, assignment]"""

    new = """try:
    from torch._opaque_base import OpaqueBase
    HAS_OPAQUE_TYPE = True
except ImportError:
    HAS_OPAQUE_TYPE = False
    OpaqueBase = object  # type: ignore[misc, assignment]"""

    if old in src:
        src = src.replace(old, new)
        open(path, 'w').write(src)
        print(f"OK: Patched {path} — torch._opaque_base import made robust")
    elif "try:" in src and "torch._opaque_base" in src:
        print(f"SKIP: {path} already patched")
    else:
        print(f"WARN: Pattern not found in {path}")
