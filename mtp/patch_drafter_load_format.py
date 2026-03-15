#!/usr/bin/env python3
"""Patch vLLM to use load_format='auto' for MTP drafter instead of fastsafetensors.

Problem: fastsafetensors opens ALL safetensors in the index in parallel (40 main
shards + MTP = 200+ GB mmap). On UMA this kills the system even though the drafter
only needs ~1.7 GB.

Fix: Force draft_load_config.load_format = 'auto' (sequential, single-shard loader).

Usage: cat patch_drafter_load_format.py | podman exec -i CONTAINER python3 -
"""

MARKER = '[MTP-LOADFMT]'

path = '/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py'

with open(path) as f:
    src = f.read()

if MARKER in src:
    print('eagle.py: already patched (load_format)')
else:
    # Patch _get_model to override load_config with load_format='auto'
    old = '                model = get_model('
    new = f'''                # {MARKER} Force auto loader for drafter (fastsafetensors mmaps ALL shards = OOM on UMA)
                import copy as _copy
                from vllm.config import LoadConfig as _LC
                _draft_lc = self.speculative_config.draft_load_config
                if _draft_lc is None:
                    _draft_lc = _copy.copy(self.vllm_config.load_config)
                if str(getattr(_draft_lc, 'load_format', '')) == 'fastsafetensors':
                    _draft_lc.load_format = 'auto'
                    import sys; sys.stderr.write(f'{MARKER} Forced drafter load_format=auto (was fastsafetensors)\\n'); sys.stderr.flush()
                model = get_model('''

    # Also add load_config parameter
    old_call = '''                model = get_model(
                    vllm_config=self.vllm_config,
                    model_config=self.speculative_config.draft_model_config,
                    load_config=self.speculative_config.draft_load_config,'''
    new_call = f'''                # {MARKER} Force auto loader for drafter (fastsafetensors mmaps ALL shards = OOM on UMA)
                import copy as _copy
                _draft_lc = self.speculative_config.draft_load_config
                if _draft_lc is None:
                    _draft_lc = _copy.copy(self.vllm_config.load_config)
                if str(getattr(_draft_lc, 'load_format', '')) == 'fastsafetensors':
                    _draft_lc.load_format = 'auto'
                    import sys; sys.stderr.write(f'{MARKER} Forced drafter load_format=auto (was fastsafetensors)\\n'); sys.stderr.flush()
                model = get_model(
                    vllm_config=self.vllm_config,
                    model_config=self.speculative_config.draft_model_config,
                    load_config=_draft_lc,'''

    if old_call in src:
        src = src.replace(old_call, new_call)
        with open(path, 'w') as f:
            f.write(src)
        print('eagle.py: patched (drafter load_format=auto)')
    else:
        print('eagle.py: pattern not found, trying simpler patch...')
        # Simpler: just add before the get_model call
        if old in src and MARKER not in src:
            src = src.replace(old, new, 1)
            with open(path, 'w') as f:
                f.write(src)
            print('eagle.py: patched (simple)')
        else:
            print('eagle.py: could not patch')
