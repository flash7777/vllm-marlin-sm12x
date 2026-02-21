# NVFP4 Benchmark Results — Qwen3-Coder-30B-A3B on DGX Spark (SM121)

Model: `Qwen3-Coder-30B-A3B-Instruct` NVFP4 (`modelopt_fp4`)
Platform: DGX Spark (GB10, SM121, 128 GB Unified Memory, 273 GB/s)
Benchmark: `bench.py --context` (deterministic, seed=42)
Port: 8011

## Benchmark Commands

```bash
# Vanilla
python3 bench.py --url http://localhost:8011 --model qwen3-coder --context \
  --label "turbo NVFP4 Vanilla (DGX)"

# EAGLE3 NST=1
python3 bench.py --url http://localhost:8011 --model qwen3-coder --context \
  --label "turbo NVFP4 +EAGLE3 NST=1 (DGX)"

# MTP NST=1
python3 bench.py --url http://localhost:8011 --model qwen3-coder --context \
  --label "turbo NVFP4 +MTP NST=1 (DGX)"

# MTP NST=2
python3 bench.py --url http://localhost:8011 --model qwen3-coder --context \
  --label "turbo NVFP4 +MTP NST=2 (DGX)"
```

---

## Baseline: vLLM-next FLASHINFER_CUTLASS (existing)

Engine: vLLM-next (26.01-py3 base, CUTLASS 4.3.5)
MoE Kernel: FLASHINFER_CUTLASS
Build: `TORCH_CUDA_ARCH_LIST="12.0a;12.1a"`, `FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"`

### vLLM-next NVFP4 Vanilla

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   | 37.3  | 64.5   | 65.0 |
| ctx=512 | 27.1  | 63.8   | 64.2 |
| ctx=2K  | 19.0  | 62.5   | 63.0 |
| ctx=8K  | 5.9   | 57.8   | 59.0 |
| ctx=16K | 3.4   | 53.5   | 54.8 |

Math: 74%

### vLLM-next NVFP4 +EAGLE3 NST=1

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   | 30.4  | 54.2   | 68.1 |
| ctx=512 | 20.5  | 52.5   | 66.0 |
| ctx=2K  | 15.2  | 51.0   | 65.2 |
| ctx=8K  | 5.7   | 44.8   | 45.5 |
| ctx=16K | 3.4   | 42.6   | 43.8 |

Math: 72%

---

## turbo (Avarok, Dual-Arch SM120a+SM121a)

Engine: vLLM 0.16.0rc2 (Avarok dgx-vllm)
MoE Kernel: VLLM_CUTLASS (v109 Pingpong)
Build: `TORCH_CUDA_ARCH_LIST="12.0a;12.1a"`
E2M1 Fallback: Threshold if-else (7 branches)
Image: `vllm-turbo:dual`
Env: `VLLM_USE_FLASHINFER_MOE_FP4=0`

### turbo SM121-only (previous build, reference)

| Context | short | medium | long | Math |
|---------|-------|--------|------|------|
| **Vanilla** | | | | |
| ctx=0   | 39.1  | 67.5   | 68.0 | 70%  |
| ctx=512 | 28.2  | 66.6   | 67.4 |      |
| ctx=2K  | 19.8  | 65.4   | 66.1 |      |
| ctx=8K  | 6.1   | 60.5   | 61.9 |      |
| ctx=16K | 3.6   | 55.9   | 57.2 |      |
| **+EAGLE3 NST=1** | | | | |
| ctx=0   | 33.3  | 56.8   | 73.9 | 78%  |
| ctx=512 | 21.6  | 54.8   | 71.4 |      |
| ctx=2K  | 15.9  | 53.3   | 70.8 |      |
| ctx=8K  | 6.0   | 46.7   | 48.4 |      |
| ctx=16K | 3.6   | 44.6   | 45.9 |      |

### turbo Dual-Arch Vanilla

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   |       |        |      |
| ctx=512 |       |        |      |
| ctx=2K  |       |        |      |
| ctx=8K  |       |        |      |
| ctx=16K |       |        |      |

Math: ?%

### turbo Dual-Arch +EAGLE3 NST=1

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   |       |        |      |
| ctx=512 |       |        |      |
| ctx=2K  |       |        |      |
| ctx=8K  |       |        |      |
| ctx=16K |       |        |      |

Math: ?%

### turbo Dual-Arch +MTP NST=1

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   |       |        |      |
| ctx=512 |       |        |      |
| ctx=2K  |       |        |      |
| ctx=8K  |       |        |      |
| ctx=16K |       |        |      |

Math: ?%

### turbo Dual-Arch +MTP NST=2

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   |       |        |      |
| ctx=512 |       |        |      |
| ctx=2K  |       |        |      |
| ctx=8K  |       |        |      |
| ctx=16K |       |        |      |

Math: ?%

---

## turbo2 (Branchless E2M1, Dual-Arch SM120a+SM121a)

Engine: vLLM 0.16.0rc2 (Avarok dgx-vllm)
MoE Kernel: VLLM_CUTLASS (v109 Pingpong)
Build: `TORCH_CUDA_ARCH_LIST="12.0a;12.1a"`
E2M1 Fallback: Branchless predicate sum (0 branches)
Image: `vllm-turbo2:dual`
Env: `VLLM_USE_FLASHINFER_MOE_FP4=0`

### turbo2 Dual-Arch Vanilla

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   |       |        |      |
| ctx=512 |       |        |      |
| ctx=2K  |       |        |      |
| ctx=8K  |       |        |      |
| ctx=16K |       |        |      |

Math: ?%

### turbo2 Dual-Arch +EAGLE3 NST=1

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   |       |        |      |
| ctx=512 |       |        |      |
| ctx=2K  |       |        |      |
| ctx=8K  |       |        |      |
| ctx=16K |       |        |      |

Math: ?%

### turbo2 Dual-Arch +MTP NST=1

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   |       |        |      |
| ctx=512 |       |        |      |
| ctx=2K  |       |        |      |
| ctx=8K  |       |        |      |
| ctx=16K |       |        |      |

Math: ?%

### turbo2 Dual-Arch +MTP NST=2

| Context | short | medium | long |
|---------|-------|--------|------|
| ctx=0   |       |        |      |
| ctx=512 |       |        |      |
| ctx=2K  |       |        |      |
| ctx=8K  |       |        |      |
| ctx=16K |       |        |      |

Math: ?%

---

## Vergleich (ctx=0 long tok/s)

| Engine | MoE Kernel | E2M1 | Vanilla | +EAGLE3 | +MTP=1 | +MTP=2 | Math V/E/M1/M2 |
|--------|-----------|------|---------|---------|--------|--------|-----------------|
| vLLM-next | FI_CUTLASS | HW+exmy | 65.0 | 68.1 | - | - | 74/72/-/- |
| turbo SM121 | VLLM_CUTLASS | if-else | 68.0 | 73.9 | - | - | 70/78/-/- |
| turbo dual | VLLM_CUTLASS | if-else | | | | | |
| turbo2 dual | VLLM_CUTLASS | branchless | | | | | |

### Delta vs vLLM-next Vanilla (65.0 tok/s)

| Variante | Vanilla | +EAGLE3 | +MTP=1 | +MTP=2 |
|----------|---------|---------|--------|--------|
| turbo SM121 | +4.6% | +13.7% | - | - |
| turbo dual | | | | |
| turbo2 dual | | | | |

---

## Container Start-Befehle

### turbo/turbo2 Vanilla
```bash
podman run -d --replace --name vllm-turbo \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -p 8011:8000 \
  -v /data/tensordata:/data/tensordata \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  localhost/vllm-turbo:dual \
  vllm serve /data/tensordata/Qwen3-Coder-30B-A3B-Instruct-NVFP4 \
    --host 0.0.0.0 --port 8000 \
    --served-model-name qwen3-coder \
    --gpu-memory-utilization 0.05 --kv-cache-memory-bytes 10G \
    --max-model-len 32768 \
    --trust-remote-code
```

### +EAGLE3 NST=1
```bash
  ... (wie oben) plus:
    --speculative-model /data/tensordata/Qwen3-Coder-30B-A3B-EAGLE3 \
    --speculative-method eagle3 \
    --num-speculative-tokens 1
```

### +MTP NST=1
```bash
  ... (wie oben) plus:
    --speculative-method mtp \
    --num-speculative-tokens 1
```

### +MTP NST=2
```bash
  ... (wie oben) plus:
    --speculative-method mtp \
    --num-speculative-tokens 2
```
