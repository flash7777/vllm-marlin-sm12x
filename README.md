# vllm-marlin-sm12x

Patches for [vllm-project/vllm](https://github.com/vllm-project/vllm) enabling Marlin W4A8-FP8 on SM121 (DGX Spark GB10) + TMA load module for SM120/SM121.

## Base

- **Upstream**: `vllm-project/vllm` @ `fd267bc7b7cd3d001ac5a893eacb9e56ff256822`
- **Branch**: `main`

## Patches

### `marlin_sm12x.patch`

Applies to 2 files:

| File | Change |
|------|--------|
| `csrc/quantization/marlin/marlin.cu` | SM121 fix: `major_capability == 12` statt `== 120` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py` | `is_device_capability_family(120)` statt `is_device_capability(120)` |

### `marlin_tma.cuh`

New file: TMA (Tensor Memory Accelerator) load primitives for Blackwell (SM120/SM121).
Place at `csrc/quantization/marlin/marlin_tma.cuh`.

## Apply

```bash
cd vllm
git apply ../marlin_sm12x.patch
cp ../marlin_tma.cuh csrc/quantization/marlin/
```

## Activate W4A8-FP8

```bash
export VLLM_MARLIN_INPUT_DTYPE=fp8
```
