# vllm-marlin-sm12x

Patches for [vllm-project/vllm](https://github.com/vllm-project/vllm) enabling Marlin W4A8-FP8 on SM121 (DGX Spark GB10) + multi-node TP=2 optimizations.

## Base

- **Upstream**: `vllm-project/vllm` @ `f97ca6717`
- **Container**: `nvcr.io/nvidia/vllm:26.01-py3`
- **Branch**: `main`

## Features

| Feature | Patch | Effect |
|---------|-------|--------|
| Marlin SM12x | `marlin_sm12x.patch` | W4A8-FP8 + W4A16 on SM120/SM121 |
| UF17 EAGER_ALLREDUCE | `patch_uf17_eager_allreduce.py` | NCCL outside CUDA Graphs (+8%) |
| **UF19v4 RDMA** | `uf19_rdma.cu` + `patch_uf19_rdma_allreduce.py` | **CUDA-graph-compatible ibverbs AllReduce (12.2 µs vs 16.4 µs NCCL, -26%)** |
| MTP + NVFP4 | `patch_mtp_nvfp4_exclusion.py` | MTP speculative decoding with NVFP4 |
| EAGLE3 | built-in | Speculative decoding NST=1 |
| GLM-4.7 compat | `patch_transformers.py` | transformers 5.0 + compressed-tensors |

## UF19v4 Architecture

```
GPU Kernels (CUDA-graph-capturable):
  wait_send_done <<<1,1>>>  ─── polls send_done (proxy completed prev RDMA)
  prepare_send   <<<N,256>>> ── copies input→send_buf, atomicAdd send_flag
  poll_recv      <<<1,1>>>  ─── polls recv_flag (peer's NIC wrote via RDMA)
  add_recv       <<<N,256>>> ── adds local + recv_buf(load_cv) → output

CPU Proxy Thread (background):
  spin-polls send_flag → ibv_post_send(data+flag) → poll CQ → store send_done
```

Benchmark (2x GB10, CX7 RoCE 200 Gbps, 4 KiB bf16):

| Method | µs/call | vs NCCL |
|--------|---------|---------|
| NCCL AllReduce | 16.4 | baseline |
| NCCL Net Plugin | 20.6 | +26% slower |
| **UF19v4 Mini-Proxy** | **12.2** | **-26% faster** |

## Build

```bash
cd build && ./build-ng.sh    # → localhost/vllm-ng
```

## Runtime (TP=2 with UF19v4)

```bash
# Container needs:
#   -v /dev/infiniband:/dev/infiniband
#   --cap-add=IPC_LOCK
# Env vars:
#   VLLM_UF_EAGER_ALLREDUCE=1
#   VLLM_UF_UF19_RDMA=1
#   VLLM_UF19_PEER_IP=192.168.0.116
```

## Patches

### `marlin_sm12x.patch`

| File | Change |
|------|--------|
| `csrc/quantization/marlin/marlin.cu` | SM121 fix: `major_capability == 12` statt `== 120` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py` | `is_device_capability_family(120)` statt `is_device_capability(120)` |

### Activate W4A8-FP8

```bash
export VLLM_MARLIN_INPUT_DTYPE=fp8
```
