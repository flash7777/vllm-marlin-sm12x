# Pinned Versions (funktionierend, 2026-03-14)

## Wheels
Gespeichert unter `/data/tensordata/wheels/eugr-20260314/`

| Package | Version | Commit | File |
|---|---|---|---|
| vLLM | 0.17.1rc1.dev158 | `600a039f5` | vllm-0.17.1rc1.dev158+g600a039f5.d20260314.cu131-cp312-cp312-linux_aarch64.whl |
| FlashInfer Python | 0.6.6 | `081b91c8` | flashinfer_python-0.6.6-py3-none-any.whl |
| FlashInfer Cubin | 0.6.6 | `081b91c8` | flashinfer_cubin-0.6.6-py3-none-any.whl |
| FlashInfer JIT Cache | 0.6.6 | `081b91c8` | flashinfer_jit_cache-0.6.6-cp39-abi3-manylinux_2_28_aarch64.whl |

## Base Image
`nvcr.io/nvidia/pytorch:26.01-py3` (PyTorch 2.10, CUDA 13.1)

## Reproduzieren
```bash
# Mit gepinnten Wheels bauen (kein Download):
cd ~/spark-vllm-docker
cp /data/tensordata/wheels/eugr-20260314/*.whl wheels/
./build-and-copy.sh -t vllm-eugr --tf5 --no-build  # nur Runner bauen
# Oder manuell:
podman build -t vllm-eugr --build-arg PRE_TRANSFORMERS=1 \
  -v $PWD/wheels:/workspace/wheels:z --target runner .
```

## Getestet mit
- Qwen3.5-397B-A17B INT4 AutoRound, TP=2 Ray, DGX+PGX
- 10 tok/s Generation, 391 = 17*23 korrekt
- fastsafetensors Loader (nach Duplikat-Fix, siehe DUPLICATE_FIX.md)
