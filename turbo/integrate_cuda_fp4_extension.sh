#!/bin/bash
# ============================================================================
# Integrate CUDA FP4 Extension for GB10 Blackwell
# ============================================================================
# Installs CUDA FP4 headers into system include path.
# Test/benchmark compilation skipped (no GPU at build time, optional files).
# ============================================================================

set -e

echo "================================"
echo "  CUDA FP4 Extension"
echo "  Integration for GB10 (SM_121)"
echo "================================"

CUDA_FP4_SRC="/workspace/dgx-vllm-build/cutlass_nvfp4"
CUDA_INCLUDE="/usr/local/cuda/include"

echo "[1/1] Installing CUDA FP4 headers..."
for f in cuda_fp4.h cuda_fp4_gemm.h nvfp4_types.cuh nvfp4_gemm_kernel.cuh \
         nvfp4_gemm_simple_hw.cuh nvfp4_tcgen05_ptx_v2.cuh \
         nvfp4_gemm_kernel_hardware.cuh nvfp4_tcgen05_ptx.cuh \
         nvfp4_tcgen05_binary.cuh; do
  if [ -f "${CUDA_FP4_SRC}/${f}" ]; then
    cp "${CUDA_FP4_SRC}/${f}" "${CUDA_INCLUDE}/"
    echo "  + ${f}"
  fi
done

echo ""
echo "Headers installed to ${CUDA_INCLUDE}"
echo "Test/benchmark compilation skipped (run inside container with GPU)"
echo "================================"
