#!/bin/bash
# AutoRound Qwen3.5-397B mit MTP-Support
# Läuft auf RTX (2x RTX PRO 6000, 503 GB RAM + 500 GB zram)
# Braucht AutoRound >= main (2026-03-13, MTP fix #1477)
# Kalibriert: iters=200, nsamples=128, GPU-accelerated (cuda:0)
# Dauer: ~8-12 Stunden (60 MoE Layers + 1 MTP Layer, 512 Experts je)
#
# Vorbereitung:
#   1. zram einrichten: echo zstd > /sys/block/zram0/comp_algorithm && echo 500G > /sys/block/zram0/disksize && mkswap /dev/zram0 && swapon -p 100 /dev/zram0
#   2. NIC offloads deaktivieren: ethtool -K enp130s0f0np0 rx off tx off tso off gso off gro off lro off
#
# Usage:
#   podman run --rm -it \
#     --device nvidia.com/gpu=all --security-opt=label=disable \
#     --hooks-dir=/usr/share/containers/oci/hooks.d \
#     --shm-size=64g \
#     -v /data/tensordata:/data/tensordata \
#     -v /tmp/autoround_397b_with_mtp.sh:/opt/run.sh:ro \
#     localhost/autoround:latest \
#     bash /opt/run.sh

set -euo pipefail

echo "=== AutoRound: Qwen3.5-397B-A17B + MTP ==="

# 1. Upgrade AutoRound to main (MTP support)
echo "Upgrading AutoRound to main branch..."
pip install --no-deps git+https://github.com/intel/auto-round.git 2>&1 | tail -3
python3 -c "import auto_round; print('AutoRound:', auto_round.__version__)"

# 2. Check MTP support
python3 -c "
from auto_round.utils import copy_missing_tensors_from_source
print('MTP support: OK (copy_missing_tensors_from_source found)')
" || { echo "ERROR: MTP support not available"; exit 1; }

# 3. Run AutoRound
MODEL=/data/tensordata/Qwen3.5-397B-A17B
OUTPUT=/data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound-MTP

echo "Model:  $MODEL"
echo "Output: $OUTPUT"
echo ""

python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
import torch

model_name = '$MODEL'
output_dir = '$OUTPUT'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print('Loading model (CPU, low_cpu_mem_usage)...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='cpu',
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print('Starting AutoRound quantization (calibrated, iters=200)...')
autoround = AutoRound(
    model=model,
    tokenizer=tokenizer,
    bits=4,
    group_size=128,
    sym=True,
    iters=200,
    nsamples=128,
    batch_size=1,
    seqlen=2048,
    device='cuda:0',
    # MTP will be auto-detected and copied/quantized
)

print('Quantizing...')
autoround.quantize()

print('Saving...')
autoround.save_quantized(
    output_dir,
    format='auto_gptq',
    inplace=True,
)

print(f'Done! Output: {output_dir}')
"
