# Qwen3 Coder Next Recipe

> A step-by-step guide to running **Qwen3 Coder Next** with adequate pace and accuracy for [opencode](https://opencode.ai/) inference using vLLM on NVIDIA DGX Spark.

---

## Prerequisites

- NVIDIA DGX Spark (or compatible GPU system)
- A Hugging Face account with an access token
- Linux (Ubuntu/Debian-based)

---

## 1. Download the int4 AutoRound Model

Place your Hugging Face token in `~/.hf_token`, then clone the download tooling:

```bash
cd ~/
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

Edit `hf-download.sh` and insert the following at **line 4**:

```bash
export HF_TOKEN=$(cat $HOME/.hf_token)
```

Run the download:

```bash
./hf-download.sh Intel/Qwen3-Coder-Next-int4-AutoRound
```

> **Note:** This will take a while depending on your connection speed.

---

## 2. Install Podman

```bash
sudo apt-get update
sudo apt-get -y install podman
```

---

## 3. Patch vLLM

Clone the Marlin SM12x patch:

```bash
cd ~/
git clone https://github.com/flash7777/vllm-marlin-sm12x.git
cd vllm-marlin-sm12x
```

Clone vLLM and apply the patch:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

git apply ../marlin_sm12x.patch
cp ../marlin_tma.cuh csrc/quantization/marlin/
```

Build the patched vLLM container:

```bash
cd ../build
./build.sh
```

Verify the image was created:

```bash
podman images
# You should see: localhost/vllm-next
```

---

## 4. Create a Launch Script

Create a file named `~/run-qwen.sh`:

```bash
#!/bin/bash

podman run -it --rm \
  --name vllm-int4 \
  --network host --gpus all --ipc=host \
  --device nvidia.com/gpu=all \
  --security-opt label=disable \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_MLA_DISABLE=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_MARLIN_INPUT_DTYPE=fp8 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e HF_HUB_OFFLINE=1 \
  vllm-next vllm \
  serve Intel/Qwen3-Coder-Next-int4-AutoRound \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --quantization auto_round \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --served-model-name qwen3-coder-next \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
#   --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

Make it executable:

```bash
chmod +x ~/run-qwen.sh
```

### Tuning Notes

- **Speculative decoding:** You can enable it by switching to `--no-enable-prefix-caching` and uncommenting the last line. However, in practice disabling prefix caching tends to be slower for agentic programming workloads.
- **Background mode:** Once everything is working, change the first line to `podman run -d` to run detached. Running interactively (`-it`) is recommended initially so you can observe the logs and watch interactions.

---

## 5. Run Inference

Launch the server — expect roughly **10 minutes** for model loading:

```bash
~/run-qwen.sh
```

To stop the server from a separate terminal:

```bash
podman stop vllm-int4
```

---

## 6. Opencode Integration

On your development machine, edit `~/.config/opencode/opencode.json` and adjust the `baseURL` to match your Spark hostname:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "lmstudio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM (local)",
      "options": {
        "baseURL": "http://dgx-spark.local:8000/v1"
      },
      "models": {
        "qwen3-coder-next": {
          "name": "Qwen3 Coder Next (local)"
        }
      }
    }
  }
}
```

You're all set — open `opencode` and select **Qwen3 Coder Next (local)** as your model.
