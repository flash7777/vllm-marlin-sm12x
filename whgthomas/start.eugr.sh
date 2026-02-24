./launch-cluster.sh -t vllm-node-20260223  --solo \
exec vllm serve Intel/Qwen3-Coder-Next-int4-AutoRound \
--max-model-len 131072 \
--gpu-memory-utilization 0.7 \
--port 8888 --host 0.0.0.0 \
--load-format fastsafetensors \
--enable-prefix-caching \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder
