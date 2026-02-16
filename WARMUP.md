# Warmup-Effekt bei vLLM + EAGLE3

## Problem

Der erste bench.py Lauf nach Container-Start liefert **~10% niedrigere** Throughput-Werte als Folgeläufe.

## Messung (Spiegel 2, RTX PRO 6000, SM120)

Qwen3-Coder-30B INT4 W4A16 + EAGLE3 NST=3, frisch gebautes vllm-next Image.

| Test | 1. Lauf (kalt) | 2. Lauf (warm) | Delta |
|---|---:|---:|---:|
| perf long (400 tok) | 283 tok/s | **310 tok/s** | +10% |
| ctx=0 long | 275 tok/s | **304 tok/s** | +11% |
| ctx=512 long | 217 tok/s | **260 tok/s** | +20% |
| perf medium (150 tok) | 174 tok/s | **183 tok/s** | +5% |

## Ursache

Beim Container-Start durchläuft vLLM mehrere JIT-Phasen:

1. **torch.compile** (~100s): Dynamo Bytecode-Transform + Graph-Kompilierung für Backbone + EAGLE Head
2. **CUDA Graph Capture**: Erste Requests triggern Graph-Capture für verschiedene Batch-Sizes
3. **FlashInfer Autotuning**: Attention-Kernel Konfigurationssuche

Beim ersten bench.py Request nach Start sind nicht alle CUDA Graphs gecaptured. Besonders EAGLE3 hat mehrere Pfade (Draft + Verify + Accept), die erst nach den ersten Requests vollständig optimiert sind.

## Empfehlung

Vor dem eigentlichen Benchmark **1-2 Warmup-Requests** senden:

```bash
# Warmup (2 kurze Requests)
curl -s http://localhost:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-coder-int4","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}' > /dev/null
curl -s http://localhost:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-coder-int4","messages":[{"role":"user","content":"Write a function"}],"max_tokens":200}' > /dev/null

# Dann bench.py
python3 bench.py --url http://localhost:8011 --model qwen3-coder-int4 --label "..." --context
```

Alternativ: bench.py `--perf-rounds 5` nutzen — die ersten 1-2 Runs sind Warmup, der Durchschnitt über 5 Runs glättet den Effekt ausreichend.
