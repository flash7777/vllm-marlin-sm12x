# TASK: Qwen3.5 AutoRound INT4 auf vllm-ng17 TP=2

## Modelle

| Modell | Params (aktiv) | INT4 Groesse | MAX_MODEL_LEN | Status DGX | Status PGX |
|--------|---------------|-------------|---------------|------------|------------|
| Qwen3.5-122B-A10B | 122B (10B) | ~67 GB | 16384 | vorhanden | vorhanden |
| Qwen3.5-REAP-262B-A17B | 262B (17B) | ~131 GB | 8192 | vorhanden | vorhanden |
| Qwen3.5-397B-A17B | 397B (17B) | ~199 GB | 131072 | nur PGX | vorhanden |

Quelle: Intel AutoRound (HuggingFace), packing_format: auto_round:auto_gptq

## Fixes (gegenueber erstem Versuch)

### 1. VLLM_MARLIN_USE_ATOMIC_ADD=1
**Kritisch!** Ohne atomare Adds hat der Marlin INT4 Kernel Race Conditions → Garbage Output.
Gesetzt in: `start.ng17_tp2.head` + `start.ng17_tp2.worker` (Container Env-Vars).

### 2. Chat Template
AutoRound-Modelle liefern KEIN chat_template in tokenizer_config.json.
Fix: chat_template vom Original-Modell (Qwen/Qwen3.5-122B-A10B) kopiert.
Manuell auf DGX + PGX angewendet.

### 3. Transformers Rope-Patch (NICHT noetig)
eugr/spark-vllm-docker patcht `modeling_rope_utils.py` (set() Wrapping fuer
ignore_keys_at_rope_validation). In vllm-ng17 mit transformers 5.3.0 bereits upstream.

### 4. vLLM Flags (aus Community)
- `--reasoning-parser qwen3` — Qwen3.5 Reasoning/Think Tokens
- `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` — Tool Calling
- `--enable-prefix-caching` — KV-Cache Prefix Sharing
- `--load-format fastsafetensors` — Schnelleres Model Loading

## Start-Befehle

```bash
# Voraussetzung: Head + Worker Container muessen laufen
./start.ng17_tp2.head       # DGX Spark (Rank 0)
./start.ng17_tp2.worker     # PGX ThinkStation (Rank 1)

# Dann eines der Serve-Scripts:
./start.ng17_tp2.serve.qwen35-122b-autoround   # 122B (~67 GB, passt auf TP=2)
./start.ng17_tp2.serve.qwen35-262b-autoround   # 262B (~131 GB, passt auf TP=2)
./start.ng17_tp2.serve.qwen35-397b-autoround   # 397B (~199 GB, knapp auf TP=2)
```

API auf Port 8011:
```bash
curl http://192.168.1.117:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-122b","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

## Erwartete Performance (Dual GB10 Spark)

| Modell | tok/s (vanilla) | Quelle |
|--------|----------------|--------|
| 397B INT4 | ~26-30 | NVIDIA Forum (gpieceoffice, Icisu) |
| 122B INT4 | ~30 | NVIDIA Forum (stefan132) |
| 262B INT4 | tbd | noch nicht getestet |

## Referenzen

- NVIDIA Forum: https://forums.developer.nvidia.com/t/361967
- Community Docker: https://github.com/eugr/spark-vllm-docker
- Intel AutoRound: https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound

## Bekannte Probleme

- 122B AutoRound: Intel sagt "has not been validated for this particular model" (seqlen=512 Kalibrierung)
- 262B REAP AutoRound: Weight-Naming fuer MoE Experts (`experts.down_proj.N` statt `experts.w2_weight`) — moeglicherweise inkompatibel mit aelteren vLLM Versionen
- 397B: Passt nur knapp auf TP=2 (199 GB auf 2x128 GB), max_model_len=131072 braucht fast allen KV-Cache
- chat_template muss manuell in tokenizer_config.json eingefuegt werden (nicht im AutoRound Modell enthalten)
