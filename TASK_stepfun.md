# TASK: StepFun Step 3.5 Flash Benchmark (vllm-ng16, TP=2)

## Ziel

Benchmark von Step 3.5 Flash auf DGX+PGX (TP=2) mit vllm-ng16:
- **Vanilla** (ohne Spekulation): math + context-matrix via bench.py
- **MTP** (num_speculative_tokens=1): math + context-matrix via bench.py

## Modell-Details

| Key | Wert |
|-----|------|
| Architektur | Step3p5ForCausalLM (MoE: 288 Experts, top_k=8) |
| Layers | 45 hidden + 3 MTP (num_nextn_predict_layers=3) |
| Hidden | 4096, 64 heads, head_dim=128 |
| MoE intermediate | 1280 per expert, share_expert_dim=1280 |
| Sliding window | 512 |
| max_position_embeddings | 262144 |
| Quantisierung | INT4 AutoRound (~110 GB) |
| INT4 Pfad | `/data/tensordata/Step-3.5-Flash-int4-AutoRound` |
| BF16-MTP Pfad | `/data/tensordata/Step-3.5-Flash-BF16-MTP` (nur MTP-Weights, 18 GB) |
| MTP im INT4 | Ja, extrahiert: `model-mtp-00001-of-00001.safetensors` (Layers 45-47) |
| Chat-Template | Fuegt automatisch `<think>\n` ein (Reasoning-Modell) |
| vLLM Registry | `Step3p5ForCausalLM` (vanilla), `Step3p5MTP` (MTP) |
| Custom code | `auto_map` mit `configuration_step3p5.py` + `modeling_step3p5.py` |

## Infrastruktur

- Image: `localhost/vllm-ng16` (vLLM 0.16.0rc2, PyTorch 2.11, transformers 5.3)
- Container: `ng16-tp2-head` (DGX), `ng16-tp2-worker` (PGX)
- Start-Script: `start.ng16_tp2.serve step3p5 [mtp]`
- serve_torchrun.py: `/opt/vllm/serve_torchrun.py` (in beiden Containern)
- Port: 8011

## Status

### Erledigt
- [x] INT4 AutoRound Modell vorhanden (110 GB)
- [x] BF16 MTP-Weights extrahiert (Layers 45-47 -> model-mtp-00001-of-00001.safetensors)
- [x] serve_torchrun.py: Timeout (120s) fuer engine.step() eingebaut (verhindert System-Freeze)
- [x] serve_torchrun.py: `<think>...</think>` Stripping eingebaut
- [x] Updated serve_torchrun.py in beide Container kopiert (head + worker)
- [x] start.ng16_tp2.serve hat step3p5 + mtp Config

### Offen
- [ ] Vanilla Step 3.5 starten und pruefen ob Generation funktioniert
- [ ] Detokenizer-Bug (TypeError: tuple+list) reproduzieren und fixen falls noetig
- [ ] bench.py math (vanilla)
- [ ] bench.py --context (vanilla)
- [ ] MTP Step 3.5 starten
- [ ] bench.py math (MTP)
- [ ] bench.py --context (MTP)

## Bekannte Probleme

### 1. Detokenizer TypeError (tuple vs list)
```
File "vllm/tokenizers/detokenizer_utils.py", line 166, in detokenize_incrementally
    output_tokens = prev_tokens + new_tokens
TypeError: can only concatenate tuple (not "list") to tuple
```
- Tritt bei Generation auf, konnte im isolierten Test NICHT reproduziert werden
- Moeglicherweise: custom Tokenizer gibt bei bestimmten Token-IDs tuple zurueck
- **Workaround**: serve_torchrun.py hat jetzt 120s Timeout, System friert nicht mehr ein
- **Fix**: Falls noetig, in detokenizer_utils.py `prev_tokens = list(prev_tokens)` einfuegen

### 2. Content leer obwohl Thinking
- Chat-Template fuegt `<think>\n` automatisch ein
- Modell generiert erst Reasoning (`<think>...</think>`), dann Content
- **Fix**: serve_torchrun.py strippt jetzt `<think>...</think>` Bloecke aus Output

### 3. System-Freeze bei TP=2
- engine.step() haengt in NCCL-Operation wenn ein Rank Fehler hat
- **Fix**: ThreadPoolExecutor mit 120s Timeout auf beiden Ranks

## Bench-Kommandos

```bash
# Vanilla
python3 bench.py --url http://localhost:8011 --model step3p5-flash \
  --label "vLLM-ng16 INT4 Step3.5 Vanilla (DGX+PGX TP=2)"

python3 bench.py --url http://localhost:8011 --model step3p5-flash \
  --label "vLLM-ng16 INT4 Step3.5 Vanilla (DGX+PGX TP=2)" --context --skip-perf --skip-math

# MTP
python3 bench.py --url http://localhost:8011 --model step3p5-flash \
  --label "vLLM-ng16 INT4 Step3.5 MTP-NST1 (DGX+PGX TP=2)"

python3 bench.py --url http://localhost:8011 --model step3p5-flash \
  --label "vLLM-ng16 INT4 Step3.5 MTP-NST1 (DGX+PGX TP=2)" --context --skip-perf --skip-math
```

## Start-Befehle

```bash
# Container starten (falls nicht running)
cd ~/vllm-marlin-sm12x
./start.ng16_tp2.head
./start.ng16_tp2.worker   # SSH auf PGX

# Vanilla serve
./start.ng16_tp2.serve step3p5

# MTP serve
./start.ng16_tp2.serve step3p5 mtp

# Logs
podman exec ng16-tp2-head cat /tmp/ng16-step3p5.log
podman exec ng16-tp2-head tail -f /tmp/ng16-step3p5.log
ssh flash@192.168.1.116 "podman logs ng16-tp2-worker" | tail -20

# Stoppen
podman exec ng16-tp2-head bash -c "pkill -f torchrun"
ssh flash@192.168.1.116 "podman exec ng16-tp2-worker bash -c 'pkill -f torchrun'"
```

## Ergebnisse

### Vanilla (INT4 W4A16)
| Test | tok/s | Math | Anmerkung |
|------|-------|------|-----------|
| perf short | | | |
| perf medium | | | |
| perf long | | | |
| math 50 | | /50 | |

### MTP NST=1 (INT4 W4A16 + BF16 MTP)
| Test | tok/s | Math | Anmerkung |
|------|-------|------|-----------|
| perf short | | | |
| perf medium | | | |
| perf long | | | |
| math 50 | | /50 | |
