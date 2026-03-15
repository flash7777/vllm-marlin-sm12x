# MTP INT4 AutoRound fuer Qwen3.5-397B-A17B

## Ziel

BF16 MTP-Gewichte aus Qwen3.5-397B isoliert zu INT4 quantisieren und in ein
INT4-AutoRound-Modell injizieren, damit vLLM MTP Speculative Decoding nutzen kann.

## Ausgangslage

| Was | Pfad | Format | Groesse |
|-----|------|--------|---------|
| BF16 Quellmodell | `/data/tensordata/Qwen3.5-397B-A17B` | BF16, 94 safetensors | ~800 GB |
| Extrahierte MTP-Gewichte | `/data/tensordata/mtp_weights_397b.safetensors` | BF16, 1553 Tensoren | 13.2 GB |
| Intel INT4 AutoRound | HuggingFace `Intel/Qwen3.5-397B-A17B-int4-AutoRound` | INT4 GPTQ | ~199 GB |

Das Intel AutoRound Modell hat **keine MTP-Gewichte** (nur layers 0-59, MTP-Layer 60 fehlt).

## MTP-Gewichte Struktur

```
mtp.fc.weight                                    # [4096, 8192] - Projektion hidden+embed -> hidden
mtp.pre_fc_norm_embedding.weight                 # [4096]       - RMSNorm
mtp.pre_fc_norm_hidden.weight                    # [4096]       - RMSNorm
mtp.layers.0.self_attn.{q,k,v,o}_proj.weight     # Attention (full_attention)
mtp.layers.0.self_attn.{q,k}_norm.weight          # QK-Norm
mtp.layers.0.mlp.experts.{0-511}.{gate,up,down}_proj.weight  # 512 Experts
mtp.layers.0.mlp.gate.weight                      # Router [512, 4096]
mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.weight    # Shared Expert
mtp.layers.0.mlp.shared_expert_gate.weight         # Shared Expert Gate
mtp.layers.0.input_layernorm.weight                # RMSNorm
mtp.layers.0.post_attention_layernorm.weight       # RMSNorm
mtp.norm.weight                                    # Final RMSNorm
```

Architektonisch identisch mit einem regulaeren Qwen3.5 MoE Decoder Layer.

## Schritt 1: AutoRound (isoliert, CPU)

**Script:** `autoround_mtp.py`

**Strategie:** MTP-Gewichte als 1-Layer Qwen3_5MoeForCausalLM tarnen:
- `mtp.layers.0.*` -> `model.layers.0.*`
- `mtp.norm.weight` -> `model.norm.weight`
- Dummy embed_tokens + lm_head (random init, werden nicht quantisiert)
- Config: `layer_types=["full_attention"]`, `num_hidden_layers=1`

AutoRound quantisiert den Decoder Layer (Attention + MoE Experts).
MTP-spezifische Gewichte (fc, norms) bleiben BF16 (zu klein fuer INT4).

**Einschraenkung:** Kalibrierung ohne echte Aktivierungen aus dem 60-Layer Trunk.
AutoRound optimiert trotzdem Rundungsrichtungen (besser als RTN), aber die
Kalibrierungsqualitaet ist limitiert.

```bash
podman run --rm -it \
    -v /data/tensordata:/data/tensordata \
    -v /root/vllm-marlin-sm12x/mtp:/workspace/mtp \
    localhost/autoround:latest \
    python3 /workspace/mtp/autoround_mtp.py \
        --mtp-weights /data/tensordata/mtp_weights_397b.safetensors \
        --tokenizer /data/tensordata/Qwen3.5-397B-A17B \
        --output /data/tensordata/mtp_weights_397b_int4 \
        --bits 4 --group-size 128 --iters 200 --seqlen 512
```

Erwartete Laufzeit: Stunden (CPU, 512 Experts, 200 Iterationen).
Mit `--iters 0` pure RTN in Minuten (schlechtere Qualitaet).

## Schritt 2: Injektion in INT4-Modell

**Script:** `inject_mtp.py`

```bash
python3 inject_mtp.py \
    --mtp-weights /data/tensordata/mtp_weights_397b_int4/mtp_int4.safetensors \
    --int4-model /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound
```

Was passiert:
1. MTP-Safetensors als `model-mtp-00001-of-00001.safetensors` in Modellverzeichnis kopiert
2. `model.safetensors.index.json` um MTP-Eintraege erweitert
3. `config.json` um `mtp_num_hidden_layers=1` ergaenzt (falls fehlend)

## Schritt 3: vLLM Serve mit MTP

```bash
vllm serve /data/tensordata/Qwen3.5-397B-A17B-int4-AutoRound \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --tensor-parallel-size 2
```

### Wie vLLM MTP laedt

vLLM nutzt `Qwen3_5MultiTokenPredictor` (aus `qwen3_5_mtp.py`):
- Laedt MTP-Gewichte aus dem **selben Modellverzeichnis** mit Prefix `mtp.*`
- Nutzt die **selbe quant_config** wie das Hauptmodell (INT4 GPTQ -> Marlin Kernel)
- MTP-Layer wird als Draft-Modell fuer Speculative Decoding verwendet
- Unterstuetzt per-Expert Gewichte (gate_proj/up_proj/down_proj einzeln)
- Unterstuetzt GPTQ-Format (qweight, scales, qzeros)

### Wichtig: quant_config Kompatibilitaet

Die INT4-MTP-Gewichte muessen im selben GPTQ-Pack-Format sein wie das
Hauptmodell (auto_round:auto_gptq, group_size=128, bits=4, sym=False).
AutoRound mit `format="auto_gptq"` stellt das sicher.

## Offene Fragen

- [ ] Kalibrierungsqualitaet: Wie stark leidet MTP-Accuracy ohne echte Trunk-Aktivierungen?
- [ ] Laufzeit: CPU AutoRound mit 512 Experts — evtl. `--iters 50` als Kompromiss?
- [ ] Embed_tokens: Dummy random ok, oder besser echte Embeddings aus BF16 kopieren (~2 GB)?
- [ ] vLLM Kompatibilitaet: Laedt der Marlin Kernel GPTQ MTP-Gewichte korrekt?
