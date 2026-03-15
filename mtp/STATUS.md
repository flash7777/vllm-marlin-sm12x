# MTP Status: Qwen3.5-397B auf Dual DGX Spark

## Ergebnis: NICHT MACHBAR auf 2× GB10 (119 GB UMA)

## Was funktioniert
- MTP INT4 Weights extrahiert + quantisiert (3.3 GB GPTQ)
- Key-Format gefixt (experts.id.proj)
- INC quant_config Patch funktioniert (MoE-Experts als INT4 Marlin erkannt)
- Kein "Unquantized MoE" mehr

## Was scheitert
PGX OOM beim Marlin Weight Repacking der MTP-Layer (512 Experts × 3 Projections).
Marlin konvertiert GPTQ → optimiertes Layout mit temporären Buffers.

## Memory-Budget
| Komponente | GB/Node |
|---|---|
| Modell INT4 | 100 |
| MTP INT4 Weights | 1.7 |
| Marlin Repacking Peak | ~3-5 (geschätzt) |
| KV-Cache | 8 |
| OS/CUDA/Ray | 5 |
| **Summe** | **~118-120** |
| **Verfügbar** | **119** |

## Mögliche Lösungen
1. **3. Node** — mehr Memory Headroom
2. **Kleineres Modell** (122B statt 397B) — MTP passt dort locker
3. **Lazy Marlin Repacking** — vLLM-Patch der Experts on-demand repackt statt alle auf einmal
4. **GPTQ-Kernel statt Marlin** — kein Repacking nötig, aber langsamer

## Dateien (funktionsfähig, bereit für stärkere Hardware)
- `mtp/01_extract_mtp_bf16.py` — Extraktion aus BF16
- `mtp/autoround_mtp.py` — INT4 Quantisierung
- `mtp/fix_mtp_key_format.py` — Key-Umbenennung
- `mtp/inject_mtp.py` — Injektion ins INT4 Modell
- `mtp/patch_mtp_quant.py` — vLLM INC Patch für INT4 MTP
- `start_cluster.sh --mtp` / `--mtp-eager` — Start-Scripts
