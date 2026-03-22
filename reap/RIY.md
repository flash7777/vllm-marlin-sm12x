# RIY — Reap It Yourself

Expert-Pruning als Laufzeit-Konfiguration statt als destruktive Modell-Transformation.

## Kernidee

Statt ein gepruntes Modell zu veröffentlichen (wie REAP-262B), wird das **Original** quantisiert und eine **Filter-Liste** bestimmt welche Experts maskiert werden. Kein Re-Indexing, keine Architektur-Änderung, alle Expert-Indices bleiben original.

## Profilformat

```json
{
  "version": 1,
  "model": "Qwen3.5-397B-A17B",
  "quantization": "int4-autoround",
  "workload": "municipal German administrative",
  "pruned_experts": [[0, 3], [0, 11], [4, 7], [12, 2]]
}
```

- `pruned_experts`: Liste von `[layer, expert_id]` Tupeln (Original-Indices)
- `workload`: Beschreibung des Anwendungsprofils (verschiedene Workloads prunen unterschiedliche Experts)
- `model`: Das Basismodell von dem die Scores stammen
- `quantization`: Ziel-Quantisierung

Verschiedene Profile auf demselben Basismodell:
- `riy_coding.json` — optimiert auf Code-Generation
- `riy_german_admin.json` — deutsche Verwaltungssprache
- `riy_math.json` — Mathematik
- `riy_general_35pct.json` — allgemein, 35% gepruned (≈ REAP-262B Äquivalent)

## Warum REAP→AutoRound ≠ AutoRound→REAP

AutoRound optimiert Scales/Zeros per Layer gegen den Layer-Output. Der Layer-Output hängt vom aktiven Routing ab:

- **AutoRound → Mask**: Scales optimiert für 512-Expert-Routing. Nach Maskierung ändert sich das Routing → Scales suboptimal.
- **Mask → AutoRound**: Scales optimiert für das tatsächliche Routing mit Maske → optimal für Inferenz.

Deshalb muss AutoRound die RIY-Maske **während der Kalibrierung** aktiv haben. Die Reihenfolge ist: Maske laden → Kalibrierung mit Maske → Quantisierung der unmaskierten Experts.

## Zielmodell-Format

```
512 Expert-Slots pro Layer (Original-Indices):
  333 × INT4 quantisiert (aktiv)
  179 × Zero-Tensor + "pruned" Markierung (maskiert)
    1 × Shared Expert BF16 (extra_config)
Router: unverändert (512 Outputs)
MTP: unverändert (512 Experts, Original-Indices)
```

Keine Architektur-Änderung. `config.json` behält `n_routed_experts: 512`. Das RIY-Profil wird als separates JSON mitgeliefert.

## Komponenten

### 1. AutoRound RIY-Support

AutoRound muss während der Kalibrierung die Maske anwenden:

```python
# Pseudocode
riy_profile = load_json("riy_general_35pct.json")
pruned_set = set((l, e) for l, e in riy_profile["pruned_experts"])

# Während Kalibrierung: maskierte Experts liefern 0, Gewichte renormalisiert
# Nur unmaskierte Experts werden quantisiert (spart Zeit + ist korrekt)
# Maskierte Expert-Slots im Output: Zero-Tensor mit Markierung
```

Vorteile:
- Kalibrierung sieht exakt die gleichen Aktivierungen wie Inferenz
- Maskierte Experts werden übersprungen → ~35% weniger Quantisierungszeit
- Shared Experts bleiben BF16 (wie bei 122B/397B Format)

### 2. vLLM RIY-Support (`vllm-ng17e-riy` Image)

Zwei separate Aktivierungsstufen — selektiv ein/ausschaltbar:

#### Profil-Maske (`--riy-expert-profile profile.json`)

- **Logit-Mask auf Router** bei Init: geprunte Experts bekommen `-inf`, Router wählt sie nie
- **Expert-Map**: logischer → physischer Index (kompakt), geprunte Weights nicht geladen
- **Speicherersparnis**: 20% Pruning → ~20% weniger GPU-RAM
- **Zero Runtime-Overhead**: Mask ist als Buffer registriert, keine Python-Checks im Hot-Path
- Aktivierung: `--riy <profile.json>` in `start_cluster.sh`

#### Monitor (`--riy-monitor`)

- **Stats-Sammlung**: `scatter_add_` auf GPU-Tensoren, pro Layer/Expert Frequenz + Routing-Weights
- **HTTP-Server** (Port 8019): `/riy/stats`, `/riy/mask`, `/riy/health`, `/riy/stats/start|stop|reset`
- **Live-Maske**: Maske per Admin-API setzen/löschen, Renormalisierung im Forward
- **~5% Performance-Overhead** — nur aktivieren wenn Profiling/Tuning nötig
- Aktivierung: `--riy-monitor` in `start_cluster.sh` → setzt `VLLM_RIY_MONITOR=1`

#### Kombinationen

| Flag | Profil-Maske | Stats | HTTP | Live-Maske | Overhead |
|------|-------------|-------|------|-----------|----------|
| (keins) | — | — | — | — | 0% |
| `--riy <p>` | aktiv | — | — | — | 0% |
| `--riy-monitor` | — | aktiv | aktiv | aktiv | ~5% |
| `--riy <p> --riy-monitor` | aktiv | aktiv | aktiv | aktiv | ~5% |

### 3. RIY-Profil-Generator

Erzeugt RIY-Profile aus REAP-Scores oder aus vLLM-Laufzeit-Statistiken:

```bash
# Aus bestehenden REAP-Scores (wenn veröffentlicht)
python3 riy_from_scores.py --scores expert_scores.json --ratio 0.35 --output riy_35pct.json

# Aus vLLM Admin-API Statistiken
python3 riy_from_stats.py --stats vllm_expert_stats.json --threshold 0.01 --output riy_lowuse.json

# Aus unserem Expert-Mapping (REAP-262B nachbauen)
python3 riy_from_mapping.py --pruned expert_mapping_pruned.json --output riy_reap262b.json
```

## Workflow

```
1. Modell ohne Profil starten (RIY-Image, kein --riy)
   → RIY-Code passiv vorhanden, Admin-API erreichbar
                ↓
2. Monitor aktivieren: bash start_cluster.sh --riy-monitor
   → Stats-Sammlung starten: curl -X POST localhost:8019/riy/stats/start
   → Workload durchlaufen lassen
   → Stats exportieren: curl localhost:8019/riy/stats > stats.json
                ↓
3. RIY-Profil erzeugen (aus Stats oder REAP-Scores)
   python3 riy_from_stats.py --stats stats.json --ratio 0.20 --output riy_20pct.json
                ↓
4. Produktiv mit Profil starten (ohne Monitor):
   bash start_cluster.sh --riy riy_20pct.json
   → Geprunte Experts nicht geladen, zero Performance-Overhead
                ↓
5. Optional: Profil + Monitor für Feintuning:
   bash start_cluster.sh --riy riy_20pct.json --riy-monitor
```

## Vorteile gegenüber REAP-Publish

| | REAP (publiziertes Modell) | RIY (Profil + Original) |
|---|---|---|
| Pruning-Grad | Fix (z.B. 35%) | Beliebig, per Profil |
| Expert-Indices | Re-indiziert | Original |
| MTP-Kompatibilität | Kaputt (andere Indices) | Erhalten |
| Quantisierung | Nachträglich, suboptimal | Mit Maske, optimal |
| Modell-Downloads | Pro Pruning-Variante ~500 GB | Einmal Original + Profile (je ~50 KB) |
| Workload-Anpassung | Nicht möglich | Profil wechseln |
| Live-Anpassung | Nicht möglich | Admin-API |

## Implementierung

- **RIY-Patch**: `riy.patch` (im Repo-Root) — patcht vLLM für Profil-Mask + Monitor
- **Image**: `vllm-ng17e-riy` (Default in `start_cluster.sh`)
- **Runtime-Patches** (Sideload bei Container-Start):
  - `mtp/patch_riy_monitor_guard.py` — Guard für Stats/HTTP hinter `VLLM_RIY_MONITOR=1`
  - `mtp/patch_mtp_riy_enable.py` — RIY Expert-Mask auch auf MTP-Drafter-Layer
- **RIY-Profile**: JSON-Dateien, z.B. `/data/tensordata/riy_profile_397b.json`
- **Ohne RIY**: `--no-riy` in `start_cluster.sh` → nutzt `vllm-ng17e` Image

## Ergebnisse (DGX Spark TP=2, Qwen3.5-397B INT4)

| Config | tok/s (medium) | Overhead |
|--------|---------------|----------|
| Ohne RIY | 28.0 | — |
| RIY 20% Profil (ohne Monitor) | 27.9 | 0% |
| RIY 20% Profil (mit Monitor) | 26.4 | 5% |
