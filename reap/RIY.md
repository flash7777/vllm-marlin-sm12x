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

### 2. vLLM RIY-Support (`vllm-riy`)

#### Phase 1 — Pruning on Air (Priorität)

- **ExpertMask-Klasse**: Hält die Liste maskierter `(layer, expert)` Tupel, thread-safe, JSON-serialisierbar
- **Hook im MoE-Dispatch** (`fused_moe.py`): Maskierte Experts bekommen Gewicht 0, verbleibende Gewichte werden renormalisiert, kein Re-Indexing
- **Bestehenden `expert_load_metrics`-Zähler** nach außen exponieren (kein neues Monitoring, nur Interfacing)
- **Admin-API** (`/admin`): Maske setzen/lesen/löschen, Stats aktivieren/deaktivieren/resetten/exportieren, Prune-Kandidaten vorschlagen

#### Phase 2 — Ladezeit-Filter

- **CLI-Parameter** `--riy-expert-profile profile.json`
- Maskierte Experts werden beim Laden als Zero-Tensor initialisiert, **nicht** aus dem Checkpoint geladen
- Kein Re-Indexing, Architektur-Metadaten unverändert
- **Phase 2b Follow-up**: Echte Speicherersparnis (Allokation der Zero-Tensoren überspringen)

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
1. Expert-Scores berechnen (REAP Kalibrierung auf dem Basismodell)
   ODER bestehende Scores/Mapping nutzen
                ↓
2. RIY-Profil erzeugen (Workload-spezifisch, gewünschte Ratio)
                ↓
3. AutoRound mit RIY-Profil (Maske aktiv während Kalibrierung)
   → INT4 Modell mit 512 Expert-Slots, 179 als Zero markiert
                ↓
4. vLLM mit --riy-expert-profile starten
   → Maskierte Experts nicht laden, Router renormalisiert
                ↓
5. Optional: Admin-API für Live-Anpassung der Maske
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

## Repositories

- **AutoRound RIY-Patch**: `flash7777/vllm-marlin-sm12x/reap/` (dieses Verzeichnis)
- **vLLM RIY-Fork**: `flash7777/vllm-riy` (noch anzulegen)
- **RIY-Profile**: Als JSON-Dateien neben dem quantisierten Modell

## Abhängigkeiten

- `expert_mapping_pruned.json` — aus `expert_mapping.py` (bereits erzeugt)
- AutoRound >= 0.12.0
- vLLM (zu forken als vllm-riy)
