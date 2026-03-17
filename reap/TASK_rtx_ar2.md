# TASK: FMAAQ auf RTX (Full-Model-Assisted AutoRound Quantization)

INT4 AutoRound der REAP-262B mit dem ungeprunten 397B-Original als Referenz.

## Zielsystem

- **Host**: Spiegel 2, `ssh -p 2020 root@10.249.0.99`
- **GPU**: 2× RTX PRO 6000 Blackwell (96 GB each, SM120)
- **RAM**: 503 GB (kein Swap)
- **Storage**: NFS `/data/tensordata/` (26 TB frei)

## Vorhandene Daten

| Modell | Pfad | Größe |
|---|---|---|
| 397B BF16 Original | `/data/tensordata/Qwen3.5-397B-A17B/` | 752 GB |
| REAP-262B BF16 | `/data/tensordata/Qwen3.5-REAP-262B-A17B/` | 488 GB |
| REAP-262B INT4 (bestehend) | `/data/tensordata/Qwen3.5-REAP-262B-A17B-int4-AutoRound/` | ~131 GB |
| AutoRound Container | `localhost/autoround:latest` | 15.4 GB |
| Expert-Mapping | DGX `/data/tensordata/expert_mapping.json` | 303 KB |

## Schritt 1: zram aufsetzen (zstd)

503 GB RAM + zram ≈ 700-800 GB effektiv. Nötig weil der 397B Teacher (752 GB BF16) plus REAP-262B Student gleichzeitig im RAM liegen müssen — zumindest Layer-weise.

```bash
# zram mit zstd Kompression
modprobe zram num_devices=1
echo zstd > /sys/block/zram0/comp_algorithm
echo 256G > /sys/block/zram0/disksize
mkswap /dev/zram0
swapon -p 5 /dev/zram0
```

Kompressionsratio bei BF16-Weights typisch 1.5-2× → 256 GB zram ≈ 380-500 GB extra.

## Schritt 2: Expert-Mapping auf RTX transferieren

```bash
# Von DGX
scp -P 2020 /data/tensordata/expert_mapping.json root@10.249.0.99:/data/tensordata/
scp -P 2020 /data/tensordata/expert_mapping_pruned.json root@10.249.0.99:/data/tensordata/
```

## Schritt 3: AutoRound patchen

### Kern-Idee

AutoRound optimiert per-Layer: `loss = MSE(int4_layer(input), bf16_layer(input))`.
FMAAQ ersetzt die Referenz: `loss = MSE(int4_reap_layer(input), bf16_original_layer(input))`.

Beide Layer-Outputs haben `hidden_size=4096` — direkt vergleichbar, kein Re-Expansion nötig.

### Patch-Strategie

1. AutoRound's `autoround.py` Kalibrierungsloop lokalisieren
2. Hook einbauen: statt eigenem BF16-Layer den 397B-Layer als Referenz laden
3. Teacher-Layer on-demand laden (Layer für Layer, nicht alles in RAM)
4. Nach dem Forward-Pass den Teacher-Layer freigeben

### Neues Startscript

```bash
# start_fmaaq.sh
podman run -d \
  --replace \
  --name autoround-fmaaq \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -v /data/tensordata:/data/tensordata \
  -v /path/to/patch:/opt/patch:ro \
  --shm-size=64g \
  localhost/autoround:latest \
  python3 /opt/patch/fmaaq_quantize.py \
    --model-path /data/tensordata/Qwen3.5-REAP-262B-A17B \
    --teacher-path /data/tensordata/Qwen3.5-397B-A17B \
    --output-dir /data/tensordata/Qwen3.5-REAP-262B-A17B-int4-FMAAQ \
    --bits 4 \
    --group-size 128 \
    --sym \
    --iters 50 \
    --seqlen 512 \
    --extra-config-shared-expert-bf16
```

## Schritt 4: `fmaaq_quantize.py` implementieren

Kern-Logik (Pseudocode):

```python
from auto_round import AutoRound

# Student: REAP-262B (wird quantisiert)
student = load_model("Qwen3.5-REAP-262B-A17B", device="cuda:0")

# Teacher: 397B Original (Referenz, layer-weise von Disk)
# NICHT komplett laden — nur den aktuellen Layer on-demand

class FMAAQAutoRound(AutoRound):
    def __init__(self, teacher_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_path = teacher_path

    def get_reference_output(self, layer_idx, input_tensor):
        """Load teacher layer, compute reference output, free memory."""
        teacher_layer = load_single_layer(self.teacher_path, layer_idx)
        teacher_layer.to(input_tensor.device)
        with torch.no_grad():
            ref_output = teacher_layer(input_tensor)
        del teacher_layer
        torch.cuda.empty_cache()
        return ref_output

# extra_config: shared experts BF16
extra_config = {}
for i in range(60):
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        key = f"model.language_model.layers.{i}.mlp.shared_expert.{proj}"
        extra_config[key] = {"bits": 16}

autoround = FMAAQAutoRound(
    teacher_path="/data/tensordata/Qwen3.5-397B-A17B",
    model=student,
    tokenizer=tokenizer,
    bits=4, group_size=128, sym=True,
    iters=50, seqlen=512,
    extra_config=extra_config,
)
autoround.quantize()
autoround.save_quantized(output_dir)
```

### Offene Fragen zur Implementierung

1. **Input-Alignment**: AutoRound propagiert Hidden States durch den Student. Für FMAAQ muss der Input zum jeweiligen Layer vom **Teacher** kommen, nicht vom Student. Sonst vergleicht man Outputs auf unterschiedlichen Inputs.

2. **Layer-Loading**: Der 397B hat 60 Layers mit je ~12.5 GB. On-demand von NFS laden ist ~1 GB/s. Pro Layer ~12s Load-Time — akzeptabel bei 50 Iterationen? Oder Teacher-Layer auf GPU cachen (96 GB reicht für ~7 Layers)?

3. **GPU-Split**: Student auf GPU 0, Teacher-Layer temporär auf GPU 1? Oder alles auf einer GPU sequenziell?

4. **Bestehender AutoRound-Code**: Wie tief greift man ein? Monkey-Patch der Loss-Funktion? Fork? Wrapper?

## Schritt 5: Quantisierung starten

```bash
# In tmux
tmux new-session -s fmaaq
podman exec -it autoround-fmaaq python3 /opt/patch/fmaaq_quantize.py ...
```

Erwartete Laufzeit: ~4-8h (50 Iterationen × 60 Layers × Teacher-Loading).

## Zielformat

Identisch zu den funktionierenden `Qwen3.5-122B-A10B-int4-AutoRound` und `Qwen3.5-397B-A17B-int4-AutoRound`:

```json
{
  "quant_method": "auto-round",
  "bits": 4,
  "group_size": 128,
  "sym": true,
  "packing_format": "auto_round:auto_gptq",
  "extra_config": {
    "model.language_model.layers.0.mlp.shared_expert_gate": {"bits": 16, "data_type": "float"},
    "model.language_model.layers.0.mlp.shared_expert.gate_proj": {"bits": 16, "data_type": "float"},
    "model.language_model.layers.0.mlp.shared_expert.up_proj": {"bits": 16, "data_type": "float"},
    "model.language_model.layers.0.mlp.shared_expert.down_proj": {"bits": 16, "data_type": "float"},
    ...
  }
}
```

Quantisiert werden: Routed Experts, Attention (Q/K/V/O), MTP Routed Experts.
BF16 bleiben: Shared Expert MLPs, Shared Expert Gate, MTP Shared Expert, LayerNorm, Embeddings.

### MTP-Erhalt

Das REAP-262B BF16 hat keine MTP-Layer (beim Pruning entfernt). Die MTP-Layer kommen vom 397B-Original (`mtp_weights_397b.safetensors`, 1553 Tensoren, BF16, 512 Experts).

Vor der Quantisierung:
1. MTP-Experts auf 333 prunen (mit `expert_mapping.json` — **nicht** identisch zum Hauptmodell-Mapping, da MTP eigene Experts hat)
2. Oder: MTP mit vollen 512 Experts beibehalten (Config muss separate `num_experts` für MTP erlauben)
3. MTP-Layer an das REAP-262B BF16 anhängen
4. AutoRound quantisiert MTP Routed Experts mit, Shared Expert bleibt BF16

→ **Option 2**: MTP mit vollen 512 Experts beibehalten. Kein Pruning der MTP-Experts, keine Qualitätseinbuße. Config braucht separate `num_experts` für MTP-Layer (512) vs Hauptmodell (333).

Kein nachträgliches Einpflanzen nötig — AutoRound schreibt das direkt korrekt, wenn `extra_config` gesetzt ist.

## Schritt 6: Auf DGX transferieren und testen

## Schritt 7: Evaluierung

Vergleichen auf Spiegel 2 (vLLM, TP=2):

| Variante | Beschreibung |
|---|---|
| Baseline | Bestehende `REAP-262B-int4-AutoRound` (standard, shared INT4) |
| Fixed | `REAP-262B-int4-AutoRound-RTX-fixed` (shared BF16 eingepflanzt) |
| FMAAQ | Neue Quantisierung mit 397B-Referenz + shared BF16 |

Metriken: Perplexity auf Eval-Set, MMLU, HumanEval, plus subjektive Chat-Qualität.

## Risiken

- **AutoRound-Internals**: Der Kalibrierungsloop ist möglicherweise nicht sauber trennbar für Teacher/Student
- **RAM**: 397B (752 GB) + 262B (488 GB) = 1.2 TB — selbst mit zram knapp. Layer-weise Loading ist Pflicht.
- **NFS-Latenz**: Teacher-Layer von NFS laden könnte langsam sein. Ggf. relevante Shards auf lokale SSD cachen.
