# NVFP4 Quantization Data Flow: SM120 vs SM121

## Hardware-Unterschied

| | SM120 (RTX PRO 6000) | SM121 (DGX Spark GB10) |
|---|---|---|
| Tensor Cores | mma.sync FP4×FP4→FP32 | mma.sync FP4×FP4→FP32 |
| tcgen05.mma | nein | nein |
| `cvt.rn.satfinite.e2m1x2.f32` | **ja** (Hardware PTX) | **nein** (fehlt!) |
| E2M1 Konvertierung | Hardware, 1 Instruktion | Software-Fallback noetig |

## CUTLASS SM120 GEMM Kernel (Hot Path)

Der CUTLASS GEMM Kernel selbst macht **keine** float→E2M1 Konvertierung.
Beide Operanden (Weights + Activations) muessen **vorher** als E2M1 vorliegen:

```
Global Memory (E2M1 packed bytes, 4 bit pro Element)
    │
    │  TMA Load
    ▼
Shared Memory (E2M1 bytes, unveraendert)
    │
    │  smem → rmem copy
    ▼
Register (E2M1 bytes)
    │
    │  Bit-Shift left by 2:  0b0000ABCD → 0b00ABCD00
    │  (Alignment fuer mma.sync F8F6F4 Format)
    ▼
mma.sync Tensor Core (FP4 × FP4 → FP32 Accumulator)
```

Quelle: `cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp` (Zeilen 145-165, 810-830)
Bit-Shift: `cute/atom/mma_traits_sm120.hpp` (`fp4_shift_A`/`fp4_shift_B`)

## Activation Quantization (VOR dem GEMM)

Die Aktivierungen kommen als BF16 aus der vorherigen Schicht.
Sie muessen zu E2M1 (FP4) quantisiert werden **bevor** der GEMM Kernel startet:

```
Vorherige Schicht
    │
    ▼  BF16 Aktivierungen
    │
    ▼  ┌─────────────────────────────────────────────────────┐
       │  Activation Quantization Kernel (float → E2M1)     │
       │                                                     │
       │  SM120: Hardware PTX cvt.rn.satfinite.e2m1x2.f32   │
       │         → 1 Instruktion, maximal schnell            │
       │                                                     │
       │  SM121: Software-Fallback (kein PTX vorhanden!)     │
       │         Option A: CUTLASS exmy_base.h (~40 SASS)    │
       │         Option B: Avarok Threshold (~7 Branches)    │
       │         Option C: Branchless Predicate-Sum (~14 SASS)│
       └─────────────────────────────────────────────────────┘
    │
    ▼  E2M1 Aktivierungen + Block Scale Factors
    │
    ▼  CUTLASS GEMM Kernel (nimmt fertige E2M1 Bytes)
```

## Wo passiert die Quantisierung?

Die float→E2M1 Konvertierung liegt **nicht** im CUTLASS GEMM Kernel,
sondern in einem **separaten Quantisierungs-Kernel** davor:

| Backend | Quantisierungs-Kernel | Nutzt CUTLASS NumericConverter? |
|---|---|---|
| FlashInfer CUTLASS (JIT) | FlashInfer-eigener Kernel | Eigene CUTLASS-Headers (JIT) |
| vLLM CUTLASS | vLLM-kompilierter Kernel | /opt/cutlass Headers (Build-Time) |
| Marlin | Marlin-eigener Kernel | Nein (eigene Implementierung) |

**Entscheidend**: Unser branchless Patch in `/opt/cutlass/` wirkt nur auf
vLLM-kompilierte Kernels. FlashInfer JIT nutzt eigene CUTLASS-Header-Kopien.

## SM121 Software-Fallback Vergleich

Alle drei Ansaetze ersetzen die fehlende PTX-Instruktion auf SM121:

| Ansatz | SASS | Branches | Quelle |
|---|---|---|---|
| CUTLASS exmy_base.h | ~40 | mehrere | NVIDIA CUTLASS generisch |
| Avarok Threshold | ~20 | 7 (if-else) | dgx-vllm nvfp4_utils.cu |
| Branchless Predicate-Sum | ~14 | 0 | vllm-marlin-sm12x (unser Patch) |

### Branchless Algorithmus

```cpp
// E2M1 hat 8 Magnitude-Stufen: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// Schwellwerte sind Mittelpunkte zwischen benachbarten Werten.
// Jeder Vergleich liefert 0 oder 1; die Summe ergibt direkt die E2M1 Magnitude.

float av = (v < 0.0f) ? -v : v;
uint8_t mag = (av > 0.25f) + (av >= 0.75f) + (av > 1.25f) +
              (av >= 1.75f) + (av > 2.5f)  + (av >= 3.5f) + (av > 5.0f);
uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
return float_e2m1_t::bitcast(sign | mag);
```

## Offene Frage

Wo genau liegt der Activation-Quantisierungs-Kernel fuer NVFP4?

- In FlashInfer: JIT-kompiliert mit eigenen CUTLASS-Headers
- In vLLM: kompiliert mit `/opt/cutlass` (unser Patch wirkt hier)
- In einem fused Operator (torch.compile Fusion)?

Der genaue Pfad bestimmt, ob unser branchless Patch die Inference-Performance
auf SM121 beeinflusst oder nicht.

## Benchmark-Status

| Test | tok/s | Anmerkung |
|---|---|---|
| vllm-next NVFP4 vanilla (Baseline) | 65.0 | SM121, FlashInfer CUTLASS |
| vllm-next2 NVFP4 + CUDA_LAUNCH_BLOCKING | 62.5 | SM121, Patches aktiv, synchron |
| vllm-next2 NVFP4 ohne BLOCKING | CRASH | illegal instruction bei CUDA Graph Capture |

Der Crash ohne BLOCKING kommt aus vLLMs eigenem kompiliertem Code (_C.abi3.so),
nicht aus FlashInfer JIT. Die CUTLASS-Patches in /opt/cutlass/ aendern vLLMs
eigene Kernel-Binary und verursachen einen asynchronen illegal instruction
waehrend CUDA Graph Capture.
