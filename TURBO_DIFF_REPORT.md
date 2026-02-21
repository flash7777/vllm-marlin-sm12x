# Turbo (Avarok dgx-vllm v22) — CUTLASS-Aenderungsbericht

Analyse aller CUTLASS-bezogenen Aenderungen gegenueber dem Upstream-vLLM v0.16.0rc2
(Commit `3b30e6150`). Jede Aenderung mit Diff-Nachweis und Wirkungsbeschreibung.

---

## 1. Software E2M1 Konvertierung (DER Kernpatch)

**Datei**: `patch_nvfp4_utils_sw_e2m1.py`
**Ziel**: `csrc/quantization/fp4/nvfp4_utils.cuh`
**Wirkung**: Ermoeglicht NVFP4-Quantisierung auf SM121 (GB10), das die PTX-Instruktion `cvt.rn.satfinite.e2m1x2.f32` nicht hat

### Diff: Neue Helper-Funktion (eingefuegt nach `namespace vllm {`)

```cpp
// NEU: Software E2M1 fuer SM121
+#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
+__device__ __forceinline__ uint8_t _sw_float_to_e2m1(float x) {
+  uint8_t sign = (uint8_t)((__float_as_uint(x) >> 28) & 8u);
+  float ax = fabsf(x);
+  uint8_t mag;
+  if      (ax <= 0.25f)  mag = 0;  // 0.0
+  else if (ax <  0.75f)  mag = 1;  // 0.5
+  else if (ax <= 1.25f)  mag = 2;  // 1.0
+  else if (ax <  1.75f)  mag = 3;  // 1.5
+  else if (ax <= 2.5f)   mag = 4;  // 2.0
+  else if (ax <  3.5f)   mag = 5;  // 3.0
+  else if (ax <= 5.0f)   mag = 6;  // 4.0
+  else                    mag = 7;  // 6.0 (satfinite)
+  return sign | mag;
+}
+#endif
```

### Diff: 3 Funktionen erhalten `#if __CUDA_ARCH__ == 1210` Guard

Beispiel (`fp32_vec8_to_e2m1(float[8])`):

```diff
 inline __device__ uint32_t fp32_vec8_to_e2m1(float (&array)[8]) {
   uint32_t val;
+#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
+  val = _sw_fp32_vec8_to_e2m1_flat(
+      array[0], array[1], array[2], array[3],
+      array[4], array[5], array[6], array[7]);
+#else
   asm volatile(
       "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
       ...);
+#endif
   return val;
 }
```

Gleiche Behandlung fuer:
- `fp32_vec8_to_e2m1(float2 (&array)[4])` — 4x float2 Ueberladung
- `fp32_vec16_to_e2m1(float2 (&array)[8])` — 16-Werte Variante (packt 2x uint32)

**Wirkung**: Ohne diesen Patch: 1.1 tok/s (Python-Fallback). Mit Patch: 35+ tok/s. **32x Speedup**.
Der Software-Pfad wird NUR auf SM121 (`__CUDA_ARCH__ == 1210`) aktiviert, andere Architekturen nutzen weiterhin Hardware-PTX.

---

## 2. FP4 Typ-Definitionen

**Datei**: `nv_fp4_dummy.h`
**Ziel**: `/usr/local/cuda/include/nv_fp4_dummy.h`
**Wirkung**: CUDA 13.0 CCCL-Header referenzieren `__nv_fp4_e2m1`, aber NVIDIA hat den Typ noch nicht offiziell definiert

### Diff: Komplett neue Datei (Auszug)

```cpp
+struct __align__(1) __nv_fp4_e2m1 {
+    unsigned char __x;  // 8-bit storage (lower 4 bits used)
+
+    __host__ __device__ constexpr __nv_fp4_e2m1() : __x(0) {}
+    __host__ __device__ constexpr __nv_fp4_e2m1(unsigned char val) : __x(val & 0x0F) {}
+
+    __host__ __device__ __forceinline__ operator float() const {
+        unsigned char sign = (__x >> 3) & 0x1;
+        unsigned char exp = (__x >> 1) & 0x3;
+        unsigned char mantissa = __x & 0x1;
+        // E2M1 Decoding: 0.0, 0.25, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
+        ...
+    }
+};
```

**Wirkung**: Ohne diesen Header scheitert die Kompilierung von CUTLASS FP4-Kernels. Er stellt 3 Typen (`__nv_fp4_e2m1`, `__nv_fp4x2_e2m1`, `__nv_fp8x4_storage_t`), 5 Intrinsics und 9 Operatoren bereit.

---

## 3. CMake Architektur-Listen (SCALED_MM_ARCHS)

**Datei**: `fix_cmake_sm120_archs_v113_corrected.sh`
**Ziel**: `CMakeLists.txt`
**Wirkung**: GB10 (CC 12.1) wird in die SM120-Kernel-Compilation eingeschlossen

### Diff

```diff
 # CUDA >= 13.0 Branch (Zeile ~533):
-cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" ...)
+cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f;12.1f" ...)

 # CUDA < 13.0 Branch (Zeile ~535, Vollstaendigkeit):
-cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a" ...)
+cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a;12.1a" ...)
```

### Zusaetzlich im Dockerfile (Zeile 96-108):

```diff
 # CUDA_SUPPORTED_ARCHS:
-set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
+set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")

 # SCALED_MM_ARCHS (3 Stellen):
-cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" ...)
+cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f;12.1f" ...)

 # MLA_ARCHS:
-cuda_archs_loose_intersection(MLA_ARCHS "10.0f;11.0f;12.0f" ...)
+cuda_archs_loose_intersection(MLA_ARCHS "10.0f;11.0f;12.0f;12.1f" ...)

 # CUTLASS_MOE_DATA_ARCHS:
-cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" ...)
+cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f;12.1f" ...)
```

**Wirkung**: Ohne diese Aenderung erkennt CMake SM121 nicht und kompiliert keine SM120-Kernels. Die GPU wird dann zur Laufzeit von allen CUTLASS-Pfaden ausgeschlossen.

---

## 4. Dispatcher Flag (ENABLE_SCALED_MM_SM120)

**Datei**: `fix_dispatcher_flag_v115.sh`
**Ziel**: `CMakeLists.txt` (Injection nach SM120-Section)
**Wirkung**: `scaled_mm_entry.cu` wird mit `-DENABLE_SCALED_MM_SM120=1` kompiliert

### Diff: CMake-Block wird nach SM120-endif() eingefuegt

```cmake
+if(ENABLE_SCALED_MM_SM120 OR TARGET_SM120_BUILT)
+  set(DISPATCHER_FILE "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu")
+  set_source_files_properties(
+    ${DISPATCHER_FILE}
+    PROPERTIES
+    COMPILE_DEFINITIONS "ENABLE_SCALED_MM_SM120=1"
+  )
+  message(STATUS "v115: Set ENABLE_SCALED_MM_SM120=1 for ${DISPATCHER_FILE}")
+endif()
```

**Wirkung**: Ohne diesen Fix werden die SM120-Kernels zwar kompiliert, aber der Dispatcher (`scaled_mm_entry.cu`) sieht sie nicht — weil `#ifdef ENABLE_SCALED_MM_SM120` zu FALSE evaluiert. **Der Dispatcher und die Kernels werden getrennt kompiliert und brauchen das Flag jeweils einzeln.**

---

## 5. Capability Routing (SM121 → SM120)

**Datei**: `fix_capability_121_v112.py`
**Ziel**: `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu`
**Wirkung**: CC 121 wird zu SM120-Kernels geroutet (statt Fehler)

### Diff

```diff
 // In der switch/if-Kette des Dispatchers:
-if (version_num == 120)
+if (version_num >= 120 && version_num < 130)
     cutlass_scaled_mm_sm120(...);

 // Fehlermeldung:
-"Required capability: 90 or 100"
+"Required capability: 90, 100, or 120"
```

**Wirkung**: vLLM erkennt SM121 als SM12x-Familie und leitet zum SM120-Codepfad weiter. Pattern: Exakte Gleichheit `==120` → Bereichspruefung `>=120 && <130`.

---

## 6. Grouped MoE Kernel v1 (SM100-Fallback)

**Datei**: `grouped_mm_gb10_native.cu` (NEUE Datei)
**Basiert auf**: `csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu`
**Wirkung**: MoE-GEMM fuer GB10 mit konservativen SM100-Parametern

### Diff gegenueber SM100-Original

```diff
 // ArchTag:
  using ArchTag = cutlass::arch::Sm100;          // UNVERÄNDERT

 // KernelSchedule:
  using KernelSchedule = KernelPtrArrayTmaWarpSpecialized1SmSm100;  // UNVERÄNDERT

 // TileShape (Default-Config):
-using TileShape = Shape<_128, _256, _128>;       // SM100 Original
+using TileShape = Shape<_64, _128, _64>;          // GB10: konservativ

 // TileShape (Small-Batch):
-using TileShape = Shape<_128, _16, _128>;         // SM100: M64-Config
+using TileShape = Shape<_64, _64, _64>;            // GB10: sehr konservativ

 // ClusterShape:
  using ClusterShape = Shape<_1, _1, _1>;          // Gleich (SM100 hat auch 1x1x1 Configs)

-// SM100 hat zusaetzlich 2SM-Variante mit ClusterShape 2x1x1 -> ENTFERNT
-// SM100 hat N-basierte Heuristik (N >= 8192) -> ENTFERNT
+// GB10: Nur M-basierte Selektion (m <= 64 ? small : default)
```

**Wirkung**: Funktioniert, aber suboptimal. Nutzt SM100-Schedule mit kleineren Tiles als Sicherheitsreserve. ~35 tok/s (vor weiteren Optimierungen).

---

## 7. Grouped MoE Kernel v109 (GB10-Native, NVIDIA-optimiert)

**Datei**: `grouped_mm_gb10_native_v109.cu` (NEUE Datei)
**Basiert auf**: NVIDIA CUTLASS Example `79d_blackwell_geforce_nvfp4_grouped_gemm.cu`
**Wirkung**: Echte GB10-Hardware-Nutzung statt SM100-Fallback

### Diff gegenueber v1 (gb10_native.cu)

```diff
 // ArchTag:
-using ArchTag = cutlass::arch::Sm100;
+using ArchTag = cutlass::arch::Sm120;   // GeForce Blackwell-spezifisch!

 // KernelSchedule:
-using KernelSchedule = KernelPtrArrayTmaWarpSpecialized1SmSm100;
+using KernelSchedule = KernelPtrArrayTmaWarpSpecializedPingpong;  // Ueberlappt Compute+Memory!

 // TileShape (Default):
-using TileShape = Shape<_64, _128, _64>;
+using TileShape = Shape<_128, _128, _128>;  // NVIDIA-Empfehlung fuer GeForce

 // TileShape (Small-Batch):
-using TileShape = Shape<_64, _64, _64>;
+using TileShape = Shape<_64, _128, _64>;    // Groesser als v1
```

### Kernunterschiede

| Eigenschaft | v1 (SM100-Fallback) | v109 (GB10-Native) |
|-------------|--------------------|--------------------|
| ArchTag | `Sm100` | **`Sm120`** |
| Schedule | 1SmSm100 | **Pingpong** |
| Default Tile | 64x128x64 | **128x128x128** |
| Small Tile | 64x64x64 | **64x128x64** |
| Basis | vLLM SM100-Code | **NVIDIA Example 79d** |

**Wirkung**: Pingpong-Schedule ueberlappt Compute- und Memory-Operationen — entscheidend fuer GB10's bandwidth-limitierte LPDDR5X. Sm120-ArchTag aktiviert GeForce-spezifische Hardware-Features. **Dies ist die einzige echte Optimierung** im gesamten Kernel-Stack.

---

## 8. Scaled MM FP8 Dispatch (SM121)

**Datei**: `scaled_mm_sm121_fp8_dispatch.cuh` (NEUE Datei)
**Basiert auf**: `c3x/scaled_mm_sm100_fp8_dispatch.cuh`
**Wirkung**: FP8-GEMM-Dispatch fuer SM121 mit vereinfachter Tile-Auswahl

### Diff gegenueber SM100-Original

```diff
 // Anzahl Configs:
-5 Configs (default, M256, M64_swap_ab, M64, M16_swap_ab)
+3 Configs (default, small, large)

 // ClusterShape (alle Configs):
-Shape<_2, _2, _1>   // SM100: 2x2 Cluster
-Shape<_2, _1, _1>   // SM100: 2x1 Cluster
-Shape<_4, _1, _1>   // SM100: 4x1 Cluster
-Shape<_1, _1, _1>   // SM100: 1x1 nur fuer M64/M16
+Shape<_1, _1, _1>   // SM121: IMMER 1x1x1 (GB10-Constraint)

 // Schedule:
-KernelPtrArrayTmaWarpSpecialized...  // SM100: explizit pro Config
+KernelScheduleAuto                    // SM121: automatisch (vereinfacht)
+EpilogueScheduleAuto                  // SM121: automatisch

 // Dispatch-Heuristik:
-if (m <= 16) → swap_ab + M16
-elif (m <= 64) → conditional swap_ab
-elif (m <= 256) → M256
-else → default
+if (m*n < 65536) → small (128x128x128)
+elif (m*n > 1048576) → large (128x256x128)
+else → default (128x256x128)

-// SM100: swap_ab Optimierung fuer kleine M -> ENTFERNT
```

**Wirkung**: Funktional korrekt, aber weniger fein abgestimmt als SM100. Die `swap_ab` Optimierung (Transponierung fuer bessere Cache-Nutzung bei kleinem M) fehlt — potenziell 10-15% Verlust bei Token-Generation (M=1).

---

## 9. Scaled MM Entry Point (SM121)

**Datei**: `scaled_mm_c3x_sm121.cu` (NEUE Datei)
**Basiert auf**: `scaled_mm_c3x_sm100.cu`
**Wirkung**: Leitet SM121 an SM100-Kernels weiter

### Diff

```diff
+#if defined ENABLE_SCALED_MM_SM121 && ENABLE_SCALED_MM_SM121
+void cutlass_scaled_mm_sm121(...) {
+  dispatch_scaled_mm(c, a, b, a_scales, b_scales, bias,
+                     vllm::cutlass_scaled_mm_sm100_fp8,    // FP8 → SM100
+                     nullptr,                                // INT8 → nicht supported
+                     vllm::cutlass_scaled_mm_blockwise_sm100_fp8);  // Blockwise → SM100
+}
+#endif
```

**Wirkung**: Reiner Wrapper — leitet ALLE SM121-Aufrufe an bestehende SM100-Kernels weiter. INT8 explizit deaktiviert (`nullptr`). **Keine eigene Kernel-Logik.**

---

## 10. NVFP4 Kernel-Kompilierung (CMake-Patch)

**Datei**: `cmake_patch_gb10_nvfp4_v6_full_kernels.sh`
**Ziel**: `CMakeLists.txt` (FP4_ARCHS Section)
**Wirkung**: Alle 5 NVFP4-Kernel-Dateien werden fuer SM121 kompiliert

### Diff: 5 Kernel-Dateien werden in FP4-Arch-Liste aufgenommen

Betroffene Dateien:
1. `nvfp4_quant_kernels.cu` — Aktivierungs-Quantisierung
2. `nvfp4_experts_quant.cu` — Per-Expert-Quantisierung
3. `activation_nvfp4_quant_fusion_kernels.cu` — SiLU+Mul+Quant Fusion
4. `nvfp4_blockwise_moe_kernel.cu` — CUTLASS MoE GEMM
5. `nvfp4_scaled_mm_sm120_kernels.cu` — CUTLASS FP4 Dense GEMM

**Wirkung**: Ohne diesen Patch werden FP4-Kernels nicht fuer SM121 kompiliert. Zusammen mit dem Software-E2M1-Patch (Punkt 1) koennen alle 5 Kernel-Dateien fuer SM121 gebaut werden — keine Python-Fallbacks, CUDA-Graph-kompatibel.

---

## Zusammenfassung

### Wirkungskette

```
Patch 2 (nv_fp4_dummy.h)     → Kompilierung moeglich
Patch 3 (CMake Archs)        → SM121 wird erkannt
Patch 10 (NVFP4 CMake)       → FP4-Kernels werden gebaut
Patch 1 (Software E2M1)      → FP4-Konvertierung funktioniert     [35 tok/s]
Patch 4 (Dispatcher Flag)    → Dispatcher kennt SM120-Kernels
Patch 5 (Capability Route)   → SM121 wird zu SM120 geroutet
Patch 7 (v109 Pingpong)      → Echte GB10-Hardware-Nutzung        [40 tok/s]
+ Marlin MoE Backend          → Effizienterer MoE-Dispatch          [42 tok/s]
+ MTP Speculation              → Multi-Token-Prediction              [67 tok/s]
```

### Klassifikation

| Patch | Typ | Wirkung |
|-------|-----|---------|
| Software E2M1 | **Innovation** | Ersetzt fehlende PTX-Instruktion, 32x Speedup |
| nv_fp4_dummy.h | **Workaround** | Liefert fehlende CUDA-Typdefinitionen |
| CMake Arch-Listen | **Konfiguration** | Aktiviert Compilation fuer SM121 |
| Dispatcher Flag | **Bugfix** | Behebt getrenntes Kompilieren von Kernels vs. Dispatcher |
| Capability Routing | **Portierung** | `==120` → `>=120 && <130` |
| v109 Pingpong Kernel | **Optimierung** | Einzige echte HW-spezifische Optimierung (Sm120 + Pingpong) |
| v1 MoE Kernel | **Portierung** | Konservative SM100-Kopie mit kleineren Tiles |
| SM121 scaled_mm | **Wrapper** | Leitet an SM100-Kernels weiter, kein eigener Code |
| NVFP4 CMake | **Konfiguration** | Aktiviert alle 5 FP4-Kernel-Dateien fuer SM121 |

---

## Fazit: Was ist wirklich neu?

### Ehrliche Bilanz

Von den 10 Patches sind **~70% Wrapping, Routing und CMake-Konfiguration**. Die
zentrale Frage ist: Was bringt Turbo, das wir nicht bereits haben?

### NVFP4 auf CUTLASS funktioniert bereits — ohne Turbo

**Fakt**: NVFP4 ueber CUTLASS laeuft auf SM120 UND SM121 bereits produktiv:

| Platform | Engine | NVFP4 CUTLASS | Methode |
|----------|--------|---------------|---------|
| SM120 (RTX PRO 6000) | vLLM-next | **157.9 tok/s** | FlashInfer CUTLASS JIT, OOTB |
| SM121 (DGX Spark) | vLLM-next | **65.0 tok/s** | FlashInfer CUTLASS JIT + `FLASHINFER_CUDA_ARCH_LIST="12.0a 12.1a"` |
| SM121 (DGX Spark) | SGLang-next | **66.0 tok/s** | sgl-kernel CUTLASS + LD_PRELOAD sm120_shim.so |

FlashInfer JIT umgeht das E2M1-Problem elegant: Es kompiliert fuer `sm_120a`
(das die Hardware-Instruktion HAT), und die CUDA Runtime waehlt die korrekte
Binary auch auf SM121. **Kein Software-E2M1 noetig.**

### Was Turbo tatsaechlich anders macht

Turbo nutzt vLLMs **nativen** CUTLASS-Pfad (VLLM_CUTLASS) statt FlashInfer JIT
fuer MoE-GEMMs. Dense GEMMs nutzen weiterhin FlashInfer CUTLASS.

Die Benchmark-Frage war:

> **Ist vLLMs nativer CUTLASS-Pfad (mit Turbo-Patches) schneller als
> FlashInfer CUTLASS JIT?**

**Antwort**: Ja, +4.6% vanilla (68.0 vs 65.0 tok/s), +1.9% mit EAGLE3 (73.9 vs 72.5).
Der v109 Pingpong-Kernel bringt einen kleinen, messbaren Vorteil.

| Engine | MoE-Kernel | Vanilla | +EAGLE3 | Math V/E |
|--------|-----------|---------|---------|----------|
| **vLLM-turbo** | **VLLM_CUTLASS (v109)** | **68.0** | **73.9** | 70%/78% |
| vLLM-next | FLASHINFER_CUTLASS | 65.0 | 72.5 | 74%/72% |

### 1. Software E2M1 — TOTER CODE (wird nie ausgefuehrt!)

**WICHTIG**: Turbo baut fuer `TORCH_CUDA_ARCH_LIST="12.0a"` — also SM120, NICHT SM121.
Das bedeutet `__CUDA_ARCH__ == 1200` zur Compile-Zeit. Der Guard im Patch:

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210  // NIE WAHR!
```

...ist **nie erfuellt**. Die Hardware-PTX `cvt.rn.satfinite.e2m1x2.f32` wird benutzt
(SM120 hat diese Instruktion). Die CUDA Runtime auf SM121 (GB10) laedt einfach den
SM120-Binary — identisches Prinzip wie FlashInfer JIT mit `sm_120a`.

Die `_sw_float_to_e2m1()` Funktion wird **nie aufgerufen**. Der behauptete "32x Speedup
durch Software E2M1" existiert nicht — der Speedup kommt allein daher, dass fuer
`sm_120a` statt `sm_121a` kompiliert wird. Jeder Build mit `TORCH_CUDA_ARCH_LIST="12.0a"`
erreicht dasselbe ohne den Patch.

`_sw_float_to_e2m1()` ersetzt die fehlende PTX-Instruktion auf SM121. Aber:

- SM120 (RTX PRO 6000) HAT die Instruktion in Hardware → Patch irrelevant
- SM121 via FlashInfer JIT braucht den Patch NICHT (Dual-Arch-Kompilierung)
- Nur relevant wenn man vLLMs nativen NVFP4-Pfad auf SM121 erzwingen will

**Verbesserungspotenzial**: Die Implementierung hat 7 if-else Branches (Warp
Divergence). Branchless Variante moeglich:

```cpp
// Aktuell (7 Branches, Warp Divergence):
if      (ax <= 0.25f)  mag = 0;
else if (ax <  0.75f)  mag = 1;
// ... 5 weitere Branches

// Besser (0 Branches, nur Praedikate + Integer-Add):
uint8_t mag = (ax > 0.25f) + (ax >= 0.75f) + (ax > 1.25f) +
              (ax >= 1.75f) + (ax > 2.5f)  + (ax >= 3.5f) + (ax > 5.0f);
```

### 2. v109 Pingpong Kernel — korrekte Konfiguration, kein neuer Algorithmus

Nutzt NVIDIAs eigene GeForce-Empfehlung (Sm120 + Pingpong + 128x128x128 Tiles).
Aber: FlashInfer JIT nutzt ebenfalls SM120-CUTLASS-Templates mit vergleichbarer
Konfiguration. Die 435 TFLOPS FP8 aus unserem CUTLASS-Profiling bestaetigen,
dass die Hardware-Leistung bereits ausgereizt wird.

### Was bleibt Wrapping/Boilerplate

| Datei | Tut | Bewertung |
|-------|-----|-----------|
| `nv_fp4_dummy.h` | Reimplementiert `__nv_fp4_e2m1` Typen | Kompatibilitaets-Shim, NVIDIA liefert das in CUDA 13.1+ |
| `scaled_mm_c3x_sm121.cu` | Leitet SM121 an SM100-Kernels | Reiner Wrapper, 0 Zeilen Logik |
| `scaled_mm_sm121_fp8_dispatch.cuh` | 3 Tile-Configs + Groessen-Heuristik | Vereinfachte SM100-Kopie ohne swap_ab |
| `grouped_mm_gb10_native.cu` (v1) | SM100-Schedule mit kleineren Tiles | Ueberholt durch v109 |
| CMake-Patches (3, 4, 10) | `;12.1f` einfuegen, Flag setzen | Konfiguration, kein Code |
| Capability Routing (5) | `==120` → `>=120 && <130` | Einzeiler |
| `cutlass_nvfp4/` Headers | FP4-Typ-Definitionen, Software-GEMM | Referenz-Code, nicht fuer Produktion |

### Vergleich mit unseren bestehenden Patches

| Feature | Unsere Patches (vllm-marlin-sm12x) | Avarok Turbo |
|---------|-----------------------------------|--------------|
| NVFP4 MoE | FlashInfer CUTLASS JIT (SM120) | Eigener v109 Kernel (Sm120+Pingpong) |
| FP8 MoE | Triton Fallback | Triton Fallback (gleich) |
| Marlin INT4 | SM12x Capability-Fix, W4A8 | Nicht enthalten |
| E2M1 Software | Nicht noetig (FlashInfer JIT) | **Toter Code** (baut fuer sm_120a, Guard prueft sm_121a) |
| EAGLE3/DFlash | Funktioniert | MTP statt EAGLE3 |
| NVFP4 Dense GEMM | FlashInfer JIT | vLLM-native CUTLASS |

### Kernaussage

**Turbo's v109 Pingpong-Kernel bringt +4.6% gegenueber FlashInfer CUTLASS JIT**
(68.0 vs 65.0 tok/s auf DGX Spark). Das ist der einzige reale Mehrwert.

Der prominenteste Patch — Software E2M1 (`_sw_float_to_e2m1`) — ist **toter Code**:
Turbo baut fuer `sm_120a` (`__CUDA_ARCH__ == 1200`), der Guard prueft
`__CUDA_ARCH__ == 1210`. Die Funktion wird nie ausgefuehrt. Der behauptete
"32x Speedup" kommt allein vom Kompilieren fuer `sm_120a` statt `sm_121a`.

INT4 Marlin + EAGLE3 (97.4 tok/s) bleibt mit Abstand die schnellste
Konfiguration auf DGX Spark — 43% schneller als Turbo NVFP4.
