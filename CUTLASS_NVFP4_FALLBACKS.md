# NVFP4 E2M1 Software-Fallbacks fuer SM121 (Blackwell GB10)

## Problem

SM121 (DGX Spark / PGX ThinkStation, GB10) hat **keine** `cvt.rn.satfinite.e2m1x2.f32` PTX-Instruktion.
SM120 (RTX PRO 6000, Blackwell Desktop) **hat** diese Instruktion.

Beide GPUs haben FP4 Tensor Cores (`mma.e2m1`). Nur die **float→E2M1 Konvertierung**
(Aktivierungs-Quantisierung) braucht den Software-Fallback auf SM121.

Kompiliert man fuer `sm_120a`, ist `__CUDA_ARCH__ == 1200` — Hardware-PTX wird genutzt.
Kompiliert man fuer `sm_121a`, ist `__CUDA_ARCH__ == 1210` — Software-Fallback wird aktiviert.

## Drei Ansaetze

### 1. Avarok / turbo (Threshold if-else)

Spezialisierte Implementierung mit 7 vorberechneten Schwellwerten.
Jeder E2M1-Wert hat einen definierten "Fangbereich" — einfacher Vergleich genuegt.

```cpp
// Quelle: turbo/patch_nvfp4_utils_sw_e2m1.py
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
__device__ __forceinline__ uint8_t _sw_float_to_e2m1(float x) {
    uint8_t sign = (uint8_t)((__float_as_uint(x) >> 28) & 8u);
    float ax = fabsf(x);
    uint8_t mag;
    if      (ax <= 0.25f)  mag = 0;   // -> 0.0
    else if (ax <  0.75f)  mag = 1;   // -> 0.5
    else if (ax <= 1.25f)  mag = 2;   // -> 1.0
    else if (ax <  1.75f)  mag = 3;   // -> 1.5
    else if (ax <= 2.5f)   mag = 4;   // -> 2.0
    else if (ax <  3.5f)   mag = 5;   // -> 3.0
    else if (ax <= 5.0f)   mag = 6;   // -> 4.0
    else                    mag = 7;   // -> 6.0 (satfinite)
    return sign | mag;
}
#endif
```

**Eigenschaften:**
- 7 Branches (if-else Kette)
- Compiler optimiert vermutlich zu praedizierten `FSETP` + `SEL` (kein echtes Branching im SASS)
- ~14 SASS-Instruktionen
- Round-to-nearest (nicht round-to-nearest-even)
- Nur E2M1

### 2. turbo2 (Branchless Predicate Sum)

Optimierte Version: Jeder Vergleich liefert 0 oder 1, Summe ergibt direkt den E2M1-Code.
Nutzt aus, dass die 8 E2M1-Magnitudes (0-7) aufsteigend sortiert sind.

```cpp
// Quelle: turbo2/patch_nvfp4_utils_sw_e2m1.py
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
__device__ __forceinline__ uint8_t _sw_float_to_e2m1(float x) {
    uint8_t sign = (uint8_t)((__float_as_uint(x) >> 28) & 8u);
    float ax = fabsf(x);
    uint8_t mag = (ax > 0.25f)
                + (ax >= 0.75f)
                + (ax > 1.25f)
                + (ax >= 1.75f)
                + (ax > 2.5f)
                + (ax >= 3.5f)
                + (ax > 5.0f);
    return sign | mag;
}
#endif
```

**Eigenschaften:**
- 0 Branches — 7 Praedikat-Additionen
- Garantiert keine Warp-Divergenz
- ~14 SASS-Instruktionen (`FSETP.GT` / `FSETP.GE` + `IADD3`)
- Gleiche Schwellwerte wie turbo, gleiche Ergebnisse
- Round-to-nearest (nicht round-to-nearest-even)
- Nur E2M1

### 3. CUTLASS exmy_base.h (Generische IEEE-Bit-Manipulation)

CUTLASS' eigener Software-Fallback — generisch fuer **alle** FP-Formate (E2M1, E2M3, E3M2,
E4M3, E5M2, ...). Wird aktiviert wenn `CUDA_PTX_FP4FP6_CVT_ENABLED` nicht definiert ist.

```
Aufrufkette:
  NumericArrayConverter<float_e2m1_t, float, N>::convert()
    -> NumericConverter<float_e2m1_t, float>::convert()
      -> static_cast<float_e2m1_t>(x)
        -> float_exmy_base::convert_from_float(x)
          -> FpBitRepresentation<uint8_t, 4, 2, 1, NONE>::convert_from(fp32_bits)
            -> FpBitRepresentation::convert(src_encoding, src_val, dst_encoding)
```

```
Pseudocode der convert() Funktion (~200 Zeilen Template):

  sign, exp, mantissa = extract_ieee754_bits(float_input)
  if (is_nan)  -> nan         // E2M1: unmoeglich (NanInfEncoding::NONE)
  if (is_inf)  -> satfinite   // E2M1: clamp auf 6.0
  if (is_zero) -> signed_zero

  // Denormale normalisieren
  while (hidden_bit == 0) { mantissa <<= 1; exp--; }

  if (exp > MAX_EXP) -> satfinite  // Overflow -> 6.0

  // Mantisse anpassen: shift = dst_mantissa_bits - src_mantissa_bits = 1 - 23 = -22
  mantissa = round_nearest_even(mantissa, shift_amount)
    // Guard/Round/Sticky Bits:
    //   guard_bit  = bit an Position shift_amount
    //   round_bit  = bit an Position shift_amount - 1
    //   sticky_bit = OR aller Bits unterhalb round_bit
    //   Aufrunden wenn: (sticky && round) || (guard && round && !sticky)

  // Post-Rounding Overflow
  if (hidden_bits > 1) { mantissa >>= 1; exp++; }

  return pack_bits(sign, exp, mantissa)
```

**Eigenschaften:**
- ~8 Branches + 1 while-Loop (Denormal-Normalisierung)
- ~40+ SASS-Instruktionen (Bit-Extraktion, Shifts, Masks, Rounding)
- **Round-to-nearest-even** (IEEE-754 konform, Guard/Round/Sticky)
- Generisch fuer alle ExMy-Formate
- Wird in CUTLASS intern genutzt wenn `CUDA_PTX_FP4FP6_CVT_ENABLED == 0`

## Vergleichstabelle

| Eigenschaft | turbo (Avarok) | turbo2 (Branchless) | CUTLASS exmy_base |
|---|---|---|---|
| **Ansatz** | Spezialisiert | Spezialisiert | Generisch |
| **Branches** | 7 if-else | **0** | ~8 + while |
| **SASS (geschaetzt)** | ~14 | ~14 | ~40+ |
| **Warp-Divergenz** | Compiler-abhaengig | **Garantiert keine** | Ja (while, if) |
| **Rounding** | Nearest | Nearest | Nearest-**even** |
| **Latenz** | ~7 Zyklen | ~7 Zyklen | ~20+ Zyklen |
| **Codegroesse** | 20 Zeilen | 8 Zeilen | ~200 Zeilen |
| **Formate** | Nur E2M1 | Nur E2M1 | Alle ExMy |
| **IEEE-konform** | Annaehernd* | Annaehernd* | **Formal korrekt** |

\* An exakten Mittelpunkten (0.75, 1.75, 3.5) kann turbo/turbo2 um 1 ULP von
round-to-nearest-even abweichen. Bei FP4 mit 8 Werten praktisch irrelevant.

## Wo wird was genutzt?

| Pfad | SM120a (HW PTX) | SM121a (SW Fallback) |
|---|---|---|
| vLLM `nvfp4_utils.cuh` (3 Funktionen) | `cvt.rn.satfinite.e2m1x2.f32` | turbo / turbo2 |
| CUTLASS `NumericArrayConverter` | `cvt.rn.satfinite.e2m1x2.f32` | exmy_base.h |
| FlashInfer CUTLASS JIT | `cvt.rn.satfinite.e2m1x2.f32` | exmy_base.h |

## Aktivierung

```cpp
// Guard in nvfp4_utils.cuh (turbo + turbo2):
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
  // Software E2M1 (turbo oder turbo2)
#else
  // Hardware PTX: cvt.rn.satfinite.e2m1x2.f32
#endif

// Guard in CUTLASS float_subbyte.h (nach Patch):
// SM121 aus Bedingung entfernt -> CUDA_PTX_FP4FP6_CVT_ENABLED == 0 -> exmy_base.h
#if (defined(CUTLASS_ARCH_MMA_SM120A_ENABLED))  // SM121 ENTFERNT
#  define CUDA_PTX_FP4FP6_CVT_ENABLED 1
#endif
```

## Build-Konfiguration (Dual-Arch)

```dockerfile
ENV TORCH_CUDA_ARCH_LIST="12.0a;12.1a"
# SM120a: Hardware E2M1 + native CUTLASS PTX
# SM121a: Software E2M1 (turbo/turbo2) + CUTLASS exmy_base Fallback
```
