# TMA Concept for Marlin Kernel

## Problem

Marlin loads INT4-packed weights from global memory via `cp.async.cg` — **16 bytes per load instruction** (= 32 INT4 values). Each load requires software address computation in registers. For a single weight tile, dozens of load instructions are needed.

## Idea

Replace `cp.async` with TMA (`cp.async.bulk.tensor`) — hardware-accelerated bulk transfers available on SM90+ and SM120+.

|  | cp.async (current) | TMA (proposed) |
|---|---|---|
| Transfer per instruction | 16 bytes | up to 64 KB |
| Address computation | Software (registers) | Hardware (tensor descriptor) |
| Synchronization | `cp.async.commit_group` / `wait_group` | `mbarrier` (hardware barrier) |
| Instruction overhead | Many loads per tile | 1 TMA op per tile |

## Architecture of `marlin_tma.cuh`

Three layers:

### 1. Host-Side: TMA Descriptor Creation

```
create_weight_tma_desc()  →  cuTensorMapEncodeTiled (2D: K × N)
create_scale_tma_desc()   →  cuTensorMapEncodeTiled (1D: N)
```

TMA descriptors encode the tensor geometry (dimensions, strides, tile sizes). The hardware uses this to compute addresses autonomously — no register pressure for address math in the kernel.

### 2. Kernel-Side: Barrier + Load Primitives

```
barrier_init()       →  mbarrier.init.shared::cta
barrier_expect_tx()  →  mbarrier.arrive.expect_tx (tell barrier how many bytes to expect)
tma_load_2d()        →  cp.async.bulk.tensor.2d (one thread fires, hardware loads the tile)
tma_load_1d()        →  cp.async.bulk.tensor.1d (for scales/zero-points)
barrier_wait()       →  mbarrier.try_wait.parity (spin until transfer complete)
```

Flow: one thread issues the TMA load, all threads wait on the barrier.

### 3. Pipeline: Multi-Stage Double Buffering

```cpp
TmaPipeline<NumStages> pipeline;
pipeline.init();
// Stage 0: load tile A
pipeline.expect_tx(0, tile_bytes);
tma_load_2d(..., pipeline.get_barrier(0), ...);
// Stage 1: load tile B (overlapped)
pipeline.expect_tx(1, tile_bytes);
tma_load_2d(..., pipeline.get_barrier(1), ...);
// Process stage 0 while stage 1 loads
pipeline.wait(0);
compute(tile_A);
```

Phase toggling (`phase ^= 1`) enables barrier reuse across iterations.

## SM120/SM121 Specifics

- `shared::cta` scope (SM120 has no cluster/multicast, cluster is always 1×1×1)
- Same `cuTensorMapEncodeTiled` API as SM90
- `CUTE_ARCH_TMA_SM120_ENABLED` confirmed in CUTLASS `config.hpp`

## Integration Blocker

Marlin uses **scattered per-thread loads** with a custom permutation layout. Each thread computes its own addresses and loads from different locations in the weight buffer. The layout is optimized for the `cp.async` → register → `mma` pipeline.

TMA requires **contiguous rectangular tiles** — the hardware loads a complete box described by the TMA descriptor. This is fundamentally incompatible with Marlin's scattered access pattern.

Integration would require:

1. **Weight relayout**: Repack INT4 weights from Marlin's permuted format into contiguous K×N tiles at model load time
2. **New kernel path**: Separate `#ifdef MARLIN_USE_TMA` code in `marlin_template.h` with TMA loads replacing the `cp_async4` calls
3. **Shared memory redesign**: Barrier-based sync instead of `cp.async.commit_group`/`wait_group`, different shared memory buffer layout for TMA tile shapes
4. **Scale/ZP loads**: These are smaller and simpler — could be migrated to TMA independently as a first step

## Potential Payoff

The payoff is highest for **large batch sizes** where the kernel becomes more compute-bound and the reduction in load instructions frees up issue slots. For batch=1 (generation), the kernel is memory-bandwidth-bound regardless of load mechanism — TMA won't help much there.

## Files

| File | Purpose |
|---|---|
| `marlin_tma.cuh` | TMA primitives (this module) |
| `marlin.cuh` | Current `cp.async` load primitives |
| `marlin_template.h` | Kernel template (would need `#ifdef` TMA path) |
| `marlin.cu` | Host dispatch (would create TMA descriptors) |
