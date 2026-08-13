# GPU-fast H100 optimization — investigation & plan

Branch: `perf/gpu-fast-h100` (off `gpu-fast`). Status legend:
`[ ]` planned · `[x]` done · `[~]` in progress · `[✗]` tried, rejected.

## Baseline (2026-08-13, H100 80GB "GPU 0", float64, jax 0.6.2 cuda12)

First GPU measurements of `tomas_jax.fast` (`benchmarks/python/bench_fast_1m.py`):

| cells | sim | flags | wall | throughput | peak mem |
|---|---|---|---|---|---|
| 10k | 1h | — | 8.5 s | 1.2e4 cell-steps/s | 0.3 GiB |
| 100k | 6h | 4 chunks, sorted | 85 s | 7.0e4 cell-steps/s | 1.5 GiB |
| 1M | 6h | 8 chunks, sorted | **547 s** | 1.1e5 cell-steps/s | 14.7 GiB |

Target 1M × 6h < 10 s → **miss by ~55×**. Sulfur budget 2e-9 (correct);
coagulation substep cap (256) hit on all 60 steps. GPU time-limited, not
memory-limited (65 GiB headroom).

## Measured root causes (per-process timing, chunk C=125k, GPU 0)

| component | time | notes |
|---|---|---|
| coagulation_step @ n_sub=256 | 2553 ms | stiffest sorted chunk, per outer step |
| — of which mnfix (per substep) | 7.3 ms of ~10 ms | **~73% of substep cost** |
| condensation_step | 9.2 ms | fine |
| kernel kij (C,40,40) build | 1.1 ms | hoisted per outer step, fine |
| water / nucleation / chemistry | <0.3 ms | negligible |

1. **mnfix one-hot scatter materialized (C,40,40)** (`fast/mnfix.py`
   `_drift_sweep`): f64 one-hot (1.6 GB at C=125k) contracted via 2
   einsums, ×2 sweeps, per call — and mnfix runs **every coag substep**.
   HLO dump confirmed 16 `f64[125000,40,40]` buffers + 8 dots in the
   optimized module. Logically a 40-way scatter-add.
2. **tril/triu/diag of frozen kij rebuilt every substep**
   (`physics/coagulation_rates.py` inside the `fast/coagulation.py`
   fori_loop): 2 extra (C,40,40) materializations + 3 separate GEMVs
   reading `kij_lower` 3× per substep.
3. **Substep-demand tail wastes batch-max work**: per-cell demand
   n_raw=ceil(dt·λ/0.05) over the 1M benchmark ICs: p50=2, p90=49,
   p99=194, max=708; 0.39% of cells >256 cap. Sorted 8-chunk batch-maxes
   `[1,1,1,2,5,12,38,256]` → one chunk carries ~80% of all substeps.
4. **Sorting is stale after t=0**: t=0 sort predicts ~3.4 s/step but
   measured 9.1 s/step — stiffness evolves (nucleation bursts) and the
   one-time sort stops confining it. No per-chunk n_sub diagnostics
   existed to observe this.

## Phase 1 — algorithm-preserving fixes

- [x] **mnfix scatter-add** (`fast/mnfix.py`): replace one-hot einsum with
  batched `.at[rows, kk].add()`; no (C,40,40) intermediate. Conservation
  algebra identical; `tests/test_fast_mnfix.py` bit-equality passes.
- [x] **Hoist frozen-kernel decomposition** (`fast/coagulation.py` +
  `physics/coagulation_rates.py`): optional `kij_parts` argument;
  tril/triu/diag computed once per outer step. Fused the three
  lower-triangular GEMVs into one GEMM (reads `kij_lower` once) — same
  dot products, results unchanged.
- [x] **Per-segment re-sort + n_sub diagnostics** (`fast/run.py`,
  `fast/step.py`): `sort_by_coag_cost=True` now re-sorts by current λ
  before every scan segment (composite permutation restored at the end);
  `diags["coag_n_sub"]` records the per-step shared substep count
  (max over chunks).
- [x] **Cache the jitted segment runner** (`fast/run.py`): run_fast
  built a fresh `jax.jit` closure per call → every chunk×segment
  re-traced and re-compiled an identical program (48 recompiles ×
  ~6-7 s inside the 1M timed run — this was the single largest cost,
  and pre-existed for plain chunked runs at 8 recompiles). Now one
  module-level `lru_cache` runner per (dt, step_kwargs), shared by all
  chunks/segments/calls.
- [x] Tests: `kij_parts` path ≡ plain path; scatter ≡ one-hot deposit;
  sorted/chunked ≡ unchunked routing (`tests/test_fast_perf_refactor.py`).
- [x] Full CPU test suite green.
- [x] GPU re-benchmark: **1M × 6h = 121.5 s** (was 547 s, **4.5×**);
  peak memory 14.7 → 8.7 GiB; sulfur budget 2.0e-9 unchanged.
- [x] Adversarial code review (10 findings) + fixes: forcings normalized
  /uploaded once and time-sliced before cell gathers in the sorted
  runner (was: full-time-axis gather + re-upload per chunk×segment);
  constant forcings broadcast per segment, never dense (n_steps, C);
  `n_steps=0` no-op instead of IndexError; shared `_merge_diags` with a
  loud KeyError on unlisted diag keys (was: duplicated merge tuples that
  silently dropped unlisted keys); `_permute_state` reused for chunk
  slicing; Optional typing + ValueError guard on kij/kij_parts; docs
  contradictions fixed; GPU scatter-add non-bitwise-reproducibility
  documented in gpu_fast.md.

## Phase 2 — bigger levers (pending Phase 1 results)

- [x] Pallas fused coagulation substep kernel — landed as the opt-in
  `fast/coagulation_pallas.py` (`coagulation_step_pallas`, or
  `fast_step(..., coag_pallas=True)`). **3.0× on the substep loop**:
  4.11 → 1.37 ms/substep at C=125k, n_sub=256. Details below.
- [ ] MNFIX cadence inside coag substeps (positivity clamp per substep,
  full mnfix every K) — physics-affecting, needs recalibration against
  the adversarial cases in docs/gpu_fast.md.
- [x] `c_max` 0.05 → 0.1 — DONE (`afe1343`), now the default: measured
  identical sig-bin errors at both settings (the 256-substep cap, not
  c_max, limits the stiffest cells); 1M×6h went 102.5 → 71.1 s.
- [~] Substep-cap reduction 256 → 64: error study done (below), default
  change pending sign-off; end-to-end timing with Pallas in progress.

### Error metric: population-weighted, not raw per-bin (2026-08-13)

Raw per-bin relative error wildly overstates degradation: the worst
"errors" (up to 38×) sit in bins holding ~1e-11 particles/cm³ — MNFIX
NEPS seeds, not physical populations (e.g. bins 0-7, 1.9-9.6 nm, whose
share of the cell's N is 1e-13..1e-16). The canonical accuracy metric
for substep policies is therefore **population-weighted relative
error** = |ΔN_k|/N_k,ref × (N_k,ref/ΣN_ref), plus the per-cell
total-number error. Bin-resolved study (stiffest 2000 cells, 1 h,
default policy vs converged 0.0125/cap-4096 reference): mode bins
(97-390 nm, 7-14% of N each) err at 1e-4 worst; weighted max 1e-5;
per-cell total-N max 5.3e-5.

### Substep-cap study (2026-08-13, weighted metric)

Stiffest 2000 cells, 1 h, c_max=0.1, vs converged reference. Error is
first-order in 1/cap, as expected for Euler:

| cap | weighted p50 | weighted p99 | weighted max | total-N max |
|---|---|---|---|---|
| 256 (default) | 1.6e-10 | 4.9e-6 | 1.0e-5 | 5.3e-5 |
| 128 | 3.3e-10 | 1.0e-5 | 2.1e-5 | 1.1e-4 |
| 64 | 6.6e-10 | 2.1e-5 | 4.3e-5 | 2.2e-4 |
| 32 | 1.3e-9 | 4.2e-5 | 8.6e-5 | 4.5e-4 |

Stiff-chunk cost is proportional to the cap, and stiff chunks dominate
wall time, so cap 64 ≈ another ~2-3× on the pre-Pallas headline for a
weighted error still ≤ 4.3e-5. Only the stiffest ~0.2-3% of cells are
capped at all.

### Pallas kernel (2026-08-13)

Benchmark setup (`benchmarks/python/bench_coag_pallas.py`): the 1M-cell
seed-0 benchmark ICs, sorted by max loss frequency as `fast/run.py`
does, stiffest 125k cells, one full `coagulation_step` at dt=360 s. That
chunk demands the full **n_sub=256** cap (with the current c_max=0.1
default) — all per-substep numbers below are at that explicit n_sub,
not "the default demand" (a cap-64 default is under discussion; the
per-substep advantage is unchanged since kernel cost is linear in
n_sub).

| path | step (n_sub=256) | ms/substep | speedup |
|---|---|---|---|
| XLA (`coagulation_step`) | 1052 ms | 4.11 | 1× |
| Pallas v1 — (64,64) register tiles | 28.3 s @4 warps / 5.9 s @16 | 110 / 23 | **0.04× / 0.18×** |
| Pallas v2 — vector lanes, fori j-loop | 417 ms | 1.63 | 2.5× |
| v2 + persistent CTAs | 427-447 ms | 1.67-1.75 | 2.4× |
| v2 + unrolled j-loop | 333 ms | 1.30 | 3.2× |
| v2 + cond-skipped deposit (**landed**) | 350 ms | **1.37** | **3.0×** |

Equivalence (vs the XLA path, same n_sub by construction —
`tests/test_fast_coag_pallas.py`, CPU interpret + GPU Triton): max rel
err over a full step on mixed median+stiff states ≤ 5.7e-14 on Nk/Mk,
≤ 4e-16 on overflow (bar: 1e-10). A full `fast_step` (coag + all other
processes) differs by ≤ 1.1e-10 relative, worst case an NEPS-scale
empty bin (2.8e-4 particles in a 1.6e10-particle cell).

**Landed architecture** (the docstring of `fast/coagulation_pallas.py`
has the full story): persistent grid of ~2.6k CTAs, one cell at a time
per CTA, 2 warps; all per-bin state in (64,) register vectors (bins
padded 40→64, Triton pow2; padded lanes kept inert); kij transposed
once per outer step and its columns streamed from L1/L2 inside a
**fully unrolled** source-bin loop (4 masked FMA accumulators);
`shift_right` and the MNFIX deposit via a CTA-private GMEM scratch row
+ CTA barrier (masked gather loads / masked f64 `atomic_add`), the
deposit wrapped in `lax.cond` on a CTA-uniform "any lane shifted"
predicate so quiescent substeps skip its 3 barriers entirely; MNFIX
target bins via exact IEEE-754 exponent-field ceil/floor(log2) (no
libdevice log); `xnew` gathered from the true xk table because the
`jnp.power` grid is NOT bit-exact powers of two (1 ulp off at bins
11/21/29/31/39 — an `xk0*exp2(k)` reconstruction would not be
bit-identical).

**Dead ends & measured findings:**

1. **(64,64) register-tile kernel (v1) — rejected.** The natural
   formulation (load kij once into registers; one-hot masks for the
   tril/triu matvecs, shift_right, xk gathers, and the deposit scatter)
   is 5-25× *slower* than XLA: ~20 tile-sized f64 temporaries per
   substep cannot fit the 256 KB/SM register file, so every op
   round-trips local memory (~4% of peak f64). Component attribution at
   C=125k, w16 (ms/substep): full 23.0; mnfix alone 28.3; rates alone
   5.7; mnfix without the two one-hot xk gathers 9.9; mnfix with
   gather+log+deposit all stubbed 1.5. Data-dependent one-hot
   contractions are the poison; loop-invariant masks are fine.
2. **Persistent CTAs bought nothing on their own** (1.63 → 1.73 ms):
   the kernel is not HBM/L2-capacity-bound on kij re-reads as first
   modeled — sweeping the grid 1.3k→42k CTAs moves the time < 5%. Kept
   anyway: scratch shrinks from (C,8,64) to (P,8,64) and large-P
   results are marginally best. The real bottleneck is **latency**:
   GMEM stores are write-through (L1-invalidating), so every scratch
   round-trip costs ~L2 latency, and a `fori_loop` j-loop serializes
   ~120 such scalar loads per substep.
3. **Unrolling the 40-iteration j-loop** (Python loop → one basic
   block) was the single biggest v2 win (1.73 → 1.30 ms): all 40 column
   loads + 120 scalar loads issue independently and memory-level
   parallelism covers the latency.
4. **The cond-skipped deposit is ~5% slower on the all-stiff chunk**
   (1.30 → 1.37 ms; shifts are frequent there) but skips 3 barriers +
   6 row ops per sweep on quiescent cells — the common case in median
   chunks; kept.
5. `num_warps=2` (one thread per padded lane) is optimal: 1.37 @w2,
   1.53 @w1, 2.63 @w4 (the v1 tile kernel instead wanted w16).
6. **Pallas/Triton f64 support is complete for this kernel**: add/mul/
   div, sqrt/exp/exp2/log (libdevice), ceil/floor, comparisons, where,
   full/axis reductions, bitcast to int64 + shifts/masks, masked
   gather/scatter `pl.load`/`pl.store` with int-vector indices, f64
   `pl.atomic_add`, `debug_barrier`, dynamic-bound `fori_loop`, nested
   loops, `lax.cond` — all lower. `tl.dot` was never needed (explicit
   FMA accumulation instead of f64 MMA).
7. **API potholes** (jax 0.6.2): `lax.slice` of in-kernel tensors and
   `reduce_or` (`jnp.any`) don't lower (use iota+where, and an int-sum
   `> 0` for any()); `compiler_params` must be
   `pallas.triton.CompilerParams`, not a dict; a `fori_loop` carry must
   be a traced value, not a Python int (`scf.yield ... is not a
   Value`); interpret-mode discharge rejects Python-int index
   components (wrap as `jnp.asarray(i, int64)`), mixed int32/int64
   index tuples (cast `program_id` up), and masked atomics (mask only
   in compiled mode — masking only skips 0.0-adds).
8. The remaining 1.37 ms/substep is barrier/latency-dominated, not
   bandwidth-bound: a further ~2-3× likely needs SMEM-resident kij +
   shuffle reductions, i.e. a custom CUDA kernel via `jax.ffi` (or the
   Mosaic-GPU Pallas backend once its f64 story is clear). Not pursued
   — 3× met the target and the XLA path remains the default.

Follow-up: wire `coag_pallas=True` through a full `run_fast` 1M×6h
benchmark (the stiff chunks it accelerates carry ~80% of all substeps,
so the headline gain should approach the coag share of wall time).

## Results log

| date | change | 1M×6h wall | notes |
|---|---|---|---|
| 2026-08-13 | baseline | 547 s | 8 chunks, t=0 sort; includes 8 hidden recompiles |
| 2026-08-13 | + mnfix scatter, kij hoist/fusion, per-segment re-sort | 420 s | recompiles ballooned to 48 (re-sort restructure) and masked the kernel gains |
| 2026-08-13 | + cached segment runner | **121.5 s** | 2.02 s/step, 4.9e5 cell-steps/s, 8.7 GiB peak |
| 2026-08-13 | + review fixes, 16 chunks | **102.5 s** | finer stiffness bucketing buys ~15%; 6.3 GiB peak |
| 2026-08-13 | + c_max 0.1 default | **71.1 s** | identical weighted errors (cap-limited cells unaffected) |
| 2026-08-13 | + Pallas coag kernel (opt-in `coag_pallas=True`) | **31.1 s** | 2.3× end-to-end; 17.6× vs baseline |
| 2026-08-13 | + cap 64 (experiment, pending sign-off) | **23.2 s** | 23.6× vs baseline; weighted err ≤ 4.3e-5 |

Scaling sweep (post-Phase 1, sorted chunks sized ~62-125k cells):

| cells | chunks | wall (6h sim) | throughput | peak mem |
|---|---|---|---|---|
| 100k | 4 | 17.5 s | 3.4e5 cell-steps/s | 1.3 GiB |
| 1M | 8 | 121.5 s | 4.9e5 | 8.7 GiB |
| 1M | 16 | 102.5 s | 5.9e5 | 6.3 GiB |
| 4M | 32 | 393.6 s | 6.1e5 | 21.3 GiB |

Throughput saturates ~6e5 cell-steps/s (≈1.7 µs/cell-step) past ~1M
cells; wall time is linear in cells beyond that. Memory would allow
~15M cells on one 80 GB H100 — cells are time-limited, not
memory-limited. The dominant remaining cost is the coagulation substep
loop's (C,40,40) kij HBM traffic → Phase 2 lever #1 (Pallas).

Per-substep kernel costs at C=125k (median/stiff chunk): mnfix 7.3 → 1.35 ms,
coag substep 10 → 4.2 ms, condensation_step 9.2 → 3.4 ms, stiff-chunk outer
step 2553 → 1065 ms.

**Lesson recorded:** a fresh `jax.jit` closure per call means re-trace +
re-compile of an identical program; at ~6-7 s per compile of the segment
program this dominated everything until cached. Always hoist jit-wrapped
closures out of per-call scopes (or lru_cache them keyed on the static
config).

## Environment notes

- Venv: `.venv/` (jax 0.6.2 + cuda12 wheels). Driver lives in the GPU
  Operator container: `export LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu`.
- Pin to the free GPU: `CUDA_VISIBLE_DEVICES=0` (GPUs 1-7 run training).
