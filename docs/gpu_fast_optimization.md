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

- [ ] Pallas fused coagulation substep kernel: compute kij on the fly in
  SMEM (12.8 KB/cell), fuse rates + Euler + clamp; removes all (C,40,40)
  HBM traffic (docs/gpu_fast.md lever #1, ~10×).
- [ ] MNFIX cadence inside coag substeps (positivity clamp per substep,
  full mnfix every K) — physics-affecting, needs recalibration against
  the adversarial cases in docs/gpu_fast.md.
- [ ] `c_max` 0.05 → 0.1 experiment (halves substeps, ~2× coarser in
  stiffest bins only).

## Results log

| date | change | 1M×6h wall | notes |
|---|---|---|---|
| 2026-08-13 | baseline | 547 s | 8 chunks, t=0 sort; includes 8 hidden recompiles |
| 2026-08-13 | + mnfix scatter, kij hoist/fusion, per-segment re-sort | 420 s | recompiles ballooned to 48 (re-sort restructure) and masked the kernel gains |
| 2026-08-13 | + cached segment runner | **121.5 s** | 2.02 s/step, 4.9e5 cell-steps/s, 8.7 GiB peak |
| 2026-08-13 | + review fixes, 16 chunks | **102.5 s** | finer stiffness bucketing buys ~15%; 6.3 GiB peak |

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
