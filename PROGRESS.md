# TOMAS-JAX Development Progress Report

This file tracks all significant changes to the TOMAS-JAX codebase. Entries are in reverse chronological order (newest first).

---

## 2026-03-02 (Sun) — Fortran TFL vs PPM 24h Benchmark Comparison

**Time**: ~14:30 PST

### Summary

Ran 49 scenarios (S01-S49) x 24 hours x 2 modes (cond_only, combined) for both Fortran TFL and Fortran PPM, then generated 8 comparison plots and a detailed summary.

### Key Results

- **22 of 49 scenarios** exercise PPM transport (Path 1 where mcond > 1e-3 * total_dry_mass)
- **Cond-only (active only)**: N_tot error median=1.5e-6, max=1.2e-3; M_dry error median=5.1e-6, max=1.9e-4
- **Combined (active only)**: N_tot error median=6.0e-3, max=4.9e-2; M_dry error median=5.2e-6, max=1.7e-4
- **N conservation**: TFL perfect (1.0000); PPM median=0.9999, worst=0.9996 (cond-only)
- **Per-bin differences**: Up to 100% in individual bins — expected algorithmic difference (PPM is less diffusive, produces sharper distributions; TFL smears the growth front across bins)
- **Timing**: PPM ~0.93x TFL for cond-only (69ms vs 74ms); equal for combined (~0.32s, dominated by coagulation)
- **Scenario 50**: Crashes both methods (known extreme condition issue)

### Files Created
- `benchmarks/python/compare_fortran_tfl_ppm.py` — Comparison script (8 plots + summary)
- `benchmarks/results/fortran_tfl_vs_ppm/` — 8 PNG figures + summary.txt

### Next Steps
- Investigate PPM N conservation loss (worst case 0.04% at 24h) — likely from positivity clamping
- Consider running with `-O2` for more realistic timing comparison
- Compare Fortran PPM against JAX PPM to verify cross-language consistency

---

## 2026-03-02 (Sun) — Port PPM Condensation to Fortran

**Time**: ~14:00 PST

### Summary

Ported the PPM (Piecewise Parabolic Method) condensation algorithm from JAX/Python (`tomas_jax/physics/condensation_ppm.py`) to Fortran. The PPM Fortran code lives in a separate `tomas_fortran/src_ppm/` directory so both TFL and PPM methods coexist and can be benchmarked head-to-head in pure Fortran.

### Files Created
- `tomas_fortran/src_ppm/tmcond_ppm.f` — 8 PPM subroutines (~590 lines):
  - `PPM_RECONSTRUCT` — 4th-order interface interpolation + Colella-Woodward limiting + positivity
  - `PPM_EDGE_VELOCITY` — Upwind velocity at bin edges via DMDT_INT
  - `PPM_NUMBER_FLUX` — Departure-point parabola integration for number flux
  - `PPM_MASS_FLUX` — Analytical mass-weighted flux (∫m(η)n(η)dη with exp antiderivatives)
  - `PPM_DRY_MASS_ANALYTICAL` — Exact dry mass from PPM coefficients (moment integrals I0, I1, I2)
  - `PPM_SPECIES_FLUX` — Upwind donor ratio × dry mass flux for all 44 species
  - `PPM_COMPUTE_SUBSTEPS` — CFL-limited substep count (C_max = 0.8)
  - `PPM_CONDENSATION_STEP` — Main orchestrator (freeze WR, substep loop, call all above)
- `tomas_fortran/src_ppm/ezcond_ppm.f` — PPM-aware ezcond driver (~170 lines):
  - Same 3-path decision tree as ezcond.f (CS check, significance thresholds)
  - Path 1 calls PPM_CONDENSATION_STEP with dt=1.0, then adds condensed mass via sinkfrac
  - Mass conservation check and correction (same as ezcond.f)
- `tomas_fortran/harness/benchmark_24h_ppm.f` — PPM benchmark harness (~230 lines):
  - Same as benchmark_24h.f but calls ezcond_ppm instead of ezcond
  - Output files prefixed with `ppm_` (e.g., `ppm_s01_cond_hour01_Nk.csv`)
  - Timing CSV: `output/24h/timing_fortran_ppm.csv`

### Files Modified
- `tomas_fortran/Makefile` — Added `SRC_PPM`, `PPM_OBJS`, compile rules, `benchmark_24h_ppm` target, `benchmark_both` convenience target
- `tomas_fortran/README.md` — Documented src_ppm/ and new build targets
- `CLAUDE.md` — Added src_ppm/ files to file layout, updated Fortran source mapping

### Verification
- `make benchmark_24h_ppm` compiles cleanly (no warnings)
- `make benchmark_24h` still compiles (TFL unchanged)
- Both executables produced: `benchmark_24h_ppm.exe` (121KB), `benchmark_24h.exe` (103KB)

### Next Steps
- Run single scenario cond-only and compare against JAX PPM JIT output
- Run full 50-scenario benchmark for timing comparison
- Add Python comparison script for TFL vs PPM Fortran output

---

## 2026-03-02 (Sun) — Bring TOMAS Fortran into tomas_fortran/

**Time**: ~22:00 PST

### Summary

Made the repository self-contained by copying the original TOMAS Fortran source into `tomas_fortran/` with a proper build system. Previously, the Fortran source was referenced via fragile relative paths to `../../original-models/TOMAS/backup/src/`.

### Files Created
- `tomas_fortran/src/` — 14 core Fortran source files (multicoag.f, initbounds.f, mnfix.f, aerodens.f, loginit.f, ezcond.f, tmcond.f, dmdt_int.f, getCondSink.f, gasdiff.f, eznh3eqm.f, ezwatereqm.f, waterso4.f, waternacl.f)
- `tomas_fortran/include/sizecode.COM` — Common block definitions
- `tomas_fortran/harness/benchmark_24h.f` — 24h benchmark driver (copied from benchmarks/fortran/)
- `tomas_fortran/harness/benchmark_harness.f` — Single-scenario driver (copied from benchmarks/fortran/)
- `tomas_fortran/Makefile` — Self-contained build system (targets: all, benchmark_24h, run_24h, clean)
- `tomas_fortran/README.md` — Build instructions and file descriptions

### Files Modified
- `benchmarks/fortran/Makefile` — Updated TOMAS_SRC and TOMAS_INC to point to `../../tomas_fortran/src/` and `../../tomas_fortran/include/`
- `CLAUDE.md` — Added tomas_fortran/ to file layout, updated Fortran source path reference

### Next Steps
- Remove symlink dependency from benchmarks/fortran/ (sizecode.COM still symlinked for its local compile)
- Consider adding `.gitignore` for `tomas_fortran/output/` and `*.o` files

---

## 2026-03-02 (Sun) — GMD Paper Draft Populated

**Time**: ~18:00 PST

### Summary

Populated the full GMD-style paper draft at `docs/paper/draft.md` (~5000 words) with:
- Complete prose for all 8 sections + 4 appendices
- LaTeX-style equations extracted from source code (dmdt_int, PPM reconstruction, analytical mass-weighted flux, Fuchs-Sutugin, condensation sink, MNFIX partial transfer)
- 6 validation/timing tables with exact numbers from benchmark summary
- 9 planned figure captions and 6 table references
- 30+ references in GMD bibliography style
- Cross-verified all formulas against source code and all benchmark numbers against `summary.txt` — no inconsistencies found

### Files Modified
- `docs/paper/draft.md` — complete rewrite from outline to full paper

### Next Steps
- Generate actual figures from benchmark data
- Convert to LaTeX for GMD submission
- Add nucleation and organic condensation sections when implemented

---

## 2026-03-02 (Sun) — PPM Analytical Mass-Weighted Flux Fix + Vectorized Species Transport

**Time**: ~11:00 PST

### Problem

PPM condensation had two critical issues:
1. **81% N loss over 24h** — The `passive_flux` function used `F_M = F_N * r_avg_donor`, assigning average mass to all departing particles. But PPM selectively removes particles from the bin edge (where they're 2x heavier), causing systematic mass underestimate → M/N mismatch → MNFIX clips → N loss death spiral.
2. **~150x slower than Fortran** — Species transport used `jax.lax.fori_loop` over 44 species, creating convoluted XLA control flow instead of matrix math.

### Fix: Analytical Mass-Weighted Integrals

Replaced naive `F_M = F_N * r_avg` with exact analytical integral of `m(η)*n(η)` over the departure region:
- `m(η) = m_L * exp(a*η)` is the dry mass at position η within a bin (mass-doubling: 2x variation)
- `n(η) = n_L + b*η - n_6*η²` is the PPM number density parabola
- Antiderivatives: `∫ η^k * exp(aη) dη` for k=0,1,2

New functions in `condensation_ppm.py`:
- `_mass_antideriv(eta)` — evaluates antiderivatives of η^k * exp(aη)
- `_integrate_mass_parabola_right/left()` — mass integrals over departure regions
- `ppm_mass_flux()` — computes F_M_dry at each edge using mass-weighted integrals
- `species_flux()` — vectorized, all 44 species in one matrix multiply (no loops)

Key normalization insight: `species_flux` uses `M_dry_analytical` from `dry_mass_from_ppm_number()` (not tracked mass) to ensure exact conservation when C=1.

### Results (49 scenarios, 24h)

| Mode | N_tot error (median) | Mass conservation | JAX wall time | JAX/Fortran |
|------|---------------------|-------------------|---------------|-------------|
| Cond PPM_JIT | 2.08e-7 | 8.6e-16 | 0.26s | 3.4x |
| Cond TFL_JIT | 2.08e-7 | 5.9e-15 | 0.47s | 6.2x |
| Combined PPM_JIT | 2.18e-3 | 1.9e-10 | 0.41s | 1.3x |
| Coag-Only | 1.83e-3 | 1.9e-10 | 0.15s | 0.57x |

PPM_JIT is **1.8x faster than TFL_JIT** for condensation, with machine-precision mass conservation and perfect N conservation.

### Files Modified

- `tomas_jax/physics/condensation_ppm.py` — Added analytical mass-weighted flux functions, replaced `passive_flux` with vectorized `species_flux`, removed `closure_update`, eliminated `fori_loop` over species
- `tests/test_ppm_condensation.py` — Updated imports and tests for new API (removed `passive_flux`/`closure_update`, added `species_flux`/`ppm_mass_flux` tests)
- `benchmarks/python/run_ppm_analytical_benchmark.py` — **NEW**: 49 scenarios x 5 modes benchmark

### Output

- `benchmarks/results/2026-03-02-ppm-analytical/` — 9 figures + summary.txt

---

## 2026-03-01 (Sat) — Fix TFL Condensation Oscillation (Root Cause: MNFIX + Condensation Sink)

**Time**: ~00:30 PST

### Root Cause

TFL condensation produced oscillating/pulsing size distributions (bins emptying completely then all particles jumping to the next bin). Two bugs were identified:

1. **MNFIX was doing "move-all" instead of Fortran's "partial-transfer"** (THE ROOT CAUSE)
   - Our MNFIX: when avg mass > upper bin boundary → move ALL particles to next bin → source bin empties → pulse
   - Fortran MNFIX: split the bin — some particles stay at geometric mean mass, excess moves to neighbor → smooth gradual redistribution
   - Fortran uses `nshift = (drymass - xold*number) / (xnew - xold)` for the partial transfer, keeping remaining particles at `xold = sqrt(xk[k]*xk[k+1])` and shifting `nshift` particles at `xnew = xk[kk+1]/1.1`

2. **Condensation sink Neps threshold was wrong** (SECONDARY)
   - Fortran getCondSink.f: `Neps = 1e10` — bins with < 1e10 particles use default `density=1500`, `mp=1.4*xk[k]`
   - Our code: `Neps = 1e-20` — computed actual density/mp for ALL bins, giving different `sinkfrac` distributions

3. **Mass conservation correction threshold was too permissive** (MINOR)
   - Fortran ezcond.f: `abs(1-ratio) < 1.0` → ratio in (0, 2)
   - Our ezcond.py: `abs(ratio) < 100.0` → allowed extreme amplification

### Files Modified

- `tomas_jax/core/mnfix_jax.py` — **Complete rewrite** to match Fortran mnfix.f:
  - Phase 1: Fix empty bins (Neps=1e-5, matching Fortran)
  - Phase 2: Fix extreme out-of-range (avg > max grid or avg < min grid)
  - Phase 3: Partial transfer using Fortran's split algorithm (nshift formula)
  - Uses `jax.lax.fori_loop` for sequential processing matching Fortran order
  - Multi-bin drift handled for 1-2 bin shifts (common case)

- `tomas_jax/physics/condensation_sink.py` — Match Fortran getCondSink.f Neps=1e10:
  - Bins with Nk > 1e10: compute actual density and mp from composition
  - Bins with Nk <= 1e10: use default density=1500, mp=1.4*xk[k]
  - Now directly computes Dpk instead of calling calc_particle_properties

- `tomas_jax/physics/ezcond.py` — Fixed mass correction threshold:
  - Changed from `abs(ratio) < 100.0` to `ratio > 0.0 and ratio < 2.0` (matching Fortran)

- `tomas_jax/core/config.py` — Added `xk_boundaries()` utility function and `XK0` constant

### Results (1-hour, 60 steps)

| Scenario | Before Fix | After Fix |
|----------|-----------|-----------|
| S10 (peak bins) | 2-7% error | 0.02-12% error |
| S20 (all bins) | **100% error (empty bins!)** | **0.3-1.9% error** |
| S39 (peak bins) | **93% error** | **0.2-2.4% error** |

### Tests

All 73 non-cached tests pass (15 TFL JIT + 43 PPM + 13 PPM JIT + 2 coagulation).

### Known Remaining Issues

- S39 lower-tail bins (6-8) still have 14-39% error — these have very few particles and are affected by the Neps threshold boundary
- PPM still has fundamental N conservation issues (separate from MNFIX) due to passive_flux using average donor mass instead of position-dependent mass
- The 24h benchmark NPZ files need regeneration with the fixed code

---

## 2026-02-28 (Fri) — JIT-Compile TFL Condensation (Fortran-Matching, 43x Speedup)

**Time**: ~00:00 PST

### Summary
Created a pure-JAX, JIT-compilable version of the TFL (Tzivion-Feingold-Levin) condensation algorithm. The sequential TFL matches Fortran output well (unlike PPM which produces overly narrow distributions), so JIT-compiling TFL gives us both accuracy and speed.

### Approach
- Vectorized top-hat construction and dmdt_int translation across all bins
- `jax.lax.fori_loop` over 36 source bins with vectorized overlap-fraction scatter inside
- Condensing species uses inverse dmdt_int at bin boundaries for trapezoidal interpolation, matching Fortran's 1.5*YM for full middle bins
- Three-way branch via `jax.lax.cond` (tmcond / simple-add / no-op)
- `jax.lax.scan` fuses the 1440-step time loop into a single XLA program

### Files Created
- `tomas_jax/physics/condensation_tfl_jax.py` — Pure-JAX TFL: `tmcond_jax()` (vectorized bin remapping) + `ezcond_tfl_jax()` (condensation driver). ~370 lines.
- `tests/test_tfl_jit_condensation.py` — 15 tests: tmcond zero-tau/shift/JIT, ezcond zero-mcond/JIT, full-step no-nan/gas-depletion/mass-conservation/JIT/1-step-match/60-step-match, scan runs/scan-vs-loop/N-conservation, dispatcher. All passing.

### Files Modified
- `tomas_jax/solvers/condensation.py` — Added `condensation_step_tfl_jax()`, `condensation_step_tfl_jit`, `run_condensation_scan_tfl()`. New `method='tfl_jit'` dispatcher.
- `run_box_model.py` — Added `--method tfl_jit` option.
- `benchmarks/python/run_24h_scenarios.py` — Added `tfl_jit` method with JIT warmup and scan-fused fast path.
- `benchmarks/python/compare_methods.py` — Added TFL_JIT to 5-way comparison (Fortran/TFL/TFL_JIT/PPM/PPM_JIT).
- `tomas_jax/solvers/__init__.py` — Updated docstring.

### Performance Results (S01 cond-only, 24h)
| Method | Wall Time | vs Fortran | vs Sequential TFL |
|--------|-----------|------------|-------------------|
| Fortran | 0.091s | 1.0x | — |
| TFL (sequential) | 13.942s | 153x slower | 1.0x |
| **TFL JIT** | **0.323s** | **3.5x slower** | **43x faster** |
| PPM (sequential) | 131.327s | 1443x slower | — |
| PPM JIT | 0.321s | 3.5x slower | — |

### Validation
- TFL JIT matches sequential TFL exactly after 1 step (N_tot rel_diff=0.00e+00, M_dry rel_diff=4.79e-16)
- TFL JIT matches sequential TFL exactly after 60 steps (N_tot rel_diff=0.00e+00, M_dry rel_diff=3.02e-16, all significant bins identical)
- TFL JIT matches sequential TFL after 6 hours in full comparison (N_ratio=0.999846 for both)
- Mass conservation error < 1e-4 relative at hour 6
- All 15 TFL JIT tests pass, all 43 PPM tests pass, all 13 PPM JIT tests pass

### PPM Accuracy Issue Identified
PPM produces overly narrow size distributions (1-bin spike) that don't match Fortran. TFL matches Fortran well. This is a pre-existing algorithmic issue with PPM (not a JIT bug — PPM JIT matches PPM exactly). The TFL JIT approach bypasses this entirely.

### Known Limitations
- TFL JIT is 3.5x slower than Fortran (Fortran uses hand-optimized sequential loops; JAX pays XLA dispatch + fori_loop overhead)
- JIT compilation takes ~30-60s on first call (subsequent calls reuse cached XLA)
- PPM accuracy issue remains open (not addressed in this change)

---

## 2026-02-27 (Thu) — JIT-Compile PPM Condensation Pipeline

**Time**: ~22:00 PST

### Summary
Rewrote the PPM condensation wrappers as pure JAX to enable full JIT compilation of the condensation pipeline. The PPM algorithm itself was already JIT-compiled, but it was wrapped in numpy code with 36+ array conversions and Python for-loops per call, executed 1440 times per 24h run.

### Files Created
- `tomas_jax/physics/ezcond_ppm_jax.py` — Pure-JAX ezcond replacement (~120 lines). Vectorized TAU computation, `jax.lax.cond` for three-way branch (PPM/simple-add/no-op), no internal MNFIX.

### Files Modified
- `tomas_jax/solvers/condensation.py` — Added `condensation_step_jax()` (pure-JAX single step), `condensation_step_jit` (pre-compiled JIT wrapper), `run_condensation_scan()` (scan-fused 1440-step time loop). New `method='ppm_jit'` dispatcher.
- `run_box_model.py` — Added `--method ppm_jit` option.
- `benchmarks/python/run_24h_scenarios.py` — Added `ppm_jit` method with JIT warmup and scan-fused fast path for cond-only mode.
- `tomas_jax/physics/__init__.py`, `tomas_jax/solvers/__init__.py` — Updated docstrings.

### Files Created (Tests)
- `tests/test_ppm_jit_condensation.py` — 13 tests: ezcond_ppm_jax equivalence, JIT compilation, mass/N conservation, scan vs loop equivalence, dispatcher routing. All passing.

### Performance Results
| Config | Old PPM | JIT PPM | Fortran |
|--------|---------|---------|---------|
| Per-step | 21.7ms | 0.1ms | — |
| 24h cond-only | ~12s | 0.29s | 0.08s |
| Speedup | — | ~40x | — |

### Existing Tests
- All 43 existing PPM tests still pass
- All 13 new JIT tests pass
- 89 pre-existing 24h scenario test failures unchanged

---

## 2026-02-27 (Thu) — Fix PPM Number Conservation: Replace Closure with Direct Transport

**Time**: ~17:00 PST

### Root Cause

The PPM closure step (`M_cs = M_dry_integral - M_noncond`) systematically lost ~2% of SO4 mass per step because the PPM integral uses a reconstructed parabola whose effective bin-average mass differs from the discrete geometric mean. Over 1440 steps, this destroyed mass and caused the mass conservation correction to produce absurd ratios (e.g., -192,379), which were then skipped, leaving MNFIX to aggressively clip bins and destroy N.

### Fix: Transport All Species + Add Mass Separately

1. **`condensation_ppm.py`**: Modified `ppm_condensation_step` to transport ALL species (including condensing species) using upwind flux. Removed the closure step entirely (no more `dry_mass_from_ppm_number` + `closure_update` inside substeps).

2. **`ezcond_ppm.py`**: After PPM transport, add condensed mass proportionally to sinkfrac (`Mk2[k, spec] += mcond * sinkfrac[k] / totsinkfrac`). Removed the broken ratio-based mass conservation correction. Reduced MNFIX calls from 4+ to 2 (input + final).

### Why It Works
- Upwind transport preserves total mass (flux telescoping: `sum(F[k+1]-F[k]) = 0`)
- Adding `mcond * sinkfrac` adds exactly `mcond`
- N-M consistency: same `F_N` used for both N and M transport, so `xbar = M/N` stays in bounds, no MNFIX clipping needed

### Results
- **N loss per step**: 0.0000e+00 (was ~2% before)
- **Projected 24h N loss**: 0.0% (was 33% before)
- **Mass conservation**: exact (mcond gained = mcond added)
- **All 43 PPM tests pass** (one test updated to reflect new design)

### Files Modified
- `tomas_jax/physics/condensation_ppm.py` — Transport all species, remove closure
- `tomas_jax/physics/ezcond_ppm.py` — Add mass after PPM, remove broken correction, reduce MNFIX
- `tests/test_ppm_condensation.py` — Updated `test_cspecies_increases_with_growth` → `test_cspecies_conserved_by_transport`
- `diagnose_ppm_n_loss.py` — Updated for new code path

---

## 2026-02-27 (Thu) — 24h Benchmark: Bug Fixes, Fortran Timing, Full 50-Scenario Run

**Time**: ~09:00–15:00 PST

### Bug Fixes

1. **NaN propagation in condensation sink** (`tomas_jax/physics/condensation_sink.py`)
   - **Root cause**: Empty bins have `Dpk = 0` → `Kn = 2*mfp/0 = inf` → `beta = NaN` → `CS = NaN` → gas-phase mass dumped to bin 0 → mass/number blowup
   - **Fix**: Added `safe_beta = jnp.where(Dpk > 0.0, beta, 0.0)` at line 71. Bins with no diameter contribute zero to condensation sink.
   - **Impact**: S01 PPM N_tot went from 746,000x blowup to 0.66 ratio at 24h.

2. **MNFIX empty-bin detection** (`tomas_jax/core/mnfix_jax.py`)
   - **Root cause**: `mask_empty` only checked `Nk_new < TINY_N`, missing bins with number but zero dry mass (created by PPM mass conservation rescaling zeroing out mass)
   - **Fix**: Changed to `mask_empty = (Nk_new < TINY_N) | ((Nk_new >= TINY_N) & (drymass_new < NEPS))` at line 129
   - **Impact**: Prevents NaN propagation from bins with number but no mass

### Fortran Timing

- **Modified**: `benchmarks/fortran/benchmark_24h.f` — Changed timing CSV write from end-of-program to incremental after each scenario (survives crashes)
- **Result**: Fortran timing captured for 49/50 scenarios (S50 crashes in ezcond due to mass conservation STOP)
- **Average times**: coag 0.27s, cond 0.08s, combined 0.33s per 24h scenario

### 50-Scenario JAX Run

- Deleted all old NPZ files and re-ran with both bug fixes applied
- Run covers 50 scenarios × 3 modes × 2 methods (TFL + PPM) = 250 NPZ files
- Small-GMD scenarios (S45 GMD=0.018um, S46 GMD=0.012um) are very slow for PPM due to CFL substep overhead

### Plots Generated (with 41 scenarios)

- All 8 summary figures + 1 timing bar chart generated
- Key findings:
  - TFL mass conservation: excellent (1e-15 to 1e-8 relative error)
  - PPM mass conservation: problematic (10-100% error for many scenarios)
  - TFL vs Fortran N_tot: ~1e-3 median relative error
  - PPM vs Fortran N_tot: ~1e-2 median relative error

### Files Modified
- `tomas_jax/physics/condensation_sink.py` — NaN guard on empty bins
- `tomas_jax/core/mnfix_jax.py` — Empty-bin detection includes zero-dry-mass case
- `benchmarks/fortran/benchmark_24h.f` — Incremental timing CSV write
- `benchmarks/python/compare_24h.py` — Added M_dry keys to save_comparison_npz
- `benchmarks/python/plot_24h_summary.py` — Figure 2 styling (distinct line styles for Fortran/TFL/PPM)

### Known Issues
- **PPM N conservation**: 33% drift for small-GMD scenarios. Eulerian advection + MNFIX clipping breaks 0th moment conservation. Needs moment-preserving scheme (Moving Center or similar).
- **PPM mass conservation**: Many scenarios show >1% dry mass error. Related to N drift issue.
- **JAX speed**: TFL is ~30-100x slower than Fortran (sequential Python loops). PPM varies widely (fast for large GMD, very slow for small GMD due to CFL substeps).
- **Fortran S50**: ezcond calls STOP on mass conservation error, killing the benchmark.
- **Bin-0 mass dump**: When CS is too small, condensation dumps mass to bin 0 — this is nucleation, not condensation. Should be removed before nucleation is implemented.

### Next Steps
- Complete 50-scenario run and regenerate all plots
- Generate timing comparison (Fortran vs JAX-TFL vs JAX-PPM)
- Fix PPM N conservation (Moving Center method or moment-preserving advection)
- Remove bin-0 mass dump from condensation pathway

---

## 2026-02-26 (Wed) — 24h Benchmark Suite: Implementation

**Time**: ~10:00–22:00 PST

### New Files Created

1. **Scenario Generation** (`benchmarks/python/scenarios.py`)
   - 50 Latin Hypercube scenarios via pyDOE2
   - Parameters: N_total, GMD, GSD, temp, pres, RH, Gc_SO4, H2SO4_prod
   - Seed=42 for reproducibility
   - Writes `benchmarks/fortran/scenarios.csv`

2. **JAX 24h Runner** (`benchmarks/python/run_24h_scenarios.py`)
   - Runs all 50 scenarios × 3 modes × 2 methods (TFL + PPM)
   - Saves hourly snapshots to NPZ (Nk, Mk, Gc, N_tot, M_tot, M_dry)
   - Records wall-clock timing per run
   - Writes `benchmarks/results/24h/timing_jax.csv`

3. **3-Way Comparison Engine** (`benchmarks/python/compare_24h.py`)
   - Loads Fortran CSV + JAX NPZ results
   - Computes per-hour relative errors for N_tot, M_tot, M_dry
   - Saves comparison NPZ files per mode

4. **Summary Plot Generator** (`benchmarks/python/plot_24h_summary.py`)
   - Figure 1: Total N timeseries (coag/cond/combined panels)
   - Figure 2: Total dry mass timeseries
   - Figure 3: Error evolution (TFL/PPM vs Fortran, 2×3 grid)
   - Figure 4: PPM vs TFL scatter at hour 24
   - Figure 5: Mass conservation error bar chart
   - Figure 6: Size distribution evolution (3 representative scenarios)
   - Figure 7: All 50 scenarios hour-24 size distribution (10×5 grid)
   - Figure 8: Per-bin error heatmap (50 scenarios × 36 bins)

5. **Timing Plot Generator** (`benchmarks/python/plot_24h_timing.py`)
   - Bar chart: average wall-clock time per mode (Fortran/TFL/PPM)
   - Scatter: per-scenario Fortran vs JAX timing

6. **Fortran Benchmark Harness** (`benchmarks/fortran/benchmark_24h.f`)
   - Reads 50 scenarios from CSV
   - Runs 24h simulation with hourly CSV output
   - Three modes: coag_only, cond_only, combined
   - CPU timing per scenario/mode

7. **Test Suite** (`tests/test_24h_scenarios.py`)
   - Parametrized over all 50 scenarios
   - Tests: TFL vs Fortran (N, M), PPM vs Fortran (N, M), PPM vs TFL, mass conservation
   - Tolerance table: 50% N (different integrators), 5% M (coag), 1% conservation

8. **Shell Runner** (`benchmarks/run_24h.sh`)
   - End-to-end: scenarios → Fortran → JAX → compare → plots → timing

9. **Documentation** (`docs/24h_benchmark.md`)
   - Full benchmark documentation (overview, scenarios, conservation, running, output, plots, tests, FAQ)

### Mass Conservation Fix
- Changed `M_tot` tracking to use dry mass: `M_dry = sum(Mk[:, :SRTH2O])` where SRTH2O=43 (water index)
- Water is diagnostic (set by equilibrium), not conserved — including it creates spurious sources/sinks

### Files Modified
- `benchmarks/python/compare_24h.py` — Full rewrite for 3-way comparison
- `benchmarks/python/plot_24h_summary.py` — Full rewrite with 8 figures
- `tests/test_24h_scenarios.py` — Full rewrite with parametrized tests

---

## 2026-02-25 (Tue) — PPM Condensation: Phase 2 Complete

**Time**: Full day

### New Files Created

1. **PPM Algorithm** (`tomas_jax/physics/condensation_ppm.py`)
   - Piecewise Parabolic Method (Colella & Woodward 1984) in log-mass space
   - JIT-compatible via `jax.lax.fori_loop`
   - 3rd-order accuracy, CFL-based automatic substepping (C_max=0.8)
   - Monotonicity + positivity limiters

2. **PPM ezcond Driver** (`tomas_jax/physics/ezcond_ppm.py`)
   - Same interface as TFL ezcond
   - Computes TAU for full mcond (PPM handles its own CFL substepping)
   - Post-hoc mass conservation correction

3. **Method Selection** (`tomas_jax/solvers/condensation.py`)
   - Added `method='tfl'|'ppm'` parameter to `condensation_step()`
   - TFL: sequential numpy (Fortran-faithful)
   - PPM: JIT-compiled JAX

4. **PPM Tests** (`tests/test_ppm_condensation.py`)
   - 43 tests covering PPM reconstruction, flux, advection, species transport
   - All passing

5. **PPM Documentation** (`docs/ppm_condensation.md`)
   - Full algorithm description with mathematical formulation
   - Edge velocity, species transport, dry mass closure

### Performance
- PPM is ~100-180x faster than TFL for condensation portion
- JIT compilation overhead on first call

---

## 2026-02-24 (Mon) — Condensation: Phase 1 Complete (TFL Port)

**Time**: Full day

### New Files Created

1. **Gas Properties** (`tomas_jax/physics/gas_properties.py`)
   - Fuller-Schettler-Giddings diffusivity
   - Mean molecular speed, mean free path
   - Knudsen number, Fuchs-Sutugin correction

2. **Condensation Sink** (`tomas_jax/physics/condensation_sink.py`)
   - First-order gas loss rate CS [s^-1]
   - Per-bin sink fractions

3. **TFL Condensation** (`tomas_jax/physics/condensation.py`)
   - `dmdt_int` — analytic growth (Stevens et al. 1996)
   - `tmcond` — TFL bin redistribution (Tzivion/Feingold/Levin 1989)
   - Sequential Python loops matching Fortran line-by-line

4. **ezcond Driver** (`tomas_jax/physics/ezcond.py`)
   - Simple condensation driver (Jeff Pierce 2007)
   - Adaptive sub-stepping, mass conservation correction

5. **NH3 Equilibrium** (`tomas_jax/physics/nh3_equilibrium.py`)
   - Stoichiometric NH4:SO4 = 2:1 partitioning

6. **Water Equilibrium** (`tomas_jax/physics/water_equilibrium.py`)
   - Piecewise polynomial fits to ISORROPIA (273K)
   - Ammonium bisulfate + sea salt (disabled)

7. **Architecture Docs** (`docs/architecture.md`)
   - Full condensation pipeline documentation

### Validation
- H2SO4 diffusivity at 298K: 1.15e-5 m²/s (correct)
- dmdt_int: verified growth/shrink/neutral behavior
- Mass conservation: aerosol gained = gas lost within correction

---

## 2025-11-29 (Fri) — Initial Coagulation Implementation

**Commits**: 592ee01 → f41202e

### Features
- Working TOMAS coagulation in JAX
- JIT-compiled via diffrax (Tsit5 adaptive solver)
- Brownian coagulation kernel with Fuchs correction
- TFL mass-conserving algorithm
- MNFIX vectorized in JAX
- Sensitivity analysis via automatic differentiation
- README and basic project structure

### Files
- `tomas_jax/core/config.py` — Constants, dimensions
- `tomas_jax/core/state.py` — TomasState NamedTuple
- `tomas_jax/core/mnfix_jax.py` — Mass-number consistency
- `tomas_jax/physics/coagulation_kernel.py` — Brownian kernel
- `tomas_jax/physics/coagulation_rates.py` — TFL solver
- `tomas_jax/solvers/diffrax.py` — ODE integrator
- `run_box_model.py` — Main driver
- `run_sensitivity_analysis.py` — AD examples

### Validation
- Mass conservation < 1e-13 relative error for coagulation
