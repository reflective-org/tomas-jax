# TOMAS-JAX Development Progress Report

This file tracks all significant changes to the TOMAS-JAX codebase. Entries are in reverse chronological order (newest first).

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
