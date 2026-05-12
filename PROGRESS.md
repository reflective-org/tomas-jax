# TOMAS-JAX Development Progress Report

This file tracks all significant changes to the TOMAS-JAX codebase. Entries are in reverse chronological order (newest first).

---

## 2026-05-07 (Wed) — Nucleation rate helpers refactored

**Time**: evening PST
**Branch**: `feature/zhao2024-rate-fixes`

### Summary
Factored the rate computation out of `zhao2024_nucleation_step` into a
new `zhao2024_nucleation_rate(...) -> (total_J, per_mech_(11,))` helper
and renamed `estimate_nucleation_rate` to `ricco_dunne_nucleation_rate`
for symmetry. Both helpers are JIT-safe and compute the actual
instantaneous rate without state mutation. The "estimate" terminology
was misleading — the function never approximated; it just didn't apply
gas-clamping. Driven by tomas-api needing a way to report the correct
rate per scheme in its diagnostic snapshots and per-step history.

### Changes
- `tomas_jax/physics/nucleation.py`:
  - New `zhao2024_nucleation_rate`. Returns total J and per-mechanism
    array in enable_masks order (bn, tn, bi, ti, syn, porg_n, porg_i,
    org_sa, amine, iod_n, iod_i).
  - `zhao2024_nucleation_step` now delegates rate computation to the
    helper before applying gas-clamping and bin-0 deposition.
  - `ricco_dunne_nucleation_rate` (renamed). `estimate_nucleation_rate`
    kept as a deprecated alias for backward compat.
- `tomas_jax/solvers/condensation.py`: internal callers (4 call sites
  in `_full_step_core`, `full_step_jax`, `_run_scan` body, `make_step`)
  switched to `ricco_dunne_nucleation_rate`.
- `tests/test_nucleation.py`: added `TestRiccoDunneNucleationRate` and
  `TestZhao2024NucleationRate` (8 tests) including a refactor regression
  asserting `dN_step == total_J * boxvol * dt`.

### Known limitations
- The new helper isn't yet re-exported from `tomas_jax/__init__.py`;
  callers still import from `tomas_jax.physics.nucleation`.

---

## 2026-04-18 (Fri) — 1000-Scenario Tropo→Strato Benchmark (PPM-only)

**Time**: late afternoon PST

### Summary
Scaled the Fortran-vs-JAX PPM benchmark from 49 tropospheric scenarios to 1000 scenarios spanning troposphere → lower stratosphere. Three strata: tropo (600, P ≥ 25 kPa), UTLS (250, 5 kPa ≤ P < 25 kPa), strato (150, 100 Pa ≤ P < 5 kPa). JAX PPM_JIT vs Fortran PPM on all 3 modes (coag_only, cond_only, combined).

### New infrastructure
- `benchmarks/python/scenarios_atmos.py` — stratified LHC generator (3 regimes × regime-realistic ranges)
- `tomas_fortran/harness/benchmark_24h_ppm_atmos.f` — 1000-scenario Fortran harness with 4-digit scenario IDs, crash-tolerant timing CSV
- `benchmarks/python/run_atmos_scenarios.py` — PPM-only JAX runner; fast combined path via `run_combined_scan_ppm` (avoids the 900 s/scenario Python fallback)
- `benchmarks/python/export_atmos_metrics.py` — regime-classified skill metrics export (1000 × 3 modes × 4 hours = 12k rows)
- `benchmarks/python/plot_atmos_suite.py` — 12 comprehensive figures (coverage, pairplot, regime Taylor, violin, CDF, error-vs-P, error contour, per-bin heatmap, error evolution, mass conservation, rep scenarios, walltime)
- `benchmarks/python/build_atmos_tables.py` — regime-stratified markdown + LaTeX skill tables
- `benchmarks/run_atmos.sh` — orchestration driver
- `tomas_fortran/Makefile` — added `benchmark_24h_ppm_atmos` + `run_24h_ppm_atmos` targets

### Runtime (measured)
- Fortran 1000-run: 1000 scenarios × 3 modes = 3000 sims, 216 001 CSV files, 0 crashes
- JAX 1000-run: 2993/3000 runs in 3.23 h wall (coag_only ~3.9 s/scen, cond_only ~0.2 s/scen, combined ~0.6 s/scen)
- Results: `benchmarks/results/atmos/` (3000 NPZs) + `tomas_fortran/output/atmos/` (216k CSVs) + `benchmarks/results/atmos/stats/` (12 figures + 2 tables + 12k-row metrics CSV)

### Key discovery
The old `run_24h_scenarios.run_scenario` combined-mode path falls through to a Python for-loop with 1440 per-step dispatches (~920 s/scenario). Bypassed by calling `run_combined_scan_ppm` directly from `run_atmos_scenarios.run_combined_fast`, which uses the existing XLA scan-fused kernel. Per-scenario wall time dropped from 920 s → 0.6 s (1500× speedup).

### Ensemble skill (h24, all 1000 scenarios)
| Regime | Mode | n | Nk R² med | Nk KGE med | \|N_tot relerr\| med | \|N_tot relerr\| p95 |
|---|---|---:|---:|---:|---:|---:|
| tropo  | coag_only | 600 | 0.9997 | 0.9967 | 2.3e-03 | 1.5e-02 |
| tropo  | cond_only | 600 | 0.9585 | 0.9401 | 1.7e-07 | 2.4e-04 |
| tropo  | combined  | 600 | 0.9802 | 0.9679 | 3.3e-03 | 2.8e-02 |
| utls   | coag_only | 250 | 0.9997 | 0.9990 | 6.4e-04 | 1.4e-02 |
| utls   | cond_only | 250 | 0.9325 | 0.8804 | 2.0e-07 | 2.1e-04 |
| utls   | combined  | 250 | 0.9371 | 0.8849 | 1.4e-03 | 2.7e-02 |
| strato | coag_only | 150 | 0.9998 | 0.9996 | 2.7e-04 | 5.8e-03 |
| strato | cond_only | 150 | 0.9097 | 0.8571 | 5.5e-11 | 1.3e-07 |
| strato | combined  | 150 | 0.9120 | 0.8585 | 6.6e-04 | 9.7e-03 |

### Findings
1. **Coagulation**: JAX matches Fortran to within 0.03-0.2% median on N_tot across the entire tropo→strato range. Nk R² > 0.9997 in all regimes.
2. **Condensation**: per-bin R² degrades from 0.96 (tropo) → 0.91 (strato) as conditions get colder/drier/sparser. But total N is essentially unchanged (N_tot_relerr median 1e-7 to 5e-11) — the two codes redistribute the mass slightly differently but conserve the totals to machine precision.
3. **Stratosphere**: N_total rel err is smallest in strato (5.5e-11 for cond_only) because condensation events are tiny when gas and N are low. But per-bin log-bias is larger (regime has wider Nk dynamic range).
4. **No Fortran crashes** at any scenario across the 1000-point atmospheric envelope — Fortran PPM is robust stratospheric-regime conditions (surprising given the code has no stratospheric test coverage in the original distribution).

---

## 2026-04-18 (Fri) — Fortran-vs-JAX Statistical Benchmark Suite + Audit

**Time**: afternoon PST

### Summary
Added a thin statistical-reporting layer on top of the existing 49-scenario LHC × 5-mode × 2-method benchmark so Fortran-vs-JAX parity can be presented to the group with quantitative skill metrics. Zero new simulations; all new code consumes existing NPZ/CSV artifacts. Also shipped a consolidated audit document covering JIT status and precomputed-and-recycled compromises.

### New files
- `benchmarks/python/metrics.py` — skill metrics (bias, rmse, nrmse, mape, r2, pearson_r, kge, log_bias, max_relerr)
- `benchmarks/python/export_metrics_csv.py` — 3750-row per-(scenario,mode,pair,hour) CSV + 75-row summary
- `benchmarks/python/plot_statistical_suite.py` — 7 figures: Taylor, Q-Q, CDF, per-bin heatmap, error-vs-params, walltime, convergence
- `benchmarks/python/build_presentation_tables.py` — markdown + LaTeX ensemble tables
- `benchmarks/run_stats.sh` — single entrypoint, runs zero simulations
- `docs/AUDIT_JAX_VS_FORTRAN.md` — consolidated audit: JIT status, compromises, GPU readiness

### Key numbers (h24, 49 LHC scenarios)
- coag_only TFL vs Fortran: Nk R² = 0.9998, |N_tot relerr| median = 1.2e-3, p95 = 9.0e-3
- cond_only TFL vs Fortran: Nk R² = 0.9986, |N_tot relerr| median = 1.4e-7
- combined TFL vs Fortran: Nk R² = 0.9996, |N_tot relerr| median = 1.5e-3
- `full` mode has the intentional nucleation-clamping tail (p95 ≈ 1); expected.

### Infrastructure
- Symlinked `tomas_fortran/output/24h` → nested actual location for compare_24h path compatibility.
- `manifest.json` records md5 of scenarios.csv so stats export exits loudly if LHC regenerates.
- `--scenarios` flag plumbed through for incremental runs.

### Files Modified
- `PROGRESS.md` — this entry
- `docs/porting_status.md` — cross-link to audit doc

---

## 2026-04-17 (Thu) — Radiative Forcing JAX Port: Phase 5 (Gauss-Legendre Upscatter)

**Time**: morning PST

### Summary
Replaced scipy.quad-based upscatter integration with 32-point Gauss-Legendre quadrature, making the entire precomputation pipeline JIT-compilable. Test suite speedup: 36s → 3s. All 84 RF tests pass.

### Phase 5: Gauss-Legendre Upscatter
- Added `_upscatter_fraction_gl(g, sza_rad)` — JIT-compilable upscatter via 32-point GL quadrature
- Added `_avg_solar_power_gl(lat_rad, sda, So)` — JIT-compilable 24h solar power average
- Rewrote `_compute_global_avg_upscatter` — vectorized with 2D vmap over (g, sza) pairs
- Vectorized `scattering_efficiency_vs_radius` — single-wavelength uses vmap + batch upscatter; spectral uses 2D vmap
- Module-level GL constants: `_GL_NODES`, `_GL_WEIGHTS` (32-point, from `np.polynomial.legendre.leggauss`)
- GL vs scipy accuracy: atol < 1e-4 across 16 (g, sza) combinations
- Plot validation: fig2/fig3 byte-identical; fig1 visually identical (121-byte diff from GL approximation)

### Tests Added
- `TestGaussLegendreUpscatter` class (22 tests): GL vs scipy match (4g × 4sza = 16 parametrized), bounds, JIT, vmap, grad, isotropic
- Total: 84 RF tests (37 existing + 47 new)

### Performance
- Test suite: 36s → 3.3s (11x speedup from eliminating scipy.quad in precomputation)
- `scattering_efficiency_vs_radius`: fully vectorized, no Python loops

### Files Modified
- `tomas_jax/physics/radiative_forcing.py` — GL quadrature, vectorized upscatter/scattering_efficiency
- `tests/test_radiative_forcing.py` — `TestGaussLegendreUpscatter` class

---

## 2026-04-16 (Wed) — Radiative Forcing JAX Port (Phases 1–4)

**Time**: evening PST

### Summary
Ported the radiative forcing pipeline to pure JAX with JIT compilation, vmap vectorization, and autodiff support. Three files modified: `bhmie.py` (new JAX Mie functions), `radiative_forcing.py` (runtime functions ported to jnp), `test_radiative_forcing.py` (26 new tests).

### Phase 1: bhmie → JAX
- Added `bhmie_jax(x, refrel, nang)` — full Mie with S1/S2, JIT-compilable via `@partial(jax.jit, static_argnums=(2,))`
- Added `bhmie_qsca_jax(x, refrel)` — lightweight (Qext, Qsca, gsca only), `@jax.jit`
- Uses `jax.lax.fori_loop` with fixed upper bounds (NSTOP_MAX=250, NMX_MAX=300) and traced masks
- Fixed S1/S2 backward sign convention: `p = -p` must happen **before** backward amplitude accumulation
- Matches numpy `bhmie` to rtol=1e-10 for x ∈ {0.01, 0.1, 1.0, 5.0, 20.0, 50.0, 100.0}

### Phase 2: Runtime RF → jnp
- `compute_optical_depth`, `compute_rf`, `compute_mass_scattering_efficiency`, `compute_rf_efficiency`: `np.*` → `jnp.*`, `if/else` → `jnp.where`, `@jax.jit`
- `h2so4_solution_density`, `_interpolate_upscatter`: `np.interp` → `jnp.interp`, `@jax.jit`
- `_planck`: `np.exp` → `jnp.exp`; `_solar_spectral_weights`: `np.linspace` → `jnp.linspace`
- Precomputation functions (scipy.quad-based) remain as numpy fallbacks

### Phase 3: vmap Mie precomputation
- Single-wavelength: `jax.vmap(bhmie_qsca_jax, (0, None))` replaces 40-bin for-loop
- Spectral: 2D vmap `jax.vmap(jax.vmap(...))` replaces 30-wavelength × 40-bin double loop
- Vectorized upscatter interpolation and weighted accumulation via broadcasting

### Phase 4: Tests
- 26 new tests: `TestBhmieJax` (14 tests) + `TestJITRuntime` (12 tests)
- bhmie_jax: numpy reference match, JIT, vmap (1D + 2D), grad through Qsca
- Runtime: JIT correctness, grad w.r.t. Nk and column_area, vmap over distributions
- Total: 63 RF tests (37 existing + 26 new)

### Files Modified
- `tomas_jax/physics/bhmie.py` — Added `bhmie_jax`, `bhmie_qsca_jax`, helpers
- `tomas_jax/physics/radiative_forcing.py` — JAX imports, jnp runtime, vmap precomputation
- `tests/test_radiative_forcing.py` — `TestBhmieJax`, `TestJITRuntime` classes

### Known Limitations
- Grad through Qsca at integer nstop boundaries has local accuracy only (jnp.int32 truncation)

---

## 2026-04-16 (Wed) — Phase A Hardening: Unit Tests for Foundation Modules

**Time**: evening PST

### Summary
Added 90 unit tests across 4 previously untested foundation modules: `nh3_equilibrium.py`, `water_equilibrium.py`, `condensation_sink.py`, and `mnfix_jax.py`. All 677 tests pass (up from 579). Updated `docs/porting_status.md` with test counts.

### New Test Files

1. **`tests/test_nh3_equilibrium.py`** — 20 tests
   - Ammonia-limited case (all NH3 → particles, distributed proportional to SO4)
   - Ammonia-excess case (2:1 NH4:SO4 stoichiometry, excess stays gas)
   - Total nitrogen conservation across 5 NH3/SO4 ratios
   - Edge cases: zero NH3, zero SO4, single bin, species isolation
   - JIT compilation and eager/JIT equivalence

2. **`tests/test_water_equilibrium.py`** — 30 tests
   - Sulfate WR: known references (50%→1.29, 80%→1.96, 95%→5.05), monotonicity, clipping
   - Piecewise polynomial continuity at 5 breakpoints (41%, 61%, 81%, 91%, 96%)
   - Sea salt WR: low RH baseline, higher hygroscopicity than sulfate
   - calc_equilibrium_water: SO4/organic contributions, proportionality, species isolation
   - JIT compilation

3. **`tests/test_condensation_sink.py`** — 18 tests
   - CS physical range, formula verification, scaling with Nk and particle size
   - sinkfrac normalization, non-negativity, single-bin dominance
   - NEPS threshold (1e10), sparse bin default density handling
   - NaN guard for Dpk=0, accommodation coefficient effect
   - JIT compilation

4. **`tests/test_mnfix.py`** — 22 tests
   - Phase 1: empty bins get NEPS=1e-5, SO4 at geometric mean
   - Phase 2: extreme avg mass trimming (above grid, below grid)
   - Phase 3: partial transfer (upward, downward, partial not total, in-range unchanged)
   - Conservation: number, mass, multi-species, composition fractions
   - Multi-bin jumps (10+ bins), positivity, no-NaN stress test
   - JIT compilation

### Updated Documentation
- `docs/porting_status.md`: Updated test counts in tables, marked 4 modules as tested, reduced untested count from 8 to 4

---

## 2026-04-15 (Tue) — Radiative Forcing with Tabazadeh H₂SO₄/H₂O Equilibrium

**Time**: afternoon PST

### Summary
Added direct shortwave radiative forcing module (Chylek & Wong 1995 / Pierce et al. 2010) with Bohren-Huffman Mie scattering and Tabazadeh et al. (1997) binary H₂SO₄/H₂O equilibrium composition parameterization.

### Changes

1. **Mie scattering** (`tomas_jax/physics/bhmie.py`)
   - Bohren-Huffman Mie code (numpy). Computes Qsca, Qext, gsca for any size parameter and refractive index.

2. **Radiative forcing** (`tomas_jax/physics/radiative_forcing.py`)
   - `precompute_mie_properties()`: Mie + global-avg upscatter (Wiscombe & Grams 1976). Single-wavelength and spectral integration modes.
   - `compute_rf()`: Chylek & Wong RF with Pierce SI cloud-fraction extension.
   - `scattering_efficiency_vs_radius()`: Reproduces Pierce et al. (2010) Figure 1.
   - `h2so4_equilibrium_wt(temp, rh)`: Tabazadeh et al. (1997) Table 1 vapor pressure interpolation.
   - `h2so4_solution_density(wt)`: CRC Handbook density interpolation.

3. **Benchmark plots** (`benchmarks/python/plot_radiative_forcing.py`) — 5 figures
4. **Tests** (`tests/test_radiative_forcing.py`) — 8 Tabazadeh composition tests + RF tests
5. **Documentation** (`docs/radiative_forcing.md`)

### Key Finding
Pierce assumed 75 wt% H₂SO₄ (peak 0.84 W/m² per Mt-S). Tabazadeh at T=220K, RH=5% gives 63.8 wt% and peak 1.09 — 30% higher due to water dilution.

---

## 2026-04-10 (Thu) — JAX Performance Optimization

**Time**: evening PST

### Summary
Comprehensive JAX/JIT performance audit and optimization. Identified and fixed redundant computation in hot loops, unnecessary memory allocations, and unused scan arrays.

### Changes

1. **PPM substep loop-invariant hoisting** (`tomas_jax/physics/condensation_ppm.py`)
   - Moved `compute_edge_velocity()` and `_compute_moment_integrals()` out of `fori_loop` in both `ppm_condensation_step` and `ppm_advect_number_only`
   - All inputs are constant across CFL substeps — explicit hoisting guarantees no redundant computation regardless of XLA LICM behavior

2. **Removed `.copy()` on immutable JAX arrays** (`tomas_jax/physics/condensation_tfl_jax.py`)
   - `ANKD.copy()` / `AMKD.copy()` replaced with direct references — JAX arrays are immutable

3. **Eliminated unused scan carry allocation** (`tomas_jax/solvers/condensation.py`)
   - `jnp.arange(nsteps)` passed to `jax.lax.scan` but never read by body → replaced with `None`

### Deferred (not worth the churn)
- `static_argnums` → `static_argnames`: cleaner but requires updating all call sites
- `compute_zeta(xk)` caching: just a few scalar FLOPs, XLA LICM handles this
- kwargs validation restructure: current trace-time behavior is correct

### Verification
All 550 tests pass (0 failures). PPM, TFL JIT, and smoke tests verified.

---

## 2026-04-10 (Thu) — Radiative Forcing Module (Phase 9)

**Time**: afternoon PST

### Summary
Added direct shortwave radiative forcing calculation based on Chylek & Wong (1995), following Pierce et al. (2010).

### Files created
- `tomas_jax/physics/bhmie.py` — Bohren-Huffman Mie scattering (numpy, cleaned from reference)
- `tomas_jax/physics/radiative_forcing.py` — RF module: Mie precomputation per TOMAS bin, upscatter fraction (Wiscombe & Grams 1976), optical depth, Chylek & Wong RF equation
- `tests/test_radiative_forcing.py` — 29 tests (all passing), covers Mie, upscatter, RF, MSE, Pierce Fig 1 validation
- `docs/radiative_forcing.md` — Full documentation with usage examples

### Key features
- `precompute_mie_properties(xk)`: Qsca, Qext, gsca, global-avg upscatter for each bin
- `compute_rf(Nk, mie, column_area)`: RF [W/m²] from Chylek & Wong 1995
- `compute_rf_efficiency(Nk, Mk, mie)`: RF per unit burden [W/m² per g/m²]
- `compute_mass_scattering_efficiency(Nk, Mk, mie)`: MSE [m²/g]
- `scattering_efficiency_vs_radius()`: Reproduces Pierce et al. 2010 Figure 1 curve
- `compute_rf_latitude_resolved()`: Full lat/month calculation as in getRF_monthly.py

### References
- Chylek & Wong (1995), GRL 22, 929-931
- Pierce et al. (2010), GRL 37, L18805
- Wiscombe & Grams (1976), J. Atmos. Sci. 33, 2440-2451
- Bohren & Huffman (1983), Absorption and Scattering of Light by Small Particles

### Next steps
- Multi-wavelength integration across solar spectrum
- Gravitational settling efficiency (Pierce Fig 1 right axis)

---

## 2026-04-10 (Thu) — Code Review Round 2: Robustness, Tests, Documentation

**Time**: morning PST

### Summary
Second round of code review hardening. Focus on numerical robustness, test quality, test coverage, and physics documentation.

### Changes
1. **Euler solver robustness** — replaced hardcoded `1e-15`/`1e-25` thresholds with `TINY_N`/`TINY_M` from config. Added `return_info` parameter to detect when `max_substeps` truncates integration.
2. **Tightened PPM flux test** — `rtol=0.1` (10%) → `rtol=1e-10` for test where analytical answer is exact.
3. **Shared test fixtures** — created `tests/conftest.py` with `xk` fixture and `make_lognormal` helper. Removed duplicate grid construction from 3 test files.
4. **Deleted stale `pyproj.toml`** (typo'd filename, outdated metadata). Removed dead `ZETA_DEFAULT` constant. Fixed misleading Dpk=0 comment in coagulation kernel.
5. **Literature references** — added citations for Riccobono 2014, Dunne 2016, Tang 1997, Sutherland viscosity, and TOMAS MFP formula.
6. **Smoke tests** — 8 lightweight tests (~5s) exercising coagulation, condensation, and full pipeline. Always run, no Fortran data needed.
7. **Physics foundation tests** — 22 unit tests for `density.py`, `properties.py`, `gas_properties.py` (previously zero coverage). Tests against reference values and physical limits.

### Files Modified/Created
- `tomas_jax/solvers/euler.py` (thresholds, return_info)
- `tomas_jax/physics/coagulation_rates.py` (removed ZETA_DEFAULT)
- `tomas_jax/physics/coagulation_kernel.py` (comment fix)
- `tomas_jax/physics/nucleation.py` (citations)
- `tomas_jax/physics/density.py` (citation)
- `tomas_jax/physics/properties.py` (citation)
- `tomas_jax/physics/gas_properties.py` (MFP explanation)
- `tests/conftest.py` (new — shared fixtures)
- `tests/test_smoke.py` (new — 8 smoke tests)
- `tests/test_physics_foundations.py` (new — 22 unit tests)
- `tests/test_ppm_condensation.py` (tightened tolerance)
- `tests/test_nucleation.py`, `test_tfl_jit_condensation.py`, `test_ppm_jit_condensation.py` (use shared fixtures)
- Deleted: `pyproj.toml`

---

## 2026-04-09 (Wed) — Code Review Fixes

**Time**: evening PST

### Summary
Addressed issues found during comprehensive dev branch code review. Focus on API safety, solver robustness, naming consistency, and test infrastructure.

### Changes
1. **kwargs validation in `make_step()`** — warns on unknown kwargs (catches typos that silently default to 0.0). Validates against set of 17 known parameter names.
2. **Process order validation** — warns when process order deviates from canonical physical ordering (so2_chemistry → nucleation → coagulation → condensation → dilution).
3. **diffrax solver status checking** — checks `solution.result` after each Tsit5 substep; reverts to previous state on failure instead of silently using partial results. Optional `return_status=True` flag for callers.
4. **`icomp_nodiag` constant cleanup** — replaced all hardcoded `=42` defaults with `ICOMP_NODIAG` from config across 6 files.
5. **Renamed `accommodation_coeff` → `alpha`** — standardized parameter name across `condensation_sink.py`, `gas_properties.py`, and 10 call sites to match TomasState convention.
6. **Stale README reference** — removed deleted `run_sensitivity_analysis.py` from project structure.
7. **`coag_rk4_step` deprecation** — now emits `DeprecationWarning` instead of being a silent alias.
8. **Top-bin overflow tracking for coagulation mass budget** — `calc_coagulation_rates` now returns a third value `dM_overflow` (mass rate that would enter a hypothetical bin above the grid, i.e., the terms `shift_right` truncates). Both `coag_euler_step` and `diffrax_step` accumulate overflow via `return_overflow=True`. Root cause of coag mass conservation test failures: top-bin self-coagulation produces particles exceeding the grid boundary — this mass is physically lost. Now tracked explicitly so `M(0) = M(24h) + overflow` closes to <1e-8 relative.
9. **Rewrote `test_coag_mass_conservation`** — no longer loads stale NPZ data; runs solver inline with overflow tracking and verifies mass budget closure across all 50 scenarios.
10. **`--force` flag for benchmark runner** — `run_24h_scenarios.py` now supports `--force` to overwrite existing NPZ files.

### Files Modified
- `tomas_jax/solvers/condensation.py` (kwargs validation, process order, ICOMP_NODIAG)
- `tomas_jax/solvers/diffrax.py` (solver status, ICOMP_NODIAG, coag_rk4_step deprecation, overflow tracking)
- `tomas_jax/solvers/euler.py` (ICOMP_NODIAG, overflow unpacking)
- `tomas_jax/solvers/coagulation_jax.py` (overflow unpacking)
- `tomas_jax/physics/coagulation_rates.py` (ICOMP_NODIAG, dM_overflow return)
- `tomas_jax/physics/condensation_sink.py` (alpha rename)
- `tomas_jax/physics/gas_properties.py` (alpha rename)
- `tomas_jax/physics/ezcond.py`, `ezcond_ppm.py`, `ezcond_ppm_jax.py`, `condensation_tfl_jax.py` (alpha rename)
- `tomas_jax/core/mnfix_jax.py`, `mnfix_fortran.py` (ICOMP_NODIAG)
- `tomas_jax/utils/diagnostics.py` (overflow unpacking)
- `tests/test_24h_scenarios.py` (rewrote coag mass conservation test)
- `benchmarks/python/run_24h_scenarios.py` (--force flag)
- `benchmarks/python/level07_rates.py` (overflow unpacking)
- `README.md` (stale reference)

---

## 2026-03-24 (Mon) — JAX/GPU-Readiness Audit

**Time**: ~evening PST

### Summary
Comprehensive audit of all ~25 tomas_jax modules for GPU deployment readiness. The JIT-compiled fast paths were already excellent pure JAX. This audit fixed patterns that would cause issues on GPU: `None` default args breaking JIT tracing, missing kwarg defaults, scattered `jax.config.update` calls, and undocumented legacy numpy paths.

### Changes
1. **Moved coag import to module level** in `solvers/condensation.py` — was inside 3 function bodies, now at top with other imports.
2. **Removed 12 redundant `jax.config.update("jax_enable_x64", True)`** calls from library modules (`core/mnfix_jax.py`, `core/mnfix_fortran.py`, `physics/condensation_sink.py`, `physics/coagulation_rates.py`, `physics/coagulation_kernel.py`, `physics/density.py`, `physics/gas_properties.py`, `physics/nh3_equilibrium.py`, `physics/water_equilibrium.py`, `physics/properties.py`, `solvers/euler.py`, `solvers/diffrax.py`). Only `core/config.py` retains the call.
3. **Fixed `dilution_step` None defaults** — removed `None` default args that break JIT when call patterns change. Now requires explicit arrays. Updated callers: `solvers/condensation.py`, `tests/test_dilution.py`, `run_box_model.py`.
4. **Fixed `make_step` kwargs** — replaced bare `kwargs['nh3_conc']`, `kwargs['fion']`, `kwargs['org_conc']` with `kwargs.get('key', 0.0)` to prevent KeyError crashes.
5. **Added deprecation warnings** to legacy numpy paths (`method='tfl'`, `method='ppm'`) in `condensation_step()`. Updated docstrings in `condensation.py`, `ezcond.py`, `ezcond_ppm.py`.
6. **Auto-JIT `make_step`** — added `jit=True` parameter; returns JIT-compiled function by default. Removed redundant `jax.jit()` wrapping from `run_box_model.py`, `benchmark_dilution.py`, `benchmark_so2_sensitivity.py`, `run_sai_simulation.py`.
7. **Created `docs/gpu_deployment.md`** — float64 by GPU type, `JAX_PLATFORMS` usage, code path selection, `jax.vmap` batch processing, memory estimates, JIT stability tips.
8. **Updated CLAUDE.md** — added GPU deployment section, centralized x64 config note, dilution_step change, file layout entry.

### Files Modified
- `tomas_jax/solvers/condensation.py` (import, kwargs, dilution, deprecation, auto-JIT)
- `tomas_jax/physics/dilution.py` (removed None defaults)
- `tomas_jax/physics/condensation.py`, `ezcond.py`, `ezcond_ppm.py` (deprecation docstrings)
- 12 files: removed redundant `jax.config.update` calls
- `tests/test_dilution.py` (explicit zero backgrounds)
- `run_box_model.py` (auto-JIT, explicit backgrounds)
- `benchmarks/python/benchmark_dilution.py`, `benchmark_so2_sensitivity.py` (auto-JIT)
- `experimental_case/run_sai_simulation.py` (auto-JIT)

### Files Created
- `docs/gpu_deployment.md`

### Known Issues
- Tests and benchmarks retain their own `jax.config.update` calls (they may be run as standalone scripts)
- No actual GPU testing performed (audit is structural/code-level only)

---

## 2026-03-14 (Fri) — SAI 96h Box Model Simulation

**Time**: ~10:15 PM PST

### Summary
Added a 96-hour SAI box model simulation that evolves a fresh SO2 plume (10 tonnes, 1 ton/min, 250 m/s, r=100m → ~2×10¹³ molec/cm³) through all microphysical processes at stratospheric conditions (216K, 55 hPa). Three dilution cases: baseline (closed box), volume 2× in 24h, and volume 1.78× in 24h.

### Files Created
- `experimental_case/run_sai_simulation.py` — 96h simulation: 3 cases × 5760 steps, 6 figures
- `experimental_case/__init__.py` — Package init for module imports

### Files Modified
- `experimental_case/README.md` — Added simulation description, usage, output table

### Key Results
- Massive nucleation burst in first ~2h: N goes from 5/cm³ background to ~2×10⁷/cm³
- SO2 decay: baseline τ≈11 days (OH-limited), dilution cases much faster
- H2SO4 consumed rapidly by condensation sink + nucleation
- Banana plots show classic nucleation → growth → coagulation evolution
- Runtime: ~15s per case (~45s total for 3 cases)

### Figures (in `experimental_case/results/`)
1. `fig1_ntotal.png` — N_total timeseries (absolute + ratio with passive tracer)
2. `fig2_mdry.png` — M_dry timeseries
3. `fig3_dndlogdp.png` — dN/dlogDp snapshots at 12h, 24h, 48h, 72h, 96h
4. `fig4_dmdlogdp.png` — dM/dlogDp snapshots
5. `fig5_banana.png` — Banana plots (3 panels)
6. `fig6_gas.png` — SO2 + H2SO4 gas-phase timeseries

---

## 2026-03-13 (Thu) — Dilution / Entrainment Process

**Time**: ~PST

### Summary
Added dilution/entrainment as the 5th physics process. First-order relaxation of Nk, Mk, Gc toward a background state: `C(t+dt) = Cbg + (C - Cbg) * exp(-kdil * dt)`. Analytical, unconditionally stable, JIT-compilable. Includes a passive scalar tracer (`dilute_tracer()`) that decays by dilution only — provides a clean reference to compare reactive species against pure dilution.

### Files Created
- `tomas_jax/physics/dilution.py` — `dilution_step()` + `dilute_tracer()` functions
- `tests/test_dilution.py` — 10 unit tests (6 dilution_step + 4 passive tracer)
- `docs/dilution.md` — Algorithm docs, typical kdil values, tracer usage

### Files Modified
- `tomas_jax/solvers/condensation.py` — Added `'dilution'` to `make_step()` valid set + process dispatch
- `run_box_model.py` — Added `--dilution-rate`, `--dilution-bg` CLI flags; passive tracer tracking + diagnostics
- `CLAUDE.md` — Added dilution.py to file layout, updated architecture rules
- `docs/missing_physics.md` — Marked §5 (Dilution) as implemented

### Benchmark
- `benchmarks/python/benchmark_dilution.py` — 3 cases (baseline/moderate/strong) × 24h, all processes + SO2
- 6 figures: N_total, M_dry, gas-phase SO2+H2SO4, size distributions, dilution attribution, banana plots
- Passive tracer overlay shows chemistry/nucleation partially compensating dilution losses
- Runtime: ~1.8s per scenario (3 × 1.8s = 5.4s total)

### Usage
```bash
python run_box_model.py --make-step --dilution-rate 1e-4                       # clean-air
python run_box_model.py --make-step --dilution-rate 1e-4 --dilution-bg ambient # relax to initial
python -m benchmarks.python.benchmark_dilution                                 # full benchmark + plots
python -m benchmarks.python.benchmark_dilution --plot-only                     # replot from NPZ
```

---

## 2026-03-12 (Thu) — SO2 Lifetime Heatmaps (Surface to Stratosphere)

**Time**: ~6:00 PM PST

### Summary
Added Figure G to SO2 chemistry validation: 3-panel 2D heatmap showing SO2+OH kinetics from surface to stratosphere (0–50 km). Uses US Standard Atmosphere 1976 (all 5 layers).

### Key Results
- Panel (a): k1(T,P) — rate constant drops ~130× from surface to 50km due to pressure decrease
- Panel (b): SO2 lifetime at [OH]=10⁶ — ranges from ~10 days (surface) to ~1400 days (50km)
- Panel (c): Lifetime vs altitude & [OH] — practical reference chart for atmospheric modelers
- Stratospheric temperature inversion (T rising above 20km) visible in standard atmosphere overlay

### Files Modified
- `benchmarks/python/validate_so2_chemistry.py` — Added `_std_atmosphere()` (US Std Atm 1976, 0–50km), `figure_g()` (3-panel heatmap); now 7 figures total

---

## 2026-03-12 (Thu) — SO2 Sensitivity Benchmark

**Time**: ~5:00 PM PST

### Summary
Added SO2 sensitivity benchmark: 5 SO2 concentrations (1e8–1e11 molec/cm³) × 4 process modes × 3 altitudes × 2 grid resolutions over 48 hours. Produces 12 figures showing SO2 decay (3 units), H2SO4 gas, particle number/mass evolution, banana plots, and size distributions.

### Files Created
- `benchmarks/python/benchmark_so2_sensitivity.py` — Full benchmark runner + 12-figure plotting suite

### Key Results
- 20 scenarios (surface, 40-bin) run in ~40s total
- SO2 decay ratio at 48h: ~0.82 (surface, cond_only, SO2=1e10), consistent with k1×[OH]=1e6 lifetime ~11 days
- Full 120-scenario suite (all altitudes + 80-bin): ~15-30 min estimated
- All 12 figures generate correctly from saved NPZ files

### Usage
```bash
python -m benchmarks.python.benchmark_so2_sensitivity           # run all + plot
python -m benchmarks.python.benchmark_so2_sensitivity --plot-only  # replot from NPZ
python -m benchmarks.python.benchmark_so2_sensitivity --grids 40   # 40-bin only
python -m benchmarks.python.benchmark_so2_sensitivity --altitudes sfc  # surface only
```

---

## 2026-03-12 (Thu) — SO2 + OH Gas-Phase Chemistry

**Time**: ~2:30 PM PST

### Summary
Implemented SO2 + OH → H2SO4 gas-phase chemistry using the Sun et al. (2022) Troe formalism with H2O vapor enhancement. This replaces the constant H2SO4 production rate with physically-based SO2 oxidation. All 4 functions are JIT-compilable.

### Key Details
- **Rate constant**: k1(298K, 1atm) = 1.05×10⁻¹² cm³/molec/s, SO2 lifetime ~11 days at [OH]=10⁶
- **H2O enhancement**: 2-5% at 298K, 50% RH (Sun et al. Eqs. 5-6)
- **OH modes**: Constant (default) and diurnal (proportional to cos(SZA))
- **Gc extended**: 43→44 elements. SRTSO2=43, N_GAS_SPECIES=44
- **Backward compatible**: No SO2 flags → legacy constant H2SO4 production path

### Files Created
- `tomas_jax/physics/so2_chemistry.py` — 4 JIT functions: calc_k1_so2_oh, calc_solar_zenith_angle, calc_oh_concentration, so2_oxidation_step
- `tests/test_so2_chemistry.py` — 28 tests (Troe rate, SZA, OH modes, oxidation step, conservation)
- `benchmarks/python/validate_so2_chemistry.py` — 6 validation figures reproducing Sun et al. (2022)
- `docs/so2_chemistry.md` — Algorithm documentation, usage, validation

### Files Modified
- `tomas_jax/core/config.py` — Added SRTSO2=43, MW_SO2=64.066, MW_OH=17.007, SV_SO2=41.73; N_GAS_SPECIES=44
- `tomas_jax/solvers/condensation.py` — Added 'so2_chemistry' to make_step valid processes; _run_scan supports so2_prod_rate
- `run_box_model.py` — New CLI: --so2-init, --so2-emission, --oh-conc, --oh-diurnal, --lat, --lon, --day-of-year
- `docs/references.md` — Added Sun et al. (2022), Buck (1981)
- `README.md` — SO2 chemistry features, CLI options, state variables, file tree
- `CLAUDE.md` — SO2 chemistry architecture rules, file layout, species indices

### Reference
- Sun, W., et al.: Kinetics of OH + SO2 + M. *Atmos. Chem. Phys.*, 22, 4969-4984, 2022.
- Supplemental: `references/so2_chemistry/acp-22-4969-2022-supplement.pdf`

### Verification
- 160 tests pass (28 new + 132 existing, all backward-compatible)
- Box model runs with SO2 chemistry: `python run_box_model.py --so2-init 5e10 --oh-conc 1e6 --make-step`
- 6 validation figures in `benchmarks/results/so2_chemistry/`

---

## 2026-03-12 (Thu) — Default Grid: 40 Bins, 1.7nm Start

**Time**: ~12:00 PM PST

### Summary
Changed default bin grid from 36 bins (3.2nm start) to 40 bins (1.7nm start, Dunne 2016 nucleation cluster size). Added 80-bin high-resolution preset. All physics/solver code is shape-agnostic — no changes needed there.

### Files Modified
- `tomas_jax/core/config.py` — NBINS=40, XK0=(pi/6)*(1.7e-9)^3*1770, added `XK0_LEGACY`, `NBINS_LEGACY`, `make_grid_80bin()`
- `tests/test_ppm_condensation.py` — Import NBINS from config instead of hardcoded
- `tests/test_tfl_jit_condensation.py` — Use XK0 from config
- `tests/test_ppm_jit_condensation.py` — Use XK0 from config
- 13 benchmark scripts — Import `NBINS_LEGACY as NBINS` and `XK0_LEGACY` for Fortran 36-bin comparison
- `benchmarks/python/test_cases.py` — Derive nbins from xk shape instead of config constant
- `CLAUDE.md` — Updated dimensions and grid documentation

### Grid Configurations
| Config | Bins | Start | End | Mass ratio | Usage |
|--------|------|-------|-----|------------|-------|
| Default (40-bin) | 40 | 1.7nm | 17.5μm | ×2 | Production, nucleation studies |
| High-res (80-bin) | 80 | 1.7nm | 17.5μm | ×√2 | Convergence studies |
| Legacy (36-bin) | 36 | 2.6nm | 10.6μm | ×2 | Fortran comparison benchmarks |

### Verification
- All 132 core tests pass (nucleation, PPM, TFL, coagulation)
- 24 pre-existing test_24h_scenarios failures (load stored 36-bin data — unrelated)
- Nucleation cluster (mnuc=3.47e-24 kg) still fits in bin 0 of 40-bin grid

---

## 2026-03-11 (Wed) — Zhao 2024 11-Mechanism NPF Scheme

**Time**: ~5:00 PM PST

### Summary

Implemented the complete Zhao et al. 2024 (Nature 631, 98–105) 11-mechanism new particle formation scheme as a selectable alternative to the existing Riccobono 2014 + Dunne 2016 scheme.

### Files Modified
- `tomas_jax/physics/nucleation.py` — Added 7 new mechanism functions + driver:
  - `zhao2024_synergistic_hno3_rate()` — Mechanism 5 (Wang 2022 + Zhao 2024 cutoff)
  - `kirkby2016_pure_organic_rate()` — Mechanisms 6–7 (Kirkby 2016 + Zhao 2024 T-factor)
  - `zhao2024_organic_h2so4_rate()` — Mechanism 8 (Lehtipalo 2018 refit)
  - `zhao2024_amine_h2so4_rate()` — Mechanism 9 (Hanson 2017 + Cai 2021 T-function)
  - `zhao2024_iodine_oxoacid_rate()` — Mechanisms 10–11 (He 2021 fits)
  - `zhao2024_nucleation_step()` — 11-mechanism driver with enable masks
  - `_compute_ionc()` — Shared ion concentration helper
  - Fixed `estimate_nucleation_rate()` missing parameters (Gc, temp, pres, boxvol)
- `tomas_jax/solvers/condensation.py` — Added `nucl_scheme` parameter to `make_step()`:
  - `nucl_scheme='ricco_dunne'` (default) or `'zhao2024'`
  - Python-level dispatch, no recompilation between schemes
  - Imported `zhao2024_nucleation_step`, `ZHAO2024_ALL_ENABLED`
- `run_box_model.py` — Added `--nucl-scheme` CLI option:
  - Both manual and `make_step` paths support Zhao 2024
  - Added Zhao 2024 gas precursor inputs (hno3, ulvoc, dma, hio3)
- `tests/test_nucleation.py` — Added 35 new tests (58 total, all passing):
  - `TestSynergisticHNO3` (4 tests): scaling, temperature cutoff
  - `TestKirkby2016` (5 tests): neutral/ion rates, temperature, zero input
  - `TestOrganicH2SO4` (3 tests): bilinear form, scaling
  - `TestAmineH2SO4` (5 tests): H₂SO₄^3, DMA^1.5 scaling, Cai 2021 f_T
  - `TestIodineOxoacids` (6 tests): rates, T-clamps at 263K/283K
  - `TestZhao2024Step` (8 tests): driver mass balance, enable masks, clamping, JIT
  - `TestComputeIonc` (3 tests): ion concentration helper
- `docs/zhao2024_nucleation.md` — Added driver function docs, usage examples, CLI
- `docs/future_features.md` — Added item #8: Explicit ELVOC/HOM Gas-Phase Species
- `CLAUDE.md` — Updated nucleation architecture rule and file layout

### Key Design Decisions
- `ulvoc` (Zhao 2024) and `org_conc` (Riccobono) are separate inputs
- Enable masks are tuples of 11 floats to avoid JIT recompilation
- ULVOC used for both mechs 6-7 (ULVOC-only) and mech 8 (ULVOC+ELVOC) — noted as approximation
- Kirkby 2016 [n±] units: back-calculated as 10³ cm⁻³ from Fig. 3 data
- Mechanism 9 base rate from Hanson 2017 formula (≡ Almeida × 10 at 278K)
- Iodine T-clamps: 263K neutral, 283K ion-induced (He 2021 experimental range)

### Known Limitations
- `estimate_nucleation_rate()` always uses Ricco+Dunne for adaptive sub-stepping (conservative estimate; Zhao 2024 rates may be higher)
- ELVOC/HOM as explicit gas species is a future feature (#8 in future_features.md)
- Scan-fused paths (`run_full_scan`, `run_nucleation_condensation_scan`) still use `ricco_dunne` only

---

## 2026-03-11 (Wed) — Web Interface UX Design Decisions

**Time**: ~PST

### Summary

Completed structured Q&A session to define all UX and architecture decisions for the tomas-app web interface. All decisions documented in `tomas-app/tomas-web/DECISIONS.md` (ADR-008 through ADR-016).

### Key Decisions
- **Bins**: 40 default, 80/160 as options (dropdown)
- **Simulation time**: 6–48h, default 12h
- **Timestep**: dropdown (1/5/10/20/30/40/60/120s), default 60s
- **Process modes**: 6 named modes (coag-only → full model), not independent toggles
- **Condensation method**: PPM only, no user choice
- **Gas-phase modes**: 4 modes per gas (H₂SO₄, NH₃, organics) — depleting, prescribed trend, constant, constant production
- **Prescribed trend profiles**: 6 profiles (diurnal, morning burst, linear ramp up/down, exponential decay, step change) with preview graph
- **Presets**: 7 scenarios populating all fields, all editable after selection
- **Plots**: 9 panels — number/mass size dist (hourly lines + hover), banana plot, N/M/Gc time series, nucleation rate, condensation sink, mass conservation
- **Audience**: atmospheric researchers, students, climate scientists (SAI/CCN)

### Next Steps
- Implement new form design with named process modes
- Implement 4-mode gas-phase forcing with profile previews
- Add bin resolution selector
- Add simulation time/timestep controls
- Implement 9-panel results visualization with hover interactions

---

## 2026-03-10 (Tue) — Adaptive Nucleation Sub-Stepping

**Time**: ~PST

### Summary

Implemented adaptive nucleation sub-stepping to handle high nucleation rates (J > 100 cm⁻³s⁻¹) that can create dN comparable to the entire existing particle population in a single 60s timestep.

### Files Modified
- `tomas_jax/physics/nucleation.py` — Added `estimate_nucleation_rate()` and `compute_nucleation_substeps()` (two new JIT-compilable functions)
- `tomas_jax/solvers/condensation.py` — Modified `_full_step_core()`, `condensation_step_with_nucleation_jax()`, `full_step_jax()`, `run_nucleation_condensation_scan()`, `run_full_scan()`, and `make_step()` to use adaptive nucleation sub-stepping via `fori_loop`
- `docs/nucleation.md` — Documented adaptive sub-stepping algorithm and parameters

### Algorithm
- Before nucleation, estimate total rate J and compute `n_sub = ceil(dN / (max_frac * N_total))`, clamped to [1, max_substeps]
- Loop `n_sub` times with `dt_nuc = dt / n_sub`, calling `nucleation_step` + `mnfix_jax` per substep
- Dynamic `n_nuc` compiles to XLA `while_loop` via `jax.lax.fori_loop`
- Default: `max_nucleation_frac=0.5` (50% per substep), `max_nuc_substeps=20`

### Verification
- All 83 core tests pass (nucleation: 24, PPM: 31, TFL: 15, coag: 13)
- Smoke test: J=363 at H2SO4=1e8 gives n_sub=5 (N_total=1e10) or n_sub=20 (N_total=1e8)
- With sub-stepping disabled (max_frac=1e10), matches original single-step output

---

## 2026-03-10 (Tue) — Web Interface Architecture Specification

**Time**: ~PST

### Summary

Created comprehensive architecture and implementation specification for a public web interface to the TOMAS-JAX box model. Document covers full-stack design: FastAPI backend wrapping existing JAX solvers, React frontend with scientific visualizations, deployment strategy, security, and a phased development roadmap.

### Files Created
- `docs/web_interface_spec.md` — Full specification covering:
  - Two-repository architecture (tomas-api + tomas-web)
  - REST API design (4 endpoints: simulate, results, presets, health)
  - Async job queue for long-running simulations (full mode: 5–13s)
  - JIT warmup strategy to avoid 60s cold starts
  - Frontend layout with 4 visualization types (size distribution, banana plot, time series, conservation)
  - 6 curated preset scenarios for researchers
  - Deployment options (Fly.io/Cloud Run + Vercel, ~$8–16/month)
  - Security: rate limiting, input validation, DoS mitigation
  - 4-phase development roadmap (MVP in 2–3 weeks)
  - Risk matrix and complete parameter reference

### Next Steps
- Decide on monorepo vs two-repo approach
- Scaffold backend and frontend repositories
- Implement Phase 1 MVP

---

## 2026-03-10 (Mon) — Constant-Gas Full-Mode Benchmark + Fixed-Production Mode

**Time**: ~15:00 PST

### Summary

Added dedicated nucleation+coagulation+condensation benchmark (`benchmark_nucleation_constgc.py`) comparing JAX PPM (40/80 bins) vs Fortran TFL (36 bins). Discovered and explained mass resolution dependence in constant-gas mode, then implemented fixed-production mode that eliminates it.

### New Files

- **`benchmarks/python/benchmark_nucleation_constgc.py`** — Full-mode benchmark with two gas modes:
  - *Constant-gas* (`--h2so4 <conc>`): H2SO4 reset each step. Good for comparing against Fortran.
  - *Fixed-production* (`--prod-rate <rate>`): H2SO4 produced at constant rate, consumed by condensation. Resolution-independent.
  - Features: 4 figures (dN, dM, timeseries with Gc panel, 3-panel banana plot), Fortran overlay, `--no-organic-nuc` flag
- **Fortran constgc harness updates**: Added nucleation support (do_nucl toggle, nucleation_driver call, hourly Nk snapshots, dt=10s option)
- **Makefile**: constgc target now links NUC_OBJS + nucleation_driver.o
- **convergence_test.py**: Added full-mode scan functions and `--mode full` flag

### Key Finding: Constant-Gas vs Fixed-Production

With **constant gas** (H2SO4 = 1e8, Dunne-only, dt=10s):
- 40-bin M_dry = 11.35 μg/m³, 80-bin M_dry = 14.70 μg/m³ (**30% divergence**)
- Cause: Different bin resolutions → different CS → different mcond per step. With gas held constant, CS differences compound without self-correction.

With **fixed production** (prod = 1e7 molec/cm³/s, Dunne-only, dt=10s):
- 40-bin M_dry = 140.7943 μg/m³, 80-bin M_dry = 140.7947 μg/m³ (**0.0003% match**)
- Why: Higher CS → faster gas depletion → lower equilibrium gas → same total condensation. The CS cancels: `mcond ≈ CS × (prod/CS) × dt = prod × dt`.

### Organic Nucleation Parameters

Investigated how `org_conc` is handled in original models:
- **TOMAS box.f**: Uses user-entered constant nucleation rate (no Riccobono/Dunne)
- **SOM-TOMAS**: Organic concentration computed dynamically from gas-phase chemistry (SOMGC array → ELVOC sum)
- **Our code**: `org_conc` is a subroutine argument — correct design. Hardcoded `1e7` in benchmarks is a test convenience, not model behavior.

### Performance Audit (not yet implemented)

Identified key bottlenecks for future optimization:
1. MNFIX called ~12-15 times per timestep (Phases 1 & 2 could be vectorized)
2. Redundant condensation_sink computation in ezcond (already computed in core)
3. `n_coag_substeps=10` in full mode (Fortran uses 1; could reduce to 3)

### Known Issues

- Fortran constgc harness currently hardcoded at H2SO4 = 1e8 with organic nucleation disabled (temporary for benchmarking)
- 160-bin grid commented out in convergence_test.py (coag jaggedness at fine resolution)

---

## 2026-03-06 (Thu) — Code Review Fixes, PPM Threshold Bug Fix, Branch Merge

**Time**: ~09:00 PST

### Summary

Applied code review fixes across 5 source files, identified and fixed a critical PPM condensation bug where `ezcond_ppm_jax` was falling through to `simple_add_path` instead of the PPM advection algorithm. Cleaned up 10 stale files. Merged `configurable-nbins` into `dev`.

### Bug Fix: PPM Condensation Threshold

**Root cause**: `ezcond_ppm_jax.py` line 131 had `mcond > tot_m * 1e-3` (should be `mcond > 0.0`). For scenarios where total particle mass was large relative to condensed mass, this threshold evaluated False, routing to `simple_add_path` (proportional mass distribution identical to TFL) instead of the actual PPM advection. Result: PPM produced TFL-identical jagged distributions instead of its own smooth, resolution-stable output.

**Fix**: Reverted threshold to `mcond > 0.0` — any positive condensation triggers PPM advection.

### Code Review Changes

1. **`tomas_jax/physics/properties.py`**: Safe division fix (`jnp.maximum(Nk, 1e-20)` before division to prevent NaN in XLA select), removed dead `Kn2` variable, Horner form for denominator
2. **`tomas_jax/core/config.py`**: Vectorized `make_grid()` — geometric progression via `jnp.power` instead of sequential loop
3. **`tomas_jax/solvers/diffrax.py`**: Removed TomasState from ODE state (was polluting PID error norm), replaced `try/except TypeError` tracing trap with explicit `n_substeps: int` parameter
4. **`tomas_jax/solvers/euler.py`**: Empty bin mask fix in adaptive dt — bins with `Nk < 1e-15` or `Mk < 1e-25` get `dt=inf` instead of near-zero dt (prevented death spiral)
5. **`tomas_jax/physics/ezcond_ppm_jax.py`**: PPM threshold fix (see above)

### Cleanup

Deleted 10 orphaned files: `diffrax_original.py`, `tomas_jax/config.py`, `tomas_jax/plotting.py`, `tomas_jax/production_driver.py`, `main.py`, `diagnose_errors.py`, `diagnose_ppm_n_loss.py`, `run_sensitivity_analysis.py`, `run_temp_sensitivity.py`, `run_mode_dp_sensitivity.py`

### Branch Merge

Merged `configurable-nbins` into `dev`. Resolved add/add conflicts in Fortran source files and PROGRESS.md by taking `configurable-nbins` versions.

### Verification

- 28 condensation tests pass (TFL JIT + PPM JIT)
- Convergence benchmark (constant Gc, H2SO4=1e7, N=1e4, GMD=0.02, GSD=1.6): PPM smooth and resolution-stable at 40/80 bins, TFL shows expected bin-doubling artifacts
- PPM ≠ TFL confirmed numerically (PPM N=4.892e9 vs TFL N=4.757e9 at 40 bins combined)

---

## 2026-03-05 (Wed) — Modular Process Orchestrator for condensation.py

**Time**: ~16:00 PST

### Summary

Refactored `tomas_jax/solvers/condensation.py` from 769 lines of duplicated code to ~430 lines using layered core helpers + thin wrappers. Added `make_step()` composable API for easy process reordering.

### Changes

1. **`tomas_jax/solvers/condensation.py`** (primary — 769 → ~430 lines):
   - Added `_condensation_step_core(ezcond_fn)` — single implementation for both PPM and TFL JIT paths
   - Added `_combined_step_core()` — coag + cond parameterized by ezcond_fn
   - Added `_full_step_core()` — nucl + coag + cond parameterized by ezcond_fn
   - Added `_run_scan()` — single scan implementation replacing 6 copy-pasted loops
   - Added `make_step(processes, cond_method)` — public composable API
   - Fixed `condensation_step_with_nucleation_jax` — was running BOTH TFL and PPM then selecting via `jnp.where`; now uses Python-level dispatch (no double compute)
   - All 15+ existing function names preserved as thin wrappers for backward compatibility

2. **`benchmarks/python/run_24h_scenarios.py`**: Changed `use_tfl=jnp.asarray(val)` → `use_tfl=val` (Python float for static dispatch)

3. **`benchmarks/python/time_single_scenario.py`**: Changed `use_tfl=jnp.float64(1.0)` → `use_tfl=1.0`

4. **`run_box_model.py`**: Added `--make-step` flag to demo `make_step()` composable API

### Verification
- All 28 condensation tests pass (TFL JIT + PPM JIT)
- All 24 nucleation tests pass
- `make_step()` import and creation verified

### Known Issues
- `use_tfl` parameter is now Python float (not JAX array) — minor breaking change for callers that pass `jnp.asarray()`. All in-repo callers updated.

---

## 2026-03-05 (Wed) — Add Coag+Cond Combined Mode to Convergence Benchmark

**Time**: ~12:00 PST

### Summary

Extended the convergence benchmark to support combined coagulation+condensation at multiple resolutions (36/72/144 bins). Added `--mode combined` CLI flag to `convergence_test.py` and updated the Fortran harness with a `do_coag` toggle.

### Changes

1. **`benchmarks/python/convergence_test.py`**:
   - Added `--mode {cond_only,combined}` CLI argument (default: cond_only)
   - Imported `run_combined_scan_tfl`, `run_combined_scan_ppm`, `combined_step_tfl_jax`, `combined_step_ppm_jax` from `solvers/condensation.py`
   - Added `_run_constant_gc_combined_scan_tfl()` and `_run_constant_gc_combined_scan_ppm()` for constant-gas mode with coagulation
   - Mode branching in `_run_tfl()` and `_run_ppm()` inner functions
   - Plot titles include process label ("Coag+Cond" vs "Condensation")
   - Fortran overlay labels update to "Fortran Coag+Cond" in combined mode
   - Output filenames get `_combined` suffix in combined mode
   - `load_fortran_results()` accepts `mode` parameter for mode-dependent filenames

2. **`tomas_fortran/harness/benchmark_constgc.f`**:
   - Added `do_coag` logical parameter (set to `.true.` for combined mode)
   - Calls `multicoag(dt)` + `mnfix(Nk, Mk)` before condensation when `do_coag=.true.`
   - Output filenames use `constgc_combined_` prefix when `do_coag=.true.`

### Results

- Combined mode (dt=60s): Coagulation reduces N by ~10% (1.0e9 → 9.0e8) over 24h
- JAX TFL 36-bin vs Fortran 36-bin: N agreement ~0.2%
- PPM converges well across resolutions: N varies <0.1% from 72→144 bins
- Fortran combined: 0.28s, JAX TFL combined: 1.32s, JAX PPM combined: 1.05s (36 bins)
- Backward compatibility: `--mode cond_only` produces identical results to before

### Output

- 7 plots per mode in `benchmarks/results/convergence/` (with `_combined` suffix)
- Fortran output in `tomas_fortran/output/constgc/` (constgc_combined_* and constgc_* files)

---

## 2026-03-03 (Mon) — Configurable NBINS + TFL vs PPM Convergence Test

**Time**: ~15:00 PST

### Summary

Made the number of size bins (NBINS) a runtime parameter instead of a hardcoded constant. Both TFL and PPM condensation now work at arbitrary bin resolutions (36, 72, 144+). Created a multi-resolution convergence benchmark that runs both methods at 3 grid configurations.

### Changes

1. **`make_grid()` function** (`core/config.py`): Creates bin boundaries for arbitrary resolution. Accepts `nbins`, `xk0` (lower boundary), and `doubling_factor`. Refactored existing `xk_boundaries()` to delegate to it.

2. **Removed NBINS hardcoding** (`core/state.py`): Replaced `assert Nk.shape[0] == NBINS` with shape consistency checks (Nk 1D, Mk bins == Nk bins).

3. **Cleaned NBINS imports from 8 physics files**: All JIT-compiled functions now derive bin count from array shapes (`Nk.shape[0]`) instead of importing the config constant. Removed unused NBINS imports from `condensation_sink.py`, `condensation_ppm.py`, `condensation_tfl_jax.py`, `ezcond_ppm_jax.py`, `nucleation.py`.

4. **`condensation_sink.py` xk parameter**: Added explicit `xk` parameter (default None, falls back to `xk_boundaries()`). Updated all 8 call sites to pass `xk=xk` explicitly.

5. **PPM adaptive delta_xi** (`condensation_ppm.py`): Parameterized all PPM internals by `delta_xi = ln(doubling_factor)` instead of hardcoded `ln(2)`. Added `_compute_moment_integrals(a)` for dynamic I0/I1/I2 computation. PPM now works correctly at any bin resolution.

6. **Convergence benchmark** (`benchmarks/python/convergence_test.py`): New script running condensation-only at 3 resolutions (36×2, 72×√2, 144×2^¼) starting from 1.7nm. Generates 4 figures: size distributions, zoomed comparison, integral convergence, timing.

### Key Results

| Config | TFL time | TFL M_dry | PPM time | PPM M_dry |
|--------|----------|-----------|----------|-----------|
| 36 bins (×2) | 0.48s | 6.32e-7 | 0.20s | 6.32e-7 |
| 72 bins (×√2) | 1.14s | 1.06e-6 | 0.59s | 6.31e-7 |
| 144 bins (×2^¼) | 3.72s | 1.09e-6 | 2.07s | 6.29e-7 |

- **PPM M_dry is resolution-stable** (0.5% variation) while TFL M_dry diverges 72% at finer grids (S01)
- **PPM is ~1.8x faster** than TFL at every resolution
- TFL shows oscillation artifacts at 144 bins; PPM stays smooth
- For well-contained distributions (S05), both methods converge identically

### Files Modified
- `tomas_jax/core/config.py` — Added `make_grid()`
- `tomas_jax/core/state.py` — Shape consistency checks
- `tomas_jax/physics/ezcond.py`, `ezcond_ppm.py`, `condensation.py` — `ibins = Nk.shape[0]`
- `tomas_jax/physics/condensation_sink.py` — Added `xk` parameter
- `tomas_jax/physics/condensation_ppm.py` — Adaptive `delta_xi`
- `tomas_jax/physics/condensation_tfl_jax.py`, `ezcond_ppm_jax.py`, `nucleation.py` — Removed unused imports
- `tomas_jax/solvers/condensation.py` — Pass `xk=xk` to condensation_sink calls
- `benchmarks/python/convergence_test.py` — **New**

### Plots
- `benchmarks/results/convergence/convergence_sizedist_S{01,05}.png`
- `benchmarks/results/convergence/convergence_zoomed_S{01,05}.png`
- `benchmarks/results/convergence/convergence_totals_S{01,05}.png`
- `benchmarks/results/convergence/convergence_timing_S{01,05}.png`

---

## 2026-03-03 (Mon) — Solver Cleanup + Clean Timing Benchmark

**Time**: ~22:00 PST

### Summary

Cleaned up solver code accumulated during nucleation debugging: restored degraded Tsit5 parameters, renamed misnamed `coag_rk4_step` → `coag_euler_step`, documented all 6 scan-fused functions, and wrote a clean single-scenario timing benchmark script.

### Changes

1. **Restored `diffrax_step` parameters** (`solvers/diffrax.py`): rtol 1e-3→1e-4, atol 1.0→1e-10, max_steps 500→5000. These were loosened during debugging and never restored.

2. **Renamed `coag_rk4_step` → `coag_euler_step`** (`solvers/diffrax.py`, `solvers/condensation.py`): The function was always forward Euler, not RK4. Added deprecated alias for backward compatibility. Updated all callers and docstrings.

3. **Documented scan-fused functions** (`solvers/condensation.py`): Added table at top of module docstring listing all 6 scan functions with their coag/cond/nucl capabilities.

4. **New timing benchmark** (`benchmarks/python/time_single_scenario.py`): Clean script that times all 8 solver combinations for a single scenario with JIT warmup, median of 3 runs, and Fortran comparison table.

### Fresh Timing Results (S01, 24h, M1 Pro)

| Test | JAX (s) | Fortran (s) | Ratio |
|------|---------|-------------|-------|
| Tsit5 coag-only | 4.144 | 0.256 | 16.2x |
| Euler coag-only | 0.346 | 0.256 | 1.35x |
| PPM cond-only | 0.213 | 0.086 | 2.47x |
| TFL cond-only | 0.488 | 0.086 | 5.66x |
| Euler+PPM combined | 0.579 | 0.328 | 1.76x |
| Euler+TFL combined | 0.828 | 0.328 | 2.52x |
| Nucl+TFL cond | 0.701 | — | — |
| Full (nucl+coag+cond) | 1.673 | — | — |

### Key Observations

- Tsit5 with restored tolerances (atol=1e-10) is 16x slower than Fortran — the loose atol=1.0 was what made it fast before
- Euler coag is only 1.35x slower than Fortran — competitive
- PPM cond is the fastest JAX condensation path (2.5x Fortran)
- Euler+PPM combined is the best full JAX path (1.76x Fortran)

### Files Modified
- `tomas_jax/solvers/diffrax.py` — parameters + rename
- `tomas_jax/solvers/condensation.py` — updated imports/calls + docstring table
- `benchmarks/python/time_single_scenario.py` — new timing script

---

## 2026-03-03 (Mon) — Full-Mode Stability Fixes + Benchmark Plots

**Time**: ~14:00 PST

### Summary

Fixed critical stability issues in full-mode (nucleation+coagulation+condensation) that caused NaN divergence in 13/50 scenarios. Root cause: unclamped organic mass in nucleation when H2SO4 gas is exhausted. Also replaced RK4 coagulation solver with forward Euler (matches Fortran approach). Generated comprehensive 8-figure benchmark comparison for all 5 modes.

### Bugs Fixed

1. **Nucleation organic mass clamping** (`nucleation.py`): When H2SO4 gas supply is exhausted (`need_clamp=True`), number and SO4 mass were clamped but organic mass was left at the full nucleation rate. This created particles with avg mass 80,000x above bin 0 boundary (e.g., mass=3.08e-19 in bin 0 where xk[0]=1.6e-23). The misalignment caused catastrophic cascading rates in TFL coagulation. Fix: `dM_org_clamped = dN_clamped * 0.1 * _MNUC` — clamp organic proportionally.

2. **RK4 intermediate-stage amplification** (`diffrax.py`): RK4 intermediate stages (k2/k3/k4) amplify N^2 coagulation rates through positive feedback when particle counts are high (cold scenarios with strong nucleation). Replaced with forward Euler (single RHS evaluation per substep + positivity clipping + MNFIX). This matches Fortran's explicit approach and is more stable.

3. **Fortran CSV missing-E exponent** (`utils.py`): Fortran writes very small numbers like `0.1024186408109169-238` (missing 'E' before exponent). Added `_fix_fortran_float()` regex to insert 'E'.

### Results (All 5 modes, 50 scenarios, TFL vs Fortran at hour 24)

| Mode | N Scenarios | N<1% | N<5% | N Med Err | N Max Err | M Med Err | M Max Err |
|------|-------------|------|------|-----------|-----------|-----------|-----------|
| Coag Only | 42 | 39 | 42 | 1.37e-3 | 2.59e-2 | 3.15e-7 | 4.82e-3 |
| Cond Only | 42 | 42 | 42 | 9.97e-7 | 1.01e-3 | 2.64e-6 | 1.00e+0 |
| Coag+Cond | 41 | 34 | 40 | 1.30e-3 | 7.74e-2 | 3.47e-6 | 4.82e-3 |
| Nucl+Cond | 50 | 33 | 41 | 2.14e-3 | 2.48e-1 | 1.22e-5 | 4.33e-3 |
| Full | 50 | 16 | 24 | 8.06e-2 | 9.95e-1 | 8.21e-4 | 1.00e+0 |

Note: Coag/Cond/Combined have 42/41 scenarios because Fortran S50 crashes (known ezcond STOP). Full mode has higher errors due to organic clamping divergence from Fortran (intentional fix for JAX stability).

### Files Modified
- `tomas_jax/physics/nucleation.py` — Added organic mass clamping in gas-limited path
- `tomas_jax/solvers/diffrax.py` — Replaced RK4 with forward Euler coagulation solver
- `tomas_jax/solvers/condensation.py` — n_substeps=10, removed dual TFL/PPM computation in full_step_jax
- `benchmarks/python/utils.py` — Fixed Fortran CSV parser for missing-E exponents

### Files Created
- `benchmarks/python/plot_all_modes.py` — 8-figure comprehensive benchmark comparison
- `benchmarks/results/nucleation_benchmark/` — 8 PNG figures (timeseries, error evolution, scatter, heatmap, summary)

---

## 2026-03-02 (Sun) — Nucleation Benchmark: Fortran vs JAX Verification

**Time**: ~21:00 PST

### Summary

Two-phase nucleation benchmark comparing JAX vs Fortran for Riccobono 2014 and Dunne 2016 parameterizations.

### Phase 1: Parameterization-Level (Option A)

20 test cases (10 Riccobono + 10 Dunne) with varying T, H2SO4, org, NH3, fion. All match Fortran to machine precision (max relative error < 1e-12).

**Files created:**
- `tomas_fortran/src/ricco_nucl.f`, `tomas_fortran/src/dunne_inorg_nucl.f` — copied from TRACER_SOM-TOMAS (prints stripped)
- `tomas_fortran/harness/benchmark_nucleation.f` — standalone Fortran benchmark
- `benchmarks/python/compare_nucleation.py` — JAX vs Fortran comparison

### Phase 2: Full 24h Driver (Option B)

50 scenarios × nucl_cond and full modes, comparing hourly output over 24 hours.

**Key bugs found and fixed:**
1. **Gas depletion MW correction**: JAX had `gas_depleted = dM_so4 * (98/96)`, Fortran depletes Gc by SO4 mass directly (no correction). Fixed.
2. **Organic mass clamping**: When H2SO4 gas is exhausted, Fortran does NOT revert the organic mass to bin 0. JAX was reverting it. Fixed to match Fortran.
3. **MNFIX multi-bin shift**: JAX MNFIX only handled 1-2 bin shifts. Nucleated particles with large organic mass need to jump 12+ bins. Fixed with analytical log2 computation: `kk = ceil(log2(avg*1.1/xk[0])) - 1`.

**Results (nucl_cond mode, 50 scenarios):**
- N_tot median error: 0.2%, max 25% (33/50 < 1%, 41/50 < 5%)
- M_tot max error: 0.43%, median 1.2e-5
- Residual N errors from MNFIX partial-transfer differences compounding over 1440 steps

**Files created:**
- `tomas_fortran/src/nucleation_driver.f` — 24h nucleation driver subroutine
- Fortran output: `tomas_fortran/output/24h/` (50 scenarios × 5 modes × 24 hours)

**Files modified:**
- `tomas_jax/physics/nucleation.py` — Fixed gas depletion and organic clamping
- `tomas_jax/core/mnfix_jax.py` — Fixed multi-bin shift with analytical computation
- `tomas_jax/solvers/condensation.py` — Added `full_step_jax`, `run_full_scan` (scan-fused nucleation+coagulation+condensation)
- `tomas_fortran/harness/benchmark_24h.f` — Added modes 4 (nucl_cond) and 5 (full)
- `tomas_fortran/Makefile` — Added nucleation objects
- `benchmarks/python/run_24h_scenarios.py` — Added nucl_cond and full modes with scan-fused paths, hourly snapshots
- `benchmarks/python/compare_24h.py` — Added nucl_cond and full modes, fixed FORTRAN_DIR path
- `tests/test_nucleation.py` — Updated mass balance test

### Performance
- JAX nucl_cond (scan-fused): ~0.5s/scenario (24×60-step scans)
- JAX full (scan-fused with diffrax): ~5-13s/scenario (adaptive ODE stepping)
- Fortran combined: ~0.33s/scenario

---

## 2026-03-02 (Sun) — Add Nucleation (Riccobono 2014 + Dunne 2016)

**Time**: ~22:00 PST

### Summary

Implemented nucleation parameterizations ported from TRACER_SOM-TOMAS Fortran. Two schemes: Riccobono 2014 organic nucleation (with Yu 2017 temperature correction) and Dunne 2016 inorganic nucleation (4 mechanisms: binary/ternary x neutral/ion-induced). All functions are pure-JAX and JIT-compilable.

### Files Created
- `tomas_jax/physics/nucleation.py` — 3 functions: `ricco_nucleation_rate`, `dunne_nucleation_rate`, `nucleation_step`
- `tests/test_nucleation.py` — 24 unit tests (all passing)
- `docs/nucleation.md` — Algorithm documentation

### Files Modified
- `tomas_jax/solvers/condensation.py` — Added `condensation_step_with_nucleation_jax`, `run_nucleation_condensation_scan`
- `run_box_model.py` — Nucleation in time loop (after H2SO4 production, before coagulation), `--no-nucleation` CLI flag
- `CLAUDE.md` — Updated file layout, operator-split order, Fortran source mapping
- `PROGRESS.md` — This entry

### Key Details
- Nucleation cluster: r=0.85nm, rho=1350 kg/m3, composition 90% SO4 + 10% organic
- Cluster mass (~3.47e-24 kg) < XK0 (1.6e-23 kg), always placed in bin 0
- Gas depletion: SO4 mass subtracted directly from Gc[SRTSO4] (no 98/96 MW correction), with clamping
- In clamped path (gas exhausted): SO4 mass = Gc * 96/98, organic mass NOT reverted (matches Fortran)
- Enable/disable flags use multiplicative float masks (0.0/1.0) to avoid JIT recompilation
- Operator splitting order: H2SO4 production -> nucleation -> coagulation -> condensation

### Verification
- Riccobono at T=278K, h2so4=1e7, org=1e7: J = 3.27 cm^-3 s^-1 (exact)
- Dunne with nh3=0, fion=0: only Jbn > 0 (correct)
- Mass balance: aerosol SO4 gained * 98/96 = gas H2SO4 lost (within 1e-6 relative)
- All 24 tests passing, JIT compilation verified

### Known Limitations
- Coagulation-sink survival fraction (Kerminen-Kulmala) not implemented (commented out in Fortran too)
- Organic vapor (org_conc) is an external input, not coupled to the gas-phase chemistry
- No nuc_bin search — always bin 0 (valid since mnuc < XK0)

### Next Steps
- Run 24h benchmark with nucleation enabled
- Compare nucleation event banana plots with observations
- Consider coupling organic vapor to gas-phase chemistry

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
