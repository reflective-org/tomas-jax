# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

TOMAS-JAX is a JAX re-implementation of the TOMAS (TwO-Moment Aerosol Sectional) aerosol microphysics model. It implements SO2 chemistry (Sun et al. 2022, JIT-compiled), coagulation (JIT-compiled), condensation (TFL/PPM, JIT-compiled), and nucleation (Riccobono 2014 + Dunne 2016, JIT-compiled) for a 40-bin (1.7nm start), 44-component aerosol size distribution.

## Build & Run

```bash
# Install (uv recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"

# Run box model
python run_box_model.py

# Run tests
python -m pytest tests/

# Quick import check
python -c "from tomas_jax import TomasState, CoagulationSolver"
```

## Key Architecture Rules

- **float64 everywhere.** `config.py` sets `jax_enable_x64 = True` before any JAX import. Never use float32 for aerosol microphysics.
- **config.py is the single source of truth** for dimensions (NBINS=40, ICOMP=44), species indices, and physical constants. Default grid: 40 bins starting at 1.7nm (Dunne 2016 cluster size), mass-doubling, covering 1.7nm–17.5μm. Use `make_grid(nbins, xk0, doubling_factor)` for custom bin grids. `make_grid_80bin()` returns the 80-bin high-resolution grid (same range, √2 mass ratio). Legacy 36-bin grid: `NBINS_LEGACY=36`, `XK0_LEGACY=1.6033e-23`. JIT-compiled functions must derive bin count from array shapes (`Nk.shape[0]`), never from the config NBINS constant.
- **TomasState is a NamedTuple** with fields: Nk, Mk, xk, temp, pres, boxvol, Gc, rh, alpha. Use `.create()` factory for initialization, `.update()` for modification.
- **Coagulation has two solvers:** `diffrax_step` (Tsit5 adaptive ODE, rtol=1e-4, atol=1e-10, for standalone coagulation) and `coag_euler_step` (forward Euler + MNFIX, for scan-fused loops). The forward Euler solver is more stable for high-N scenarios — higher-order methods amplify N^2 coagulation rates. Legacy alias `coag_rk4_step` emits `DeprecationWarning`. The coagulation_rhs returns zero derivatives for Gc, rh, alpha — these are preserved unchanged through the ODE solve. Both solvers support `return_overflow=True` to track mass lost at the top bin boundary.
- **Coagulation top-bin overflow:** `calc_coagulation_rates` returns `(dNdt, dMdt, dM_overflow)` where `dM_overflow` is the mass rate [kg/cell/s] that would enter a hypothetical bin above the grid (the TFL `shift_right` truncation). Top-bin self-coagulation produces particles exceeding the grid boundary — this mass is physically lost. Track it to close mass budgets: `M(0) = M(t) + cumulative_overflow`. Coagulation mass error should remain < 1e-8 relative when accounting for overflow.
- **Condensation has four methods:** `method='tfl'` (default, sequential Fortran-faithful), `method='tfl_jit'` (fully JIT-compiled TFL, Fortran-matching), `method='ppm'` (PPM with numpy wrapper), or `method='ppm_jit'` (fully JIT-compiled PPM). Both TFL_JIT and PPM_JIT are fast paths. PPM_JIT uses analytical mass-weighted fluxes for exact conservation and is ~1.8x faster than TFL_JIT for condensation-only. TFL matches Fortran output exactly. Use `run_condensation_scan_tfl()` or `run_condensation_scan()` for scan-fused time loops.
- **Nucleation is JIT-compiled** with two selectable schemes:
  - `ricco_dunne` (default): Riccobono 2014 (organic) + Dunne 2016 (inorganic, 4 mechanisms). Enable/disable via `enable_organic`/`enable_inorganic` float masks (0.0/1.0).
  - `zhao2024`: Zhao et al. 2024 11-mechanism scheme. Mechanisms 1–4 reuse Dunne 2016; adds synergistic HNO₃ (5), pure-organic Kirkby 2016 (6–7), organic-H₂SO₄ (8), amine-H₂SO₄ (9), iodine oxoacids (10–11). Per-mechanism enable via `enable_masks` tuple of 11 floats. Extra inputs: `hno3`, `ulvoc`, `dma`, `hio3`.
  - Select via `make_step(processes, nucl_scheme='zhao2024')` or `--nucl-scheme zhao2024` CLI flag.
  - Nucleated clusters go to bin 0 (90% SO4, 10% organic). Gas depletion: SO4 mass subtracted directly from Gc (no 98/96 MW correction). Organic mass is clamped proportionally when gas is exhausted (diverges from Fortran, but necessary for JAX coagulation stability).
- **Adaptive nucleation sub-stepping** prevents particle creation surges: `estimate_nucleation_rate()` computes J, `compute_nucleation_substeps()` returns n_sub = ceil(dN/(frac*N_total)), clamped to [1, max_substeps]. Each substep runs nucleation_step + MNFIX. Default: max_frac=0.5, max_substeps=20. Applies to `_full_step_core`, `full_step_jax`, `condensation_step_with_nucleation_jax`, `make_step`, and all scan-fused loops.
- **SO2 chemistry is JIT-compiled** using the Sun et al. (2022) Troe formalism with H2O enhancement. `calc_k1_so2_oh(temp, pres, rh)` returns the rate constant [cm³/molec/s]. `so2_oxidation_step()` applies analytical pseudo-first-order decay: SO2(t+dt) = SO2(t) × exp(-k1 × [OH] × dt). Sulfur is conserved (ΔH2SO4 = ΔSO2 × MW_H2SO4/MW_SO2). OH can be constant or diurnal (proportional to cos(SZA)). SO2 stored in Gc[SRTSO2=43]; N_GAS_SPECIES=44 (was 43). When SO2=0, falls back to existing constant prod_rate path.
- **Dilution is JIT-compiled** first-order relaxation toward background: `C(t+dt) = Cbg + (C - Cbg) * exp(-kdil * dt)`. Applies to Nk, Mk, Gc independently. When Cbg=0, simplifies to exponential decay. Pass `kdil`, `Nk_bg`, `Mk_bg`, `Gc_bg` as kwargs to `make_step`.
- **Operator splitting:** Each timestep runs: (1) SO2 chemistry, (2) nucleation, (3) coagulation (JIT), (4) condensation, (5) dilution independently. Use `make_step(['so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'], cond_method='ppm_jit')` for composable process ordering. `make_step()` validates kwargs (warns on typos) and process order (warns if non-canonical).
- **Condensation orchestrator uses layered cores:** `_condensation_step_core(ezcond_fn)` is the single implementation for PPM/TFL; `_combined_step_core()` adds coag; `_full_step_core()` adds nucl+coag; `_run_scan()` is the single scan loop. All public functions are thin wrappers.
- **MNFIX multi-bin shift:** Uses analytical log2 computation to find target bin for large mass shifts (e.g., nucleated particles jumping 12+ bins). Formula: `kk = ceil(log2(avg*1.1/xk[0])) - 1`.
- **Scan-fused modes:** `run_condensation_scan_tfl()` (cond-only), `run_nucleation_condensation_scan()` (nucl+cond), `run_full_scan()` (nucl+coag+cond). All compile into single XLA programs for zero Python dispatch overhead.
- **GPU deployment:** See `docs/gpu_deployment.md`. Use only `*_jit` methods and `make_step()`. Legacy numpy paths (`method='tfl'`, `method='ppm'`) emit `DeprecationWarning` and are not GPU-compatible. `make_step()` returns a JIT-compiled function by default (`jit=True`). For batch/ensemble runs, use `jax.vmap(step_fn)`.
- **Centralized float64 config:** Only `core/config.py` calls `jax.config.update("jax_enable_x64", True)`. All other library modules reference it via comment. Do not add redundant x64 config calls.
- **dilution_step requires explicit background arrays** (no `None` defaults). Pass `jnp.zeros_like(Nk)` etc. for clean-air dilution. This ensures JIT safety.
- **`tomas_jax/fast/` is the GPU-fast reduced model** (this branch): natively batched over a leading cell axis, aerosol SO4+H2O only (ICOMP=2: SRTSO4=0, SRTH2O=1), gases H2SO4+SO2 only (Gc 2-wide: GH2SO4=0, GSO2=1). Do NOT edit `core/config.py` to ICOMP=2 — species indices are imported module-level across the full model and JAX silently clamps out-of-range indices. The fast package has its own config/state/mnfix/density/water and reuses the shape-agnostic full-model kernels via vmap. Rules: loop trip counts are global batch reductions with static caps (never per-cell traced loops); MNFIX is the vectorized `fast/mnfix.py` (exact-conservation scatter-add, 2 sweeps — never materialize a (C,B,B) one-hot); nucleation is analytic (closed-form gas ODE); gas↔aerosol transfers use S-conserving 98/96 MW conversions. The jitted segment runner in `fast/run.py` is `lru_cache`d at module level — never create a fresh `jax.jit` closure per call (each recompile of the segment program costs ~6-7 s on GPU). `sort_by_coag_cost=True` re-sorts by current coagulation stiffness before every scan segment; `diags["coag_n_sub"]` reports the shared substep count per step. See docs/gpu_fast.md for deviations and calibration, docs/gpu_fast_optimization.md for the H100 optimization log (Phases 1-2: 547→23.1 s at 1M cells × 6 h; defaults: auto Pallas coag kernel on GPU, coag_sub_cap=64 — pass 256 for the lower-error setting). Benchmark: `python -m benchmarks.python.bench_fast_1m --cells 1000 --hours 1 --cpu-smoke`.

## File Layout

```
tomas_jax/
  core/config.py              — Constants, dimensions (NBINS=40, 1.7nm start), species indices, make_grid(), make_grid_80bin()
  core/state.py               — TomasState NamedTuple
  core/mnfix_jax.py           — Mass-number drift correction (vectorized JAX)
  physics/condensation.py     — TFL condensation (dmdt_int + tmcond)
  physics/condensation_ppm.py — PPM condensation (Eulerian advection, JIT)
  physics/ezcond.py           — TFL ezcond driver
  physics/ezcond_ppm.py       — PPM ezcond driver (numpy wrapper)
  physics/ezcond_ppm_jax.py   — Pure-JAX PPM ezcond driver (JIT-compilable)
  physics/condensation_tfl_jax.py — Pure-JAX TFL condensation (tmcond_jax + ezcond_tfl_jax, JIT)
  physics/gas_properties.py   — Gas diffusivity, MFP, Fuchs-Sutugin correction
  physics/condensation_sink.py — Condensation sink CS [s^-1] and per-bin fractions
  physics/nh3_equilibrium.py  — NH3/NH4 stoichiometric equilibrium
  physics/water_equilibrium.py — Hygroscopic water uptake (ISORROPIA fits)
  physics/bhmie.py            — Bohren-Huffman Mie scattering (numpy, precomputation)
  physics/radiative_forcing.py — Direct SW radiative forcing (Chylek & Wong 1995, Pierce et al. 2010) + Tabazadeh (1997) H₂SO₄/H₂O equilibrium
  physics/nucleation.py       — Nucleation: ricco_dunne (Riccobono+Dunne) + zhao2024 (11-mechanism) schemes (JIT-compilable)
  physics/so2_chemistry.py    — SO2+OH chemistry: Sun et al. (2022) Troe formalism, SZA, diurnal OH (JIT-compilable)
  physics/dilution.py         — Dilution/entrainment: first-order relaxation toward background (JIT-compilable)
  solvers/diffrax.py          — Coagulation solvers: Tsit5 adaptive (diffrax_step), forward Euler (coag_euler_step)
  solvers/condensation.py     — Process orchestrator: core helpers + thin wrappers + make_step() composable API + scan-fused loops
  fast/config.py              — GPU-fast reduced model: dims (ICOMP=2), compact species/gas indices
  fast/state.py               — FastState: batched NamedTuple (leading cell axis C)
  fast/mnfix.py               — Vectorized MNFIX (O(1) depth, exact-conservation scatter)
  fast/density.py             — Binary H2SO4/H2O Tang density
  fast/water.py               — Tabazadeh 1997 water equilibrium (pure JAX)
  fast/nucleation.py          — Dunne Jbn neutral-binary, analytic gas-ODE integration
  fast/coagulation.py         — Adaptive-capped Euler coagulation (loss-frequency criterion)
  fast/coagulation_pallas.py  — Opt-in fused Triton kernel for the coag substep loop (GPU, ~3x)
  fast/condensation.py        — Batched PPM driver, global-max capped CFL substeps
  fast/step.py                — Process composition (chem→nucl→mnfix→water→coag→cond→water→mnfix)
  fast/run.py                 — jit(scan) driver: donation, cell chunking, stiffness sorting

benchmarks/
  fortran/benchmark_24h.f     — Fortran 24h benchmark harness
  python/scenarios.py         — LHC scenario generator (50 scenarios)
  python/run_24h_scenarios.py — JAX 24h runner (3 modes × 3 methods)
  python/run_ppm_analytical_benchmark.py — PPM analytical flux benchmark (49 scenarios × 5 modes)
  python/compare_24h.py       — 3-way comparison (Fortran vs TFL vs PPM, 5 modes)
  python/compare_nucleation.py — Parameterization-level nucleation comparison
  python/plot_24h_summary.py  — 8 summary plots (coag/cond/combined)
  python/plot_all_modes.py    — 8 benchmark plots for all 5 modes (incl. nucleation)
  python/plot_24h_timing.py   — Timing comparison plots
  python/time_single_scenario.py — Clean single-scenario timing benchmark (all solver combos vs Fortran)
  python/convergence_test.py  — Multi-resolution TFL vs PPM convergence (40/80 bins)
  python/benchmark_nucleation_constgc.py — Nucleation full-mode benchmark (constant-gas + fixed-production)
  python/validate_so2_chemistry.py — SO2 chemistry validation (7 figures vs Sun et al. 2022, incl. stratospheric lifetime heatmaps)
  python/benchmark_so2_sensitivity.py — SO2 sensitivity benchmark (5 SO2 × 4 modes × 3 altitudes × 2 grids, 48h, 12 figures)
  python/benchmark_dilution.py — Dilution benchmark (3 cases × 24h, all processes + SO2 + dilution, 6 figures)
  python/bench_fast_1m.py     — GPU-fast reduced model benchmark (1M cells × 6h target <10s; --cpu-smoke mode)
  python/bench_coag_pallas.py — XLA vs Pallas coagulation substep timing (stiff sorted chunk)

tomas_fortran/
  src/                        — 14 core TOMAS Fortran source files (TFL condensation)
  src_ppm/tmcond_ppm.f        — PPM condensation (8 subroutines, ported from JAX)
  src_ppm/ezcond_ppm.f        — PPM-aware ezcond driver
  include/sizecode.COM        — Common block definitions
  harness/benchmark_24h.f     — 24h TFL benchmark driver
  harness/benchmark_24h_ppm.f — 24h PPM benchmark driver
  harness/benchmark_harness.f     — Single-scenario benchmark driver
  harness/benchmark_nucleation.f  — Nucleation parameterization benchmark
  harness/benchmark_constgc.f     — Constant-gas full-mode benchmark (nucl+coag+cond)
  src/ricco_nucl.f                — Riccobono 2014 nucleation (from TRACER_SOM-TOMAS)
  src/dunne_inorg_nucl.f          — Dunne 2016 inorganic nucleation
  src/nucleation_driver.f         — Nucleation driver for 24h harness
  Makefile                        — Self-contained build (no external paths)
  README.md                   — Build instructions and file descriptions

docs/
  gpu_deployment.md           — GPU deployment guide (float64, JAX_PLATFORMS, vmap, code paths)
  architecture.md             — Condensation pipeline architecture
  ppm_condensation.md         — PPM algorithm documentation
  24h_benchmark.md            — 24h benchmark suite documentation
  nucleation.md               — Nucleation algorithm documentation (ricco_dunne scheme)
  zhao2024_nucleation.md      — Zhao 2024 11-mechanism NPF scheme documentation
  so2_chemistry.md            — SO2+OH chemistry (Sun et al. 2022), validation, usage
  dilution.md                 — Dilution/entrainment algorithm, parameters, usage
  gpu_fast.md                 — GPU-fast reduced model: architecture, physics deviations, calibration, usage
  gpu_fast_optimization.md    — H100 optimization log: measured root causes, phase checklist, results, lessons
  missing_physics.md          — Gap analysis of unimplemented physics
  future_features.md          — Planned improvements: AD, GPU, vmap, multi-species, surrogates
```

## Species Indices (0-based)

- 0 = SO4, 1-41 = Organics, 42 = NH4, 43 = H2O
- Gas array Gc has 44 elements (N_GAS_SPECIES=44): indices 0-42 (aerosol species except water) + index 43 (SO2, SRTSO2=43)

## Fortran Source Mapping

The condensation modules are direct ports of TOMAS Fortran:
- `condensation.py` (dmdt_int, tmcond) ← dmdt_int.f, tmcond.f
- `condensation_ppm.py` → tmcond_ppm.f (PPM advection, Fortran port in src_ppm/)
- `ezcond.py` ← ezcond.f
- `ezcond_ppm.py` — PPM driver (same interface as ezcond, uses PPM internally)
- `ezcond_ppm_jax.py` — Pure-JAX PPM driver (JIT-compilable version of ezcond_ppm.py)
- `nh3_equilibrium.py` ← eznh3eqm.f
- `water_equilibrium.py` ← waterso4.f, waternacl.f, ezwatereqm.f
- `gas_properties.py` ← gasdiff.f, getCondSink.f
- `condensation_sink.py` ← getCondSink.f
- `nucleation.py` (ricco_nucleation_rate) ← ricco_nucl.f (TRACER_SOM-TOMAS)
- `nucleation.py` (dunne_nucleation_rate) ← dunne_inorg_nucl.f (TRACER_SOM-TOMAS)
- `nucleation.py` (nucleation_step) ← nucleation.f (TRACER_SOM-TOMAS)

Original Fortran source is in `tomas_fortran/src/` (self-contained copy).

## Common Patterns

- Units: Nk in [#/grid cell], Mk in [kg/grid cell], xk in [kg], boxvol in [cm^3]
- Converting molec/cm3 to kg/grid cell: `conc * boxvol * (MW/1000) / AVOGADRO`
- MNFIX should be called after any operation that changes mass or number
- Empty bins get tiny values (NEPS=1e-20, TINY_N=1e-10) to prevent NaN

## Testing

When modifying condensation code, verify:
1. `dmdt_int`: grows with TAU>0, shrinks with TAU<0, unchanged at TAU=0
2. H2SO4 diffusivity at 298K should be ~1.15e-5 m^2/s
3. Mass conservation: aerosol gained = gas lost (check after ezcond correction)
4. PPM ≠ TFL: PPM must produce different (smoother) distributions than TFL. If PPM=TFL, check `ezcond_ppm_jax.py` threshold (`mcond > 0.0` triggers PPM; `mcond > tot_m * 1e-3` is WRONG — causes fallthrough to simple_add_path)
5. Convergence benchmark: `python -m benchmarks.python.convergence_test --constant-gc --h2so4 1e7 --n-total 1e4 --gmd 0.02 --gsd 1.6 --temp 298 --pres 101325 --mode cond_only` (PPM should be smooth and resolution-stable at 40/80 bins)
6. Coagulation mass budget: `M(0) = M(24h) + overflow` should close to < 1e-8 relative (use `return_overflow=True`)

## Benchmark & Testing Rules

- **Never skip or hide failures.** If a model run crashes, the benchmark must fail loudly — never produce empty/placeholder figures or silently skip scenarios. A crash means there is a bug that must be diagnosed and fixed before proceeding.
- **Always run the full benchmark end-to-end** before declaring success. Partial test runs (e.g., single grid resolution, single altitude) do not validate the full parameter space.
- **Plot functions must error when data is missing**, not silently produce empty axes. Use explicit checks and raise informative errors (e.g., `raise FileNotFoundError(f"Missing NPZ: {fname}. Run simulations first.")`).

## Benchmark & Testing Rules

- **Never skip or hide failures.** If a model run crashes, the benchmark must fail loudly — never produce empty/placeholder figures or silently skip scenarios. A crash means there is a bug that must be diagnosed and fixed before proceeding.
- **Always run the full benchmark end-to-end** before declaring success. Partial test runs (e.g., single grid resolution, single altitude) do not validate the full parameter space.
- **Plot functions must error when data is missing**, not silently produce empty axes. Use explicit checks and raise informative errors (e.g., `raise FileNotFoundError(f"Missing NPZ: {fname}. Run simulations first.")`).

## Documentation Requirements

**Always document changes in the repository.** This is mandatory, not optional.

1. **Progress Report (`PROGRESS.md`)**: After completing any significant work, add a dated entry to `PROGRESS.md` with:
   - Date and time (PST timezone)
   - Summary of what was changed (files modified/created, bugs fixed, features added)
   - Known issues or limitations discovered
   - Next steps identified

2. **Technical Docs (`docs/`)**: When adding or modifying a feature:
   - Update the relevant doc in `docs/` (architecture.md, ppm_condensation.md, 24h_benchmark.md)
   - If the feature is new and doesn't fit existing docs, create a new `.md` file in `docs/`
   - Include: algorithm description, file locations, usage examples, known limitations

3. **CLAUDE.md**: Keep this file up to date when:
   - New modules are added to the file layout
   - New architecture rules are established
   - New testing requirements are identified

4. **README.md**: Update when user-facing interfaces change (new CLI options, new scripts, new dependencies)

5. **Inline comments**: Add comments only where logic is non-obvious. Do not add docstrings to code you didn't change.
