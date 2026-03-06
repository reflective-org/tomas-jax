# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

TOMAS-JAX is a JAX re-implementation of the TOMAS (TwO-Moment Aerosol Sectional) aerosol microphysics model. It implements coagulation (JIT-compiled), condensation (TFL/PPM, JIT-compiled), and nucleation (Riccobono 2014 + Dunne 2016, JIT-compiled) for a 36-bin, 44-component aerosol size distribution.

## Build & Run

```bash
# Install
pip install -e .

# Run box model
python run_box_model.py

# Run tests
python -m pytest tests/

# Quick import check
python -c "from tomas_jax import TomasState, CoagulationSolver"
```

## Key Architecture Rules

- **float64 everywhere.** `config.py` sets `jax_enable_x64 = True` before any JAX import. Never use float32 for aerosol microphysics.
- **config.py is the single source of truth** for dimensions (NBINS=36, ICOMP=44), species indices, and physical constants. Use `make_grid(nbins, xk0, doubling_factor)` for custom bin grids. JIT-compiled functions must derive bin count from array shapes (`Nk.shape[0]`), never from the config NBINS constant.
- **TomasState is a NamedTuple** with fields: Nk, Mk, xk, temp, pres, boxvol, Gc, rh, alpha. Use `.create()` factory for initialization, `.update()` for modification.
- **Coagulation has two solvers:** `diffrax_step` (Tsit5 adaptive ODE, rtol=1e-4, atol=1e-10, for standalone coagulation) and `coag_euler_step` (forward Euler + MNFIX, for scan-fused loops). The forward Euler solver is more stable for high-N scenarios — higher-order methods amplify N^2 coagulation rates. Legacy alias `coag_rk4_step` still works. The coagulation_rhs returns zero derivatives for Gc, rh, alpha — these are preserved unchanged through the ODE solve.
- **Condensation has four methods:** `method='tfl'` (default, sequential Fortran-faithful), `method='tfl_jit'` (fully JIT-compiled TFL, Fortran-matching), `method='ppm'` (PPM with numpy wrapper), or `method='ppm_jit'` (fully JIT-compiled PPM). Both TFL_JIT and PPM_JIT are fast paths. PPM_JIT uses analytical mass-weighted fluxes for exact conservation and is ~1.8x faster than TFL_JIT for condensation-only. TFL matches Fortran output exactly. Use `run_condensation_scan_tfl()` or `run_condensation_scan()` for scan-fused time loops.
- **Nucleation is JIT-compiled** with two parameterizations: Riccobono 2014 (organic, `ricco_nucleation_rate`) and Dunne 2016 (inorganic, 4 mechanisms, `dunne_nucleation_rate`). Enable/disable via float masks (0.0/1.0) to avoid recompilation. Nucleated clusters go to bin 0 (90% SO4, 10% organic). Gas depletion: SO4 mass subtracted directly from Gc (no 98/96 MW correction). Organic mass is clamped proportionally when gas is exhausted (diverges from Fortran, but necessary for JAX coagulation stability).
- **Operator splitting:** Each timestep runs: (1) H2SO4 production, (2) nucleation, (3) coagulation (JIT), (4) condensation independently. Use `make_step(['nucleation', 'coagulation', 'condensation'], cond_method='ppm_jit')` for composable process ordering.
- **Condensation orchestrator uses layered cores:** `_condensation_step_core(ezcond_fn)` is the single implementation for PPM/TFL; `_combined_step_core()` adds coag; `_full_step_core()` adds nucl+coag; `_run_scan()` is the single scan loop. All public functions are thin wrappers.
- **MNFIX multi-bin shift:** Uses analytical log2 computation to find target bin for large mass shifts (e.g., nucleated particles jumping 12+ bins). Formula: `kk = ceil(log2(avg*1.1/xk[0])) - 1`.
- **Scan-fused modes:** `run_condensation_scan_tfl()` (cond-only), `run_nucleation_condensation_scan()` (nucl+cond), `run_full_scan()` (nucl+coag+cond). All compile into single XLA programs for zero Python dispatch overhead.

## File Layout

```
tomas_jax/
  core/config.py              — Constants, dimensions, species indices
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
  physics/nucleation.py       — Nucleation: Riccobono 2014 + Dunne 2016 (JIT-compilable)
  solvers/diffrax.py          — Coagulation solvers: Tsit5 adaptive (diffrax_step), forward Euler (coag_euler_step)
  solvers/condensation.py     — Process orchestrator: core helpers + thin wrappers + make_step() composable API + scan-fused loops

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
  python/convergence_test.py  — Multi-resolution TFL vs PPM convergence (36/72/144 bins)

tomas_fortran/
  src/                        — 14 core TOMAS Fortran source files (TFL condensation)
  src_ppm/tmcond_ppm.f        — PPM condensation (8 subroutines, ported from JAX)
  src_ppm/ezcond_ppm.f        — PPM-aware ezcond driver
  include/sizecode.COM        — Common block definitions
  harness/benchmark_24h.f     — 24h TFL benchmark driver
  harness/benchmark_24h_ppm.f — 24h PPM benchmark driver
  harness/benchmark_harness.f     — Single-scenario benchmark driver
  harness/benchmark_nucleation.f  — Nucleation parameterization benchmark
  src/ricco_nucl.f                — Riccobono 2014 nucleation (from TRACER_SOM-TOMAS)
  src/dunne_inorg_nucl.f          — Dunne 2016 inorganic nucleation
  src/nucleation_driver.f         — Nucleation driver for 24h harness
  Makefile                        — Self-contained build (no external paths)
  README.md                   — Build instructions and file descriptions

docs/
  architecture.md             — Condensation pipeline architecture
  ppm_condensation.md         — PPM algorithm documentation
  24h_benchmark.md            — 24h benchmark suite documentation
  nucleation.md               — Nucleation algorithm documentation
  future_features.md          — Planned improvements: AD, GPU, vmap, multi-species, surrogates
```

## Species Indices (0-based)

- 0 = SO4, 1-41 = Organics, 42 = NH4, 43 = H2O
- Gas array Gc has 43 elements (all species except water)

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
4. Coagulation mass error should remain < 1e-13 relative

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
