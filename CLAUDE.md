# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

TOMAS-JAX is a JAX re-implementation of the TOMAS (TwO-Moment Aerosol Sectional) aerosol microphysics model. It implements coagulation (JIT-compiled) and condensation (sequential Fortran-faithful port) for a 36-bin, 44-component aerosol size distribution.

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
- **config.py is the single source of truth** for dimensions (NBINS=36, ICOMP=44), species indices, and physical constants. Never hardcode these elsewhere.
- **TomasState is a NamedTuple** with fields: Nk, Mk, xk, temp, pres, boxvol, Gc, rh, alpha. Use `.create()` factory for initialization, `.update()` for modification.
- **Coagulation is JIT-compiled** via diffrax (Tsit5 solver). The coagulation_rhs returns zero derivatives for Gc, rh, alpha — these are preserved unchanged through the ODE solve.
- **Condensation has four methods:** `method='tfl'` (default, sequential Fortran-faithful), `method='tfl_jit'` (fully JIT-compiled TFL, Fortran-matching), `method='ppm'` (PPM with numpy wrapper), or `method='ppm_jit'` (fully JIT-compiled PPM). TFL_JIT is the recommended fast path — matches Fortran output exactly while being 43x faster than sequential TFL. Use `run_condensation_scan_tfl()` for scan-fused time loops. Note: PPM produces overly narrow distributions; TFL matches Fortran.
- **Operator splitting:** Each timestep runs coagulation (JIT) then condensation independently.

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
  solvers/diffrax.py          — Coagulation ODE solver (Tsit5 + MNFIX)
  solvers/condensation.py     — Condensation driver (method='tfl'|'tfl_jit'|'ppm'|'ppm_jit') + scan-fused loops

benchmarks/
  fortran/benchmark_24h.f     — Fortran 24h benchmark harness
  python/scenarios.py         — LHC scenario generator (50 scenarios)
  python/run_24h_scenarios.py — JAX 24h runner (3 modes × 3 methods)
  python/compare_24h.py       — 3-way comparison (Fortran vs TFL vs PPM)
  python/plot_24h_summary.py  — 8 summary plots
  python/plot_24h_timing.py   — Timing comparison plots

docs/
  architecture.md             — Condensation pipeline architecture
  ppm_condensation.md         — PPM algorithm documentation
  24h_benchmark.md            — 24h benchmark suite documentation
```

## Species Indices (0-based)

- 0 = SO4, 1-41 = Organics, 42 = NH4, 43 = H2O
- Gas array Gc has 43 elements (all species except water)

## Fortran Source Mapping

The condensation modules are direct ports of TOMAS Fortran:
- `condensation.py` (dmdt_int, tmcond) ← dmdt_int.f, tmcond.f
- `condensation_ppm.py` — PPM advection (new, no Fortran equivalent)
- `ezcond.py` ← ezcond.f
- `ezcond_ppm.py` — PPM driver (same interface as ezcond, uses PPM internally)
- `ezcond_ppm_jax.py` — Pure-JAX PPM driver (JIT-compilable version of ezcond_ppm.py)
- `nh3_equilibrium.py` ← eznh3eqm.f
- `water_equilibrium.py` ← waterso4.f, waternacl.f, ezwatereqm.f
- `gas_properties.py` ← gasdiff.f, getCondSink.f
- `condensation_sink.py` ← getCondSink.f

Original Fortran lives at: `../../original-models/TOMAS/backup/src/`

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
