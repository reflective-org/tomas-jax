# Condensation Pipeline Architecture

This document describes the condensation/evaporation implementation in TOMAS-JAX, including data flow, Fortran mapping, and design decisions.

## Overview

Condensation is implemented as an **operator-split** step that runs after the JIT-compiled coagulation solver. Two bin redistribution methods are available:

- **TFL** (Phase 1, default): Sequential Python loops (numpy) matching the original Fortran line-by-line
- **PPM** (Phase 2): JIT-compatible Eulerian advection in log-mass space via `jax.lax.fori_loop`

```
                    Model Timestep (e.g., 60 s)
                    ===========================

 H2SO4 Source ──> Coagulation (JIT) ──> Condensation ──> Output
                       |                     |
                   diffrax.py          condensation.py (solver)
                   Tsit5 + MNFIX          method='tfl' | method='ppm'
                                    ┌──────────┼──────────────┐
                                    |          |              |
                              H2SO4 cond   NH3 eqm     Water eqm
                                    |      (eznh3eqm)  (ezwatereqm)
                               ┌────┴────┐
                               |         |
                          method='tfl' method='ppm'
                               |         |
                          ezcond()   ezcond_ppm()
                               |         |
                          tmcond()   ppm_condensation_step()
                          (TFL alg.) (PPM advection, JIT)
```

## Data Flow

### Entry Point: `condensation_step()`

**File:** `tomas_jax/solvers/condensation.py`

```
Input:  Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
Output: Nk_new, Mk_new, Gc_new
```

**Sequence:**

1. **Condensation Sink** (`calc_condensation_sink`)
   - Computes CS [s^-1] = first-order gas loss rate
   - Returns per-bin sink fractions for mass distribution

2. **Gas Depletion**
   - `mcond = Gc[SO4] * (1 - exp(-CS * dt))` — exponential decay
   - `Gc[SO4] -= mcond`

3. **ezcond** — Distributes `mcond` across bins
   - MNFIX on input
   - Adaptive sub-stepping (limits mass increase to ~10%)
   - Calls `tmcond` for TFL redistribution or simple mass addition
   - Post-hoc mass conservation correction

4. **NH3 Equilibrium** (`eznh3eqm`)
   - Stoichiometric NH4:SO4 = 2:1 partitioning

5. **Water Equilibrium** (`calc_equilibrium_water`)
   - Polynomial fits to ISORROPIA results

6. **MNFIX** — Final mass-number consistency check

### Internal Call Graph

```
condensation_step(method='tfl')          condensation_step(method='ppm')
├── calc_condensation_sink()             ├── calc_condensation_sink()
│   ├── calc_gas_diffusivity()           │   └── (same as TFL)
│   ├── calc_particle_properties()       │
│   ├── calc_knudsen_number()            ├── ezcond_ppm()              [ezcond_ppm.py]
│   └── calc_fuchs_sutugin_correction()  │   ├── mnfix_jax()
│                                        │   ├── calc_condensation_sink()
├── ezcond()                             │   └── ppm_condensation_step() [condensation_ppm.py]
│   ├── mnfix_jax()                      │       ├── compute_wr()
│   ├── calc_condensation_sink()         │       ├── compute_substeps()
│   └── tmcond()         [condensation]  │       ├── compute_edge_velocity()
│       └── dmdt_int()   [condensation]  │       │   └── dmdt_int()
│                                        │       ├── ppm_reconstruct()
├── eznh3eqm()    [nh3_equilibrium]      │       ├── ppm_flux()
├── calc_equilibrium_water()             │       ├── advect_totals()
└── mnfix_jax()   [mnfix_jax]            │       ├── passive_flux()
                                         │       ├── dry_mass_from_ppm_number()
                                         │       └── closure_update()
                                         │
                                         ├── eznh3eqm()
                                         ├── calc_equilibrium_water()
                                         └── mnfix_jax()
```

## Module Details

### `gas_properties.py` — Gas-Phase Transport Properties

**Fortran source:** `gasdiff.f`, `getCondSink.f` (lines 77-84)

| Function | Formula | Reference |
|---|---|---|
| `calc_gas_diffusivity` | Fuller-Schettler-Giddings: Di = 1e-7 * T^1.75 * mwf / (P/bar) / Svf | gasdiff.f L39-41 |
| `calc_mean_molecular_speed` | ms = sqrt(8*R*T / (pi*MW_kg)) | S&P eqn 9.2 |
| `calc_mean_free_path` | mfp = 2*Di/ms (TOMAS-specific, NOT kinetic theory) | getCondSink.f L84 |
| `calc_knudsen_number` | Kn = 2*mfp/Dp | getCondSink.f L111 |
| `calc_fuchs_sutugin_correction` | beta = (1+Kn)/(1+2*Kn*(1+Kn)/alpha) | getCondSink.f L112 |

**Key note:** The mean free path formula `mfp = 2*Di/ms` is specific to TOMAS. It differs from the standard kinetic theory formula. This choice is deliberate for consistency with the Dahneke/Fuchs-Sutugin correction.

### `condensation_sink.py` — First-Order Loss Rate

**Fortran source:** `getCondSink.f` (lines 77-135)

```
CS = 2 * pi * Di * sum(Dpk * Nk * beta) / (boxvol * 1e-6)
sinkfrac[k] = (Dpk[k] * Nk[k] * beta[k]) / sum(Dpk * Nk * beta)
```

The condensation sink `CS` has units [s^-1] and represents the pseudo-first-order rate constant for gas-phase loss to aerosol. Larger particles and higher number concentrations increase CS.

`sinkfrac` distributes the condensing mass proportionally across bins based on each bin's contribution to the total sink.

### `condensation.py` — TFL Condensation Algorithm

**Fortran source:** `tmcond.f` (lines 52-366), `dmdt_int.f`

This is the core physics. The TFL (Tzivion-Feingold-Levin) algorithm is a semi-Lagrangian scheme:

#### `dmdt_int(M0, TAU, WR)` — Analytic Growth

Stevens et al. 1996, Eq. A3:

```
MH2O = (WR - 1) * M0
X = ((M0 + MH2O)^(2/3))
X = sqrt(max(0, (2/3)*TAU + X))
result = X^3 / WR
```

- `TAU > 0`: condensation (particle grows)
- `TAU < 0`: evaporation (particle shrinks)
- `TAU = 0`: no change
- `WR` keeps the wet/dry ratio constant during growth

#### `tmcond(TAU, xk, Mkd, Nkd, cspecies, moxd)` — Bin Redistribution

Algorithm per source bin L:

1. **Preprocessing** (lines 82-110)
   - Empty bins get tiny SO4 mass to prevent division by zero
   - Overflow bins (avg mass > upper boundary) transfer 10% to neighbor

2. **Top-hat construction** (lines 179-200)
   - Average mass: `AVG = AMKDRY / ANKD`
   - Shape factor: `XI = 0.5 + XX*(1.5 - XX)` where `XX = xk[L]/AVG`
   - Width: blend of shape-based `W1` and boundary-based `W2` (50/50)
   - Edges: `XU = AVG + WTH/2`, `XL = AVG - WTH/2`

3. **Translation** (lines 219-236)
   - `YU = dmdt_int(XU, TAU, WR)` — translate upper edge
   - `YL = dmdt_int(XL, TAU, WR)` — translate lower edge
   - `c1/c2` corrections (disabled for SO4 with `moxd=0`)
   - Condensing species mass at edges: `YUC`, `YLC`

4. **Below-grid handling** (lines 243-274)
   - If `YL < xk[0]`: keep particles in original bin

5. **Remapping** (lines 279-360)
   - Search for target bins where translated top-hat overlaps
   - Number proportional to overlap fraction
   - Non-condensing species: proportional remapping
   - Condensing species: trapezoidal interpolation using edge masses

### `ezcond.py` — Simple Condensation Driver

**Fortran source:** `ezcond.f` (Jeff Pierce, May 2007)

Three condensation paths based on mass ratio:

| Condition | Path | Method |
|---|---|---|
| `mconds > tot_m * 1e-3` | Full TFL | TAU calculation + `tmcond()` |
| `mconds > tot_s * 1e-12` | Simple add | Distribute mass by `sinkfrac`, MNFIX |
| Otherwise | Skip | No condensation |

**Adaptive sub-stepping:** The number of sub-steps is chosen so that no bin's mass increases by more than ~10% per step. This prevents numerical instability in the TFL remapping.

**Mass conservation correction:** After all sub-steps, the actual mass change is compared to `mcond`. A correction factor `ratio = (tot_f - tot_i) / mcond` is applied to adjust each bin's species mass.

### `nh3_equilibrium.py` — Ammonium-Sulfate Equilibrium

**Fortran source:** `eznh3eqm.f` (Jeff Pierce, April 2007)

Simple stoichiometric equilibrium:

- Total NH3 (kmoles) = `Gc[NH4]/17 + sum(Mk[:,NH4]/18)`
- Total SO4 (kmoles) = `sum(Mk[:,SO4]/96)`

**Case 1:** `tot_nh3/2 < tot_so4` (ammonia-limited)
- No gas-phase NH3
- Particle NH4 distributed proportionally to sulfate

**Case 2:** `tot_nh3/2 >= tot_so4` (excess ammonia)
- Particles filled to 2:1 NH4:SO4 stoichiometry
- Excess NH3 stays in gas phase

### `water_equilibrium.py` — Hygroscopic Water Uptake

**Fortran source:** `waterso4.f`, `waternacl.f`, `ezwatereqm.f`

Piecewise polynomial fits to ISORROPIA thermodynamic model results at 273 K:

- **Ammonium bisulfate** (`water_uptake_sulfate`): 6 RH ranges, quadratic/cubic polynomials
- **Sea salt** (`water_uptake_seasalt`): 5 RH ranges, cubic polynomials

Water mass per bin: `water = (so4_mass * 1.2 + org_mass) * (wr_so4 - 1)`

The factor 1.2 converts SO4 mass to NH4HSO4 mass. Organic aerosol is assumed to have the same water uptake as sulfate. Sea salt is currently disabled.

## Design Decisions

### Phase 1: Sequential Loops

The condensation modules (`condensation.py`, `ezcond.py`) use numpy arrays and Python for-loops. This matches the Fortran structure exactly, making verification against the original code straightforward. The tradeoff is that these functions are **not JIT-compatible** and run on CPU only.

### Operator Splitting

Coagulation and condensation are decoupled into separate operator-split steps. This allows:
- Coagulation to use the JIT-compiled diffrax solver (fast)
- Condensation to use the sequential Fortran-faithful implementation (correct)
- Independent verification of each process

### JAX/numpy Boundary

The condensation driver (`solvers/condensation.py`) bridges JAX and numpy:
- Receives JAX arrays from the main loop
- Converts to numpy for sequential condensation
- Converts back to JAX for equilibrium calculations and MNFIX
- Returns JAX arrays

This boundary crossing adds overhead but keeps the code clean.

### c1/c2 Corrections Disabled for SO4

In `ezcond.py`, tmcond is called with `moxd=zeros` for sulfate condensation. The c1/c2 corrections (added by Win Trivitayanurak, 5/25/06) are only needed for multi-component condensation where the condensing species mass flux must match a prescribed value. For pure SO4 condensation, no correction is needed.

## Phase 2: PPM Condensation

The PPM (Piecewise Parabolic Method) is now implemented as an alternative to TFL:

- **File:** `tomas_jax/physics/condensation_ppm.py` — PPM algorithm (JIT-compatible)
- **File:** `tomas_jax/physics/ezcond_ppm.py` — PPM-aware ezcond driver
- **Usage:** Pass `method='ppm'` to `condensation_step()` or `--method ppm` to `run_box_model.py`
- **Docs:** See `docs/ppm_condensation.md` for full algorithm description

Key differences from TFL:
- 3rd-order accuracy (vs 1st-order top-hat)
- Eulerian flux-form advection in log-mass space
- JIT-compilable via `jax.lax.fori_loop`
- CFL-based automatic substepping (C_max=0.8)

## Bug Fixes and Guards

### Condensation Sink NaN Guard (`condensation_sink.py`)

Empty bins have `Dpk = 0`, which causes division by zero in the Knudsen number calculation (`Kn = 2*mfp/Dpk`). This propagates NaN through the Fuchs-Sutugin correction and into the condensation sink, destroying the entire simulation.

**Guard**: `safe_beta = jnp.where(Dpk > 0.0, beta, 0.0)` — bins with zero diameter contribute nothing to the condensation sink.

### MNFIX Empty-Bin Detection (`mnfix_jax.py`)

After PPM mass conservation rescaling, bins can have positive number but zero dry mass. The original empty-bin check (`Nk_new < TINY_N`) missed these. They cause `xbar = 0`, triggering spurious downward bin shifts.

**Fix**: `mask_empty = (Nk_new < TINY_N) | ((Nk_new >= TINY_N) & (drymass_new < NEPS))`

## Future Roadmap

- **Fix PPM N conservation**: Moving Center method or moment-preserving advection to prevent 0th moment drift
- **Remove bin-0 mass dump**: Condensation should not create new particles (that's nucleation)
- **Fused time loop**: `jax.lax.scan` over all 1440 timesteps to eliminate Python dispatch overhead
- **Multi-species condensation** with VBS partitioning and Kelvin effect
- **GPU acceleration** for the full coagulation + condensation pipeline
- **End-to-end JIT** for condensation + coagulation in a single compiled step
