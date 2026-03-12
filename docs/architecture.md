# TOMAS-JAX Architecture

This document describes the complete TOMAS-JAX system: how every module connects, all four condensation methods, critical bug fixes, and how to run everything.

---

## 1. System Overview

TOMAS-JAX implements three aerosol microphysics processes via operator splitting:

```
                         Model Timestep (e.g., 60 s)
                         ===========================

  H2SO4 Source ──> Coagulation (JIT) ──> Condensation ──> Output
                        |                     |
                    diffrax.py          condensation.py (solver)
                    Tsit5 + MNFIX       method = 'tfl' | 'tfl_jit' | 'ppm' | 'ppm_jit'
```

Each timestep:

1. **H2SO4 source** — Add gas-phase production to `Gc[SO4]`
2. **Coagulation** — JIT-compiled Tsit5 ODE solve (diffrax)
3. **Condensation** — Gas depletion + bin redistribution + equilibria
4. **Output** — Nk, Mk, Gc arrays

---

## 2. Complete Module Map

```
tomas_jax/
│
├── core/
│   ├── config.py                  ─── Constants, NBINS=40 (1.7nm start), ICOMP=44, grid presets, species indices
│   ├── state.py                   ─── TomasState NamedTuple
│   └── mnfix_jax.py               ─── Mass-number consistency (Fortran partial-transfer)
│
├── physics/
│   ├── coagulation_kernel.py      ─── Brownian kernel + Fuchs correction
│   ├── coagulation_rates.py       ─── TFL coagulation solver
│   ├── properties.py              ─── Particle diameter, diffusivity, thermal speed
│   ├── density.py                 ─── Mixed-salt aerosol density
│   ├── gas_properties.py          ─── Gas diffusivity, MFP, Fuchs-Sutugin
│   ├── condensation_sink.py       ─── CS [s^-1] and per-bin sinkfrac
│   ├── condensation.py            ─── TFL algorithm: dmdt_int + tmcond (sequential)
│   ├── condensation_ppm.py        ─── PPM advection (3rd-order, JIT via fori_loop)
│   ├── condensation_tfl_jax.py    ─── TFL JIT: tmcond_jax + ezcond_tfl_jax
│   ├── ezcond.py                  ─── TFL ezcond driver (numpy, sequential)
│   ├── ezcond_ppm.py              ─── PPM ezcond driver (numpy wrapper, slow)
│   ├── ezcond_ppm_jax.py          ─── PPM ezcond driver (pure JAX, JIT-compilable)
│   ├── nh3_equilibrium.py         ─── NH3/NH4 stoichiometric equilibrium
│   └── water_equilibrium.py       ─── Hygroscopic water uptake (ISORROPIA fits)
│
├── solvers/
│   ├── diffrax.py                 ─── Coagulation ODE integrator (Tsit5 + MNFIX)
│   └── condensation.py            ─── Condensation driver (all 4 methods + scan loops)
│
└── utils/
    ├── plotting.py                ─── Visualization tools
    └── diagnostics.py             ─── Coagulation rate diagnostics
```

---

## 3. Condensation Pipeline — All Four Methods

### 3.1 Method Comparison

| Property | `tfl` | `tfl_jit` | `ppm` | `ppm_jit` |
|----------|-------|-----------|-------|-----------|
| Algorithm | Semi-Lagrangian top-hat | Same | Eulerian PPM advection | Same |
| Accuracy | 1st order | 1st order | 3rd order | 3rd order |
| Implementation | numpy + Python loops | Pure JAX + fori_loop | JAX core, numpy wrapper | Pure JAX |
| JIT-compiled | No | **Yes** | Partially | **Yes** |
| Scan-fusable | No | **Yes** | No | **Yes** |
| Speed (cond, 24h) | ~12 s | **~0.47 s** | ~131 s | **~0.26 s** |
| Fortran match | Exact | Exact | Different scheme | Different scheme |
| Fortran time | 0.08 s | 0.08 s | 0.08 s | 0.08 s |

**Recommendations:**
- Use `tfl_jit` when Fortran-matching output is needed
- Use `ppm_jit` for maximum speed (1.8x faster than TFL_JIT for condensation)
- Use `tfl` or `ppm` only for debugging (they are 25-500x slower)

### 3.2 Data Flow Diagram

```
                    condensation_step()  [solvers/condensation.py]
                    ════════════════════
                            │
          ┌─────────────────┼─────────────────────────────┐
          │                 │                             │
      method='tfl'    method='tfl_jit'              method='ppm_jit'
      method='ppm'                                  (also 'ppm' via numpy)
          │                 │                             │
          ▼                 ▼                             ▼
  ┌───────────────┐  ┌──────────────────┐   ┌──────────────────────────┐
  │ numpy path    │  │ condensation_    │   │ condensation_step_jax()  │
  │ (sequential)  │  │ step_tfl_jax()   │   │ [solvers/condensation.py]│
  └───────┬───────┘  │ [condensation_   │   └────────────┬─────────────┘
          │          │  tfl_jax.py]      │                │
          │          └────────┬─────────┘                │
          │                   │                          │
          ▼                   ▼                          ▼
  ┌──────────────┐  ┌────────────────────┐   ┌────────────────────────┐
  │ ezcond()     │  │ ezcond_tfl_jax()   │   │ ezcond_ppm_jax()       │
  │ [ezcond.py]  │  │ [condensation_     │   │ [ezcond_ppm_jax.py]    │
  │ or           │  │  tfl_jax.py]       │   │                        │
  │ ezcond_ppm() │  └────────┬───────────┘   └────────────┬───────────┘
  │ [ezcond_ppm] │           │                            │
  └──────┬───────┘           │                            │
         │                   ▼                            ▼
         │          ┌────────────────────┐   ┌────────────────────────┐
         │          │ tmcond_jax()       │   │ ppm_condensation_step()│
         │          │ [condensation_     │   │ [condensation_ppm.py]  │
         │          │  tfl_jax.py]       │   │                        │
         │          │  ├─ dmdt_int()     │   │  ├─ compute_edge_vel() │
         │          │  ├─ top-hat constr │   │  ├─ ppm_reconstruct()  │
         │          │  └─ fori_loop remap│   │  ├─ advect_number()    │
         │          └────────────────────┘   │  ├─ ppm_mass_flux()    │
         │                                   │  ├─ species_flux()     │
         │                                   │  └─ advect_totals()    │
         │                                   └────────────────────────┘
         │
         ▼
  ┌──────────────┐
  │ tmcond()     │
  │ [condensation│
  │  .py]        │
  │  ├─ dmdt_int │
  │  └─ Python   │
  │     loops    │
  └──────────────┘

                 ═══════════════════════════════
                 After ezcond (all methods):
                 ═══════════════════════════════

                 eznh3eqm()          [nh3_equilibrium.py]
                     │
                     ▼
                 calc_equilibrium_    [water_equilibrium.py]
                 water()
                     │
                     ▼
                 mnfix_jax()          [mnfix_jax.py]
                     │
                     ▼
                 Return (Nk, Mk, Gc)
```

### 3.3 Scan-Fused Time Loops

For 24h simulations (1440 × 60s steps), scan-fused loops eliminate all Python dispatch:

```
                  ┌─────────────────────────────────────────┐
                  │           jax.lax.scan                  │
                  │  ┌─────────────────────────────────┐   │
                  │  │  step_fn (runs 1440 times):     │   │
                  │  │                                  │   │
                  │  │  1. Gc[SO4] += prod_rate * dt    │   │
                  │  │  2. condensation_step_jax()      │   │
                  │  │     or condensation_step_tfl_jax │   │
                  │  │  3. Return (Nk, Mk, Gc)         │   │
                  │  │                                  │   │
                  │  └─────────────────────────────────┘   │
                  │                                         │
                  │  Compiled to a SINGLE XLA program       │
                  │  No Python overhead between steps       │
                  └─────────────────────────────────────────┘

  API:
    run_condensation_scan()       # PPM_JIT scan
    run_condensation_scan_tfl()   # TFL_JIT scan
```

---

## 4. Shared Physics Modules

These are called by all condensation methods:

### 4.1 `gas_properties.py` — Gas-Phase Transport

**Fortran source:** `gasdiff.f`, `getCondSink.f`

| Function | Formula | Reference |
|----------|---------|-----------|
| `calc_gas_diffusivity` | Fuller-Schettler-Giddings | gasdiff.f |
| `calc_mean_molecular_speed` | `sqrt(8RT / (pi*MW))` | S&P eqn 9.2 |
| `calc_mean_free_path` | `2*Di/ms` (TOMAS-specific) | getCondSink.f L84 |
| `calc_knudsen_number` | `2*mfp/Dp` | getCondSink.f L111 |
| `calc_fuchs_sutugin_correction` | `(1+Kn)/(1+2Kn(1+Kn)/alpha)` | getCondSink.f L112 |

### 4.2 `condensation_sink.py` — First-Order Loss Rate

**Fortran source:** `getCondSink.f`

```
CS = 2*pi*Di * sum(Dpk * Nk * beta) / (boxvol * 1e-6)    [s^-1]
sinkfrac[k] = (Dpk[k] * Nk[k] * beta[k]) / sum(...)
```

**Critical threshold (Neps=1e10):** Bins with `Nk <= 1e10` use default `density=1500 kg/m3` and `mp=1.4*xk[k]` instead of computing from actual composition. This matches Fortran `getCondSink.f` and is essential for reproducing Fortran sinkfrac distributions.

### 4.3 `nh3_equilibrium.py` — Ammonium-Sulfate Equilibrium

**Fortran source:** `eznh3eqm.f`

Stoichiometric NH4:SO4 = 2:1 partitioning:
- **Ammonia-limited** (`tot_nh3/2 < tot_so4`): No gas NH3, particle NH4 ∝ SO4
- **Excess ammonia** (`tot_nh3/2 >= tot_so4`): Particles at 2:1, excess in gas

### 4.4 `water_equilibrium.py` — Hygroscopic Water Uptake

**Fortran source:** `waterso4.f`, `waternacl.f`, `ezwatereqm.f`

Piecewise polynomial fits to ISORROPIA at 273K. Water mass = `(so4*1.2 + org) * (wr-1)`.

### 4.5 `mnfix_jax.py` — Mass-Number Consistency

**Fortran source:** `mnfix.f`

Corrects bins where the average dry mass is outside bin boundaries. Uses Fortran's **partial-transfer** algorithm (not move-all):

```
For each bin k:
  xbar = drymass / Nk                    # average dry mass
  if xbar > xk[k+1]:                     # too heavy → shift some UP
      nshift = (drymass - xold*Nk) / (xnew - xold)
      keep (Nk - nshift) at xold = sqrt(xk[k]*xk[k+1])
      move nshift to bin k+1 at xnew = xk[k+2]/1.1
  if xbar < xk[k]:                       # too light → shift some DOWN
      nshift = (drymass - xold*Nk) / (xnew - xold)
      keep (Nk - nshift) at xold
      move nshift to bin k-1
```

This ensures smooth, gradual redistribution instead of emptying entire bins.

---

## 5. TFL Condensation — Sequential and JIT

### 5.1 TFL Algorithm (shared physics)

The TFL (Tzivion-Feingold-Levin) algorithm is a semi-Lagrangian moving-center scheme:

1. **Top-hat construction** — Each bin's mass distribution is approximated as a top-hat (uniform) with width computed from shape factor and bin boundaries
2. **Analytic translation** — `dmdt_int()` grows/shrinks the top-hat edges using the Stevens et al. (1996) growth equation
3. **Remapping** — The translated top-hat is redistributed across destination bins proportionally to overlap

### 5.2 Sequential TFL (`method='tfl'`)

**Files:** `ezcond.py` → `condensation.py` (tmcond, dmdt_int)

Direct Fortran port using numpy arrays and Python for-loops. Useful for verification against Fortran, but ~150x slower than JIT paths. This was Phase 1 of development.

### 5.3 JIT TFL (`method='tfl_jit'`)

**Files:** `condensation_tfl_jax.py` (tmcond_jax, ezcond_tfl_jax)

Pure-JAX rewrite that produces **identical output to Fortran**:

- `tmcond_jax()` — Vectorized top-hat construction + `jax.lax.fori_loop` for bin-by-bin remapping
- `ezcond_tfl_jax()` — Pure-JAX ezcond with `jax.lax.cond` for the three condensation paths
- `condensation_step_tfl_jax()` — Full pipeline: CS → gas depletion → ezcond → NH3 → water → MNFIX
- `run_condensation_scan_tfl()` — Scan-fused 1440-step time loop

**Performance:** 43x faster than sequential TFL, matches Fortran output to machine precision.

---

## 6. PPM Condensation — Numpy Wrapper and JIT

### 6.1 PPM Algorithm (shared physics)

The PPM (Piecewise Parabolic Method) is a 3rd-order Eulerian flux-form advection scheme in log-mass space. See `docs/ppm_condensation.md` for the full algorithm.

Key components in `condensation_ppm.py`:
- **PPM reconstruction** — 4th-order edge interpolation + monotonicity/positivity limiting
- **CFL substepping** — Automatic substep count from max Courant number (C_max=0.8)
- **Analytical mass-weighted flux** — Exact integral `∫m(η)n(η)dη` over PPM departure regions
- **Vectorized species transport** — All 44 species in one matrix multiply (no loops)

### 6.2 Numpy Wrapper PPM (`method='ppm'`)

**Files:** `ezcond_ppm.py` → `condensation_ppm.py`

The PPM core is JIT-compiled, but wrapped in numpy code that forces JAX↔numpy conversions and Python for-loops per call. ~500x slower than the JIT path. Only useful for debugging.

### 6.3 JIT PPM (`method='ppm_jit'`)

**Files:** `ezcond_ppm_jax.py` → `condensation_ppm.py`

Pure-JAX pipeline:
- `ezcond_ppm_jax()` — Pure-JAX ezcond driver with `jax.lax.cond` branches
- `condensation_step_jax()` — Full pipeline: CS → gas depletion → ezcond → NH3 → water → MNFIX
- `condensation_step_jit` — Pre-compiled `jax.jit(condensation_step_jax)`
- `run_condensation_scan()` — Scan-fused 1440-step time loop

**Performance:** 1.8x faster than TFL_JIT for condensation-only. Machine-precision mass conservation.

### 6.4 Analytical Mass-Weighted Flux (Critical)

The PPM number flux `F_N` tells us how many particles cross each edge. But to transport mass, we need `F_M` — how much mass crosses.

**Naive approach (broken):** `F_M = F_N * (M/N)_donor` — assigns average mass to all departing particles. But PPM selectively removes particles from the bin edge where they're 2x heavier (mass-doubling bins). This causes systematic mass underestimate → M/N mismatch → MNFIX clips → N loss death spiral (81% loss over 24h).

**Correct approach (analytical):** Integrate `m(η)*n(η)` exactly over the departure region:
- `m(η) = m_L * exp(a*η)` is the mass at position η (a = ln 2 for mass-doubling)
- `n(η) = n_L + b*η - n_6*η²` is the PPM number density parabola
- Antiderivatives: `∫ η^k * exp(aη) dη` for k=0,1,2

All species are then transported proportionally: `F_M_all = F_M_dry[:, None] * donor_ratios`, where `donor_ratios = Mk / M_dry_analytical` uses the analytical dry mass (from the same PPM parabola integral) for exact conservation.

---

## 7. Critical Bug Fixes

### 7.1 MNFIX Partial-Transfer Rewrite (2026-03-01)

**Root cause:** Original MNFIX moved ALL particles to the next bin when average mass exceeded the boundary. This caused bins to empty completely, creating pulsing/oscillating size distributions.

**Fortran behavior:** `mnfix.f` uses partial transfer — splits the bin so some particles stay at the geometric mean mass and only the excess moves. This gives smooth, gradual redistribution.

**Fix:** Complete rewrite of `mnfix_jax.py` to implement Fortran's partial-transfer algorithm via `jax.lax.fori_loop`. Result: S20 errors dropped from 100% to 0.3-1.9%.

### 7.2 Condensation Sink Neps Threshold (2026-03-01)

**Root cause:** Our code used `Neps=1e-20`, computing actual density/mp for ALL bins. Fortran `getCondSink.f` uses `Neps=1e10` — bins with `Nk <= 1e10` use default `density=1500`, `mp=1.4*xk[k]`.

**Impact:** Different `sinkfrac` distributions caused different mass allocation across bins, producing different size distribution evolution.

**Fix:** Match Fortran's `Neps=1e10` in `condensation_sink.py`.

### 7.3 Mass Conservation Correction Threshold (2026-03-01)

**Root cause:** `ezcond.py` used `abs(ratio) < 100.0` for the mass conservation correction, allowing extreme amplification. Fortran uses `abs(1-ratio) < 1.0` (ratio must be in 0 to 2).

**Fix:** Match Fortran's threshold in `ezcond.py`.

### 7.4 Condensation Sink NaN Guard (2026-02)

Empty bins have `Dpk=0` → `Kn=inf` → `beta=NaN` → `CS=NaN`.

**Fix:** `safe_beta = jnp.where(Dpk > 0.0, beta, 0.0)` in `condensation_sink.py`.

### 7.5 PPM Analytical Mass Flux (2026-03-02)

The naive `F_M = F_N * r_avg_donor` caused 81% N loss over 24h. Replaced with exact analytical mass-weighted integrals. See Section 6.4 above.

---

## 8. How to Run Everything

### 8.1 Basic Usage

```bash
# Install
pip install -e .

# Box model (coagulation + condensation, 24h)
python run_box_model.py                       # default: TFL sequential
python run_box_model.py --method tfl_jit      # TFL JIT (Fortran-matching, fast)
python run_box_model.py --method ppm_jit      # PPM JIT (fastest condensation)

# Run all tests
python -m pytest tests/ -v
```

### 8.2 Condensation Methods in Python

```python
from tomas_jax.solvers.condensation import (
    condensation_step,           # Dispatcher for all 4 methods
    condensation_step_jit,       # Pre-compiled PPM JIT
    run_condensation_scan,       # Scan-fused PPM JIT (1440 steps)
    run_condensation_scan_tfl,   # Scan-fused TFL JIT (1440 steps)
)

# Single step (any method)
Nk, Mk, Gc = condensation_step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                                method='ppm_jit')

# Scan-fused 24h loop (fastest — single XLA program)
Nk, Mk, Gc, N_history = run_condensation_scan(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
    dt=60.0, prod_rate=prod_rate, n_steps=1440
)
```

### 8.3 Benchmarks

```bash
# Comprehensive 49-scenario benchmark (PPM analytical, 5 modes)
python benchmarks/python/run_ppm_analytical_benchmark.py
# Output: benchmarks/results/2026-03-02-ppm-analytical/

# Original 24h benchmark suite (50 scenarios × 3 modes × multiple methods)
python benchmarks/python/run_24h_scenarios.py

# Compare JAX vs Fortran
python benchmarks/python/compare_24h.py
python benchmarks/python/plot_24h_summary.py
python benchmarks/python/plot_24h_timing.py

# Fortran benchmark (requires gfortran)
cd benchmarks/fortran && make && ./benchmark_24h
```

### 8.4 Tests

```bash
python -m pytest tests/test_ppm_condensation.py -v      # PPM unit tests (31 tests)
python -m pytest tests/test_tfl_jit_condensation.py -v   # TFL JIT tests (15 tests)
python -m pytest tests/test_24h_scenarios.py -v          # 24h scenario tests
```

---

## 9. Performance Summary

Median wall time per 24h scenario (49 scenarios):

| Mode | Fortran | TFL_JIT | PPM_JIT | JAX/Fortran |
|------|---------|---------|---------|-------------|
| Coag-only | 0.27 s | — | — | 0.15 s (0.57x) |
| Cond-only | 0.08 s | 0.47 s | 0.26 s | 3.4x (PPM) |
| Combined | 0.33 s | — | 0.41 s | 1.27x (PPM) |

Notes:
- Coagulation is **faster than Fortran** (0.57x) due to JIT + vectorization
- PPM_JIT condensation is 1.8x faster than TFL_JIT
- Combined (coag + cond) is only 1.27x slower than Fortran
- First JIT compilation takes ~30-60s (add warmup call)

---

## 10. Design Decisions

### Operator Splitting

Coagulation and condensation are decoupled. This allows independent JIT compilation, verification, and method selection.

### JAX/numpy Boundary

The `tfl` and `ppm` methods use a JAX→numpy→JAX boundary (adds overhead but enables debugging). The `tfl_jit` and `ppm_jit` methods are pure JAX throughout.

### c1/c2 Corrections Disabled

In `ezcond.py`, tmcond is called with `moxd=zeros` for SO4 condensation. The c1/c2 corrections are only needed for multi-component condensation.

### Mass Conservation Uses Dry Mass

Water (species 43) is diagnostic — recomputed each step by equilibrium. Mass conservation checks use dry mass only (species 0-42).

---

## 11. Fortran Source Mapping

| Python Module | Fortran Source | Author |
|---|---|---|
| `condensation.py` (dmdt_int) | dmdt_int.f | Stevens et al. 1996 |
| `condensation.py` (tmcond) | tmcond.f | Tzivion/Feingold/Levin, Adams |
| `condensation_tfl_jax.py` | tmcond.f (JIT rewrite) | — |
| `condensation_ppm.py` | (new, no Fortran equivalent) | — |
| `ezcond.py` | ezcond.f | Jeff Pierce, May 2007 |
| `nh3_equilibrium.py` | eznh3eqm.f | Jeff Pierce, April 2007 |
| `water_equilibrium.py` | waterso4.f, waternacl.f | Peter Adams, 2000-2001 |
| `gas_properties.py` | gasdiff.f, getCondSink.f | Pierce 2007, Perry's |
| `condensation_sink.py` | getCondSink.f | Jeff Pierce, May 2007 |
| `mnfix_jax.py` | mnfix.f | — |

---

## 12. Modular Process Orchestrator

`solvers/condensation.py` uses a layered architecture to eliminate duplication while preserving backward compatibility:

### 12.1 Internal Core Helpers

```
Layer 1: _condensation_step_core(ezcond_fn)
         Single condensation implementation parameterized by ezcond function.
         Both PPM and TFL share this exact code path.

Layer 2: _combined_step_core(ezcond_fn)     — coag + condensation
         _full_step_core(ezcond_fn)          — nucl + coag + condensation

Layer 3: _run_scan(step_fn, nsteps, ...)    — single scan loop
         Replaces 6 copy-pasted scan implementations.

Layer 4: make_step(processes, cond_method)   — public composable API
```

### 12.2 Composable API: `make_step()`

```python
from tomas_jax.solvers.condensation import make_step

# Build a step function with any process combination/ordering
step = make_step(['nucleation', 'coagulation', 'condensation'],
                 cond_method='ppm_jit', n_coag_substeps=10)

# JIT-compile and run
step_jit = jax.jit(step)
Nk, Mk, Gc = step_jit(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                        org_conc=org_conc, nh3_conc=nh3_conc, fion=fion)

# Easy to reorder or skip processes
cond_only = make_step(['condensation'], cond_method='tfl_jit')
coag_cond = make_step(['coagulation', 'condensation'])
```

The `for process in processes` loop is Python-level — unrolled at JAX trace time, so there is zero runtime overhead.

### 12.3 Backward Compatibility

All 15+ existing public function names are preserved as thin wrappers:
- `condensation_step_jax()` → `_condensation_step_core(..., ezcond_fn=ezcond_ppm_jax)`
- `condensation_step_tfl_jax()` → `_condensation_step_core(..., ezcond_fn=ezcond_tfl_jax)`
- `combined_step_ppm_jax()` → `_combined_step_core(..., ezcond_fn=ezcond_ppm_jax)`
- `run_condensation_scan()` → `_run_scan(...)` with PPM step closure
- etc.

---

## 13. Future Roadmap

- **Remove bin-0 mass dump**: Condensation should not create new particles (that's nucleation)
- **Multi-species condensation** with VBS partitioning and Kelvin effect
- **GPU acceleration** for the full pipeline
- **End-to-end JIT** for coagulation + condensation in a single compiled step
