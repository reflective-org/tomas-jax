# TOMAS-JAX: Coagulation + Condensation

**High-Performance, Differentiable Aerosol Microphysics in JAX**

A modern re-implementation of the TOMAS (TwO-Moment Aerosol Sectional) microphysics algorithm, leveraging Google's JAX for:

- **Extreme Speed:** JIT compilation via XLA for CPU, GPU, and TPU.
- **Automatic Differentiation:** Gradients of the final state with respect to any input parameter.
- **Modern Solvers:** `diffrax` for adaptive, high-order ODE integration (Tsit5).
- **Vectorized Physics:** Full SIMD implementation of the TFL coagulation algorithm.

---

## Features

- **Coagulation:** Brownian coagulation with Fuchs correction (transition regime). JIT-compiled, fully differentiable.
- **Condensation:** H2SO4 condensation onto size-resolved aerosol via TFL moving-center algorithm (Fortran-faithful port of tmcond.f).
- **NH3 Equilibrium:** Stoichiometric ammonium-sulfate partitioning (eznh3eqm.f).
- **Water Uptake:** Piecewise polynomial fits for ammonium bisulfate and sea salt hygroscopic growth (ISORROPIA-based).
- **Precision:** Enforced float64 everywhere. Mass conservation < 10^-13 relative error for coagulation.
- **Operator Splitting:** Decouples expensive kernel calculations (O(N^2)) from the integration loop.
- **MNFIX-JAX:** Vectorized, JIT-compatible mass-number drift correction.
- **Diagnostics:** Built-in plotting for size distributions, banana plots, and time series.

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone <repo-url>
cd tomas-jax-coagulation
pip install -e .
```

> For NVIDIA GPU support, install CUDA-enabled JAX first (see [JAX docs](https://github.com/google/jax#installation)).

---

## Quick Start

### Box Model (Coagulation + Condensation)

```bash
python run_box_model.py
```

Runs a 24-hour 0-D box model with:
- Log-normal initial distribution (100 nm GMD, 100,000 particles/cm3)
- Constant H2SO4 production (10^7 molec/cm3/s)
- Operator-split coagulation (JIT) + condensation (sequential)
- Console output + plots (banana plot, size distributions, time series)

### Coagulation Only

```python
from tomas_jax.solvers.diffrax import diffrax_step
Nk_new, Mk_new = diffrax_step(Nk, Mk, xk, temp, pres, boxvol, dt, icomp_nodiag)
```

### Condensation Only

```python
from tomas_jax.solvers.condensation import condensation_step
Nk_new, Mk_new, Gc_new = condensation_step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt)
```

### Sensitivity Analysis (Automatic Differentiation)

```bash
python run_sensitivity_analysis.py
python run_temp_sensitivity.py
```

---

## Project Structure

```
tomas-jax-coagulation/
├── run_box_model.py                 # Main driver (coag + cond)
├── run_sensitivity_analysis.py      # AD examples
├── pyproject.toml                   # Build configuration
│
├── tomas_jax/                       # Source package
│   ├── core/
│   │   ├── config.py                # Global constants, dimensions, species indices
│   │   ├── state.py                 # TomasState (Nk, Mk, xk, Gc, rh, alpha, ...)
│   │   └── mnfix_jax.py             # Mass-number drift correction (vectorized)
│   │
│   ├── physics/
│   │   ├── coagulation_kernel.py    # Brownian coagulation (Fuchs)
│   │   ├── coagulation_rates.py     # TFL coagulation solver
│   │   ├── properties.py            # Particle diameter, diffusivity, thermal speed
│   │   ├── density.py               # Mixed-salt aerosol density (Tang 1997)
│   │   ├── gas_properties.py        # Gas diffusivity, MFP, Fuchs-Sutugin correction
│   │   ├── condensation_sink.py     # First-order gas loss rate to aerosol
│   │   ├── condensation.py          # TFL condensation: dmdt_int + tmcond
│   │   ├── ezcond.py                # Simple condensation driver
│   │   ├── water_equilibrium.py     # Hygroscopic water uptake (ISORROPIA fits)
│   │   └── nh3_equilibrium.py       # NH3/NH4 stoichiometric equilibrium
│   │
│   ├── solvers/
│   │   ├── diffrax.py               # Coagulation ODE integrator (Tsit5 + MNFIX)
│   │   └── condensation.py          # Condensation operator-split driver
│   │
│   └── utils/
│       ├── plotting.py              # Visualization tools
│       └── diagnostics.py           # Coagulation rate diagnostics
│
├── tests/                           # Test suite
│   ├── test_ppm_condensation.py     # PPM condensation tests (43 tests)
│   └── test_24h_scenarios.py        # 24h benchmark tests (50 scenarios)
│
├── benchmarks/                      # Performance benchmarks
│   ├── fortran/                     # Original Fortran benchmark harness
│   │   ├── benchmark_24h.f          # 24h Fortran driver
│   │   └── output/24h/             # Fortran CSV output
│   ├── python/                      # Python benchmark scripts
│   │   ├── scenarios.py             # LHC scenario generator
│   │   ├── run_24h_scenarios.py     # JAX 24h runner
│   │   ├── compare_24h.py           # 3-way comparison engine
│   │   ├── plot_24h_summary.py      # 8 summary plots
│   │   └── plot_24h_timing.py       # Timing comparison plots
│   └── results/                     # Generated plots and NPZ files
│
├── docs/
│   ├── architecture.md              # Condensation pipeline architecture
│   ├── ppm_condensation.md          # PPM algorithm documentation
│   └── 24h_benchmark.md            # 24h benchmark suite documentation
│
├── CLAUDE.md                        # Development instructions for Claude Code
└── PROGRESS.md                      # Development progress report (changelog)
```

---

## Physics Overview

### Coagulation (JIT-compiled)

- Brownian coagulation kernel with Fuchs non-continuum correction
- TFL (Tzivion-Feingold-Levin) mass-conserving algorithm
- Solved as an ODE via `diffrax.Tsit5` with adaptive stepping
- MNFIX applied periodically to correct mass-number bin drift
- Mass conservation: < 10^-13 relative error

### Condensation (Phase 1 - Sequential)

- **Gas properties:** Fuller-Schettler-Giddings diffusivity, Fuchs-Sutugin correction
- **Condensation sink:** CS = 2*pi*Di * sum(Dpk * Nk * beta) / boxvol
- **Gas depletion:** Exponential decay Gc(t) = Gc(0) * exp(-CS*t)
- **Bin redistribution:** TFL moving-center algorithm (tmcond.f)
  - Top-hat construction per bin
  - Analytic growth solution (dmdt_int, Stevens et al. 1996)
  - Semi-Lagrangian remapping with species-dependent mass fractions
- **Adaptive sub-stepping:** Limits mass increase to ~10% per step
- **Mass conservation correction:** Post-hoc adjustment in ezcond

### Equilibrium Processes

- **NH3:** Stoichiometric NH4/SO4 = 2:1 equilibrium. Excess NH3 stays in gas phase.
- **Water:** Piecewise polynomial fits to ISORROPIA results at 273 K. Sulfate treated as NH4HSO4 (factor 1.2). Organics assumed same uptake as sulfate.

### Operator Splitting Strategy

Each model timestep (e.g., 60 s):

1. **H2SO4 source** — Add gas-phase production
2. **Coagulation** — JIT-compiled Tsit5 ODE solve
3. **H2SO4 condensation** — Exponential depletion + ezcond
4. **NH3 equilibrium** — Instantaneous gas-particle partitioning
5. **Water equilibrium** — Instantaneous hygroscopic uptake
6. **MNFIX** — Mass-number consistency cleanup

---

## Fortran Heritage

This codebase is a direct port of the TOMAS Fortran model. Key source mapping:

| Python Module | Fortran Source | Author |
|---|---|---|
| `condensation.py` (dmdt_int) | dmdt_int.f | Stevens et al. 1996 |
| `condensation.py` (tmcond) | tmcond.f | Tzivion/Feingold/Levin 1989, Adams |
| `ezcond.py` | ezcond.f | Jeff Pierce, May 2007 |
| `nh3_equilibrium.py` | eznh3eqm.f | Jeff Pierce, April 2007 |
| `water_equilibrium.py` | waterso4.f, waternacl.f, ezwatereqm.f | Peter Adams, 2000-2001 |
| `gas_properties.py` | gasdiff.f, getCondSink.f | Pierce 2007, Perry's Handbook |
| `condensation_sink.py` | getCondSink.f | Jeff Pierce, May 2007 |

---

## State Variables

| Variable | Shape | Units | Description |
|---|---|---|---|
| `Nk` | (36,) | #/grid cell | Number concentration per bin |
| `Mk` | (36, 44) | kg/grid cell | Mass per bin per species |
| `xk` | (37,) | kg | Bin boundary masses |
| `Gc` | (43,) | kg/grid cell | Gas-phase concentrations (all species except water) |
| `temp` | scalar | K | Temperature |
| `pres` | scalar | Pa | Pressure |
| `boxvol` | scalar | cm^3 | Grid cell volume |
| `rh` | scalar | fraction | Relative humidity (0-1) |
| `alpha` | scalar | dimensionless | Accommodation coefficient |

### Species Indices (0-based)

| Index | Species |
|---|---|
| 0 | SO4 (sulfate) |
| 1-41 | Organic aerosol (41 species) |
| 42 | NH4 (ammonium) |
| 43 | H2O (water) |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/nucleation`)
3. Commit your changes
4. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

_Built with [JAX](https://github.com/google/jax) and [Diffrax](https://github.com/patrick-kidger/diffrax)._
