# TOMAS-JAX: Coagulation + Condensation

**High-Performance, Differentiable Aerosol Microphysics in JAX**

A modern re-implementation of the TOMAS (TwO-Moment Aerosol Sectional) microphysics algorithm, leveraging Google's JAX for:

- **Extreme Speed:** JIT compilation via XLA for CPU, GPU, and TPU.
- **Automatic Differentiation:** Gradients of the final state with respect to any input parameter.
- **Modern Solvers:** `diffrax` for adaptive, high-order ODE integration (Tsit5).
- **Vectorized Physics:** Full SIMD implementation of coagulation and condensation.

---

## Features

- **Coagulation:** Brownian coagulation with Fuchs correction (transition regime). JIT-compiled, fully differentiable. Faster than Fortran.
- **Condensation (4 methods):**
  - `tfl` — Sequential Fortran-faithful port (for verification)
  - `tfl_jit` — Pure-JAX JIT-compiled TFL (matches Fortran output exactly, 43x faster than sequential)
  - `ppm` — PPM with numpy wrapper (for debugging)
  - `ppm_jit` — Pure-JAX JIT-compiled PPM with analytical mass-weighted flux (fastest, 1.8x faster than TFL_JIT)
- **NH3 Equilibrium:** Stoichiometric ammonium-sulfate partitioning (eznh3eqm.f).
- **Water Uptake:** Piecewise polynomial fits for ammonium bisulfate and sea salt hygroscopic growth (ISORROPIA-based).
- **Precision:** Enforced float64 everywhere. Mass conservation at machine precision.
- **Scan-Fused Loops:** 1440-step time loops compiled to a single XLA program via `jax.lax.scan`.
- **MNFIX-JAX:** Fortran-faithful partial-transfer mass-number drift correction.
- **Diagnostics:** Built-in plotting for size distributions, banana plots, and time series.

---

## Performance

Median wall time per 24h scenario (49 LHC scenarios):

| Mode | Fortran | JAX (best) | Ratio |
|------|---------|------------|-------|
| Coagulation only | 0.27 s | 0.15 s | **0.57x (faster)** |
| Condensation only | 0.08 s | 0.26 s (PPM_JIT) | 3.4x |
| Combined | 0.33 s | 0.41 s (PPM_JIT) | 1.27x |

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
python run_box_model.py                       # default: TFL sequential
python run_box_model.py --method tfl_jit      # TFL JIT (Fortran-matching, fast)
python run_box_model.py --method ppm_jit      # PPM JIT (fastest)
```

### Python API

```python
from tomas_jax.solvers.diffrax import diffrax_step
from tomas_jax.solvers.condensation import (
    condensation_step,
    run_condensation_scan,       # PPM_JIT scan-fused loop
    run_condensation_scan_tfl,   # TFL_JIT scan-fused loop
)

# Coagulation (JIT-compiled)
Nk_new, Mk_new = diffrax_step(Nk, Mk, xk, temp, pres, boxvol, dt, icomp_nodiag)

# Condensation (single step)
Nk, Mk, Gc = condensation_step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                                method='ppm_jit')

# Scan-fused 24h loop (fastest — single XLA program for 1440 steps)
Nk, Mk, Gc, N_history = run_condensation_scan(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
    dt=60.0, prod_rate=prod_rate, n_steps=1440
)
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
├── run_box_model.py                 # Main driver (--method tfl|tfl_jit|ppm|ppm_jit)
├── run_sensitivity_analysis.py      # AD examples
├── pyproject.toml                   # Build configuration
│
├── tomas_jax/                       # Source package
│   ├── core/
│   │   ├── config.py                # Constants (NBINS=36, ICOMP=44), species indices
│   │   ├── state.py                 # TomasState NamedTuple
│   │   └── mnfix_jax.py             # Mass-number correction (Fortran partial-transfer)
│   │
│   ├── physics/
│   │   ├── coagulation_kernel.py    # Brownian coagulation (Fuchs)
│   │   ├── coagulation_rates.py     # TFL coagulation solver
│   │   ├── properties.py            # Particle diameter, diffusivity, thermal speed
│   │   ├── density.py               # Mixed-salt aerosol density (Tang 1997)
│   │   ├── gas_properties.py        # Gas diffusivity, MFP, Fuchs-Sutugin
│   │   ├── condensation_sink.py     # First-order gas loss rate to aerosol
│   │   ├── condensation.py          # TFL condensation: dmdt_int + tmcond (sequential)
│   │   ├── condensation_ppm.py      # PPM advection (3rd-order, JIT)
│   │   ├── condensation_tfl_jax.py  # TFL JIT: tmcond_jax + ezcond_tfl_jax
│   │   ├── ezcond.py                # TFL ezcond driver (sequential)
│   │   ├── ezcond_ppm.py            # PPM ezcond driver (numpy wrapper)
│   │   ├── ezcond_ppm_jax.py        # PPM ezcond driver (pure JAX, JIT)
│   │   ├── water_equilibrium.py     # Hygroscopic water uptake (ISORROPIA fits)
│   │   └── nh3_equilibrium.py       # NH3/NH4 stoichiometric equilibrium
│   │
│   ├── solvers/
│   │   ├── diffrax.py               # Coagulation ODE integrator (Tsit5 + MNFIX)
│   │   └── condensation.py          # Condensation driver (4 methods + scan loops)
│   │
│   └── utils/
│       ├── plotting.py              # Visualization tools
│       └── diagnostics.py           # Coagulation rate diagnostics
│
├── tests/
│   ├── test_ppm_condensation.py     # PPM unit tests (31 tests)
│   ├── test_tfl_jit_condensation.py # TFL JIT tests (15 tests)
│   └── test_24h_scenarios.py        # 24h benchmark validation
│
├── benchmarks/
│   ├── fortran/                     # Original Fortran benchmark harness
│   │   ├── benchmark_24h.f          # 24h Fortran driver
│   │   └── output/24h/             # Fortran NPZ/CSV output
│   ├── python/                      # Python benchmark scripts
│   │   ├── scenarios.py             # LHC scenario generator (50 scenarios)
│   │   ├── run_24h_scenarios.py     # JAX 24h runner
│   │   ├── run_ppm_analytical_benchmark.py  # PPM analytical benchmark (49×5 modes)
│   │   ├── compare_24h.py           # 3-way comparison engine
│   │   ├── plot_24h_summary.py      # 8 summary plots
│   │   └── plot_24h_timing.py       # Timing comparison plots
│   └── results/                     # Generated plots, NPZ files, summaries
│
├── docs/
│   ├── architecture.md              # System architecture + module connections
│   ├── ppm_condensation.md          # PPM algorithm documentation
│   └── 24h_benchmark.md            # 24h benchmark suite documentation
│
├── CLAUDE.md                        # Development instructions for Claude Code
└── PROGRESS.md                      # Development changelog
```

---

## Physics Overview

### Coagulation (JIT-compiled)

- Brownian coagulation kernel with Fuchs non-continuum correction
- TFL (Tzivion-Feingold-Levin) mass-conserving algorithm
- Solved as an ODE via `diffrax.Tsit5` with adaptive stepping
- MNFIX applied periodically to correct mass-number bin drift
- Mass conservation: < 10^-13 relative error

### Condensation (4 methods)

- **Gas properties:** Fuller-Schettler-Giddings diffusivity, Fuchs-Sutugin correction
- **Condensation sink:** CS = 2*pi*Di * sum(Dpk * Nk * beta) / boxvol
- **Gas depletion:** Exponential decay Gc(t) = Gc(0) * exp(-CS*t)
- **TFL redistribution:** Semi-Lagrangian moving-center (top-hat + analytic growth)
- **PPM redistribution:** Eulerian advection with analytical mass-weighted flux
- **Scan fusion:** 1440-step time loops as single XLA programs

### Equilibrium Processes

- **NH3:** Stoichiometric NH4/SO4 = 2:1 equilibrium. Excess NH3 stays in gas phase.
- **Water:** Piecewise polynomial fits to ISORROPIA results at 273 K. Sulfate treated as NH4HSO4 (factor 1.2). Organics assumed same uptake as sulfate.

---

## Documentation

- **[Architecture](docs/architecture.md)** — System diagram, module connections, all 4 methods, bug fixes, how to run
- **[PPM Condensation](docs/ppm_condensation.md)** — PPM algorithm: reconstruction, flux, analytical mass integrals
- **[24h Benchmarks](docs/24h_benchmark.md)** — Benchmark suite: scenarios, comparison methodology, plots

---

## Fortran Heritage

Direct port of the TOMAS Fortran model. See [Architecture](docs/architecture.md) for the complete source mapping table.

---

## State Variables

| Variable | Shape | Units | Description |
|---|---|---|---|
| `Nk` | (36,) | #/grid cell | Number concentration per bin |
| `Mk` | (36, 44) | kg/grid cell | Mass per bin per species |
| `xk` | (37,) | kg | Bin boundary masses |
| `Gc` | (43,) | kg/grid cell | Gas-phase concentrations |
| `temp` | scalar | K | Temperature |
| `pres` | scalar | Pa | Pressure |
| `boxvol` | scalar | cm^3 | Grid cell volume |
| `rh` | scalar | fraction | Relative humidity (0-1) |
| `alpha` | scalar | — | Accommodation coefficient |

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

_Built with [JAX](https://github.com/google/jax) and [Diffrax](https://github.com/patrick-kidger/diffrax)._
