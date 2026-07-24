# TOMAS-JAX

> [!WARNING]
> This project is under active development, there is no guarantee that it will operate as expected.


**High-Performance, Differentiable Aerosol Microphysics in JAX**

A modern re-implementation of the TOMAS (TwO-Moment Aerosol Sectional) aerosol microphysics model, leveraging Google's JAX for JIT compilation, automatic differentiation, and GPU/TPU portability.

---

## Features

- **SO2 Chemistry:** SO2 + OH → H2SO4 using Sun et al. (2022) Troe formalism with H2O enhancement. Constant or diurnal OH modes.
- **Nucleation:** Two selectable schemes, JIT-compiled with adaptive sub-stepping
  - `ricco_dunne` — Riccobono 2014 (organic) + Dunne 2016 (inorganic, 4 mechanisms)
  - `zhao2024` — Zhao et al. 2024 (11 mechanisms: inorganic, organic, amine, iodine)
- **Coagulation:** Brownian coagulation with Fuchs correction. Two JIT solvers: Tsit5 adaptive (diffrax) and forward Euler. Faster than Fortran.
- **Condensation:** H₂SO₄ condensation with 4 selectable methods:
  - `tfl_jit` — Pure-JAX TFL (matches Fortran output exactly)
  - `ppm_jit` — Pure-JAX PPM with analytical mass-weighted flux (fastest, machine-precision conservation)
  - `tfl` / `ppm` — Sequential reference implementations
- **Equilibrium:** NH₃/NH₄⁺ stoichiometric equilibrium + ISORROPIA-based water uptake
- **Configurable grid:** 40-bin default (1.7nm–17.5μm), 80-bin high-resolution, or custom via `make_grid()`
- **Composable API:** `make_step(['so2_chemistry', 'nucleation', 'coagulation', 'condensation'])` for any process combination
- **Scan-fused loops:** 1440-step time loops compiled to single XLA programs (zero Python overhead)
- **Precision:** float64 enforced everywhere. Mass conservation at machine precision.
- **160 tests**, all passing

---

## Installation

**Requirements:** Python 3.10+

### With uv (Recommended)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync
git clone https://github.com/aliakherati/tomas-jax.git
cd tomas-jax
uv sync --extra dev       # Install all dependencies from lockfile
```

### With pip

```bash
git clone https://github.com/aliakherati/tomas-jax.git
cd tomas-jax
pip install -e ".[dev]"   # Install with dev dependencies
```

### Verify

```bash
python -c "from tomas_jax import TomasState, CoagulationSolver; print('OK')"
python -m pytest tests/ --ignore=tests/test_24h_scenarios.py -q
```

---

## Quick Start

### Box Model (CLI)

```bash
# Full model: nucleation + coagulation + condensation (24h, 60s timestep)
python run_box_model.py
```

#### Process Combinations

The model supports any combination of nucleation, coagulation, and condensation. Use `--no-nucleation` and `--no-condensation` to disable processes (coagulation is always on in the CLI):

```bash
# All processes (default)
python run_box_model.py                                          # nucl + coag + cond

# Two-process combinations
python run_box_model.py --no-nucleation                          # coag + cond
python run_box_model.py --no-condensation                        # nucl + coag

# Coagulation only
python run_box_model.py --no-nucleation --no-condensation        # coag only
```

For arbitrary process combinations (including condensation-only or nucleation+condensation without coagulation), use the `--make-step` flag with the Python API:

```bash
# Condensation only (via make_step)
python -c "
from tomas_jax.solvers.condensation import make_step, run_condensation_scan
# ... see Python API section below
"
```

#### Condensation Methods

```bash
python run_box_model.py --method ppm_jit       # PPM (fastest, default)
python run_box_model.py --method tfl_jit       # TFL (Fortran-matching)
python run_box_model.py --method tfl           # TFL sequential (reference)
python run_box_model.py --method ppm           # PPM sequential (reference)
```

#### Nucleation Schemes

```bash
python run_box_model.py --nucl-scheme ricco_dunne   # Riccobono 2014 + Dunne 2016 (default)
python run_box_model.py --nucl-scheme zhao2024      # Zhao et al. 2024 (11 mechanisms)
```

#### SO2 Chemistry

```bash
# SO2 oxidation with constant OH
python run_box_model.py --so2-init 5e10 --oh-conc 1e6 --make-step

# With SO2 emissions
python run_box_model.py --so2-init 5e10 --so2-emission 1e7 --oh-conc 1e6 --make-step

# Diurnal OH cycle (proportional to cos(SZA))
python run_box_model.py --so2-init 5e10 --oh-conc 2e6 --oh-diurnal --lat 45 --make-step
```

#### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `ppm_jit` | Condensation method: `ppm_jit`, `tfl_jit`, `tfl`, `ppm` |
| `--nucl-scheme` | `ricco_dunne` | Nucleation scheme: `ricco_dunne` or `zhao2024` |
| `--no-nucleation` | off | Disable nucleation |
| `--no-condensation` | off | Disable condensation |
| `--make-step` | off | Use composable `make_step()` API |
| `--so2-init` | 0 | Initial SO2 concentration [molec/cm3] |
| `--so2-emission` | 0 | SO2 emission rate [molec/cm3/s] |
| `--oh-conc` | 0 | OH concentration [molec/cm3] |
| `--oh-diurnal` | off | Use diurnal OH cycle (proportional to cos(SZA)) |
| `--lat` | 45 | Latitude [degrees N] for diurnal cycle |
| `--lon` | 0 | Longitude [degrees E] for diurnal cycle |
| `--day-of-year` | 172 | Day of year for diurnal cycle |

### Python API

#### Composable Process Orchestrator (Recommended)

The `make_step()` API lets you create a step function with any combination and ordering of processes:

```python
from tomas_jax.solvers.condensation import make_step
import jax

# Full model: nucleation + coagulation + condensation
step_fn = jax.jit(make_step(
    processes=['nucleation', 'coagulation', 'condensation'],
    cond_method='ppm_jit',
    nucl_scheme='ricco_dunne',
))

# Coagulation + condensation (no nucleation)
step_fn = jax.jit(make_step(
    processes=['coagulation', 'condensation'],
    cond_method='ppm_jit',
))

# Condensation only
step_fn = jax.jit(make_step(
    processes=['condensation'],
    cond_method='ppm_jit',
))

# Coagulation only
step_fn = jax.jit(make_step(
    processes=['coagulation'],
))

# Nucleation + condensation (no coagulation)
step_fn = jax.jit(make_step(
    processes=['nucleation', 'condensation'],
    cond_method='ppm_jit',
    nucl_scheme='zhao2024',
))

# Nucleation + coagulation (no condensation)
step_fn = jax.jit(make_step(
    processes=['nucleation', 'coagulation'],
    nucl_scheme='ricco_dunne',
))

# Run one timestep (all combinations use the same signature)
Nk, Mk, Gc = step_fn(
    Nk, Mk, Gc, xk,
    temp, pres, boxvol, rh, alpha, dt,
    org_conc=1e7, nh3_conc=1e9, fion=3.0,  # needed when nucleation is enabled
)
```

#### Individual Process Calls

```python
from tomas_jax.solvers.diffrax import diffrax_step, coag_euler_step
from tomas_jax.solvers.condensation import condensation_step
from tomas_jax.physics.nucleation import nucleation_step

# Nucleation (creates particles in bin 0, depletes H2SO4 gas)
Nk, Mk, Gc = nucleation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
    org_conc=1e7, nh3_conc=1e9, fion=3.0,
)

# Coagulation — adaptive Tsit5 (standalone) or forward Euler (scan-fused)
Nk, Mk = diffrax_step(Nk, Mk, xk, temp, pres, boxvol, dt, icomp_nodiag=42)
Nk, Mk = coag_euler_step(Nk, Mk, xk, temp, pres, boxvol, dt, icomp_nodiag=42)

# Condensation (single step)
Nk, Mk, Gc = condensation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    method='ppm_jit',
)
```

#### Scan-Fused Time Loops (Fastest)

Pre-built scan-fused loops compile 1440 timesteps into a single XLA program with zero Python dispatch overhead:

```python
from tomas_jax.solvers.condensation import (
    run_condensation_scan,       # Condensation only (PPM)
    run_condensation_scan_tfl,   # Condensation only (TFL)
    run_nucleation_condensation_scan,  # Nucleation + condensation
    run_full_scan,               # Nucleation + coagulation + condensation
)

# Full 24h simulation as single XLA program
Nk, Mk, Gc, N_history = run_full_scan(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
    dt=60.0, prod_rate=prod_rate, n_steps=1440,
    org_conc=1e7, nh3_conc=1e9, fion=3.0,
)
```

#### Summary of All Process Combinations

| Processes | CLI | `make_step()` | Scan-fused |
|-----------|-----|---------------|------------|
| Nucl + Coag + Cond | `python run_box_model.py` | `['nucleation', 'coagulation', 'condensation']` | `run_full_scan()` |
| Coag + Cond | `--no-nucleation` | `['coagulation', 'condensation']` | — |
| Nucl + Coag | `--no-condensation` | `['nucleation', 'coagulation']` | — |
| Nucl + Cond | — | `['nucleation', 'condensation']` | `run_nucleation_condensation_scan()` |
| Coag only | `--no-nucleation --no-condensation` | `['coagulation']` | — |
| Cond only | — | `['condensation']` | `run_condensation_scan()` / `run_condensation_scan_tfl()` |
| Nucl only | — | `['nucleation']` | — |

### Grid Configuration

```python
from tomas_jax.core.config import make_grid, make_grid_80bin, xk_boundaries

# Default 40-bin grid (1.7nm start, mass-doubling, 1.7nm–17.5μm)
xk = xk_boundaries()           # shape (41,)

# 80-bin high-resolution (same range, √2 mass ratio)
xk = make_grid_80bin()          # shape (81,)

# Custom grid
xk = make_grid(nbins=60, xk0=4.55e-24, doubling_factor=2.0**(2/3))
```

### Initialization

```python
from tomas_jax.core.state import TomasState

# Create state from lognormal distribution
state = TomasState.create(
    nbins=40,
    n_total=1e4,       # #/cm³
    gmd=0.05e-6,       # 50 nm geometric mean diameter [m]
    gsd=1.6,           # geometric standard deviation
    temp=298.0,        # K
    pres=101325.0,     # Pa
    boxvol=1e6,        # cm³ (1 m³)
)
Nk, Mk, xk = state.Nk, state.Mk, state.xk
```

---

## Performance

Median wall time per 24h scenario (49 LHC scenarios, 36-bin legacy grid):

| Mode | Fortran | JAX (best) | Ratio |
|------|---------|------------|-------|
| Coagulation only | 0.27 s | 0.15 s | **0.57x (faster)** |
| Condensation only | 0.08 s | 0.26 s (PPM_JIT) | 3.4x |
| Combined (coag+cond) | 0.33 s | 0.41 s (PPM_JIT) | 1.27x |
| Full (nucl+coag+cond) | 0.33 s | 5–13 s (scan-fused) | — |

### GPU-fast reduced model (this branch)

`tomas_jax.fast` is a natively-batched reduced model (SO4+H2O aerosol,
H2SO4+SO2 gases, Dunne neutral-binary nucleation, PPM condensation,
coagulation, Tabazadeh 1997 water) for running ~10⁶ independent cells per
call on a datacenter GPU — built for global-model coupling. See
`docs/gpu_fast.md` and:

```bash
# GPU target: 1M cells x 6 simulated hours in < 10 s
python -m benchmarks.python.bench_fast_1m --cells 1000000 --hours 6 \
    --n-cell-chunks 8 --sort-by-coag-cost

# CPU validation (no GPU required)
python -m benchmarks.python.bench_fast_1m --cells 1000 --hours 1 --cpu-smoke
```

---

## State Variables

| Variable | Shape | Units | Description |
|---|---|---|---|
| `Nk` | (40,) | #/grid cell | Number concentration per bin |
| `Mk` | (40, 44) | kg/grid cell | Mass per bin per species |
| `xk` | (41,) | kg | Bin boundary masses |
| `Gc` | (44,) | kg/grid cell | Gas-phase concentrations (species 0-42 + SO2 at index 43) |
| `temp` | scalar | K | Temperature |
| `pres` | scalar | Pa | Pressure |
| `boxvol` | scalar | cm³ | Grid cell volume |
| `rh` | scalar | 0–1 | Relative humidity |
| `alpha` | scalar | — | Mass accommodation coefficient |

**Species indices (0-based):** 0 = SO₄, 1–41 = organics, 42 = NH₄, 43 = H₂O. Gas array `Gc` has 44 elements: indices 0–42 (aerosol species except water) + index 43 (SO2).

---

## Physics Overview

### Nucleation

Two selectable schemes create new particles from gas-phase precursors:

**Riccobono 2014 + Dunne 2016** (`ricco_dunne`):
- Organic nucleation: J = k_m × [H₂SO₄]² × [Org] × f(T) (Yu 2017 T-correction)
- Inorganic: 4 mechanisms (binary/ternary × neutral/ion-induced)
- Inputs: H₂SO₄, organic vapor, NH₃, ion-pair production rate

**Zhao et al. 2024** (`zhao2024`):
- 11 mechanisms including synergistic HNO₃, pure-organic Kirkby 2016, amine-H₂SO₄, iodine oxoacids
- Per-mechanism enable/disable via float masks
- Additional inputs: HNO₃, ULVOC, dimethylamine, HIO₃

Nucleated clusters: d ≈ 1.7 nm, placed in bin 0, 90% SO₄ + 10% organic. Adaptive sub-stepping prevents particle creation surges (up to 20 substeps per timestep).

### Coagulation

- Brownian coagulation kernel with Fuchs non-continuum correction
- **Tsit5 adaptive** (`diffrax_step`): 5th-order, error-controlled, for standalone use
- **Forward Euler** (`coag_euler_step`): 10 substeps + MNFIX, for scan-fused loops (more stable at high N)
- Mass conservation: < 10⁻¹³ relative error

### Condensation

- H₂SO₄ condensation onto existing aerosol. Gas depletion: Gc(t) = Gc(0) × exp(−CS×t)
- **TFL** (Tzivion-Feingold-Levin): semi-Lagrangian moving-center with top-hat reconstruction
- **PPM** (Piecewise Parabolic Method): Eulerian advection with analytical mass-weighted flux
- Condensation sink computed from Fuchs-Sutugin corrected diffusion

### Equilibrium

- **NH₃:** Stoichiometric NH₄/SO₄ = 2:1 partitioning
- **Water:** Piecewise polynomial fits to ISORROPIA (273 K reference)

### SO2 Chemistry

- SO2 + OH (+M) → H2SO4 using Sun et al. (2022) Troe formalism
- Rate constant k1(T,p) with H2O vapor enhancement
- Constant or diurnal OH modes (proportional to cos(SZA))
- k1(298K, 1atm) ≈ 1.05×10⁻¹² cm³/molec/s, SO2 lifetime ~11 days at [OH]=10⁶

### Operator Splitting Order

Each timestep: (1) SO2 + OH chemistry → (2) Nucleation → (3) Coagulation → (4) Condensation

---

## Project Structure

```
tomas-jax/
├── run_box_model.py                 # Main driver (see CLI options above)
│
├── tomas_jax/                       # Source package
│   ├── core/
│   │   ├── config.py                # NBINS=40, ICOMP=44, grid presets, species indices
│   │   ├── state.py                 # TomasState NamedTuple
│   │   └── mnfix_jax.py             # Mass-number correction (Fortran partial-transfer)
│   ├── physics/
│   │   ├── nucleation.py            # Nucleation: ricco_dunne + zhao2024 (JIT)
│   │   ├── coagulation_kernel.py    # Brownian coagulation (Fuchs)
│   │   ├── coagulation_rates.py     # TFL coagulation rate computation
│   │   ├── condensation.py          # TFL condensation: dmdt_int + tmcond
│   │   ├── condensation_ppm.py      # PPM advection (3rd-order, JIT)
│   │   ├── condensation_tfl_jax.py  # TFL JIT: tmcond_jax + ezcond_tfl_jax
│   │   ├── ezcond.py                # TFL ezcond driver
│   │   ├── ezcond_ppm_jax.py        # PPM ezcond driver (pure JAX, JIT)
│   │   ├── condensation_sink.py     # First-order gas loss rate to aerosol
│   │   ├── gas_properties.py        # Gas diffusivity, MFP, Fuchs-Sutugin
│   │   ├── properties.py            # Particle diameter, thermal speed
│   │   ├── density.py               # Mixed-salt density (Tang 1997)
│   │   ├── so2_chemistry.py          # SO2+OH chemistry (Sun et al. 2022 Troe)
│   │   ├── water_equilibrium.py     # Hygroscopic water uptake (ISORROPIA)
│   │   └── nh3_equilibrium.py       # NH3/NH4 equilibrium
│   ├── solvers/
│   │   ├── condensation.py          # Process orchestrator: make_step() + scan loops
│   │   └── diffrax.py               # Coagulation: diffrax_step + coag_euler_step
│   └── utils/
│       ├── plotting.py              # Size distributions, banana plots, time series
│       └── diagnostics.py           # Coagulation rate diagnostics
│
├── tests/                           # 160 tests
│   ├── test_nucleation.py           # Nucleation parameterization tests (24 tests)
│   ├── test_so2_chemistry.py        # SO2 chemistry tests (28 tests)
│   ├── test_ppm_condensation.py     # PPM algorithm tests (31 tests)
│   ├── test_ppm_jit_condensation.py # PPM JIT pipeline tests
│   ├── test_tfl_jit_condensation.py # TFL JIT pipeline tests (15 tests)
│   ├── test_coagulation_logic.py    # Coagulation kernel tests
│   └── test_24h_scenarios.py        # 24h benchmark validation
│
├── benchmarks/
│   ├── python/                      # Benchmark & comparison scripts
│   │   ├── scenarios.py             # LHC scenario generator (50 scenarios)
│   │   ├── convergence_test.py      # Multi-resolution convergence (40/80 bins)
│   │   ├── benchmark_nucleation_constgc.py  # Nucleation full-mode benchmark
│   │   ├── validate_so2_chemistry.py  # SO2 chemistry validation (6 figures)
│   │   ├── run_24h_scenarios.py     # 24h JAX runner (3 modes × 2 methods)
│   │   └── compare_24h.py           # 3-way Fortran vs TFL vs PPM comparison
│   └── results/                     # Generated plots, NPZ files, summaries
│
├── tomas_fortran/                   # Fortran reference implementation
│   ├── src/                         # 14 core TOMAS source files
│   ├── src_ppm/                     # PPM Fortran port
│   ├── harness/                     # Benchmark drivers
│   └── Makefile
│
├── docs/
│   ├── architecture.md              # System architecture & module connections
│   ├── nucleation.md                # Nucleation algorithms & benchmarks
│   ├── zhao2024_nucleation.md       # Zhao 2024 11-mechanism scheme
│   ├── ppm_condensation.md          # PPM algorithm documentation
│   ├── 24h_benchmark.md             # 24h benchmark suite
│   ├── so2_chemistry.md             # SO2+OH chemistry (Sun et al. 2022)
│   ├── missing_physics.md           # Gap analysis: what's not yet implemented
│   └── future_features.md           # Planned JAX-specific enhancements
│
├── CLAUDE.md                        # Development instructions
└── PROGRESS.md                      # Development changelog
```

---

## Running Tests

```bash
# Full test suite (388 tests, ~2 min) 
python -m pytest tests/ -v

# Nucleation tests only
python -m pytest tests/test_nucleation.py -v

# Quick smoke test
python -m pytest tests/test_nucleation.py tests/test_ppm_condensation.py -q
```

---

## Documentation

| Doc | Contents |
|-----|----------|
| [Architecture](docs/architecture.md) | System diagram, module connections, condensation pipeline |
| [Nucleation](docs/nucleation.md) | Riccobono+Dunne algorithms, adaptive sub-stepping, benchmarks |
| [Zhao 2024 Nucleation](docs/zhao2024_nucleation.md) | 11-mechanism NPF scheme, per-mechanism parameters |
| [PPM Condensation](docs/ppm_condensation.md) | PPM algorithm: reconstruction, flux, analytical mass integrals |
| [24h Benchmarks](docs/24h_benchmark.md) | 50-scenario benchmark suite, Fortran comparison methodology |
| [SO2 Chemistry](docs/so2_chemistry.md) | Sun et al. 2022 Troe formalism, diurnal OH, validation |
| [Missing Physics](docs/missing_physics.md) | Gap analysis: VBS, Kelvin effect, deposition, chemistry, etc. |
| [Future Features](docs/future_features.md) | GPU acceleration, autodiff, vmap, learned surrogates |

---

## Fortran Heritage

Direct port of TOMAS Fortran (coagulation, condensation, nucleation). Fortran source in `tomas_fortran/src/`. See [Architecture](docs/architecture.md) for the complete source mapping table.

---

## License

Distributed under the LGPL-3.0 License. The full LGPL-3.0 text is in [`LICENSE`](LICENSE); LGPL-3.0 incorporates GPLv3 by reference, and the full GPLv3 text is in [`LICENSE.GPL`](LICENSE.GPL) so both licenses are available in the repository.

_Built with [JAX](https://github.com/google/jax) and [Diffrax](https://github.com/patrick-kidger/diffrax)._
