# TOMAS FORTRAN vs JAX Coagulation Benchmark

## Purpose

Validates the JAX implementation of TOMAS coagulation against the original FORTRAN code.
Compares component-by-component at 10 benchmark levels and documents all known differences
with scientific justification.

## Prerequisites

- **gfortran** (or compatible Fortran compiler)
- **Python 3.8+** with JAX, numpy, diffrax
- **TOMAS FORTRAN sources** at `../../original-models/TOMAS/src/`

## Quick Start

```bash
cd benchmarks
bash run_all.sh
```

This will:
1. Build the FORTRAN benchmark harness
2. Run it to generate reference data (CSV files)
3. Run 15 levels of Python comparison (including PPM benchmarks)
4. Generate `results/comparison_report.md`

To run PPM-specific benchmarks only:
```bash
python -c "from benchmarks.python import level14_ppm_vs_tfl, level15_ppm_performance; level14_ppm_vs_tfl.run('benchmarks/fortran/output'); level15_ppm_performance.run('benchmarks/fortran/output')"
```

To generate PPM comparison plots:
```bash
python -m benchmarks.python.plot_ppm_comparison
```

## Benchmark Levels

| Level | Component | Target | Notes |
|-------|-----------|--------|-------|
| 1 | xk (bin boundaries) | EXACT | Pure arithmetic, same formula |
| 2 | mu, mfp (air properties) | rtol < 1e-4 | FORTRAN uses 0.0289 vs JAX 0.0289647 |
| 3 | density (per bin) | rtol < 1e-6 | After organic mass fix |
| 4 | Dpk, Dk, ck (properties) | rtol < 1e-3 | 0.333 vs cbrt; after Cunningham fix |
| 5 | kij (coag kernel) | rtol < 1e-3 | Propagated from Dpk |
| 6 | xbar, phi, eff (TFL) | rtol < 1e-10 | Same formulas |
| 7 | dNdt, dMdt (rates) | rtol < 1e-3 | Propagated from kij |
| 8 | MNFIX before/after | EXACT (port) | Using mnfix_fortran.py |
| 9 | Single Euler step | rtol < 1e-6 | With euler.py + mnfix_fortran.py |
| 10a | Full Euler (60s) | rtol < 1e-4 | Full multicoag comparison |
| 10b | Full Tsit5 (60s) | Document | Show improvement over Euler |
| 11 | Condensation sink | rtol < 0.2 | Gas diffusivity, Fuchs-Sutugin, CS |
| 12 | Isolated condensation | rtol < 1e-2 | ezcond + NH3 eq + water eq + MNFIX |
| 13 | Combined coag+cond | rtol < 5e-2 | Full operator-split step |
| 14 | PPM vs TFL | rtol < 0.2 | PPM agreement with TFL and Fortran |
| 15 | PPM performance | Timing | Wall-clock comparison, CFL substeps |

## Known Differences (Not Bugs)

These are deliberate improvements in the JAX code:

| Item | FORTRAN | JAX | Impact |
|------|---------|-----|--------|
| Diameter exponent | `**(0.333)` | `jnp.cbrt()` (exact 1/3) | ~1e-4 rel diff in Dpk |
| Pi constant | `3.141592654` | `3.141592653589793` | ~1e-10 |
| kB constant | `1.38e-23` | `1.380649e-23` | ~5e-4 |
| Molar mass air | `0.0289` | `0.0289647` | ~2e-3 in mfp |

## Bug Fixes Applied

1. **Cunningham slip correction** (`properties.py:79`):
   Fixed polynomial — `Kn2 * (6 + 18*Kn)` to `Kn * (6 + 18*Kn)`.
   Corrected numerator from `5+4Kn+6Kn^3+18Kn^4` to `5+4Kn+6Kn^2+18Kn^3`.

2. **Density organic mass** (`density.py:44`):
   Added organic mass to sulfate in density calculation,
   matching FORTRAN `aerodens(Mk(k,srtso4)+orgmass, ...)`.

3. **Density mtot** (`density.py:97`):
   Changed to use raw component masses instead of reconstructed compounds,
   matching FORTRAN `mtot = mso4+mno3+mnh4+mnacl+mh2o`.

## Integrator Comparison

The FORTRAN code uses Forward Euler with CFL-like adaptive sub-stepping.
The JAX production solver uses Tsit5 (5th-order Runge-Kutta) from diffrax.

| Metric | FORTRAN Euler | JAX Euler | JAX Tsit5 |
|--------|---------------|-----------|-----------|
| Order of accuracy | 1st | 1st | 5th |
| Error control | CFL-like (dtlimit/itlimit) | CFL-like (matching) | Embedded RK pair |
| Mass conservation | Depends on dt | Depends on dt | ~1e-13 relative |
| Speed (40 bins) | Baseline | Comparable | 10-100x faster via JIT |

## 24-Hour Comprehensive Benchmark (50 Scenarios)

In addition to the component-level benchmarks above, a comprehensive 24-hour
benchmark suite runs 50 Latin Hypercube scenarios across diverse atmospheric
conditions, comparing PPM vs TFL vs Fortran.

### Quick Start

```bash
cd benchmarks
bash run_24h.sh              # Full pipeline (Fortran + JAX + plots)
bash run_24h.sh --skip-fortran  # JAX only (skip Fortran)
```

### Parameter Ranges (Latin Hypercube, seed=42)

| Parameter | Min | Max | Scale |
|-----------|-----|-----|-------|
| N_total (#/cm3) | 1e2 | 1e6 | log |
| GMD (um) | 0.01 | 1.0 | log |
| GSD | 1.2 | 2.5 | linear |
| Temperature (K) | 220 | 320 | linear |
| Pressure (Pa) | 20000 | 101325 | linear |
| RH | 0.05 | 0.95 | linear |
| Gc_SO4 (kg/cell) | 1e-16 | 1e-10 | log |
| H2SO4 prod (molec/cm3/s) | 1e5 | 1e8 | log |

### Three Simulation Modes

1. **Coagulation-only:** 1440 x 60s steps of multicoag
2. **Condensation-only:** 1440 x 60s steps of ezcond + equilibria + MNFIX
3. **Combined:** Coagulation + condensation (operator split)

### Tolerance Targets

| Test | Tolerance | Rationale |
|------|-----------|-----------|
| Coag N (TFL vs Fortran) | 50% | Tsit5 vs Euler, 24h propagation |
| Coag M (TFL vs Fortran) | 5% | Mass well-conserved |
| Cond N (TFL/PPM vs Fortran) | 50% | Different redistribution |
| Cond M (TFL/PPM vs Fortran) | 10% | Strong conservation |
| PPM vs TFL N | 20% | Two different methods |
| Mass conservation | 1% | By construction |

### Test Commands

```bash
# Full suite
pytest tests/test_24h_scenarios.py -v --tb=short

# By mode
pytest tests/test_24h_scenarios.py -v -m coag_only
pytest tests/test_24h_scenarios.py -v -m cond_only
pytest tests/test_24h_scenarios.py -v -m combined
```

### Output

- **Fortran CSVs:** `fortran/output/24h/s{01-50}_{coag|cond|combined}_hour{01-24}_{Nk|Mk|Gc}.csv`
- **JAX NPZs:** `results/24h/s{01-50}_{coag|cond|combined}_{tfl|ppm}.npz`
- **Plots:** `results/24h_*.png` (6 summary figures)

## Directory Structure

```
benchmarks/
├── fortran/
│   ├── benchmark_harness.f       # Standalone FORTRAN benchmark (levels 1-13)
│   ├── benchmark_24h.f           # 24-hour benchmark (50 scenarios)
│   ├── scenarios.csv             # LHC scenario parameters (auto-generated)
│   ├── Makefile                  # Build system
│   └── output/                   # Generated CSV reference data
│       └── 24h/                  # 24-hour benchmark output
├── python/
│   ├── __init__.py
│   ├── utils.py                  # Comparison utilities
│   ├── scenarios.py              # Latin Hypercube scenario generator
│   ├── compare_all.py            # Master comparison (levels 1-15)
│   ├── compare_24h.py            # 24-hour 3-way comparison engine
│   ├── run_24h_scenarios.py      # JAX 24-hour simulation runner
│   ├── plot_24h_summary.py       # 24-hour summary plots (6 figures)
│   ├── level01_initbounds.py ... level15_ppm_performance.py
│   ├── test_cases.py             # 5 test case definitions
│   ├── plot_condensation.py      # Condensation benchmark plots
│   └── plot_ppm_comparison.py    # PPM vs TFL comparison plots
├── results/
│   ├── 24h/                      # 24-hour JAX results (NPZ)
│   └── comparison_report.md      # Level 1-15 report
├── run_all.sh                    # Master script (levels 1-15)
├── run_24h.sh                    # Master script (24-hour benchmark)
└── README.md                     # This file
```
