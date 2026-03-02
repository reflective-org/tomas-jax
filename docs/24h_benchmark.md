# 24-Hour Benchmark Suite

## 1. Overview

The 24-hour benchmark evaluates TOMAS-JAX against the original Fortran TOMAS implementation across 50 diverse atmospheric scenarios. It compares three implementations:

- **Fortran** — Original TOMAS Fortran (multicoag + ezcond)
- **JAX-TFL** — JAX port using the sequential TFL condensation (`method='tfl'`)
- **JAX-TFL_JIT** — Pure-JAX JIT-compiled TFL (`method='tfl_jit'`, matches Fortran output)
- **JAX-PPM_JIT** — Pure-JAX JIT-compiled PPM with analytical mass-weighted flux (`method='ppm_jit'`)

Each scenario runs for 24 simulated hours with 60-second timesteps (1440 steps), producing hourly snapshots of the aerosol size distribution, gas-phase concentrations, and diagnostic totals.

## 2. Scenario Design

### Latin Hypercube Sampling

50 scenarios are generated via Latin Hypercube Sampling (LHS) to span a wide range of atmospheric conditions:

| Parameter | Min | Max | Scale |
|-----------|-----|-----|-------|
| N_total [#/cm3] | 1e2 | 1e6 | log |
| GMD [um] | 0.01 | 1.0 | log |
| GSD | 1.2 | 2.5 | linear |
| Temperature [K] | 220 | 320 | linear |
| Pressure [Pa] | 20000 | 101325 | linear |
| RH | 0.05 | 0.95 | linear |
| Gc_SO4 [kg/cell] | 1e-16 | 1e-10 | log |
| H2SO4 prod [molec/cm3/s] | 1e5 | 1e8 | log |

Seed = 42 for reproducibility.

### Initialization

Each scenario initializes a lognormal size distribution across 36 TOMAS bins (mass-doubling boundaries starting at 1.6033e-23 kg). Bins with N < 0.001 are set to NEPS with minimal sulfate mass.

### Three Modes

Each scenario runs in three modes:

1. **coag_only** — Coagulation only (tests number reduction, mass conservation)
2. **cond_only** — Condensation only (tests mass transfer gas -> aerosol)
3. **combined** — Coagulation + condensation (full microphysics)

## 3. Mass Conservation

### Why Dry Mass?

The TOMAS model tracks 44 species per bin. Species index 43 is water (H2O), which is a **diagnostic** quantity set by `calc_equilibrium_water()` / `ezwatereqm()` at each timestep. Water mass appears and disappears based on RH with no gas-phase water counterpart in the Gc array.

Including water in mass conservation checks creates spurious sources/sinks. The correct conservation law uses **dry mass** only:

```
dry_mass = sum(Mk[:, 0:43])    # species 0-42 (SO4, organics, NH4)
```

### Conservation Laws

**Coagulation only:**
```
M_dry(t=24h) = M_dry(t=0)    (exactly, within numerical precision ~1e-6)
```
Coagulation redistributes particles across bins but conserves total mass and number (modulo numerical diffusion).

**Condensation only:**
```
M_dry(t=24h) + Gc(t=24h) = M_dry(t=0) + Gc(t=0) + Production(0->24h)
```
where `Production = h2so4_prod_kg_per_s * 24 * 3600`.

### What Moves Where

| Process | Gas (Gc) | Dry Aerosol (Mk[:,:43]) | Water (Mk[:,43]) |
|---------|----------|-------------------------|------------------|
| H2SO4 production | +prod*dt | — | — |
| Condensation | -CS*dt | +CS*dt | — |
| Coagulation | — | conserved | — |
| Water equilibrium | — | — | set by RH |
| NH3 equilibrium | changes | changes | — |

## 4. Running the Benchmark

### Quick Start

```bash
cd benchmarks
bash run_24h.sh --skip-fortran   # JAX only (no Fortran compiler needed)
```

### Step-by-Step

```bash
# 1. Generate scenario CSV
python -m benchmarks.python.scenarios

# 2. Run JAX simulations (all 50 scenarios x 3 modes x 2 methods)
python -m benchmarks.python.run_24h_scenarios

# 3. Run comparisons
python -m benchmarks.python.compare_24h

# 4. Generate plots
python -m benchmarks.python.plot_24h_summary

# 5. Generate timing plots
python -m benchmarks.python.plot_24h_timing

# 6. Run tests
python -m pytest tests/test_24h_scenarios.py -v --tb=short -m slow
```

### Selective Runs

```bash
# Only scenarios 1-5, PPM method, condensation mode
python -m benchmarks.python.run_24h_scenarios \
    --scenarios 1 2 3 4 5 \
    --method ppm \
    --mode cond_only
```

## 5. Output Files

### Fortran Output (`benchmarks/fortran/output/24h/`)

```
s{01-50}_{coag|cond|combined}_hour{01-24}_Nk.csv   # (36,) number per bin
s{01-50}_{coag|cond|combined}_hour{01-24}_Mk.csv   # (36,44) mass per bin/species
s{01-50}_{coag|cond|combined}_hour{01-24}_Gc.csv   # (43,) gas-phase concentrations
timing_fortran.csv                                   # Per-scenario CPU times
```

### JAX Output (`benchmarks/results/24h/`)

```
s{01-50}_{coag_only|cond_only|combined}_{tfl|ppm}.npz
  - Nk:         (24, 36)     hourly number snapshots
  - Mk:         (24, 36, 44) hourly mass snapshots
  - Gc:         (24, 43)     hourly gas-phase snapshots
  - N_tot:      (24,)        total number per hour
  - M_tot:      (24,)        total mass per hour (including water)
  - M_dry:      (24,)        dry mass per hour (excluding water)
  - wall_time_s: scalar      wall-clock time for simulation loop

timing_jax.csv                                       # Per-run wall-clock times
```

### Plots (`benchmarks/results/`)

| Figure | File | Description |
|--------|------|-------------|
| 1 | `24h_total_N_timeseries.png` | Total N evolution (3 panels: coag/cond/combined) |
| 2 | `24h_total_M_timeseries.png` | Total dry mass evolution (3 panels) |
| 3 | `24h_error_evolution.png` | Relative error vs Fortran over time (2x3 grid) |
| 4 | `24h_ppm_vs_tfl_scatter.png` | PPM vs TFL scatter at hour 24 |
| 5 | `24h_mass_conservation.png` | Dry mass conservation error at hour 24 |
| 6 | `24h_size_distribution_evolution.png` | Size dist evolution (3 representative scenarios) |
| 7 | `24h_size_distribution_24h_all.png` | Size dist at hour 24 (all 50 scenarios, 10x5 grid) |
| 8 | `24h_size_distribution_error_heatmap.png` | Per-bin PPM vs Fortran error heatmap |
| A | `24h_timing_bars.png` | Average timing bar chart |
| B | `24h_timing_scatter.png` | Per-scenario Fortran vs JAX timing scatter |

## 6. Interpreting the Plots

### Figure 1: Total Number Timeseries
Shows N_tot normalized to hour 1. Coag-only mode should show monotonic decrease (particles merge). Cond-only has constant N (condensation doesn't change number). Combined shows both effects.

### Figure 2: Total Dry Mass Timeseries
Shows dry mass (excluding water) normalized to hour 1. Coag-only should be flat (mass conserved). Cond-only and combined show mass growth from H2SO4 production + condensation.

### Figure 3: Error Evolution
Per-hour relative errors (TFL vs Fortran, PPM vs Fortran) with individual scenario traces and median line. Errors should stay below ~0.5 for N_tot.

### Figure 4: PPM vs TFL Scatter
Hour-24 total number from PPM vs TFL. Points on the 1:1 line indicate agreement. Dashed bands show 20% tolerance.

### Figure 5: Mass Conservation
Absolute relative error in dry mass + gas budget at hour 24. Target: < 1% for both TFL and PPM.

### Figure 6: Size Distribution Evolution
dN/dlogDp vs Dp at hours 1, 6, 12, 24 for three representative scenarios (small/medium/large GMD). Overlays Fortran, TFL, PPM.

### Figure 7: Hour-24 Size Distributions (All Scenarios)
10x5 small multiples showing Fortran (gray dashed), TFL (blue), PPM (red) at hour 24 combined mode. Gives a visual overview of agreement across all parameter space.

### Figure 8: Per-Bin Error Heatmap
(50 scenarios x 36 bins) heatmap of log10(relative error) for PPM vs Fortran at hour 24. Reveals systematic patterns — e.g., large errors in empty bins vs populated bins.

### Figures A & B: Timing
Bar chart of average wall-clock time per mode, and per-scenario scatter with speedup annotations.

## 7. Test Suite

The test suite in `tests/test_24h_scenarios.py` parametrizes over all 50 scenarios.

| Test | Mode | Tolerance | Rationale |
|------|------|-----------|-----------|
| `test_coag_tfl_vs_fortran_number` | coag_only | 50% | Different ODE integrators (Tsit5 vs Euler) |
| `test_coag_tfl_vs_fortran_mass` | coag_only | 5% | Mass is better conserved than number |
| `test_coag_mass_conservation` | coag_only | 1e-6 | Coag conserves dry mass exactly |
| `test_cond_tfl_vs_fortran_number` | cond_only | 50% | Condensation + equilibria differences |
| `test_cond_tfl_vs_fortran_mass` | cond_only | 10% | Mass errors from different condensation schemes |
| `test_cond_ppm_vs_fortran_number` | cond_only | 50% | PPM is a different algorithm |
| `test_cond_ppm_vs_tfl_number` | cond_only | 20% | PPM vs TFL should agree better than either vs Fortran |
| `test_cond_ppm_vs_tfl_mass` | cond_only | 10% | Same conservation law |
| `test_cond_mass_conservation_tfl` | cond_only | 1% | Dry mass + gas budget |
| `test_cond_mass_conservation_ppm` | cond_only | 1% | Dry mass + gas budget |
| `test_combined_*` | combined | various | Combined tolerances |

### Running Subsets

```bash
# Only coagulation tests
pytest tests/test_24h_scenarios.py -v -m coag_only

# Only condensation tests
pytest tests/test_24h_scenarios.py -v -m cond_only

# Only mass conservation tests
pytest tests/test_24h_scenarios.py -v -k conservation

# Single scenario
pytest tests/test_24h_scenarios.py -v -k "sid1"
```

## 8. Performance

### Measured Timing (Apple M-series, single core)

| Mode | Fortran | JAX-TFL (seq) | JAX-TFL_JIT | JAX-PPM_JIT | Best JAX/Fortran |
|------|---------|---------------|-------------|-------------|------------------|
| Coag Only | 0.27s | ~3.5s | — | — | 0.15s (**0.57x**) |
| Cond Only | 0.08s | ~12s | 0.47s | **0.26s** | 3.4x |
| Combined | 0.33s | ~16s | — | **0.41s** | 1.27x |

### Performance Notes

- **Coagulation is faster than Fortran** (0.57x ratio) thanks to JIT compilation + vectorization
- **PPM_JIT is 1.8x faster than TFL_JIT** for condensation-only
- **Combined (coag + PPM_JIT cond)** is only 1.27x slower than Fortran
- Sequential methods (`tfl`, `ppm`) are 25-500x slower — use only for debugging
- First JIT compilation takes ~30-60s; add a warmup call
- Scan-fused loops (`run_condensation_scan`, `run_condensation_scan_tfl`) compile the entire 1440-step time loop to a single XLA program

### Remaining Speed Gap

- Coagulation uses Tsit5 (adaptive RK45) vs Fortran's forward Euler — more accurate but more work per step
- PPM CFL substeps for small-GMD scenarios (< 0.03 um) require up to 200 substeps
- End-to-end JIT (coag + cond in one compiled step) not yet implemented

## 9. FAQ

**Q: Why are coag-only TFL and PPM identical?**
A: Coagulation doesn't use the condensation method. Only one run is performed with the `tfl` label.

**Q: Why does M_tot include water but M_dry doesn't?**
A: `M_tot = sum(Mk[:, :])` includes water (species 43), which is set diagnostically by equilibrium. `M_dry = sum(Mk[:, :43])` excludes water and satisfies the conservation law.

**Q: Why is the coag N tolerance so loose (50%)?**
A: JAX uses Tsit5 (5th-order adaptive Runge-Kutta) while Fortran uses forward Euler. Over 24 hours with 60-second steps, the different integrators can diverge significantly in total number, especially for coagulation which is highly nonlinear.

**Q: Can I run just a few scenarios?**
A: Yes. Use `--scenarios 1 2 3` to select specific scenario IDs.

**Q: How do I re-run without caching?**
A: Delete the NPZ files: `rm benchmarks/results/24h/*.npz`

## 10. Known Differences (Fortran vs JAX)

| Aspect | Fortran | JAX |
|--------|---------|-----|
| ODE integrator | Forward Euler | Tsit5 (adaptive RK45) via diffrax |
| Precision | double (64-bit) | double (64-bit, via jax_enable_x64) |
| Pi constant | 3.141592654 | 3.141592653589793 |
| Boltzmann | 1.38e-23 | 1.380649e-23 |
| MNFIX | Fortran MNFIX (sequential loops) | Fortran-faithful partial-transfer (fori_loop) |
| Condensation | ezcond (TFL) | TFL, TFL_JIT (Fortran-matching), PPM, PPM_JIT |
| Water equilibrium | ezwatereqm.f | water_equilibrium.py (piecewise polynomial port) |
| NH3 equilibrium | eznh3eqm.f | nh3_equilibrium.py (stoichiometric port) |

## 11. Known Issues and Bug Fixes

### Fixed: NaN Propagation in Condensation Sink (2026-02-27)

**Problem**: Empty bins have `Dpk = 0` → `Kn = inf` → `beta = NaN` → `CS = NaN`.

**Fix**: `safe_beta = jnp.where(Dpk > 0.0, beta, 0.0)` in `condensation_sink.py`.

### Fixed: MNFIX Partial-Transfer Rewrite (2026-03-01)

**Problem**: Original MNFIX moved ALL particles to the next bin when average mass exceeded the boundary. This caused bins to empty completely → pulsing/oscillating size distributions. Fortran uses partial transfer.

**Fix**: Complete rewrite of `mnfix_jax.py` with Fortran's partial-transfer algorithm: `nshift = (drymass - xold*Nk) / (xnew - xold)`, keeping remaining particles at geometric mean mass. S20 errors dropped from 100% to 0.3-1.9%.

### Fixed: Condensation Sink Neps Threshold (2026-03-01)

**Problem**: Our code used `Neps=1e-20` for all bins. Fortran `getCondSink.f` uses `Neps=1e10` — bins with `Nk <= 1e10` use default `density=1500`, `mp=1.4*xk[k]`. Different `sinkfrac` distributions caused different size distribution evolution.

**Fix**: Match Fortran's `Neps=1e10` in `condensation_sink.py`.

### Fixed: Mass Conservation Correction Threshold (2026-03-01)

**Problem**: `ezcond.py` used `abs(ratio) < 100.0` (allowed 100x amplification). Fortran uses `abs(1-ratio) < 1.0` (ratio in 0 to 2).

**Fix**: Match Fortran's threshold.

### Fixed: PPM Number Conservation (2026-03-02)

**Problem**: PPM had 81% N loss over 24h due to naive mass flux `F_M = F_N * r_avg_donor`. PPM selectively removes particles from the bin edge where they're 2x heavier → systematic mass underestimate → M/N mismatch → MNFIX clips → N loss death spiral.

**Fix**: Analytical mass-weighted integrals `∫m(η)n(η)dη` over PPM departure regions + vectorized all-species transport. Result: machine-precision mass conservation (8.6e-16), median N_tot error 2.08e-7 across 49 scenarios.

### Open: Bin-0 Mass Dump in Condensation

**Problem**: When CS < 1e-20, mass is dumped into bin 0 (physically nucleation, not condensation).

**Status**: Deferred until nucleation module is implemented.
