# Nucleation in TOMAS-JAX

## Overview

Nucleation creates new ultrafine particles from gas-phase precursors. TOMAS-JAX implements two parameterizations ported from TRACER_SOM-TOMAS Fortran:

1. **Riccobono 2014** — Organic nucleation with Yu 2017 temperature correction
2. **Dunne 2016** — Inorganic nucleation (4 mechanisms)

Both are pure-JAX, JIT-compilable functions in `tomas_jax/physics/nucleation.py`.

## Riccobono 2014 + Yu 2017

Organic nucleation rate depending on H2SO4 and oxidized organic vapor:

```
J = km * [H2SO4]^2 * [Org] * exp(delta_H/kb * (1/T - 1/T0))
```

**Constants:**
- `km = 3.27e-21` cm^6 s^-1 (nucleation rate constant)
- `delta_H = 2.66e-19` J (enthalpy change)
- `kb = 1.38e-23` J/K (Boltzmann constant)
- `T0 = 278` K (reference temperature)

At T=278K, the T-correction is exactly 1.0. Lower temperatures increase the rate.

**Reference:** Riccobono et al., Science 344, 717-721 (2014); Yu et al., ACP 17, 4997-5005 (2017).

## Dunne 2016

Four inorganic nucleation mechanisms:

| Mechanism | Symbol | Formula |
|-----------|--------|---------|
| Binary neutral | Jbn | `kbn * cna^pbn` |
| Ternary neutral | Jtn | `ktn * ffn` |
| Binary ion-induced | Jbi | `kbi * ionc * cna^pbi` |
| Ternary ion-induced | Jti | `kti * ionc * ffi` |

Total rate: `fn = Jbn + Jtn + Jbi + Jti`

**Unit scaling:** H2SO4 and NH3 inputs (molec/cm3) are scaled by 1e-6 internally (Dunne uses 10^6 molec/cm3 units).

**Ion concentration:** Steady-state from ion-pair production rate:
```
alpha_ion = 6e-8 * sqrt(300/T) + 6e-26 * Mair * (300/T)^4
ionc = sqrt(fion / alpha_ion)
```

**Ternary f-values:** Active only when scaled NH3 > 1e-10 (10^6 molec/cm3 units):
```
ffn = nh3 * cna^ptn / (an + cna^ptn / nh3^pAn)
ffi = nh3 * cna^pti / (ai + cna^pti / nh3^pAi)
```

**Rate constants** (complex T-dependence):
```
k = exp(u - exp(v * (T/1000 - w)))
```

16 parameters (complex form):

| Param | bn | tn | bi | ti |
|-------|------|------|------|------|
| p | 3.95 | 2.89 | 3.37 | 3.14 |
| u | 9.70 | 182. | -11.5 | -23.8 |
| v | 12.6 | 1.20 | 25.5 | 37.0 |
| w | -0.00707 | -4.19 | 0.181 | 0.227 |
| pA | 8.00 | — | 3.07 | — |
| a | 1.6e-6 | — | 0.00485 | — |

**Reference:** Dunne et al., Science, DOI: 10.1126/science.aaf2649 (2016).

## Nucleation Cluster Properties

- Radius: 0.85 nm (fixed for both parameterizations)
- Density: 1350 kg/m3
- Mass: `mnuc = 4/3 * pi * (0.85e-9)^3 * 1350 = ~3.47e-24 kg`
- Always placed in bin 0 (mnuc < XK0 = 1.6e-23 kg)
- Composition: 90% SO4, 10% organic (last organic species)

## Integration in TOMAS-JAX

### Unit Conversions

Gas-phase H2SO4 (kg/grid cell -> molec/cm3):
```
h2so4 = Gc[SRTSO4] / boxvol * 1000 / 98 * 6.022e23
```

Air number density:
```
Mair = 2.69e19 * 273.15/T * P/101325
```

### Operator Splitting Order

Each timestep:
1. H2SO4 production (constant source)
2. **Nucleation** — creates particles in bin 0, depletes gas
3. Coagulation (JIT, diffrax)
4. Condensation (TFL/PPM JIT, includes MNFIX)

### Gas Depletion

SO4 mass added to aerosol is subtracted directly from Gc[SRTSO4] (no 98/96 MW correction in normal path). This matches the Fortran behavior where `Gcf(srtso4) = Gci(srtso4) - (Mkf-mold)`.

**Clamping:** If gas would go negative, all available gas is consumed. In the clamped path:
- SO4 mass = Gc * (96/98) (apply MW correction only in clamped path)
- Number = clamped SO4 mass / mnuc
- **Organic mass is clamped proportionally**: `dM_org_clamped = dN_clamped * 0.1 * mnuc`. This differs from Fortran (which does not clamp organic), but is necessary for JAX stability — without it, particles created with avg mass 80,000x above bin 0 boundary cause catastrophic cascading rates in TFL coagulation

### Enable/Disable Flags

Multiplicative float masks (0.0/1.0) avoid JIT recompilation:
- `enable_organic`: Riccobono on/off
- `enable_inorganic`: Dunne on/off
- `fn_scale`: global nucleation rate multiplier

## Files

- `tomas_jax/physics/nucleation.py` — 3 functions: `ricco_nucleation_rate`, `dunne_nucleation_rate`, `nucleation_step`
- `tomas_jax/solvers/condensation.py` — `condensation_step_with_nucleation_jax`, `run_nucleation_condensation_scan`
- `tests/test_nucleation.py` — 24 unit tests
- `run_box_model.py` — nucleation in time loop with `--no-nucleation` flag

## Fortran Source Mapping

- `ricco_nucleation_rate` <- `ricco_nucl.f`
- `dunne_nucleation_rate` <- `dunne_inorg_nucl.f`
- `nucleation_step` <- `nucleation.f` (driver, simplified: no coag-sink survival fraction)

## Verification

### Unit Tests (24 tests, all passing)
- Riccobono at T=278K, h2so4=1e7, org=1e7: J = 3.27 cm^-3 s^-1
- Dunne with nh3=0, fion=0: only Jbn > 0
- Mass conservation: aerosol SO4 gained = gas H2SO4 lost (no MW correction)
- JIT compilation: `jax.jit(nucleation_step)` succeeds

### Fortran Benchmark — Parameterization Level (Option A)
20 test cases (10 Riccobono + 10 Dunne) comparing JAX vs Fortran output. All cases match to machine precision (max relative error < 1e-12).

Files: `tomas_fortran/harness/benchmark_nucleation.f`, `benchmarks/python/compare_nucleation.py`

### Fortran Benchmark — 24h Driver (Option B)
50 Latin Hypercube scenarios × nucl_cond mode (nucleation + condensation), 24-hour simulations comparing hourly output.

**Results:**
- N_tot median error: 0.2%, max 25% (33/50 scenarios < 1%, 41/50 < 5%)
- M_tot max error: 0.43%, median 1.2e-5 (excellent)
- Residual N errors from slight differences in MNFIX partial-transfer bin placement compounding over 1440 operator-split steps

### Fortran Benchmark — Full Mode (Option B, full)
50 scenarios × full mode (nucleation + coagulation + condensation), 24-hour simulations.

**Results:**
- N_tot median error: 8.1%, max 99.5% (16/50 < 1%, 24/50 < 5%)
- M_dry median error: 0.08%, max 100% (29/50 < 1%, 30/50 < 5%)
- Higher errors due to organic mass clamping fix (necessary for JAX stability, diverges from Fortran)
- Coagulation uses forward Euler (10 substeps per 60s) instead of RK4 to avoid intermediate-stage amplification

### Key Bugs Fixed During Benchmarking
1. **Gas depletion**: JAX had `98/96` MW correction, Fortran does not in normal path
2. **Organic mass clamping**: When H2SO4 gas exhausted, unclamped organic mass creates particles with avg mass 80,000x above bin boundary, causing TFL coagulation divergence. Fixed with proportional clamping
3. **MNFIX multi-bin shift**: JAX only handled ±1-2 bins; nucleated particles need ±12 bins. Fixed with analytical log2 formula
4. **RK4 stage amplification**: RK4 intermediate stages amplify N^2 coagulation rates through positive feedback. Replaced with forward Euler (matches Fortran)
5. **Fortran CSV parsing**: Missing 'E' in exponents > 3 digits. Fixed with regex insertion

### Performance
- nucl_cond scan-fused: ~0.5s/scenario (JIT)
- full scan-fused (forward Euler coag): ~5-13s/scenario
- Fortran combined: ~0.33s/scenario

### Benchmark Plots
8 figures in `benchmarks/results/nucleation_benchmark/`:
- fig1: N_tot timeseries (all 5 modes)
- fig2: M_dry timeseries (all 5 modes)
- fig3: N_tot error evolution with median/P90
- fig4: M_dry error evolution with median/P90
- fig5: Scatter (JAX vs Fortran at hour 24)
- fig6: Per-scenario error bars (nucl_cond and full)
- fig7: Error heatmap (scenario × hour)
- fig8: Summary statistics table

## Constant-Gas vs Fixed-Production Benchmark

Dedicated full-mode benchmark comparing JAX PPM (40/80 bins) vs Fortran TFL (36 bins) over 24 hours with nucleation + coagulation + condensation.

**Script:** `benchmarks/python/benchmark_nucleation_constgc.py`
**Fortran harness:** `tomas_fortran/harness/benchmark_constgc.f`

### Two Modes

1. **Constant-gas** (`--h2so4 1e7`): H2SO4 reset to fixed concentration each timestep. Open-loop — condensation sink (CS) differences between resolutions compound without feedback, causing 30-67% mass divergence between 40/80 bins.

2. **Fixed-production** (`--prod-rate 1e7`): Constant H2SO4 production rate, gas consumed by condensation. Closed-loop — at steady state, `mcond ≈ prod × dt` regardless of CS, because higher CS → faster gas depletion → lower equilibrium gas → same total condensation. Result: 40-bin and 80-bin M_dry match to **0.0003%**.

### Why Constant-Gas Diverges

The condensation rate is `mcond = Gc * (1 - exp(-CS * dt))`. When gas is held constant:
- Different resolutions produce different CS values (different bin widths → different sinkfrac distributions)
- Higher CS → more mass condensed per step → larger particles → even higher CS → divergence compounds

When production rate is fixed:
- Gas reaches quasi-steady state where `production ≈ CS * Gc`
- At steady state: `Gc_ss = prod / CS`, so `mcond = prod * dt` (CS cancels)
- Resolution-dependent CS differences cancel out in the product `CS * Gc_ss`

### Usage

```bash
# Constant-gas mode (H2SO4 = 1e7 molec/cm3, held constant)
python -m benchmarks.python.benchmark_nucleation_constgc --h2so4 1e7

# Fixed-production mode (H2SO4 production = 1e7 molec/cm3/s)
python -m benchmarks.python.benchmark_nucleation_constgc --prod-rate 1e7
```

## Organic Nucleation Concentration

In real atmospheric models, the organic concentration for Riccobono 2014 nucleation is **not hardcoded**. It is computed dynamically from gas-phase chemistry:

- **SOM-TOMAS**: `org_conc` is computed from the SOMGC array (Secondary Organic Model gas-phase concentrations). The CALC_JNUC function extracts ELVOC from a specific SOM grid species.
- **Original TOMAS (box.f)**: Uses a simple user-entered constant nucleation rate; does not implement Riccobono/Dunne at all.
- **Our Fortran harness**: `nucleation_driver.f` takes `org_conc` as a subroutine argument — the caller provides it.

For benchmarks, we use hardcoded values (e.g., `org_conc = 1e7 molec/cm3`). In a real application, `org_conc` should come from the atmospheric model's gas-phase chemistry module.
