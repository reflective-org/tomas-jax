# SOA Condensation with Volatility Basis Set (VBS)

## Overview

TOMAS-JAX implements semi-volatile organic aerosol (SOA) condensation using
the Volatility Basis Set (VBS) framework with the kinetic gas-particle
partitioning model of Zaveri et al. (2014). The implementation builds on
soacond.f (Jeff Pierce, September 2010) with corrections and improvements.

Key features:
- **Zaveri et al. (2014) kinetic framework**: gas-phase diffusion, interfacial
  accommodation, particle-phase bulk diffusion (D_b), and particle-phase
  reaction (k_c), with automatic approximation selection
- **6-bin VBS** covering C* = 0.01 to 1000 µg/m³ (Donahue et al. 2006)
- **Kelvin effect** for organic species (size-dependent vapor pressure)
- **Bidirectional**: supports both condensation and evaporation
- **Temperature-dependent C*** via Clausius-Clapeyron
- **Particle-phase reaction**: first-order loss (k_c) for irreversible uptake
- **JIT-compilable**: fully traceable by JAX

## Architecture

Five independent modules + one integration point:

```
tomas_jax/physics/
  vbs_config.py           — VBS bin definitions, C* values, D_b, k_c, unit conversions
  kelvin_effect.py        — Kelvin correction factor (organics only)
  vbs_driving_force.py    — Ambient pressure, equilibrium pressure, driving force
  condensation_sink.py    — Organic CS: Zaveri two-film K_g + gas-side k_g + Q factor
  soa_condensation.py     — SOA condensation driver (species loop + eq correction + k_c)

tomas_jax/solvers/
  condensation.py         — make_step() dispatches 'soa_condensation'
```

## Algorithm

Based on Zaveri et al. (2014) kinetic gas-particle partitioning framework.
For each VBS species j (Python loop, unrolled at trace time):

1. **C*(T)**: Temperature-correct C* via Clausius-Clapeyron:
   `C*(T) = C*(298K) × exp(-ΔHvap/R × (1/T - 1/298))`

2. **Kelvin factor**: Per-bin vapor pressure correction for surface curvature:
   `Ke(k) = exp(4σ MW / (R T ρ Dp(k)))`

3. **Condensation sink** (Zaveri et al. 2014, Eqs. 13-24): Organic-specific
   CS with gas-phase + particle-phase diffusion in series:
   ```
   f(Kn,α) = 0.75α(1+Kn) / (Kn(1+Kn) + 0.283αKn + 0.75α)   Fuchs-Sutugin (Eq. 14)
   k_g = D_g / R_p × f(Kn, α)         gas-side mass transfer (Eq. 13)
   k_p = 5 D_b / R_p                   particle-side, kc=0 (Eq. 24)
   k_p = D_b/R_p × (q coth q - 1)/(1-Q)   particle-side, kc>0 (Eq. 23)
   K_g = R_p / (1/k_g + C*/(ρ×1e9) / k_p)  combined two-film (Eq. 20)
   ```
   Where `q = R_p √(k_c/D_b)` is the diffuso-reactive parameter (Eq. 4)
   and `Q = 3(q coth q - 1)/q²` is the quasi-steady-state ratio of average
   to surface concentration inside the particle (Eq. 8). Q=1 for liquid
   (well-mixed), Q→0 for semisolid with fast reaction.

   **Approximation selection** (automatic, based on k_c):
   - **Approximation 1** (k_c ≥ 0.01 s⁻¹, Eq. 29): CS uses gas-side k_g
     only. Particle-phase effect captured by Q in the driving force.
   - **Approximation 2** (k_c < 0.01 s⁻¹, Eq. 31): CS uses two-film K_g.
     Standard driving force (particle-phase resistance in K_g).

4. **Driving force** (Zaveri Eqs. 29/31): Per-bin supersaturation:
   ```
   psat(k) = C* × (Mk_j(k) / Mtot_org(k)) × Ke(k)
   Approx 1: dp(k) = pamb - psat(k) / Q(k)    (Q-modified, Eq. 29)
   Approx 2: dp(k) = pamb - psat(k)            (standard, Eq. 31)
   ```
   For kc=0, Q=1 and both are identical.

5. **Analytical gas depletion toward equilibrium**:
   ```
   Gc_eq = psat_eff × V × MW / (R T)
   mcond = (Gc - Gc_eq) × (1 - exp(-CS × dt))
   ```
   where `psat_eff = Σ(sinkfrac × psat_effective)` is the sink-weighted
   equilibrium (using Q-modified psat for Approx 1).

6. **Distribute to bins**: Proportional to dp-weighted sink fraction:
   `mass_to_bin(k) = mcond × sinkfrac(k) × dp(k) / Σ(sinkfrac × dp)`

7. **Equilibrium correction**: Prevents per-bin overshoot using exponential
   relaxation (soacond.f fodc scheme):
   ```
   fodc = |kinetic/equilibrium| / dt
   corrected = equilibrium × (1 - exp(-fodc × dt))
   ```
   Equilibrium mass uses Q-modified psat when Approx 1 is active.

8. **Conservation**: Clamp total change to available gas/particle mass, scale
   per-bin distribution to match.

9. **Particle-phase reaction loss** (Zaveri Eqs. 19/21: `-k_c × Ā_i`):
   When k_c > 0, apply first-order decay to the volatile species in the
   particle phase: `Mk_j -= Mk_j × (1 - exp(-k_c × dt))`. This represents
   irreversible conversion to nonvolatile products (oligomers, organic salts).
   Gas is NOT affected — the reaction is internal to the particle (Eq. 30/32).
   Note: reaction products are currently removed, not tracked in a separate
   nonvolatile species.

## VBS Configuration

Standard 6-bin VBS (Donahue et al. 2006):

| Bin | C*(298K) µg/m³ | ΔHvap kJ/mol | Category |
|-----|----------------|--------------|----------|
| 0   | 0.01           | 150          | ELVOC    |
| 1   | 0.1            | 136          | LVOC     |
| 2   | 1.0            | 122          | LVOC     |
| 3   | 10.0           | 108          | SVOC     |
| 4   | 100.0          | 94           | SVOC     |
| 5   | 1000.0         | 80           | IVOC     |

All bins: MW = 200 g/mol, σ = 0.025 N/m, ρ = 1200 kg/m³, SV = 120.

Additional VBSConfig parameters (Zaveri et al. 2014):
- `Dbk`: Particle-phase bulk diffusion coefficient [m²/s], default 1×10⁻¹⁰
  (liquid organic, ~10⁻⁶ cm²/s). Range: 10⁻¹⁸ (glassy) to 10⁻⁵ (liquid water).
- `kc`: First-order particle-phase reaction rate [s⁻¹], default 0.
  Represents irreversible conversion to nonvolatile products (oligomers, etc.).
  Triggers Approximation 1 when kc ≥ 0.01 s⁻¹.

VBS bins map to TOMAS organic species 1–6 (SRTORG1 + 0..5). Species 7–41
are reserved for future expansion.

## Solvers

Two switchable solvers, selected via `solver=` kwarg or `soa_solver=` in `make_step()`.
Each sequential solver variant also accepts a `redistribution=` kwarg (or
`soa_redistribution=` in `make_step()`) controlling how grown particles are
placed into bins.

### `solver='sequential'` (default)

Sequential Python loop over 6 VBS species (Gauss-Seidel ordering). Species j+1
sees species j's updated Mk and Gc. Includes adaptive sub-stepping when
`exp(-CS×dt) < 0.01`. Matches existing behavior and Fortran comparison.

**Redistribution options** (`redistribution=` kwarg):

| Value | Method | Description |
|-------|--------|-------------|
| `'tfl'` (default) | TFL Lagrangian remapping (`tmcond_jax`) | Matches Fortran `soacond.f`, but amplifies floating-point differences at bin boundaries → oscillatory artifacts over long runs. |
| `'ppm'` | PPM Eulerian advection (`ppm_condensation_step`) | Smoother distributions, avoids TFL oscillation artifacts. Transport first, then mass addition (same pattern as H2SO4 `ezcond_ppm_jax`). |
| `'direct'` | Direct mass addition + MNFIX | No redistribution transport. Fastest, but skips size-dependent placement. |

### `solver='coupled'`

All 6 species solved simultaneously via vectorized JAX operations. Fixed-point
iteration (n_iter=3, `jax.lax.fori_loop`) resolves Raoult coupling through
Mtot_org. Adaptive sub-stepping when `max(CS)×dt` is large. Always uses direct
mass addition (no TFL/PPM redistribution). Uses batch functions for CS, Kelvin,
and driving force.

**Key difference**: Sequential (Gauss-Seidel) vs coupled (Jacobi iteration).
Both conserve mass per species to machine precision. Numerical results differ
by 5–30% for typical conditions due to different update ordering of the Raoult
denominator (Mtot_org). Neither is more "correct" — they are different valid
operator-split approaches.

## Usage

### With make_step()

```python
from tomas_jax.solvers.condensation import make_step

# Sequential solver with TFL redistribution (default, backward-compatible)
step = make_step(['condensation', 'soa_condensation'], cond_method='ppm_jit')
Nk, Mk, Gc, bv = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt)

# Sequential solver with PPM redistribution (smoother, avoids TFL artifacts)
step = make_step(['condensation', 'soa_condensation'], cond_method='ppm_jit',
                 soa_redistribution='ppm')
Nk, Mk, Gc, bv = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt)

# Coupled solver (direct mass addition, fastest)
step = make_step(['condensation', 'soa_condensation'], cond_method='ppm_jit',
                 soa_solver='coupled')
Nk, Mk, Gc, bv = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt)

# Full chain with SOA
step = make_step(['so2_chemistry', 'nucleation', 'coagulation',
                  'condensation', 'soa_condensation', 'dilution'],
                 cond_method='ppm_jit', soa_solver='coupled')
Nk, Mk, Gc, bv = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                        org_conc=org_conc, nh3_conc=nh3_conc, fion=fion,
                        oh_conc=1e6, kdil=1e-5)
```

### Custom VBS configuration

```python
from tomas_jax.physics.vbs_config import VBSConfig
import jax.numpy as jnp

# 3-bin VBS for specific use case
cfg = VBSConfig(
    n_bins=3,
    cstar_ref_ug=jnp.array([0.1, 10.0, 1000.0]),
    delta_Hvap_kJ=jnp.array([136.0, 108.0, 80.0]),
    mw=jnp.array([180.0, 200.0, 220.0]),
    sigma=0.030,
    rho=1300.0,
    sv=100.0,
    t_ref=298.0,
    species_indices=(1, 2, 3),
)

step = make_step(['soa_condensation'])
Nk, Mk, Gc = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                    vbs_config=cfg)
```

### Direct function call

```python
from tomas_jax.physics.soa_condensation import soa_condensation_step

Nk, Mk, Gc = soa_condensation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    vbs_config=DEFAULT_VBS_CONFIG,
)
```

## Operator Split Order

SOA condensation runs AFTER H2SO4 condensation (both modify Mk):

1. SO2 chemistry (gas-phase oxidation)
2. Nucleation (new particle formation)
3. Coagulation (particle-particle collisions)
4. Condensation (H2SO4, non-volatile)
5. **SOA condensation** (organics, semi-volatile, VBS)
6. Dilution (entrainment/mixing)

## Known Differences from Fortran soacond.f

### Gas diffusivity (D_g) — fixed in Fortran

The original Fortran `soacond.f` used Chapman-Enskog kinetic theory with a
hard-coded `dorg = 1 nm` collision diameter for ALL organic species, giving
D_g ≈ 3.31×10⁻⁶ m²/s at 288K. This was inconsistent with the H2SO4 path
(`getCondSink.f` → `gasdiff.f` which uses Fuller-Schettler-Giddings).

**Fixed 2026-03-17**: Fortran `soacond.f` now calls `gasdiff(temp,pres,mw,sv,Di)`
with `sv=120` for organics, matching the Python implementation. Both give
D_g ≈ 6.75×10⁻⁶ m²/s at 288K for MW=200.

### Kelvin effect bug — fixed in Fortran

The original Fortran `soacond.f` was missing a factor of 4 in the Kelvin
exponent. Corrected 2026-03-17:
```fortran
! ORIGINAL (WRONG): exp(σ×MW_kg / (R T ρ Dp))
! CORRECTED: Ke = exp(4σ MW_kg / (R T ρ Dp))  — S&P 2016 eqn 9.40
```

### Fuchs correction coefficient — fixed in Fortran

The Zaveri et al. (2014) paper Eq. (14) gives 0.283 in the Fuchs-Sutugin
correction factor. The original Fortran used 0.238 — a transcription error.
**Fixed 2026-03-17**: Fortran now uses 0.283, matching Python. Confirmed by
Fuchs & Sutugin (1971) and Seinfeld & Pandis (2006, Table 12.1).

Note: The H2SO4 path uses a different Fuchs form entirely — the Seinfeld &
Pandis Eq. 11.35 Dahneke approximation: `beta = (1+Kn)/(1+2Kn(1+Kn)/α)`.

### C*(T) in two-film resistance — fixed in Fortran

The original Fortran used the reference C*(298K) in the Kgkk two-film
resistance calculation. **Fixed 2026-03-23**: Now uses temperature-corrected
`cstar_T = cstar(jo) * exp(-ΔHvap/R * (1/T - 1/298))`, matching Python.

### R gas constant — 8.314 in Fortran

The Fortran `soacond.f` uses `R = 8.314` (3 decimal places). Python's
`soa_condensation.py` matches this with `_R_SOA = 8.314` (not the full-precision
8.314462618 from config.py). This is intentional for Fortran compatibility.

## TFL Size-Distribution Sensitivity

The sequential solver uses TFL (Top-hat Flux Limiting) Lagrangian remapping
via `tmcond_jax` to redistribute mass after condensation growth. While total
quantities (N, M, gas) match Fortran to <0.1%, the per-bin size distribution
develops oscillatory artifacts over 24 hours:

- **Root cause**: TFL remapping amplifies ULP-level floating-point differences
  between `jnp.power()` (JAX) and `**` (Fortran). When a particle's grown mass
  falls near a bin boundary, a ~1e-15 relative difference in `dmdt_int` output
  determines whether it stays in the current bin or transfers to the next.
  These discrete remapping decisions compound over 8640 timesteps.

- **Manifestation**: Distribution tails develop secondary peaks that cascade
  inward. At hour 6: bin 15 has a spurious secondary peak (5000%+ error vs
  Fortran). By hour 24: bins 13-14 show ±35-100% errors. The overall shape
  (peak location, total number/mass) remains correct.

- **This is NOT a code bug.** All intermediates (density, Dp, CS, atau, tmcond
  I/O) match Fortran to machine precision (~1e-8%) for the first timestep.
  The divergence is a fundamental sensitivity of the Lagrangian remapping
  algorithm to floating-point implementation details.

- **Recommendation**: Use the coupled solver (`solver='coupled'`) for
  production runs. It is 12× faster, produces smoother distributions (no TFL
  remapping), and conserves mass equally well. The sequential solver remains
  available for direct Fortran comparison of the algorithm.

## Open Questions

1. **Kelvin effect for inorganic H2SO4**: Fortran soacond.f applies Kelvin
   only to organics. The H2SO4 path (ezcond.f) does not include Kelvin. Some
   literature suggests Kelvin matters for H2SO4 on sub-3nm clusters. Currently
   implemented for organics only. Future investigation needed.

2. **Number of VBS bins**: Standard 6-bin VBS may be too coarse for some
   applications. Fortran uses up to 40 organic species. The architecture
   supports arbitrary VBS configurations via `VBSConfig`.

3. **Accommodation coefficient**: Current implementation uses α = 1.0 for
   organics (same as Fortran). Literature suggests α = 0.01–0.1 for
   organic-coated particles. Zaveri et al. (2014) argues α should represent
   only interfacial accommodation (not particle-phase diffusion limitations,
   which are handled separately by D_b).

4. **Bin redistribution**: Three options available via `redistribution=` kwarg:
   `'tfl'` (default, matches Fortran), `'ppm'` (smoother, avoids oscillation),
   and `'direct'` (no transport). PPM follows the H2SO4 ezcond_ppm_jax pattern:
   transport first, then mass addition.

5. **Reaction product tracking**: When k_c > 0, the particle-phase reaction
   loss is applied (volatile species decays), but the nonvolatile reaction
   products are not tracked in a separate species. To conserve total organic
   mass within the particle, the lost mass should be added to a designated
   nonvolatile organic species index.

## Benchmark

`benchmarks/python/compare_soa_solvers.py` generates 9 comparison figures
(2 scenarios × 4 solvers: Fortran, Sequential TFL, Sequential PPM, Coupled):

1. `fig1_sizedist_log.png` — dN/dlogDp (log-log) at 0, 6, 12, 24h
2. `fig2_sizedist_linear.png` — dN/dlogDp (semilog-x) at 0, 6, 12, 24h
3. `fig3_massdist_log.png` — dM/dlogDp (log-log)
4. `fig4_massdist_linear.png` — dM/dlogDp (semilog-x)
5. `fig5_gas_evolution.png` — VBS gas-phase Gc(t) per bin
6. `fig6_vbs_particle.png` — Grouped bar: particle mass per VBS bin
7. `fig7_banana.png` — Banana plots (pcolormesh) per solver
8. `fig8_totals.png` — N_total(t) and M_dry(t) timeseries
9. `fig9_relerror.png` — Per-bin relative error vs Fortran

Results saved in `benchmarks/results/soa_comparison/`.

Run: `python -m benchmarks.python.compare_soa_solvers --run`
Plot only: `python -m benchmarks.python.compare_soa_solvers --plot-only`

## Tests

58 tests in `tests/test_soa_condensation.py`:
- VBS config: C* temperature correction, unit conversions (7 tests)
- Kelvin effect: magnitude, monotonicity, large-particle limit (5 tests)
- Driving force: positive/negative/zero, equilibrium mass (8 tests)
- SOA condensation: mass conservation, pure condensation, evaporation,
  temperature dependence, non-negative mass (9 tests)
- Integration: make_step with SOA, full chain, error handling (3 tests)
- Fortran comparison: intermediates, atau, tmcond, 24h evolution (18 tests)
- PPM redistribution: mass/number conservation, gas depletion, direction match,
  non-negative mass, invalid name raises, backward compat, make_step (8 tests)

## References

- Zaveri et al. (2014), *Modeling kinetic partitioning of secondary organic
  aerosol and size distribution dynamics*, Atmos. Chem. Phys., 14, 5153–5181
  — Framework for gas-side + particle-side mass transfer with D_b and k_c
- Donahue et al. (2006), *A two-dimensional volatility basis set*,
  Environ. Sci. Technol., 40, 2635–2643
- Epstein et al. (2010), *A semiempirical correlation between enthalpy
  of vaporization and saturation concentration*, Environ. Sci. Technol.,
  44, 743–748
- Fuller, Schettler & Giddings (1966), *New method for prediction of binary
  gas-phase diffusion coefficients*, Ind. Eng. Chem., 58, 18–27
- Seinfeld & Pandis (2016), *Atmospheric Chemistry and Physics*, 3rd ed.
  — Kelvin eqn 9.40, Fuchs-Sutugin correction
- soacond.f, Jeff Pierce, September 2010 (TOMAS Fortran reference)
