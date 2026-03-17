# SOA Condensation with Volatility Basis Set (VBS)

## Overview

TOMAS-JAX implements semi-volatile organic aerosol (SOA) condensation using
the Volatility Basis Set (VBS) framework. The implementation follows soacond.f
(Jeff Pierce, September 2010) but with cleaner modular architecture.

Key features:
- **Kinetic condensation with equilibrium correction** (not pure equilibrium)
- **6-bin VBS** covering C* = 0.01 to 1000 µg/m³ (Donahue et al. 2006)
- **Kelvin effect** for organic species (size-dependent vapor pressure)
- **Bidirectional**: supports both condensation and evaporation
- **Temperature-dependent C*** via Clausius-Clapeyron
- **JIT-compilable**: fully traceable by JAX

## Architecture

Four independent modules + one integration point:

```
tomas_jax/physics/
  vbs_config.py           — VBS bin definitions, C* values, unit conversions
  kelvin_effect.py        — Kelvin correction factor (organics only)
  vbs_driving_force.py    — Ambient pressure, equilibrium pressure, driving force
  soa_condensation.py     — SOA condensation driver (species loop + eq correction)

tomas_jax/solvers/
  condensation.py         — make_step() dispatches 'soa_condensation'
```

## Algorithm

For each VBS species j (Python loop, unrolled at trace time):

1. **C*(T)**: Temperature-correct C* via Clausius-Clapeyron:
   `C*(T) = C*(298K) × exp(-ΔHvap/R × (1/T - 1/298))`

2. **Kelvin factor**: Per-bin vapor pressure correction for surface curvature:
   `Ke(k) = exp(4σ MW / (R T ρ Dp(k)))`

3. **Condensation sink**: Species-specific CS using organic diffusivity/MW:
   `CS_j = 2π Di Σ(Dpk βk Nk) / V`

4. **Driving force**: Per-bin supersaturation/undersaturation:
   `dp(k) = pamb - C* × (Mk_j(k) / Mtot_org(k)) × Ke(k)`

5. **Analytical gas depletion toward equilibrium**:
   ```
   Gc_eq = psat_eff × V × MW / (R T)
   mcond = (Gc - Gc_eq) × (1 - exp(-CS × dt))
   ```
   where `psat_eff = Σ(sinkfrac × psat)` is the sink-weighted equilibrium.

6. **Distribute to bins**: Proportional to dp-weighted sink fraction:
   `mass_to_bin(k) = mcond × sinkfrac(k) × dp(k) / Σ(sinkfrac × dp)`

7. **Equilibrium correction**: Prevents per-bin overshoot using exponential
   relaxation (soacond.f fodc scheme):
   ```
   fodc = |kinetic/equilibrium| / dt
   corrected = equilibrium × (1 - exp(-fodc × dt))
   ```

8. **Conservation**: Clamp total change to available gas/particle mass, scale
   per-bin distribution to match.

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

VBS bins map to TOMAS organic species 1–6 (SRTORG1 + 0..5). Species 7–41
are reserved for future expansion.

## Usage

### With make_step()

```python
from tomas_jax.solvers.condensation import make_step

# H2SO4 + SOA condensation
step = make_step(['condensation', 'soa_condensation'], cond_method='ppm_jit')
Nk, Mk, Gc = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt)

# Full chain with SOA
step = make_step(['so2_chemistry', 'nucleation', 'coagulation',
                  'condensation', 'soa_condensation', 'dilution'],
                 cond_method='ppm_jit')
Nk, Mk, Gc = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
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
   organic-coated particles. Listed as future improvement.

4. **Bin redistribution**: Current implementation adds mass directly without
   PPM/TFL redistribution. MNFIX handles consistency. Full TAU-based
   redistribution could be added for cases with large per-step mass changes.

## Tests

31 tests in `tests/test_soa_condensation.py`:
- VBS config: C* temperature correction, unit conversions (7 tests)
- Kelvin effect: magnitude, monotonicity, large-particle limit (5 tests)
- Driving force: positive/negative/zero, equilibrium mass (8 tests)
- SOA condensation: mass conservation, pure condensation, evaporation,
  temperature dependence, non-negative mass (8 tests)
- Integration: make_step with SOA, full chain, error handling (3 tests)

## References

- Donahue et al. (2006), *A two-dimensional volatility basis set*,
  Environ. Sci. Technol., 40, 2635–2643
- Epstein et al. (2010), *A semiempirical correlation between enthalpy
  of vaporization and saturation concentration*, Environ. Sci. Technol.,
  44, 743–748
- soacond.f, Jeff Pierce, September 2010 (TOMAS Fortran reference)
