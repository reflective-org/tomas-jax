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

## Known Differences from Fortran soacond.f

### Gas diffusivity (D_g) — intentional improvement

The Fortran `soacond.f` uses Chapman-Enskog kinetic theory with a hard-coded
`dorg = 1 nm` collision diameter for ALL organic species (lines 171, 186-187):
```fortran
dorg = 10e-10   ! 1 nm — very large for a gas-phase molecule
Di = (2/3)*sqrt(kB*T/pi*0.5*(1/morg+1/mair)) / (pi*(0.5*(dorg+dair))^2) / Na*(R*T/P)
```
This gives D_g ≈ 3.31×10⁻⁶ m²/s at 288K for MW=200.

Our Python uses **Fuller-Schettler-Giddings** (`sv=120`), giving D_g ≈ 6.75×10⁻⁶
m²/s — about **2× larger**. Fuller is the standard method for organic molecules
(Perry's Handbook), using empirical atomic diffusion volumes that capture
molecular structure. The Fortran's 1 nm diameter is unrealistically large
(typical gas-phase organics are 0.5–0.7 nm), artificially halving the
diffusivity.

Notably, the Fortran itself is inconsistent: `getCondSink.f` (H2SO4 path) calls
`gasdiff.f` which uses Fuller (MW=98, SV=42.88), but `soacond.f` (organic path)
switches to Chapman-Enskog. The commented-out line 185 (`Di=diorg(jo)*...` —
"J's diffusivity") suggests this was a known issue.

**Decision**: Keep Fuller for organics. This is physically more appropriate for
non-polar/weakly-polar organic molecules (Van der Waals dominated) vs H2SO4
(highly polar, strong hydrogen bonding, where Lennard-Jones / Chapman-Enskog
with specific parameters is needed).

Reference: Fuller, Schettler & Giddings (1966), *Ind. Eng. Chem.*, 58, 18–27.

### Kelvin effect bug — fixed in Fortran

The original Fortran `soacond.f` was missing a factor of 4 in the Kelvin
exponent. Corrected 2026-03-17:
```fortran
! ORIGINAL (WRONG): exp(σ×MW_kg / (R T ρ Dp))
! CORRECTED: Ke = exp(4σ MW_kg / (R T ρ Dp))  — S&P 2016 eqn 9.40
```

### Fuchs correction coefficient — 0.283 (corrected from Fortran's 0.238)

The Zaveri et al. (2014) paper Eq. (14) gives 0.283 in the Fuchs-Sutugin
correction factor. The Fortran `soacond.f` uses 0.238 — likely a transcription
error. Our Python uses the paper's **0.283**, confirmed by Fuchs & Sutugin
(1971) and Seinfeld & Pandis (2006, Table 12.1). This is a ~16% correction
to the mass transfer coefficient.

Note: The H2SO4 path uses a different Fuchs form entirely — the Seinfeld &
Pandis Eq. 11.35 Dahneke approximation: `beta = (1+Kn)/(1+2Kn(1+Kn)/α)`.

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

4. **Bin redistribution**: Current implementation adds mass directly without
   PPM/TFL redistribution. MNFIX handles consistency. Full TAU-based
   redistribution could be added for cases with large per-step mass changes.

5. **Reaction product tracking**: When k_c > 0, the particle-phase reaction
   loss is applied (volatile species decays), but the nonvolatile reaction
   products are not tracked in a separate species. To conserve total organic
   mass within the particle, the lost mass should be added to a designated
   nonvolatile organic species index.

## Tests

32 tests in `tests/test_soa_condensation.py`:
- VBS config: C* temperature correction, unit conversions (7 tests)
- Kelvin effect: magnitude, monotonicity, large-particle limit (5 tests)
- Driving force: positive/negative/zero, equilibrium mass (8 tests)
- SOA condensation: mass conservation, pure condensation, evaporation,
  temperature dependence, non-negative mass (9 tests)
- Integration: make_step with SOA, full chain, error handling (3 tests)

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
