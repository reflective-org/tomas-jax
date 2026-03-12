# Missing Physics & Chemistry — Gap Analysis

This document catalogs all physics and chemistry not yet implemented in TOMAS-JAX, organized by priority for building a complete atmospheric aerosol box model.

---

## What's Already Implemented

| Process | Status | Notes |
|---------|--------|-------|
| Brownian coagulation (Fuchs correction) | Done | Two JIT solvers (Tsit5 adaptive, forward Euler) |
| H₂SO₄ condensation (TFL + PPM) | Done | JIT-compiled, machine-precision conservation |
| Nucleation (Riccobono 2014 + Dunne 2016) | Done | Adaptive sub-stepping, JIT |
| Nucleation (Zhao 2024, 11 mechanisms) | Done | HNO₃, organic, amine, iodine pathways |
| NH₃/NH₄⁺ equilibrium | Done | Stoichiometric (eznh3eqm.f port) |
| Water uptake | Done | ISORROPIA piecewise polynomial fits |
| MNFIX mass-number correction | Done | Fortran-faithful, analytical multi-bin shift |
| Condensation sink | Done | First-order gas loss rate, bin-resolved |
| 40-bin / 80-bin grids | Done | 1.7nm start, configurable resolution |

---

## Tier 1 — Essential for Realistic Box Model

### 1. SO₂ → H₂SO₄ Gas-Phase Chemistry

**What:** Oxidation of SO₂ by OH to produce H₂SO₄ vapor. Currently TOMAS-JAX uses a constant H₂SO₄ production rate — no actual chemistry.

**Minimum viable implementation:**
```
d[H₂SO₄]/dt = k_OH+SO₂ × [OH] × [SO₂] − CS × [H₂SO₄]
```
where `k_OH+SO₂ ≈ 1.5e-12 cm³/s` at 298K (Atkinson et al.), and OH can be prescribed (diurnal cycle) or computed from a simplified mechanism.

**Fortran reference:** `saprc.f` has full SAPRC-99 (100+ reactions), but a simplified 3-reaction scheme (OH+SO₂, OH+DMS, photolysis) covers most box model needs.

**Effort:** 1–3 days for simplified scheme; 2–4 weeks for full SAPRC port.

---

### 2. VBS Organic Condensation / SOA Formation

**What:** Semi-volatile organic compounds (SVOCs) partition between gas and particle phase based on saturation concentration C*. The Volatility Basis Set (VBS) framework bins organics by C* (0.01, 0.1, 1, 10, 100, 1000 μg/m³).

**Key equations:**
```
ξᵢ = (1 + Cᵢ* / COA)⁻¹                          (partitioning coefficient)
Cᵢ* = Cᵢ*,ref × exp[ΔHvap/R × (1/T - 1/Tref)]   (T-dependence)
COA = Σ ξᵢ × Cᵢ,total                             (iterative: COA depends on ξ)
```

**Stiffness concern:** The partitioning equation is implicit (COA appears on both sides). Fortran uses iterative Newton-Raphson. In JAX, options include:
- `jax.lax.while_loop` Newton iteration (JIT-compatible)
- Implicit ODE solver via diffrax (treats partitioning as a stiff ODE)
- Analytical solution for 1-bin VBS (explicit formula exists)

**Fortran reference:** `soacond.f` (32KB) — VBS with Kelvin effect, equilibrium + kinetic partitioning.

**Effort:** 1–2 weeks.

---

### 3. Kelvin Effect (Curvature Correction)

**What:** Saturation vapor pressure over a curved droplet surface is higher than over a flat surface:
```
Cᵢ*(Dp) = Cᵢ*,flat × exp(4σv / RT Dp)
```
where σ is surface tension (~0.05 N/m for organics) and v is molar volume.

**Why critical:** Without this, nanoparticles (< 10 nm) grow too fast. The Kelvin effect suppresses condensation onto the smallest particles, creating a realistic "valley of death" between nucleation-mode and Aitken-mode particles.

**Impact:** Affects growth rates by 10–100× for 1–5 nm particles. Essential for realistic NPF event simulations.

**Implementation:** Modify the condensation driving force in `ezcond` / `ezcond_ppm_jax` to multiply C* by the Kelvin factor per bin. The factor depends only on bin diameter (known from xk), so it can be precomputed.

**Effort:** 2–3 days.

---

### 4. Equilibrium Partitioning (Implicit Solver)

**What:** At each timestep, determine how much of each semi-volatile species is in the gas vs. particle phase at thermodynamic equilibrium. This is the core of the VBS scheme (#2) but also applies to HNO₃, NH₃, and other inorganic semi-volatiles.

**Stiffness in Fortran:** The Fortran solver uses fixed-point iteration or bisection, which is slow and sometimes fails to converge. JAX alternatives:
- **Newton-Raphson in `lax.while_loop`**: Quadratic convergence, JIT-compatible
- **Implicit diffrax solver**: Treat partitioning as a stiff ODE, use `Kvaerno5` or `TRBDF2`
- **Hybrid kinetic approach**: Solve condensation/evaporation kinetically at fine sub-timesteps instead of assuming equilibrium (avoids the implicit solve entirely)

**The hybrid kinetic approach may be superior in JAX:** Since JIT-compiled condensation is already fast, running kinetic condensation at 1-second substeps may converge to equilibrium naturally without needing an implicit solver. This sidesteps the stiffness problem entirely.

**Effort:** 3–5 days for Newton solver; 1 week for hybrid kinetic approach.

---

### 5. Dilution / Entrainment

**What:** Mixing of the box model air with background (clean or aged) air, representing:
- Boundary layer growth (dilution with free troposphere)
- Chamber ventilation (dilution with clean air)
- Plume dilution (entrainment during transport)

**Equation:**
```
dC/dt = −kdil × (C − Cbg)
```
where `kdil` is the dilution rate [s⁻¹] and `Cbg` is background concentration. Applies to both Nk, Mk (aerosol) and Gc (gases).

**Fortran reference:** `layer_diff.f` — height-dependent layer diffusion.

**Implementation:** Add as a process in `make_step()`. Simple first-order decay toward background state. Can be applied to individual bins or bulk.

**Effort:** 1–2 days.

---

### 6. Dry Deposition / Wall Loss

**What:** Size-dependent removal of particles by:
- **Atmospheric dry deposition**: gravitational settling + Brownian diffusion to surface
- **Chamber wall loss**: diffusion to chamber walls (size-dependent, typically ~10⁻⁴ to 10⁻³ s⁻¹)

**Equations:**
```
dNk/dt = −βk × Nk
dMk/dt = −βk × Mk
```
where `βk` is the size-dependent loss rate. For atmospheric deposition:
```
vdep(Dp) = vg(Dp) + 1/(Ra + Rb + Ra×Rb×vg)    (resistance model)
```
For chamber wall loss:
```
βwall(Dp) = (π²D/L² + vg/L) × ke              (Crump & Seinfeld)
```

**Fortran reference:** `wallloss.f` (6.3KB) — vapor + particle wall losses for chamber simulations.

**Effort:** 3–5 days.

---

## Tier 2 — Full Atmospheric Model

### 7. Emissions (Primary Particles + Gases)

**What:** Size-resolved source terms for:
- Primary sulfate (industrial, volcanic)
- Sea salt (wind-speed dependent, 0.01–10 μm)
- Mineral dust (size-distributed, episodic)
- Primary organic aerosol (biomass burning, cooking)
- Gas-phase: SO₂, DMS, NH₃, VOCs

**Implementation:** Add `'emission'` process to `make_step()`. Each source provides `dNk/dt` and `dMk/dt` per bin.

**Effort:** 1 week (framework + 2–3 source types).

---

### 8. Wet Scavenging

**What:** Removal of aerosol by cloud and rain:
- **In-cloud (nucleation) scavenging**: activated CCN are removed when cloud precipitates
- **Below-cloud (impaction) scavenging**: falling rain drops collect aerosol by inertial impaction and Brownian diffusion
- **Washout ratio**: empirical relationship between aerosol and rain concentrations

**Key equation (below-cloud):**
```
dNk/dt = −Λ(Dp) × Nk
Λ(Dp) = (3/2) × (P/Drain) × E(Dp, Drain)      (collection efficiency)
```

**Effort:** 1–2 weeks.

---

### 9. DMS Chemistry

**What:** Dimethyl sulfide (DMS) oxidation — the primary natural source of SO₂ over oceans:
```
DMS + OH → SO₂ + products     (abstraction pathway)
DMS + OH → DMSO → ... → MSA   (addition pathway)
DMS + NO₃ → SO₂ + products    (nighttime)
```

**Why important:** DMS is the dominant sulfur source in marine environments. Controls baseline CCN concentrations and the "CLAW hypothesis" feedback loop.

**Effort:** 2–3 days (3–5 reactions).

---

### 10. Organic Film / Variable Mass Accommodation

**What:** The accommodation coefficient α controls the probability that a gas molecule hitting a particle sticks. Currently α = 1.0 (every collision sticks). In reality:
- Clean inorganic particles: α ≈ 0.5–1.0
- Organic-coated particles: α ≈ 0.01–0.1 (viscous organic films resist uptake)
- Temperature-dependent: glassy state at low T reduces α further

**Impact:** Reduces H₂SO₄ condensation rate by 10–100× when organic coatings are thick. Critical for Arctic and upper troposphere conditions.

**Implementation:** Make α a function of organic mass fraction per bin: `α(k) = f(Morg(k) / Mtot(k))`.

**Effort:** 1–2 days.

---

### 11. Aqueous-Phase Chemistry (In-Cloud SO₄ Production)

**What:** Sulfate production inside cloud droplets:
```
SO₂(aq) + H₂O₂(aq) → SO₄²⁻ + H₂O     (dominant, pH-independent)
SO₂(aq) + O₃(aq) → SO₄²⁻ + O₂          (important at high pH)
```

**Why important:** Aqueous-phase production often exceeds gas-phase H₂SO₄ production. It adds mass preferentially to CCN-active bins (those that activated into cloud droplets), modifying the size distribution shape.

**Effort:** 1–2 weeks (includes Henry's law, dissociation equilibria, bin-resolved activation).

---

### 12. Surface Tension (Composition-Dependent)

**What:** Surface tension affects both the Kelvin effect (#3) and CCN activation (#13). Currently would use a fixed value. Should depend on:
- Organic mass fraction (organics lower σ from ~0.072 to ~0.03 N/m)
- Surfactant partitioning (Szyszkowski equation)

**Effort:** 2–3 days.

---

## Tier 3 — Extended Capabilities

### 13. CCN Activation (Köhler Theory)

**What:** Predict cloud droplet number concentration from the aerosol size distribution and supersaturation:
```
Sc(Dp) = exp(4Aσ / 3RTDp) / (1 + κDd³/(Dp³ − Dd³))^(1/2)
```
where κ is the hygroscopicity parameter. Particles with Sc < Smax activate into cloud droplets.

**Outputs:** CDNC (cloud droplet number), activated fraction per bin, maximum supersaturation.

**Effort:** 3–5 days.

---

### 14. Ice Nucleation (INP Parameterizations)

**What:** Parameterize the number of ice nucleating particles as a function of temperature, aerosol composition, and supersaturation:
- Immersion freezing (DeMott et al. 2010, 2015)
- Deposition nucleation
- Contact freezing

**Effort:** 1 week.

---

### 15. Optical Properties

**What:** Compute aerosol optical properties from Mie theory:
- Extinction, scattering, absorption coefficients
- Single scattering albedo
- Asymmetry parameter
- Refractive indices from composition (volume mixing rule)

**Can be post-processed:** Does not need to be inside the time loop. Compute from final Nk + Mk.

**Effort:** 3–5 days.

---

### 16. Gravitational Settling

**What:** Size-dependent sedimentation velocity:
```
vg = ρp × Dp² × g × Cc / (18μ)    (Stokes settling)
```
Important for particles > 1 μm (dust, sea salt). The 40-bin grid extends to 17.5 μm, so settling matters for the upper bins.

**Effort:** 1–2 days.

---

## Tier 4 — JAX-Specific Enhancements

These are already documented in `docs/future_features.md`:

| # | Feature | Effort | Unlocks |
|---|---------|--------|---------|
| 1 | Reverse-mode autodiff | 1 day | Parameter fitting, ML hybrids, adjoint sensitivity |
| 2 | GPU/TPU acceleration | 2–3 days | 10–100× speedup for ensembles |
| 3 | Batched vmap | 2 days | 50–1000× parallel scenarios |
| 5 | Learned coagulation surrogate | 1–2 weeks | Amortized O(N²) cost |
| 7 | Adaptive timestepping | 3–5 days | Handle nucleation bursts efficiently |

---

## SOA Solver Strategy — JAX Advantages

The stiffness of organic equilibrium partitioning is a known problem in Fortran TOMAS. JAX offers several strategies that could outperform the Fortran approach:

1. **Kinetic condensation at fine sub-timesteps:** Since JIT-compiled condensation is fast (~0.26s for 1440 steps), running 10–100 kinetic sub-timesteps per operator-split step may converge to equilibrium naturally. No implicit solver needed.

2. **Newton-Raphson in `lax.while_loop`:** Quadratic convergence, JIT-compilable. The Jacobian of the VBS system is analytically available (it's a sum of rational functions).

3. **Implicit ODE via diffrax:** `Kvaerno5` or `TRBDF2` handle stiff systems natively. The VBS partitioning ODE can be solved directly.

4. **Autodiff through the solver:** Once implemented, `jax.grad` can differentiate through the entire SOA partitioning, enabling gradient-based fitting of C* values and ΔHvap to chamber data.

---

## Implementation Roadmap (Suggested Order)

```
Phase 1 — Minimal realistic box model (2–4 weeks)
  ├── #3  Kelvin effect                    (2–3 days)
  ├── #1  SO₂ + OH → H₂SO₄ chemistry      (1–3 days)
  ├── #5  Dilution                         (1–2 days)
  ├── #6  Dry deposition / wall loss       (3–5 days)
  └── #2  VBS organic condensation         (1–2 weeks)

Phase 2 — Organic aerosol complete (2–4 weeks)
  ├── #4  Equilibrium partitioning solver  (3–5 days)
  ├── #10 Variable accommodation coeff     (1–2 days)
  ├── #12 Composition-dependent σ          (2–3 days)
  └── #9  DMS chemistry                    (2–3 days)

Phase 3 — Atmospheric processes (4–6 weeks)
  ├── #7  Emissions                        (1 week)
  ├── #8  Wet scavenging                   (1–2 weeks)
  ├── #11 Aqueous-phase SO₄ production     (1–2 weeks)
  └── #16 Gravitational settling           (1–2 days)

Phase 4 — Cloud and radiation (2–4 weeks)
  ├── #13 CCN activation                   (3–5 days)
  ├── #14 Ice nucleation                   (1 week)
  └── #15 Optical properties               (3–5 days)

Phase 5 — JAX power features (ongoing)
  ├── Reverse-mode autodiff
  ├── GPU/TPU acceleration
  ├── vmap batching
  └── Learned surrogates
```

---

## Fortran Source Reference

Key unported Fortran files in `tomas_fortran/src/` and `TRACER_SOM-TOMAS/src/`:

| File | Size | Physics |
|------|------|---------|
| `saprc.f` | 52 KB | Full SAPRC-99 gas-phase mechanism (100+ reactions) |
| `soacond.f` | 32 KB | VBS organic condensation with Kelvin effect |
| `wallloss.f` | 6.3 KB | Chamber wall loss (vapor + particles) |
| `layer_diff.f` | — | Height-dependent dilution |
| `HOMCOND.f` | — | HOM condensation (alternative to soacond) |

---

*Last updated: 2026-03-12*
