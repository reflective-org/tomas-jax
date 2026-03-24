# TOMAS-JAX Inputs and Configuration Guide

## Overview

TOMAS-JAX is a box model for aerosol microphysics. A simulation requires:

1. **Initial aerosol state** — size distribution and composition
2. **Environmental conditions** — temperature, pressure, humidity
3. **Gas-phase species** — reactive gases and SOA precursors
4. **Process selection** — which physics to include
5. **Numerical settings** — grid resolution, timestep, solver options

This document describes every configurable input, its units, typical range,
and where it enters the code.

---

## 1. Aerosol Size Distribution

The initial particle population is described by a log-normal distribution.

| Parameter | Symbol | Unit | Typical Range | Code |
|-----------|--------|------|---------------|------|
| Total number concentration | N_total | #/cm³ | 10² – 10⁶ | `create_lognormal_dist(n_total=...)` |
| Geometric mean diameter | GMD | m | 10⁻⁸ – 10⁻⁵ (10 nm – 10 µm) | `create_lognormal_dist(gmd=...)` |
| Geometric standard deviation | GSD | — | 1.2 – 2.5 | `create_lognormal_dist(gsd=...)` |

These produce two arrays per grid cell:
- **Nk** — number per bin [#/grid cell], shape `(NBINS,)`
- **Mk** — mass per bin per species [kg/grid cell], shape `(NBINS, ICOMP)`

Internally, concentrations are **per grid cell** (not per cm³). To convert:
`Nk_per_cell = Nk_per_cm3 × boxvol`.

### Initial Composition

The Mk array has 44 species columns. At initialization, the log-normal
mass is assigned to a single species (typically sulfate). To set initial
composition across species:

| Species | Index | What to set |
|---------|-------|-------------|
| Sulfate (SO4) | `Mk[:, 0]` | Dominant inorganic component |
| Organics (VBS bin 0–5) | `Mk[:, 1:7]` | Initial particle-phase organics per volatility bin |
| Organics (bins 7–41) | `Mk[:, 7:42]` | Reserved for future expansion (typically zero) |
| Ammonium (NH4) | `Mk[:, 42]` | Set by NH3 equilibrium, or initialize directly |
| Water (H2O) | `Mk[:, 43]` | Diagnostic — set by water equilibrium, do not initialize |

**For VBS organics in the particle phase**: yes, you need initial particle-phase
organic mass in addition to gas-phase VBS concentrations. If you start with a
"clean sulfate" distribution, set `Mk[:, 1:7] = 0.0` and let condensation
build up the organic mass. If you want pre-existing SOA, distribute mass
across the 6 VBS species columns (indices 1–6) based on the desired
volatility distribution.

---

## 2. Environmental Conditions

| Parameter | Symbol | Unit | Typical Range | Positional arg in `step_fn` |
|-----------|--------|------|---------------|----------------------------|
| Temperature | T | K | 220 – 320 | `temp` |
| Pressure | P | Pa | 20,000 – 101,325 | `pres` |
| Relative humidity | RH | fraction | 0.05 – 0.95 | `rh` |
| Box volume | V_box | cm³ | typically 10⁶ (= 1 m³) | `boxvol` |
| Accommodation coefficient | α | — | 0.01 – 1.0 (default 1.0) | `alpha` |
| Timestep | dt | s | 1 – 600 (typical 60) | `dt` |

**Notes:**
- Temperature and pressure are currently constant per timestep. For
  time-varying conditions, update between steps in your time loop.
- Box volume changes if dilution is active (volume expansion model).
- The accommodation coefficient applies to both H2SO4 and organic
  condensation. Zaveri et al. (2014) argues α should represent only
  interfacial accommodation (not particle-phase diffusion limitations,
  which are handled separately by D_b in the VBS framework).

---

## 3. Gas-Phase Species

Gas-phase concentrations live in the `Gc` array, shape `(N_GAS_SPECIES,)` =
`(44,)`, in units of **kg per grid cell**.

### 3.1 Species Indices in Gc

| Index | Species | Description |
|-------|---------|-------------|
| 0 | H2SO4 | Sulfuric acid (condensing species) |
| 1 | VBS organic bin 0 | C* = 0.01 µg/m³ (ELVOC) |
| 2 | VBS organic bin 1 | C* = 0.1 µg/m³ (LVOC) |
| 3 | VBS organic bin 2 | C* = 1.0 µg/m³ (LVOC) |
| 4 | VBS organic bin 3 | C* = 10 µg/m³ (SVOC) |
| 5 | VBS organic bin 4 | C* = 100 µg/m³ (SVOC) |
| 6 | VBS organic bin 5 | C* = 1000 µg/m³ (IVOC) |
| 7–42 | Reserved organics | Available for future VBS expansion |
| 43 | SO2 | Sulfur dioxide (oxidized by OH to H2SO4) |

### 3.2 Unit Conversions

Gas concentrations are stored in kg/grid cell. Common conversions:

```
# molec/cm³ → kg/grid cell
Gc_kg = conc_molec_cm3 × boxvol_cm3 × (MW_g_mol / 1000) / AVOGADRO

# µg/m³ → kg/grid cell
Gc_kg = conc_ug_m3 × 1e-9 × boxvol_cm3 × 1e-6

# ppbv → molec/cm³  (then convert to kg/cell)
conc_molec_cm3 = ppbv × 1e-9 × (P / (kB × T))
```

### 3.3 H2SO4

| Parameter | Unit | Typical Range | How to set |
|-----------|------|---------------|------------|
| Initial [H2SO4] | molec/cm³ | 10⁵ – 10¹⁰ | Convert to kg/cell → `Gc[0]` |

H2SO4 is consumed by condensation onto particles and by nucleation.
It is produced from SO2 oxidation (if SO2 chemistry is enabled).

### 3.4 SO2

| Parameter | Unit | Typical Range | How to set |
|-----------|------|---------------|------------|
| Initial [SO2] | molec/cm³ | 10⁸ – 10¹² | Convert to kg/cell → `Gc[43]` |

SO2 is oxidized to H2SO4 by OH. If SO2 = 0 and so2_chemistry is not in the
process list, H2SO4 can still be set directly or via a production rate
(legacy `h2so4_prod` path).

### 3.5 VBS Organic Gases (SOA Precursors)

Each VBS bin has a gas-phase concentration in `Gc[1:7]`. These are
semi-volatile: they condense onto particles when supersaturated and
evaporate when subsaturated, according to the VBS framework.

| VBS Bin | Gc Index | C*(298K) µg/m³ | Volatility Class | Typical Gas Init |
|---------|----------|----------------|------------------|------------------|
| 0 | 1 | 0.01 | ELVOC | 10⁻¹⁵ – 10⁻¹³ kg/cell |
| 1 | 2 | 0.1 | LVOC | 10⁻¹⁵ – 10⁻¹³ kg/cell |
| 2 | 3 | 1.0 | LVOC | 10⁻¹⁵ – 10⁻¹³ kg/cell |
| 3 | 4 | 10.0 | SVOC | 10⁻¹⁵ – 10⁻¹³ kg/cell |
| 4 | 5 | 100.0 | SVOC | 10⁻¹⁵ – 10⁻¹³ kg/cell |
| 5 | 6 | 1000.0 | IVOC | 10⁻¹⁵ – 10⁻¹³ kg/cell |

**Key point**: You should set both gas-phase (`Gc[1:7]`) and particle-phase
(`Mk[:, 1:7]`) initial conditions for the VBS bins. Common setups:

- **Clean sulfate aerosol + organic gas**: `Mk[:, 1:7] = 0`, nonzero `Gc[1:7]`
  → organics condense gradually onto sulfate cores
- **Pre-existing SOA**: Distribute initial organic mass across `Mk[:, 1:7]` bins
  proportional to the desired volatility distribution
- **Gas-particle equilibrium**: Set both gas and particle to be in equilibrium
  at the initial temperature (prevents initial burst of condensation/evaporation)

---

## 4. Nucleation Inputs

Nucleation inputs are passed as **kwargs** to the step function (not stored in
Gc). They represent gas-phase precursor concentrations that drive new particle
formation.

### 4.1 Default Scheme: `ricco_dunne`

Riccobono 2014 (organic) + Dunne 2016 (4 inorganic mechanisms).

| Parameter | Kwarg | Unit | Typical Range | Required? |
|-----------|-------|------|---------------|-----------|
| H2SO4 | — | — | — | Auto: from `Gc[0]` |
| Oxidized organic vapor | `org_conc` | molec/cm³ | 10⁵ – 10⁸ | Yes |
| NH3 (ammonia) | `nh3_conc` | molec/cm³ | 10⁸ – 10¹⁰ | Yes |
| Ion-pair production rate | `fion` | pairs/cm³/s | 2 – 30 | Yes |
| Enable organic nucleation | `enable_organic` | 0.0 or 1.0 | default 1.0 | No |
| Enable inorganic nucleation | `enable_inorganic` | 0.0 or 1.0 | default 1.0 | No |
| Rate scaling factor | `fn_scale` | — | default 1.0 | No |

**What is `org_conc`?** This is a lumped proxy for highly oxygenated molecules
(HOMs) that participate in cluster stabilization. In the Riccobono (2014)
formulation, `J = K × [H2SO4]² × [org]`. It encompasses ELVOCs, ULVOCs, and
HOM dimers. It is a prescribed external input — **not** coupled to the VBS
gas array (see Section 8 for discussion).

**What is `fion`?** The ion-pair production rate from galactic cosmic rays
(GCR). Typical values: ~2 pairs/cm³/s at the surface, ~10–30 at the
tropopause. The Dunne 2016 scheme converts this to steady-state ion
concentration internally.

### 4.2 Extended Scheme: `zhao2024`

Zhao et al. 2024 (11-mechanism scheme). Includes all `ricco_dunne` inputs
plus additional species:

| Parameter | Kwarg | Unit | Typical Range | Mechanisms |
|-----------|-------|------|---------------|------------|
| HNO3 (nitric acid) | `hno3` | molec/cm³ | 10⁷ – 10¹⁰ | #5: synergistic H2SO4-HNO3-NH3 |
| ULVOC | `ulvoc` | molec/cm³ | 10⁵ – 10⁷ | #6-7: pure organic (Kirkby 2016), #8: organic-H2SO4 |
| DMA (dimethylamine) | `dma` | molec/cm³ | 10⁶ – 10⁸ | #9: amine-H2SO4 |
| HIO3 (iodic acid) | `hio3` | molec/cm³ | 10⁵ – 10⁸ | #10-11: iodine oxoacids (coastal/marine) |
| Per-mechanism enables | `enable_masks` | tuple of 11 floats | 0.0 or 1.0 each | All |

The 11 mechanisms in Zhao 2024 are:
1. Binary neutral (H2SO4 + H2O)
2. Ternary neutral (H2SO4 + NH3 + H2O)
3. Binary ion-induced (H2SO4 + ions)
4. Ternary ion-induced (H2SO4 + NH3 + ions)
5. Synergistic HNO3 (H2SO4 + HNO3 + NH3)
6. Pure organic neutral (ULVOC, Kirkby 2016)
7. Pure organic ion-induced (ULVOC + ions)
8. Organic–H2SO4 (ULVOC + H2SO4, Lehtipalo 2018)
9. Amine–H2SO4 (DMA + H2SO4, Cai 2021)
10. Iodine neutral (HIO3, He 2021)
11. Iodine ion-induced (HIO3 + ions)

---

## 5. SO2 Chemistry Inputs

| Parameter | Kwarg | Unit | Typical Range | Default |
|-----------|-------|------|---------------|---------|
| OH concentration | `oh_conc` | molec/cm³ | 10⁵ – 10⁷ | 0.0 (no chemistry) |

SO2+OH oxidation uses the Sun et al. (2022) Troe formalism with H2O
enhancement. The rate constant depends on temperature, pressure, and
relative humidity (all from positional args).

OH can be:
- **Constant**: set `oh_conc` to a fixed value (e.g., 1×10⁶ for global mean)
- **Diurnal**: use the `calc_oh_concentration()` helper which modulates OH
  by cos(solar zenith angle)

---

## 6. Dilution / Entrainment Inputs

| Parameter | Kwarg | Unit | Typical Range | Default |
|-----------|-------|------|---------------|---------|
| Dilution rate | `kdil` | s⁻¹ | 10⁻⁵ – 10⁻³ | 0.0 (no dilution) |
| Background particle number | `Nk_bg_conc` | #/cm³, shape (NBINS,) | — | None (clean air) |
| Background particle mass | `Mk_bg_conc` | kg/cm³, shape (NBINS, ICOMP) | — | None (clean air) |
| Background gas | `Gc_bg_conc` | kg/cm³, shape (N_GAS_SPECIES,) | — | None (clean air) |

The dilution model expands the box volume: `boxvol(t+dt) = boxvol(t) × exp(kdil × dt)`.

**Clean-air dilution** (background = None or zero): total per-cell mass is
conserved; concentrations decrease as 1/V. Represents boundary layer growth
diluting a plume with clean air.

**Ambient entrainment** (nonzero background): the expanded volume brings in
background aerosol and gas. Background concentrations are in **per cm³** units.
Setup example:

```python
# Use ambient air as background
Nk_bg_conc = Nk_ambient / boxvol_ambient   # [#/cm³]
Mk_bg_conc = Mk_ambient / boxvol_ambient   # [kg/cm³]
Gc_bg_conc = Gc_ambient / boxvol_ambient   # [kg/cm³]
```

Typical dilution timescales:
- Volcanic plume: kdil = 10⁻⁴ s⁻¹ (τ ≈ 2.8 h)
- Urban plume: kdil = 10⁻³ s⁻¹ (τ ≈ 17 min)
- Free troposphere: kdil = 10⁻⁵ s⁻¹ (τ ≈ 28 h)

---

## 7. VBS Configuration (Advanced)

The default 6-bin VBS configuration (`DEFAULT_VBS_CONFIG`) is suitable for
most simulations. For advanced use, create a custom `VBSConfig`:

| VBSConfig Field | Unit | Default | Description |
|-----------------|------|---------|-------------|
| `n_bins` | — | 6 | Number of VBS volatility bins |
| `cstar_ref_ug` | µg/m³ | [0.01, 0.1, 1, 10, 100, 1000] | C* at reference temperature |
| `delta_Hvap_kJ` | kJ/mol | [150, 136, 122, 108, 94, 80] | Enthalpy of vaporization |
| `mw` | g/mol | [200, 200, 200, 200, 200, 200] | Molecular weight per bin |
| `sigma` | N/m | 0.025 | Surface tension (Kelvin effect) |
| `rho` | kg/m³ | 1200 | Organic density |
| `sv` | — | 120 | Diffusion volume (Fuller method) |
| `t_ref` | K | 298 | Reference temperature for C* |
| `species_indices` | — | (1, 2, 3, 4, 5, 6) | TOMAS species indices |
| `Dbk` | m²/s | 1×10⁻¹⁰ | Particle-phase bulk diffusion (Zaveri) |
| `kc` | s⁻¹ | 0.0 | Particle-phase reaction rate (Zaveri) |

**D_b controls particle phase state:**
- 10⁻⁵ m²/s = liquid water
- 10⁻¹⁰ m²/s = liquid organic (default)
- 10⁻¹⁵ m²/s = semi-solid
- 10⁻¹⁸ m²/s = glassy

**k_c controls irreversible uptake:**
- 0.0 s⁻¹ = no reaction (reversible partitioning only)
- 10⁻⁴ – 10⁻² s⁻¹ = slow oligomerization
- ≥ 0.01 s⁻¹ = fast reaction (triggers Zaveri Approximation 1)

Pass custom config via: `step_fn(..., vbs_config=my_config)`

---

## 8. Nucleation Organics vs VBS Organics

These are **separate, decoupled inputs** serving different purposes:

| | Nucleation `org_conc` | VBS `Gc[1:7]` |
|-|----------------------|----------------|
| **Purpose** | Drives new particle formation (cluster stabilization) | Drives condensation/evaporation onto existing particles |
| **Species** | Lumped HOMs proxy (ELVOC + ULVOC + HOM dimers) | 6 bins by volatility (C* = 0.01 to 1000 µg/m³) |
| **Consumed?** | Not tracked in Gc; gas depletion computed from formed particle mass | Yes — depleted from Gc by condensation |
| **Kwarg** | `org_conc=...` | Set in `Gc[1:7]` array |

In a real atmosphere, the lowest-volatility VBS bin (C* = 0.01 µg/m³, ELVOC)
overlaps conceptually with `org_conc`. Coupling them (so nucleation depletes
from the VBS gas pool) is a future improvement (see `docs/future_features.md`).

For now, set both independently. Typical approach:
- `org_conc` = total HOM concentration from chamber/field measurements
- `Gc[1:7]` = SOA precursor gas distribution from VBS fits to smog chamber data

---

## 9. Process Selection and Build Options

### 9.1 `make_step()` API

```python
from tomas_jax.solvers.condensation import make_step

step = make_step(
    processes,              # List of process names (order matters!)
    cond_method='ppm_jit',  # Condensation solver
    nucl_scheme='ricco_dunne',  # Nucleation scheme
    n_coag_substeps=10,     # Coagulation substeps
    max_nucleation_frac=0.5,    # Adaptive nucleation: max dN/N_total
    max_nuc_substeps=20,        # Adaptive nucleation: hard cap
)
```

### 9.2 Valid Process Names

Processes execute in the order specified. Recommended order:

```python
processes = [
    'so2_chemistry',       # 1. Gas-phase: SO2 + OH → H2SO4
    'nucleation',          # 2. New particle formation
    'coagulation',         # 3. Particle-particle collisions
    'condensation',        # 4. H2SO4 condensation (non-volatile)
    'soa_condensation',    # 5. Organic condensation/evaporation (VBS)
    'dilution',            # 6. Box volume expansion
]
```

Any subset is valid: `['condensation']`, `['coagulation', 'condensation']`, etc.

### 9.3 Condensation Solvers

| Method | JIT? | Speed | Notes |
|--------|------|-------|-------|
| `'ppm_jit'` | Yes | Fast (default) | PPM advection, smooth redistribution |
| `'tfl_jit'` | Yes | Fast | Top-hat, matches Fortran output exactly |
| `'ppm'` | No | Very slow | NumPy wrapper, for debugging only |
| `'tfl'` | No | Very slow | Sequential Python loops, for debugging only |

### 9.4 Grid Resolution

| Grid | NBINS | Start | Mass Ratio | Size Range |
|------|-------|-------|------------|------------|
| Default | 40 | 1.7 nm | 2.0 | 1.7 nm – 17.5 µm |
| High-res | 80 | 1.7 nm | √2 | 1.7 nm – 17.5 µm |
| Legacy | 36 | 3.2 nm | 2.0 | 3.2 nm – 10 µm |

```python
from tomas_jax.core.config import xk_boundaries, make_grid, make_grid_80bin

xk_40 = xk_boundaries()        # 40-bin default
xk_80 = make_grid_80bin()       # 80-bin high-res
xk_36 = make_grid(36, XK0_LEGACY, 2.0)  # 36-bin legacy (Fortran comparison)
```

---

## 10. Step Function Signature

The function returned by `make_step()` has:

```python
Nk, Mk, Gc, boxvol = step_fn(
    # === Positional (required) ===
    Nk,       # [#/cell], shape (NBINS,)
    Mk,       # [kg/cell], shape (NBINS, ICOMP)
    Gc,       # [kg/cell], shape (N_GAS_SPECIES,)
    xk,       # [kg], shape (NBINS+1,)
    temp,     # [K]
    pres,     # [Pa]
    boxvol,   # [cm³]
    rh,       # [0–1]
    alpha,    # [—]
    dt,       # [s]

    # === Keyword (process-dependent) ===
    # SO2 chemistry
    oh_conc=1e6,           # [molec/cm³]

    # Nucleation (ricco_dunne)
    org_conc=1e7,          # [molec/cm³] — HOM proxy
    nh3_conc=1e9,          # [molec/cm³]
    fion=3.0,              # [pairs/cm³/s]
    enable_organic=1.0,    # 0.0 or 1.0
    enable_inorganic=1.0,  # 0.0 or 1.0
    fn_scale=1.0,          # nucleation rate multiplier

    # Nucleation (zhao2024 extras)
    hno3=0.0,              # [molec/cm³]
    ulvoc=0.0,             # [molec/cm³]
    dma=0.0,               # [molec/cm³]
    hio3=0.0,              # [molec/cm³]
    enable_masks=...,      # tuple of 11 floats

    # SOA condensation
    vbs_config=DEFAULT_VBS_CONFIG,
    soa_redistribution='tfl',  # 'tfl', 'ppm', or 'direct'

    # Dilution
    kdil=1e-4,             # [s⁻¹]
    Nk_bg_conc=None,       # [#/cm³]
    Mk_bg_conc=None,       # [kg/cm³]
    Gc_bg_conc=None,       # [kg/cm³]
)
```

---

## 11. Complete Example

```python
import jax.numpy as jnp
from tomas_jax.core.config import (
    NBINS, ICOMP, N_GAS_SPECIES, SRTSO4, SRTSO2, AVOGADRO,
    xk_boundaries,
)
from tomas_jax.solvers.condensation import make_step
from run_box_model import create_lognormal_dist

# --- Grid ---
xk = xk_boundaries()  # 40-bin, 1.7nm start

# --- Initial size distribution ---
boxvol = 1.0e6  # 1 m³ in cm³
Nk, Mk, _ = create_lognormal_dist(
    nbins=NBINS, n_total=1e4, gmd=50e-9, gsd=1.6,
    mass_min=float(xk[0]), mass_max=float(xk[-1]),
)
Nk = Nk * boxvol   # [#/cell]
Mk = Mk * boxvol   # [kg/cell]

# --- Gas phase ---
Gc = jnp.zeros(N_GAS_SPECIES)

# H2SO4: 1e7 molec/cm³
h2so4_kg = 1e7 * boxvol * (98.0 / 1000.0) / AVOGADRO
Gc = Gc.at[SRTSO4].set(h2so4_kg)

# SO2: 1 ppbv ≈ 2.46e10 molec/cm³ at STP
so2_molec_cm3 = 1e-9 * (101325.0 / (1.38e-23 * 298.0))
so2_kg = so2_molec_cm3 * boxvol * (64.066 / 1000.0) / AVOGADRO
Gc = Gc.at[SRTSO2].set(so2_kg)

# VBS organics: 1 µg/m³ total, distributed across bins
total_org_ug_m3 = 1.0  # µg/m³
vbs_fractions = jnp.array([0.05, 0.10, 0.15, 0.25, 0.25, 0.20])
for j in range(6):
    org_kg = vbs_fractions[j] * total_org_ug_m3 * 1e-9 * boxvol * 1e-6
    Gc = Gc.at[1 + j].set(org_kg)

# --- Build process chain ---
step = make_step(
    ['so2_chemistry', 'nucleation', 'coagulation',
     'condensation', 'soa_condensation', 'dilution'],
    cond_method='ppm_jit',
    nucl_scheme='ricco_dunne',
)

# --- Time loop (24 hours, dt=60s) ---
dt = 60.0
temp, pres, rh, alpha = 288.0, 101325.0, 0.5, 1.0

for i in range(1440):
    Nk, Mk, Gc, boxvol = step(
        Nk, Mk, Gc, xk,
        jnp.float64(temp), jnp.float64(pres),
        jnp.float64(boxvol), jnp.float64(rh),
        jnp.float64(alpha), jnp.float64(dt),
        # SO2 chemistry
        oh_conc=1e6,
        # Nucleation
        org_conc=1e7, nh3_conc=1e9, fion=3.0,
        # Dilution (volcanic plume)
        kdil=1e-4,
    )
```

---

## 12. Summary Table: All Inputs

| Category | Variable | Unit | Required? | Default | Source |
|----------|----------|------|-----------|---------|--------|
| **Size dist** | N_total | #/cm³ | Yes | — | init |
| | GMD | m | Yes | — | init |
| | GSD | — | Yes | — | init |
| | Mk composition | kg/cell | Yes | Pure SO4 | init |
| **Environment** | temp | K | Yes | — | positional |
| | pres | Pa | Yes | — | positional |
| | rh | 0–1 | Yes | — | positional |
| | boxvol | cm³ | Yes | 1×10⁶ | positional |
| | alpha | — | No | 1.0 | positional |
| | dt | s | Yes | — | positional |
| **Gas phase** | Gc[0] H2SO4 | kg/cell | Yes | 0 | Gc array |
| | Gc[1:7] VBS orgs | kg/cell | For SOA | 0 | Gc array |
| | Gc[43] SO2 | kg/cell | For SO2 chem | 0 | Gc array |
| **Nucleation** | org_conc | molec/cm³ | For nucl | — | kwarg |
| | nh3_conc | molec/cm³ | For nucl | — | kwarg |
| | fion | pairs/cm³/s | For nucl | — | kwarg |
| | hno3 | molec/cm³ | zhao2024 | 0 | kwarg |
| | ulvoc | molec/cm³ | zhao2024 | 0 | kwarg |
| | dma | molec/cm³ | zhao2024 | 0 | kwarg |
| | hio3 | molec/cm³ | zhao2024 | 0 | kwarg |
| **SO2 chem** | oh_conc | molec/cm³ | For SO2 | 0 | kwarg |
| **Dilution** | kdil | s⁻¹ | For dilution | 0 | kwarg |
| | Nk_bg_conc | #/cm³ | Optional | None (clean) | kwarg |
| | Mk_bg_conc | kg/cm³ | Optional | None (clean) | kwarg |
| | Gc_bg_conc | kg/cm³ | Optional | None (clean) | kwarg |
| **VBS config** | vbs_config | VBSConfig | No | DEFAULT_VBS_CONFIG | kwarg |
| | soa_redistribution | str | No | 'tfl' | kwarg/make_step |
| **Build opts** | processes | list | Yes | — | make_step |
| | cond_method | str | No | 'ppm_jit' | make_step |
| | nucl_scheme | str | No | 'ricco_dunne' | make_step |
| | n_coag_substeps | int | No | 10 | make_step |
| | max_nucleation_frac | float | No | 0.5 | make_step |
| | max_nuc_substeps | int | No | 20 | make_step |
| **Grid** | xk | kg | Yes | xk_boundaries() | array |
