# Radiative Forcing Calculation

## Overview

TOMAS-JAX includes a direct shortwave radiative forcing (RF) module based on the Chylek & Wong (1995) simplified equation, following the implementation in Pierce et al. (2010). Given a TOMAS aerosol size distribution, it computes the top-of-atmosphere radiative forcing from aerosol-radiation interactions (scattering).

## Algorithm

### Pipeline

1. **Mie scattering** (precomputed once per grid): For each TOMAS size bin, compute the scattering efficiency Q_sca, extinction efficiency Q_ext, and asymmetry parameter g using Bohren-Huffman Mie theory.

2. **Upscatter fraction** (precomputed once per grid): For each bin's asymmetry parameter, compute the fraction of scattered light directed back to space using the Wiscombe & Grams (1976) equation 22, averaged over latitudes, months, and hours.

3. **Optical depth**: From the number concentration and scattering cross section:
   ```
   τ_k = (N_k / A) × π × r_k² × Q_sca,k
   ```
   where A is the column area [m²].

4. **Radiative forcing** (Chylek & Wong 1995 / Pierce et al. 2010):
   ```
   RF = -(S₀/4) × T_a² × (1-A) × (1-R)² × 2β × τ
   ```
   where:
   - **S₀** = Solar constant [W/m²]. S₀/4 is global average insolation.
   - **T_a** = Atmospheric transmittance above the aerosol layer. Fraction of sunlight reaching the aerosol after passing through the atmosphere above it. For stratospheric aerosol, T_a = 1.0 (nothing above to attenuate). For tropospheric aerosol, T_a ≈ 0.85.
   - **A** = Cloud fraction. Fraction of sky covered by clouds, which block the aerosol's scattering effect. Only the clear-sky fraction (1-A) contributes to RF.
   - **R** = Clear-sky surface albedo. Over bright surfaces, upscattered light would have been reflected anyway, reducing the net cooling. The (1-R)² term accounts for the double pass through the aerosol layer above a non-black surface.
   - **β** = Upscatter fraction. Fraction of scattered light directed back to space, computed from the Henyey-Greenstein phase function averaged over solar zenith angles (Wiscombe & Grams 1976 eqn 22).
   - **τ** = Scattering optical depth of the aerosol layer.

### Pierce et al. (2010) SI parameters (stratospheric aerosol)

| Parameter | Symbol | Value | Rationale |
|-----------|--------|-------|-----------|
| Solar constant | S₀ | 1370 W/m² | Standard value |
| Atmospheric transmittance | T_a | 1.0 | Stratosphere: no attenuation above |
| Cloud fraction | A | 0.6 | Global average cloud cover |
| Surface albedo (clear-sky) | R | 0.15 | Global average without clouds |
| Refractive index | n | 1.4 + 1e-8i | ~65 wt% H₂SO₄/H₂O solution |

### Key assumptions
- Purely scattering aerosol (negligible absorption for sulfate at visible wavelengths)
- Solar-spectrum-weighted Mie scattering (Planck 5778 K, 300–2500 nm, 30 bands). Single-wavelength (500 nm) overestimates peak RF by ~36%.
- Thin aerosol limit (RF linear in optical depth)
- Global annual average solar geometry, albedo, and upscatter

## Files

| File | Description |
|------|-------------|
| `tomas_jax/physics/bhmie.py` | Bohren-Huffman Mie scattering (numpy) |
| `tomas_jax/physics/radiative_forcing.py` | RF calculation: Mie precomputation, upscatter, optical depth, Chylek & Wong |
| `tests/test_radiative_forcing.py` | 29 tests covering Mie, upscatter, RF, MSE, Pierce Fig 1 |
| `references/radiative-forcing/` | Pierce's original getRF_monthly.py, bhmie.py, and paper |

## Usage

### Basic: RF from a TOMAS state

```python
from tomas_jax.physics.radiative_forcing import (
    precompute_mie_properties, compute_rf
)

# Precompute Mie properties (once per grid)
mie = precompute_mie_properties(state.xk)  # ~10s with upscatter averaging

# Compute RF
# column_area = boxvol [m³] / layer_depth [m]
# e.g., boxvol=1e20 cm³, layer_depth=1000m → column_area=1e11 m²
column_area = float(state.boxvol) * 1e-6 / 1000.0  # [m²]
rf_total, rf_per_bin = compute_rf(state.Nk, mie, column_area)
print(f"RF = {rf_total:.4f} W/m²")
```

### RF efficiency (intensive property)

```python
from tomas_jax.physics.radiative_forcing import compute_rf_efficiency

# RF per unit dry aerosol burden [W/m² per g/m²]
rf_eff = compute_rf_efficiency(state.Nk, state.Mk, mie)
```

### Scattering efficiency vs radius (Pierce Figure 1)

```python
from tomas_jax.physics.radiative_forcing import scattering_efficiency_vs_radius

radii, rf_per_burden = scattering_efficiency_vs_radius()
# Plot: radii [m] vs rf_per_burden [W/m² per kg/m²]
```

### Mass scattering efficiency

```python
from tomas_jax.physics.radiative_forcing import compute_mass_scattering_efficiency

mse, mse_per_bin = compute_mass_scattering_efficiency(state.Nk, state.Mk, mie)
print(f"MSE = {mse:.2f} m²/g")
```

### Latitude-resolved RF

```python
from tomas_jax.physics.radiative_forcing import compute_rf_latitude_resolved

rf_global, rf_by_lat, lats = compute_rf_latitude_resolved(
    state.Nk, mie, column_area
)
```

## H₂SO₄/H₂O Equilibrium Composition

For stratospheric aerosol, particle composition is computed from the Tabazadeh et al. (1997) binary equilibrium parameterization, which gives the H₂SO₄ weight percent as a function of temperature and relative humidity.

```python
from tomas_jax.physics.radiative_forcing import (
    h2so4_equilibrium_wt, h2so4_solution_density
)

wt_pct = h2so4_equilibrium_wt(temp=220, rh_percent=5)   # → 63.8%
rho = h2so4_solution_density(wt_pct)                      # → 1541 kg/m³
```

### Composition at T=220K

| RH (%) | wt% H₂SO₄ | Density (kg/m³) | Peak RF, spectral (W/m² per Mt-S) | Peak RF, 500nm (W/m² per Mt-S) |
|--------|-----------|-----------------|-----------------------------------|-------------------------------|
| 1      | 71.9      | 1633            | 0.58                              | 0.91                          |
| 2      | 68.9      | 1598            | 0.63                              | 0.97                          |
| 5      | 63.8      | 1541            | 0.70                              | 1.09                          |
| 10     | 58.9      | 1487            | 0.79                              | 1.22                          |
| 20     | 52.2      | 1417            | 0.94                              | 1.45                          |

Single-wavelength (500 nm) overestimates peak RF by ~36% compared to full solar-spectrum-weighted Mie calculation. Pierce et al. (2010) used 500 nm with an assumed 75 wt% (peak 0.84). The spectral peak also shifts from ~0.19 μm (500 nm) to ~0.23 μm.

## Physical parameters

| Parameter | Default | Source |
|-----------|---------|--------|
| Solar constant S₀ | 1361 W/m² | Kopp & Lean (2011) |
| Atmospheric transmittance T_atm | 0.85 | Pierce et al. (2010) |
| Surface albedo α | 0.30 | Global average |
| Wavelength λ | 550 nm | Peak solar scattering |
| Refractive index n | 1.4 + 1e-8i | ~65% H₂SO₄/H₂O |
| Particle density ρ | Computed | Tabazadeh (1997) + CRC |

## Column area

The `column_area` parameter connects the box model geometry to optical depth. For a box of volume V [cm³] representing a layer of depth Δz [m]:

```
column_area = V [cm³] × 1e-6 [m³/cm³] / Δz [m]  →  [m²]
```

For stratospheric geoengineering scenarios, typical values:
- Layer depth ~1 km → column_area = boxvol × 1e-9 m²
- For burden-based calculations, use `compute_rf_efficiency()` which is independent of geometry

## References

1. Bohren, C. F. and D. R. Huffman (1983), *Absorption and Scattering of Light by Small Particles*, Wiley-Interscience.
2. Chylek, P. and J. Wong (1995), Effect of absorbing aerosols on global radiation budget, *GRL*, 22, 929-931.
3. Pierce, J. R. et al. (2010), Efficient formation of stratospheric aerosol for climate engineering, *GRL*, 37, L18805, doi:10.1029/2010GL043975.
4. Tabazadeh, A. et al. (1997), A new parameterization of H₂SO₄/H₂O aerosol composition: Atmospheric implications, *GRL*, 24(15), 1931-1934.
5. Wiscombe, W. J. and G. W. Grams (1976), The backscattered fraction in two-stream approximations, *J. Atmos. Sci.*, 33, 2440-2451.
