# Experimental Case: Aircraft SO2 Release

## Scenario

Stratospheric aerosol injection (SAI) scenario where SO2 is released from an aircraft at lower-stratosphere altitude. The plume is modeled as a cylinder whose cross-section is a circle of given radius and whose length per unit time equals the aircraft speed.

### Default Parameters

| Parameter | Value |
|-----------|-------|
| Total SO2 released | 10 tonnes |
| Release rate | 1 ton/min (16.67 kg/s) |
| Aircraft speed | 250 m/s |
| Plume radius (after initial mixing) | 100 m |
| Temperature | 216 K |
| Pressure | 5,500 Pa (55 hPa) |

## Concentration Calculation

Per unit time, the aircraft deposits SO2 into a cylinder of length `v × dt` and cross-section `π × r²`. The mass concentration is:

```
C₀ = (release rate [kg/s]) / (speed [m/s] × π × r² [m²])
```

The default scenario gives:

```
C₀ = 16.67 / (250 × π × 100²) = 2.12 × 10⁻⁶ kg/m³
```

### Default Result

| Units | Value |
|-------|-------|
| µg/m³ | **2,122** |
| molec/cm³ | **2.0 × 10¹³** |
| ppm | **10.8** |
| ppb | **10,816** |

### Derived Plume Geometry (default scenario)

| Parameter | Value |
|-----------|-------|
| Release duration | 10 min |
| Plume length | 150 km |
| Plume cross-sectional area | 31,416 m² |
| Plume volume | 4.71 × 10⁹ m³ |

## Usage

### 1. Run with default parameters

```bash
python experimental_case/plume_concentration.py
```

Output:

```
================================================================
Aircraft SO2 Release — Initial Plume Concentration
================================================================

--- Input Parameters ---
  Release rate       : 1.00 ton/min = 16.67 kg/s
  Total SO2          : 10.0 tonnes
  Aircraft speed     : 250 m/s
  Plume radius       : 100 m
  Plume cross-section: 31,416 m²

  Release duration   : 10.0 min
  Plume length       : 150 km
  Plume volume       : 4.71e+09 m³

--- Ambient Conditions ---
  Temperature        : 216 K
  Pressure           : 5500 Pa (55.0 hPa)
  Air number density : 1.844e+18 molec/cm³

--- SO2 per meter of flight path ---
  0.0667 kg/m = 66.67 g/m

================================================================
  INITIAL SO2 CONCENTRATION
================================================================
  2,122 µg/m³
  1.995e+13 molec/cm³
  10.82 ppm
  10,816 ppb
================================================================
```

### 2. Run with custom parameters

Change any combination of inputs:

```bash
# Smaller release, slower aircraft, tighter plume, slightly higher altitude
python experimental_case/plume_concentration.py \
    --total-so2 5 --release-rate 0.5 --speed 200 --radius 50 \
    --temp 220 --pressure 7000
```

Output:

```
================================================================
  INITIAL SO2 CONCENTRATION
================================================================
  5,305 µg/m³
  4.987e+13 molec/cm³
  21.64 ppm
  21,639 ppb
================================================================
```

### 3. Use as a Python library

```python
from experimental_case.plume_concentration import plume_so2_concentration

# Default stratospheric scenario
result = plume_so2_concentration(
    release_rate_ton_per_min=1.0,
    aircraft_speed_m_per_s=250.0,
    plume_radius_m=100.0,
    temp_K=216.0,
    pres_Pa=5500.0,
    total_so2_ton=10.0,   # optional, for plume geometry
)

print(f"  {result['ug_per_m3']:,.0f} µg/m³")
print(f"  {result['molec_per_cm3']:.2e} molec/cm³")
print(f"  {result['ppm']:.2f} ppm")
```

The returned dict contains all computed quantities:

| Key | Units | Description |
|-----|-------|-------------|
| `ug_per_m3` | µg/m³ | Mass concentration |
| `molec_per_cm3` | molec/cm³ | Number concentration |
| `ppm` | ppm | Volume mixing ratio |
| `ppb` | ppb | Volume mixing ratio |
| `kg_per_m3` | kg/m³ | Mass concentration (SI) |
| `n_air_per_cm3` | molec/cm³ | Air number density |
| `plume_area_m2` | m² | Plume cross-sectional area |
| `mass_per_m_kg` | kg/m | SO2 mass per meter of flight path |
| `release_rate_kg_per_s` | kg/s | Release rate in SI |
| `duration_min` | min | Release duration (if `total_so2_ton` given) |
| `plume_length_km` | km | Total plume length (if `total_so2_ton` given) |
| `plume_volume_m3` | m³ | Total plume volume (if `total_so2_ton` given) |

### CLI Arguments

| Argument | Default | Units | Description |
|----------|---------|-------|-------------|
| `--total-so2` | 10.0 | tonnes | Total SO2 released |
| `--release-rate` | 1.0 | ton/min | Release rate |
| `--speed` | 250.0 | m/s | Aircraft speed |
| `--radius` | 100.0 | m | Plume radius |
| `--temp` | 216.0 | K | Ambient temperature |
| `--pressure` | 5500.0 | Pa | Ambient pressure |

## Dilution Rates

Two dilution scenarios based on plume volume growth over 24 hours:

| Scenario | Volume growth | kdil (s⁻¹) | τ (hours) |
|----------|--------------|------------|-----------|
| Volume doubles in 24h | 2× | ln(2)/86400 = 8.02 × 10⁻⁶ | 34.7 |
| Volume +78% in 24h | 1.78× | ln(1.78)/86400 = 6.68 × 10⁻⁶ | 41.6 |

Where τ = 1/kdil is the e-folding time — the time for the concentration anomaly (C − C_bg) to decay to 37% of its initial value.

## 96h Box Model Simulation

`run_sai_simulation.py` runs a 4-day (96h) box model with all processes (SO2 chemistry, nucleation, coagulation, condensation, dilution) starting from the plume SO2 concentration above.

### Initial Conditions

| Parameter | Value |
|-----------|-------|
| Duration | 96h (5760 steps at dt=60s) |
| Temperature | 216 K |
| Pressure | 5,500 Pa (55 hPa) |
| Relative humidity | 0.01 |
| Initial SO2 | ~2.0 × 10¹³ molec/cm³ (from plume calculator) |
| Initial H2SO4 | 1 × 10⁵ molec/cm³ |
| Background aerosol | 5 #/cm³ (80nm, GSD=1.6) |
| OH concentration | 1 × 10⁶ molec/cm³ |

### Three Cases

| Case | kdil (s⁻¹) | Description |
|------|-----------|-------------|
| `baseline` | 0.0 | No dilution (closed box) |
| `vol_2x` | ln(2)/86400 = 8.02e-6 | Volume doubles in 24h, τ ≈ 34.7h |
| `vol_178x` | ln(1.78)/86400 = 6.68e-6 | Volume +78% in 24h, τ ≈ 41.6h |

### Usage

```bash
# Run simulation + generate plots (~45s total)
python -m experimental_case.run_sai_simulation

# Regenerate plots from existing data
python -m experimental_case.run_sai_simulation --plot-only
```

### Output

NPZ files and 6 PNG figures in `experimental_case/results/`:

| Figure | Description |
|--------|-------------|
| `fig1_ntotal.png` | N_total vs time (absolute + ratio with passive tracer) |
| `fig2_mdry.png` | M_dry vs time (absolute + ratio) |
| `fig3_dndlogdp.png` | dN/dlogDp snapshots at 12h, 24h, 48h, 72h, 96h |
| `fig4_dmdlogdp.png` | dM/dlogDp snapshots at same times |
| `fig5_banana.png` | Banana plots (one panel per case) |
| `fig6_gas.png` | SO2 + H2SO4 gas-phase timeseries |

## Files

| File | Description |
|------|-------------|
| `plume_concentration.py` | Concentration calculator (function-based + CLI) |
| `run_sai_simulation.py` | 96h SAI box model simulation (3 cases, 6 figures) |
| `results/` | Output directory (NPZ data + PNG figures) |
| `literature_review.md` | Stratospheric background aerosol literature review (Deshler 2003, Norgren 2024) |
| `README.md` | This document |
