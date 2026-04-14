# Literature Review: Stratospheric Background Aerosol

Reference data for designing SAI box model experiment initial conditions.

---

## 1. Deshler et al. (2003) — 30-Year Laramie Size Distribution Record

**Citation:**
Deshler, T., M. E. Hervig, D. J. Hofmann, J. M. Rosen, and J. B. Liley (2003),
Thirty years of in situ stratospheric aerosol size distribution measurements from
Laramie, Wyoming (41°N), using balloon-borne instruments,
*J. Geophys. Res.*, 108(D5), 4167, doi:10.1029/2002JD002514.

### Key Findings

- **30-year record (1971–2001)** of size-resolved aerosol concentrations at Laramie, Wyoming (41°N)
- Balloon-borne optical particle counters (OPC) measuring CN and particles with r >= 0.15 μm in 12 size classes
- Three major volcanic eruptions dominated 20 of the past 30 years: Fuego (1974), El Chichón (1982), Pinatubo (1991)
- **Present background period** began in 1997 — the longest volcanically quiescent period in the record
- Background state is not significantly different from 1979 observations

### Background Aerosol Size Distribution (Figure 5b)

**Measurement date:** 1999-04-20 (990420)
**Location:** Laramie, Wyoming (41°N)
**Altitude:** 20 km, potential temperature 469 K
**Condition:** Volcanically quiescent background

Bimodal lognormal fit (equation 2 in paper):

| Parameter | Mode 1 (nucleation/Aitken) | Mode 2 (accumulation) |
|-----------|---------------------------|----------------------|
| N (cm⁻³) | 7.67 | 0.006 |
| r_median (μm) | 0.069 | 0.21 |
| r_median (nm) | 69 | 210 |
| σ (= ln σ_g) | 0.31 | 0.42 |
| σ_g (geometric std dev) | exp(0.31) = 1.36 | exp(0.42) = 1.52 |

**Note on σ notation:** The paper defines σᵢ in equation (2) as the parameter appearing inside ln²σᵢ, i.e., σᵢ IS the geometric standard deviation. However, Figure 5b lists σ₁ = 0.31 and σ₂ = 0.42, which are < 1 and therefore must be ln(σ_g). Converting: σ_g1 = exp(0.31) ≈ 1.36, σ_g2 = exp(0.42) ≈ 1.52.

### Volcanic Aerosol Size Distribution (Figure 5a)

**Measurement date:** 1993-03-19 (930319) — 21 months post-Pinatubo
**Altitude:** 20 km, potential temperature 487 K

| Parameter | Mode 1 | Mode 2 |
|-----------|--------|--------|
| N (cm⁻³) | 3.99 | 1.63 |
| r_median (μm) | 0.15 | 0.41 |
| σ (= ln σ_g) | 1.06 | 0.49 |

### Surface Area and Moments

- **Background surface area:** ~1.0 (±0.6) μm² cm⁻³ in the 15–25 km layer
- **Volcanic surface area (post-Pinatubo):** 5–10 μm² cm⁻³ (from text, where N₂O₅ hydrolysis saturates)
- Aerosol surface area increases above ~5 μm² cm⁻³ become critical for NOₓ chemistry

### Measurement Uncertainties

| Parameter | Uncertainty |
|-----------|------------|
| Concentration (N) | ±N⁻⁰·⁵ (Poisson), minimum ±10% |
| Sizing (r) | ±10% |
| Lognormal fitting parameters (r_median) | ±30% |
| Distribution width (σ) | ±20% |
| Surface area and volume | ±40% |

### Vertical Structure (Figures 3b, 6, 7)

Background vertical profile at Laramie (1999):

| Altitude (km) | CN (cm⁻³) | r > 0.15 μm (cm⁻³) | r > 0.25 μm (cm⁻³) |
|---------------|-----------|---------------------|---------------------|
| 10–12 (tropopause) | ~300–1000 | ~1–10 | ~0.1–1 |
| 15 | ~10–30 | ~1–5 | ~0.1–0.5 |
| 20 (Junge layer peak) | ~5–15 | ~1–3 | ~0.1–0.3 |
| 25 | ~3–10 | ~0.5–2 | ~0.05–0.2 |
| 30 | ~2–5 | ~0.2–1 | ~0.01–0.1 |

**Note:** CN measures particles with r > 0.01 μm (condensation nuclei).

### Key Quotes

> "The present period, beginning in 1997, represents the longest such quiet period. These midlatitude data suggest that there have not been significant changes in aerosol loading under background conditions over the past 30 years." (§6, ¶40)

> "Present aerosol surface areas are ~1.0 (0.6) μm² cm⁻³ in the 15–20 (20–25) km layer based on in situ measurements." (Abstract)

---

## 2. Norgren, Kalnajs, and Deshler (2024) — STAC Total Aerosol Record

**Citation:**
Norgren, M., L. E. Kalnajs, and T. Deshler (2024),
Measurements of total aerosol concentration in the stratosphere: A new balloon-borne
instrument and a report on the existing measurement record,
*J. Geophys. Res. Atmospheres*, 129, e2024JD040992,
doi:10.1029/2024JD040992.

### Key Findings

- **STAC instrument** (Stratospheric Total Aerosol Counter): CPC measuring particles with r > 10 nm
- **256 flights** from 1989–2023 across 14 global sites
- Total aerosol is dominated by nucleation mode particles (peak near 50 nm radius)
- Extends the Wyoming CN Counter record (1974–2019) with a modernized instrument

### Global Measurement Summary (Table 1)

| Region | Sites | # Flights | Years | Mean ceiling (km) |
|--------|-------|-----------|-------|-------------------|
| Tropics | Teresina, Darwin, Niamey, Hyderabad, Bauru | 9 | 1997–2022 | 24–31 |
| Mid-latitudes | Osan, Boulder, Laramie, Gap, Lauder, Aire-sur-l'Adour | 149 | 1989–2022 | 24–33 |
| Polar | Kiruna, Andoya, McMurdo | 97 | 1989–2023 | 28–31 |

### Stratospheric Total Aerosol Concentrations

#### By altitude and region (from Figures 4, 5, 6, 7)

**Mid-latitudes (Laramie, 41°N) — Figure 7:**

| Altitude (km) | Total aerosol (cm⁻³) | Mixing ratio (mg⁻¹) | Notes |
|---------------|---------------------|---------------------|-------|
| 10–12 (tropopause) | 100–1000+ | 800–1,050 | Peak near LRT |
| 15 | 10–50 | 100–300 | Minimum above LRT |
| 20 | 5–20 | 50–200 | Junge layer region |
| 25 | 5–15 | 100–400 | Increasing mixing ratio |
| 30 | 3–10 | 200–1000 | Increasing with altitude |

**Polar (McMurdo, 78°S) — Figure 6:**

| Altitude (km) | Total aerosol (cm⁻³) | Notes |
|---------------|---------------------|-------|
| 10–15 | 10–100 | Near tropopause |
| 15–20 | 5–50 | Seasonal variation |
| 20–25 | 10–100+ | New particle layers in spring |
| 25–30 | 10–1000 | Meteoric influence |

#### 500 K layer (~21–23 km, Laramie) — 30-year record (Figure 8)

| Period | C_TA (cm⁻³) | C_>150nm (cm⁻³) | C_<150nm/C_>150nm |
|--------|------------|-----------------|-------------------|
| Pre-Pinatubo background | ~4 | ~0.2 | ~20 |
| Post-Pinatubo peak (1991) | ~20 | ~3 | ~5 |
| Current quiescent (2004+) | ~4–6 | ~0.1–0.2 | ~20–30 |

### Mixing Ratio Profiles (key values)

- **Peak total aerosol mixing ratio** just below LRT: 800–2,000 mg⁻¹
- **Tropics:** typically > 2,000 mg⁻¹ in upper troposphere
- **Mid-latitudes:** 800–1,050 mg⁻¹ near tropopause
- **Above Junge layer minimum (~20 km):** mixing ratios increase with altitude
- **Minimum above LRT:** ~70 mg⁻¹ (coincides with Junge layer maximum in larger particles)

### Volcanic Influence (Figure 8)

- Pinatubo (June 1991): C_TA increased by factor of 4, C_>150nm by factor 13–15
- Post-eruption: new small particles (C_<150nm) decreased for ~243 days as they were scavenged by larger particles
- Elevated C_>150nm persisted ~5 years; reduced C_<150nm persisted ~6 years

### Seasonal Variations (Figures 6, 7)

**Laramie (mid-latitudes):**
- Summer/autumn: peak in total aerosol below LRT (wildfire smoke in upper troposphere)
- Spring: enhanced total aerosol mixing ratio between 24–36 km (new particle layer from re-condensing H₂SO₄ transported from Arctic)
- Above LRT: less seasonal variation than below

**McMurdo (polar):**
- Spring: new particle layer near 24 km (~610 K) from homogeneous nucleation
- Winter: increasing mixing ratio with altitude (meteoric smoke, polar vortex subsidence)
- Non-volatile component ~60% throughout stratosphere (meteoric material)

### Measurement Uncertainties

| Source | Magnitude |
|--------|-----------|
| Systematic low bias | up to 25% (size-dependent) |
| Random counting uncertainty | ±10% |
| Lower particle size limit | ~10 nm radius (STAC), ~3 nm (Wyoming CNC for r ≥ 3 nm at P > 50 hPa) |

### Key Quotes

> "Typical total aerosol profiles exhibit a peak in mixing ratio, with values between 800 and 2,000 particles per mg of air (mg⁻¹), just below the lapse rate tropopause (LRT)." (Abstract)

> "Above the tropopause, a minimum in total aerosol mixing ratio of ~70 mg⁻¹ coincides with a maximum in the Junge layer particle concentration." (§4, Conclusion)

> "The total aerosol measurement record, demonstrating that volcanic injections of sulfur have minimal direct impact on the total aerosol concentration, but significantly increase the mass of larger aerosol particles." (Key Points)

---

## 3. Derived Parameters for TOMAS-JAX Experiments

### Recommended Background Initial Conditions

Based on the two papers above, the following parameters are recommended for stratospheric background aerosol in TOMAS-JAX SAI box model simulations.

#### Altitude: 20 km (~55 hPa, ~216 K) — SAI injection altitude

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| **Temperature** | 216 K | Standard atmosphere | Consistent with current run_sai_simulation.py |
| **Pressure** | 5,500 Pa (55 hPa) | Standard atmosphere | Consistent with current run_sai_simulation.py |
| **Total N (cm⁻³)** | 5–15 | Deshler 2003 Fig 5b; Norgren 2024 Fig 7 | Background CN at 20 km |
| **Mode 1 N (cm⁻³)** | 7.67 | Deshler 2003 Fig 5b (990420) | Nucleation/Aitken mode |
| **Mode 1 r_median (nm)** | 69 | Deshler 2003 Fig 5b | = 138 nm diameter |
| **Mode 1 σ_g** | 1.36 | Deshler 2003 Fig 5b | = exp(0.31) |
| **Mode 2 N (cm⁻³)** | 0.006 | Deshler 2003 Fig 5b | Accumulation mode |
| **Mode 2 r_median (nm)** | 210 | Deshler 2003 Fig 5b | = 420 nm diameter |
| **Mode 2 σ_g** | 1.52 | Deshler 2003 Fig 5b | = exp(0.42) |
| **Surface area** | ~1.0 μm² cm⁻³ | Deshler 2003 Abstract | 15–25 km layer |
| **Composition** | Pure sulfate (H₂SO₄/H₂O) | Both papers | >75% sulfuric acid at 20 km |

#### Comparison with Current run_sai_simulation.py

| Parameter | Current value | Literature value | Status |
|-----------|-------------|-----------------|--------|
| N_BG | 5 cm⁻³ | 7.67 cm⁻³ (Mode 1 only) | Low by ~50% |
| GMD_BG | 80 nm (r=40 nm) | 138 nm (r=69 nm) | Low by ~70% |
| GSD_BG | 1.6 | 1.36 | Higher (wider distribution) |
| Composition | Pure SO4 | Pure SO4 | Correct |
| 2nd mode | Not included | 0.006 cm⁻³ at r=210nm | Missing (negligible N) |

**Key differences:**
1. Current simulation uses N_BG = 5 cm⁻³ which is at the low end of the 5–15 cm⁻³ range
2. Current GMD is 80 nm (radius 40 nm), but Deshler shows peak at r = 69 nm (diameter 138 nm) — the current value represents smaller particles than observed
3. Current GSD = 1.6 is wider than Deshler's Mode 1 (σ_g = 1.36)
4. The second mode (accumulation, r=210 nm) has negligible number (0.006 cm⁻³) but carries significant surface area and mass

#### Altitude Sensitivity Cases

For parameter sweeps across different injection altitudes:

| Altitude | Pressure (hPa) | Temp (K) | N_total (cm⁻³) | Notes |
|----------|---------------|----------|----------------|-------|
| 15 km | ~120 | ~217 | 10–50 | Near tropopause, higher N |
| 20 km | ~55 | ~216 | 5–15 | Standard SAI altitude |
| 25 km | ~25 | ~222 | 3–10 | Upper Junge layer |

---

## 4. Additional References

### Kremser et al. (2016) — Stratospheric Aerosol Review

**Citation:**
Kremser, S., L. W. Thomason, L. von Hobe, M. Hermann, M. Deshler, C. Timmreck, et al. (2016),
Stratospheric aerosol — Observations, processes, and impact on climate,
*Rev. Geophys.*, 54(2), 278–335, doi:10.1002/2015RG000511.

Comprehensive review of stratospheric aerosol observations, processes, and climate impact. Key reference for understanding the Junge layer, Brewer-Dobson circulation effects, and the sulfur cycle in the stratosphere.

### Campbell and Deshler (2014) — CN Layer Modeling

**Citation:**
Campbell, P., and T. Deshler (2014),
Condensation nuclei measurements in the midlatitude (1982–2012) and Antarctic (1986–2010) stratosphere between 20 and 35 km,
*J. Geophys. Res. Atmospheres*, 119(1), 137–152, doi:10.1002/2013JD019710.

Documents the springtime new particle layer near 24 km from homogeneous nucleation of H₂SO₄ and H₂O. Important for understanding nucleation events in the stratosphere.

### Kalnajs and Deshler (2022)

**Citation:**
Kalnajs, L. E., and T. Deshler (2022),
A new instrument for balloon-borne in situ aerosol size distribution measurements, the continuation of a 50 year record of stratospheric aerosols measurements,
*J. Geophys. Res. Atmospheres*, 127(24), e2022JD037485, doi:10.1029/2022JD037485.

Describes the modernized OPC instrument used alongside STAC.

### Deshler (2023) — Data Archive

**Citation:**
Deshler, T. (2023),
University of Wyoming stratospheric aerosol measurements,
University of Wyoming, doi:10.15786/c.6379371.v1.

Public archive of all total and size-resolved aerosol data from 256 sounding flights, maintained by the University of Wyoming Libraries. Includes temperature, pressure, and potential temperature profiles.

**Data access:** https://ndacc.larc.nasa.gov/stations/laramie-wy-united-states

---

## 5. Summary for Experiment Design

### What we know well (constraints):
- Background total aerosol at 20 km: **5–15 cm⁻³** (mid-latitudes, quiescent period)
- Dominant mode peaks at **r ≈ 50–70 nm** (diameter 100–140 nm)
- Size distribution is approximately **bimodal lognormal** with a narrow nucleation mode and a very minor accumulation mode
- **Pure sulfate composition** (H₂SO₄/H₂O) dominates at mid-latitudes
- Surface area density: **~1 μm² cm⁻³** (background at 15–25 km)
- After Pinatubo-scale eruptions, total N only increases ~4x but large particle mass increases ~15x

### What is uncertain:
- **H₂SO₄ gas-phase concentration** at 20 km — depends on photochemistry, not directly measured in these papers. Typical literature values: 0.01–1 pptv (~10⁴–10⁶ molec/cm³)
- **Exact σ_g interpretation** — Deshler's notation in figures vs. text may differ; σ_g likely 1.3–1.6
- **Small particle tail** below 10 nm — STAC detection limit is r > 10 nm, particles below this are not counted
- **Seasonal and interannual variability** — ±factor of 2 in total concentration depending on season and volcanic history
- **Nucleation rates** in the stratosphere — controlled by H₂SO₄, H₂O, and temperature; observed new particle events are episodic
