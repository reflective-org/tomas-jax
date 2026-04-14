# Zhao et al. 2024 — 11-Mechanism NPF Scheme

**Reference:** Zhao, B. et al. "Global variability in atmospheric new particle formation mechanisms." *Nature* **631**, 98–105 (2024). https://doi.org/10.1038/s41586-024-07547-1

## Overview

This scheme extends the existing Riccobono 2014 + Dunne 2016 parameterizations with 7 additional mechanisms, for a total of 11 NPF mechanisms. It is implemented in `tomas_jax/physics/nucleation.py` alongside the existing `ricco_dunne` scheme and selectable via `scheme='zhao2024'`.

## Mechanism Summary

| # | Name | Species | Source |
|---|------|---------|--------|
| 1 | H₂SO₄–H₂O neutral | H₂SO₄ | Dunne 2016 |
| 2 | H₂SO₄–NH₃–H₂O neutral | H₂SO₄, NH₃ | Dunne 2016 |
| 3 | H₂SO₄–H₂O ion-induced | H₂SO₄, ions | Dunne 2016 |
| 4 | H₂SO₄–NH₃–H₂O ion-induced | H₂SO₄, NH₃, ions | Dunne 2016 |
| 5 | H₂SO₄–HNO₃–NH₃ synergistic | H₂SO₄, HNO₃, NH₃ | Wang et al. 2022 + Zhao 2024 cutoff |
| 6 | Pure-organic neutral | ULVOC | Kirkby et al. 2016 |
| 7 | Pure-organic ion-induced | ULVOC, ions | Lehtipalo et al. 2018 |
| 8 | Organic–H₂SO₄ | H₂SO₄, ULVOC | Zhao et al. 2024 (revised) |
| 9 | Amine–H₂SO₄ | H₂SO₄, DMA | Almeida et al. 2013 + Kürten + Cai 2021 |
| 10 | Iodine oxoacids neutral | HIO₃ | He et al. 2021 / Zhao 2024 |
| 11 | Iodine oxoacids ion-induced | HIO₃, ions | He et al. 2021 / Zhao 2024 |

Mechanisms 1–4 reuse the existing `dunne_nucleation_rate()` function unchanged.

---

## Mechanism 5: H₂SO₄–HNO₃–NH₃ Synergistic

### Source
Wang, M. et al. "Synergistic HNO₃–H₂SO₄–NH₃ upper tropospheric particle formation." *Nature* **605**, 483–489 (2022).
Reference PDF: `references/nucleation/wang2022_synergistic_HNO3_H2SO4_NH3.pdf`

### Parameterization
From Wang et al. 2022, equation 9 (temperature-dependent fit to CLOUD chamber data at 223 K):

```
J₁.₇ = 2.9 × 10⁻⁹⁸ · exp(14000/T) · [H₂SO₄]³ · [HNO₃]² · [NH₃]⁴
```

- Concentrations in cm⁻³, J in cm⁻³ s⁻¹, T in K
- Exponents fitted from CLOUD experiments: H₂SO₄ power = 3, HNO₃ power = 2, NH₃ power = 4
- Activation energy E/R = 14,000 K (very strong T-dependence — rate becomes negligible above ~250 K)

### Temperature Cutoff (Zhao 2024 addition)
Zhao et al. 2024 (Methods, "NPF module with 11 nucleation mechanisms") states:

> *"...we assumed that the parameterization of Wang et al. should be applied only to temperatures <238 K. At higher temperatures, the NPF rate was set to zero and **smooth transition was implemented near 238 K** to avoid abrupt change."*

The exact functional form is not given in the paper. **Implementation decision:** we use a logistic sigmoid:

```
f_cut(T) = 1 / (1 + exp((T − 238) / 2))
```

Width = **2 K** was chosen as the narrowest smooth option. Sensitivity to the width is small in practice because the Arrhenius term `exp(14000/T)` already suppresses the rate rapidly above 240 K (see figure below).

**Figure:** `docs/figures/nucleation/zhao2024_mech5_synergistic_sigmoid_cutoff.png`
Shows sigmoid shape (panel 1), full J vs T with cutoff (panel 2), and zoom near 238 K (panel 3) for widths of 1, 2, 5, 10 K, using Asian monsoon UT conditions ([H₂SO₄]=4×10⁶, [HNO₃]=1.5×10⁹, [NH₃]=5×10⁸ cm⁻³).

**Conclusion:** Width=2K is effectively indistinguishable from width=1K (hard cutoff) above 242 K, but avoids a discontinuity at the boundary. The choice of width is documented as an approximation not specified in Zhao 2024.

### Full expression implemented
```python
J_syn = 2.9e-98 * exp(14000/T) * [H₂SO₄]³ * [HNO₃]² * [NH₃]⁴
        * (1 / (1 + exp((T − 238) / 2)))
```

---

## Mechanisms 6–7: Pure-Organic Neutral and Ion-Induced

### Sources
- Kirkby, J. et al. "Ion-induced nucleation of pure biogenic particles." *Nature* **533**, 521–526 (2016).
  Reference PDF: `references/nucleation/kirkby2016_pure_organic_ion_induced.pdf`
- Temperature dependence: Zhao et al. 2024 Methods ("NPF module with 11 nucleation mechanisms").

### Parameterization (Kirkby 2016, eq. 4)

```
J_n   = a₁ · c^(a₂ + a₃/c)
J_ion = 2·[n±]·a₄ · c^(a₄ + a₅/c)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| a₁ | 0.04001 | Neutral prefactor |
| a₂ | 1.848 | Neutral power-law exponent |
| a₃ | 0.001366 | Neutral low-concentration correction |
| a₄ | 1.566 | Ion-induced coefficient (appears in both prefactor and exponent) |
| a₅ | 0.1863 | Ion-induced low-concentration correction |
| c | [ULVOC] in **10⁶ cm⁻³** | Input concentration (unit stated in Kirkby 2016) |
| [n±] | ion concentration in **10³ cm⁻³** | Unit NOT stated in Kirkby 2016; back-calculated from Fig. 3 — see backlog note in `docs/future_features.md` |

**Note on a₄:** The parameter a₄ = 1.566 appears both as a multiplicative prefactor and as the leading term of the exponent in J_ion. This is the published empirical form from Kirkby 2016 eq. 4 — not a typo. The constrained functional form uses 2 parameters (a₄, a₅) for the ion-induced rate.

### Verification against Kirkby 2016 Fig. 3 (T = 278 K, [n±] = 700 cm⁻³ for GCR)

| [HOM] (cm⁻³) | J_n (cm⁻³ s⁻¹) | J_gcr (cm⁻³ s⁻¹) |
|-------------|---------------|-----------------|
| 10⁶ | 4.0 × 10⁻² | 2.23 |
| 10⁷ | 2.82 | 87.1 |
| 10⁸ | 199 | 3195 |

These match Fig. 3 of Kirkby 2016 within the stated systematic uncertainty of +80%/−45%.

### Temperature Dependence (Zhao 2024 addition)

Zhao 2024 Methods: temperature-dependence determined from quantum chemistry calculations and the buffering effect of volatility shift with temperature:

```
J_T = J_278K · exp(-(T - 278) / 13)
```

This gives a factor of 2.15× per 10 K temperature decrease (verified: ratio J(268K)/J(278K) = 2.158).

Zhao 2024 also tested a weaker dependence (scale = 20 K, factor of 1.6×/10K) as a sensitivity experiment; the default 13 K scale is used here.

### Input: ULVOC instead of total HOM (Zhao 2024)

The original Kirkby 2016 parameterization uses total HOM as input. Zhao 2024 substitutes **ULVOC** (ultralow-volatility organics, O:C > 0.4, C* < 3×10⁻⁹ μg m⁻³) because:
> *"they have been shown to be better indicators of nucleating organics, especially considering the large range of variations in atmospheric temperature"* (Zhao 2024 Methods)

Zhao 2024 notes they "adjusted the original parameterizations to suit the new inputs," but the specific adjusted constants are not published. **Implementation decision:** use the original Kirkby 2016 constants with `ulvoc` as input, treating ULVOC concentrations as the nucleating-active fraction of HOMs.

### Full expression implemented

```python
c = ulvoc * 1e-6    # [10^6 cm^-3]
n_pm = ionc * 1e-3  # [10^3 cm^-3]
f_T = exp(-(T - 278) / 13)

J_n   = 0.04001 * c^(1.848 + 0.001366/c) * f_T
J_ion = 2 * n_pm * 1.566 * c^(1.566 + 0.1863/c) * f_T
```

Function: `kirkby2016_pure_organic_rate(temp, ulvoc, ionc)` in `tomas_jax/physics/nucleation.py`.

---

## Mechanism 8: Organic–H₂SO₄ Nucleation

### Source
Zhao et al. 2024 Methods, refitted from Lehtipalo et al. 2018 (Sci. Adv. 4, eaau5363) CLOUD data.
Reference PDF: `references/nucleation/lehtipalo2018_pure_organic_neutral.pdf`

### Parameterization

Zhao 2024 tested two functional forms against the Lehtipalo 2018 CLOUD data:
- `J = k[H₂SO₄]^a[ORG]` (R² = 0.61) — rejected
- `J = k[H₂SO₄][ORG]` (R² = 0.80) — **adopted**

The fitted parameterization (bilinear form):
```
J_org-SA = 1.85×10⁻¹⁴ · [H₂SO₄] · [ORG] · exp(-(T - 278) / 13)
```

- [H₂SO₄] and [ORG] in cm⁻³, J in cm⁻³ s⁻¹
- Temperature dependence: same as mechanisms 6–7 (Zhao 2024 quantum chemistry + volatility)

### Input: ULVOC + ELVOC (vs. ULVOC-only for mechanisms 6–7)

Zhao 2024 uses [ORG] = sum of ULVOCs + ELVOCs (C* < 3×10⁻⁵ μg m⁻³, O:C > 0.4) for this mechanism, because:
> *"ULVOCs and ELVOCs largely overlap with non-nitrate HOM dimer at the temperature (278 K) under which the experiments of Lehtipalo et al. were conducted."*

This is a slightly broader class than the ULVOC-only input (C* < 3×10⁻⁹ μg m⁻³) used by mechanisms 6–7. **Implementation approximation:** the same `ulvoc` input is used for all three mechanisms (6, 7, 8). This is noted in `docs/future_features.md` as a motivation for adding explicit ULVOC/ELVOC gas-phase species.

### Spot-check values (T = 278 K, f_T = 1)

| [H₂SO₄] (cm⁻³) | [ULVOC] (cm⁻³) | J (cm⁻³ s⁻¹) |
|-----------------|----------------|--------------|
| 10⁶ | 10⁶ | 1.85×10⁻² |
| 10⁷ | 10⁶ | 1.85×10⁻¹ |
| 10⁷ | 10⁷ | 1.85 |

### Full expression implemented
```python
J_org_sa = 1.85e-14 * [H2SO4] * ulvoc * exp(-(T - 278) / 13)
```

Function: `zhao2024_organic_h2so4_rate(temp, h2so4, ulvoc)` in `tomas_jax/physics/nucleation.py`.

---

## Mechanism 9: Amine–H₂SO₄ Nucleation

### Sources
- Almeida, J. et al. "Molecular understanding of sulphuric acid–amine particle nucleation in the atmosphere." *Nature* **502**, 359–363 (2013).
- Kürten, A. et al. "New particle formation in the sulfuric acid–dimethylamine–water system: reevaluation of CLOUD chamber measurements." *Atmos. Chem. Phys.* **18**, 845–863 (2018). → ×10 correction for self-coagulation.
- Hanson, D. et al. (2017) eq. 11 in Kürten 2018 → explicit formula for collision-controlled rate.
- Cai, R. et al. "Sulfuric acid–amine nucleation in urban Beijing." *Atmos. Chem. Phys.* **21**, 2457–2468 (2021) → temperature-dependence function via Zhao 2024.
- Reference PDFs: `references/nucleation/almeida2013_amine_H2SO4.pdf`, `references/nucleation/kurten2018_amine_correction.pdf`

### Parameterization

**Base rate at 278 K** (Hanson et al. 2017, eq. 11 from Kürten 2018):
```
J_278K = exp(-129 + 16200/278) · [H₂SO₄]³ · [DMA]^1.5
       ≈ 1.92×10⁻³¹ · [H₂SO₄]³ · [DMA]^1.5
```
All concentrations in cm⁻³, J in cm⁻³ s⁻¹. Valid for DMA = 2–16 pptv (5×10⁷–4×10⁸ cm⁻³), [H₂SO₄] = 10⁶–2×10⁷ cm⁻³ at 278 K, 38% RH.

This is equivalent to the Almeida et al. 2013 parameterization ×10 (Kürten correction for self-coagulation), as demonstrated in Kürten 2018 Fig. 1.

**Temperature dependence** (Zhao 2024, from Cai et al. 2021 cluster kinetics):
```
f_T = 1.576·exp(-((T - 250.6)/23.18)²) + 0.6956·exp(-((T - 273.1)/13.01)²)
```

Sum of two Gaussians, not Arrhenius — captures the non-monotonic T-dependence of DMA-H₂SO₄ cluster stability. f_T(278K) = 0.993 ≈ 1 (reference temperature confirmed). Peaks at ~250–260 K (f_T ≈ 1.6), drops sharply above 290 K (clusters unstable), near zero above 310 K.

**Note on the base formula:** The Zhao 2024 main text says "scaling factor of 10 to the parameterization of Almeida et al." but does not give the explicit base-rate formula (it is in their supplementary materials). The Hanson 2017 formula at 278 K gives rates equivalent to Almeida ×10 (demonstrated by Kürten 2018 Fig. 1), so we use it directly.

### Spot-check values (T = 278 K, [DMA] = 10⁹ cm⁻³ ≈ 40 pptv)

| [H₂SO₄] (cm⁻³) | J (cm⁻³ s⁻¹) |
|-----------------|--------------|
| 10⁶ | 6.0 |
| 3×10⁶ | 163 |
| 10⁷ | 6040 |

These match the Kürten-corrected J₁.₇ₙₘ data in Kürten 2018 Fig. 1 (red filled triangles).

### Full expression implemented
```python
J_ref = exp(-129 + 16200/278)  # ≈ 1.92e-31
f_T = 1.576*exp(-((T-250.6)/23.18)²) + 0.6956*exp(-((T-273.1)/13.01)²)
J = J_ref * [H₂SO₄]³ * [DMA]^1.5 * f_T
```

Function: `zhao2024_amine_h2so4_rate(temp, h2so4, dma)` in `tomas_jax/physics/nucleation.py`.

---

## Mechanisms 10–11: Iodine Oxoacids Nucleation

### Source
- He, X.-C. et al. "Role of iodine oxoacids in atmospheric aerosol nucleation." *Science* **371**, 589–595 (2021).
- Zhao et al. 2024 Methods — derived parameterizations by fitting the He et al. 2021 CLOUD data, with Arrhenius temperature-dependence functions.

### Parameterization (Zhao 2024 Methods)

**Neutral (mechanism 10):**
```
J_n = 2.57×10⁻³² · [HIO₃]^4.23 · 1.40×10⁻⁴⁶ · exp(29900/max(T, 263))
```

**Ion-induced (mechanism 11):**
```
J_ion = 1.28×10⁻¹⁸ · [HIO₃]^2.48 · ([ION]/700) · 1.40×10⁻⁴⁶ · exp(29900/max(T, 283))
```

- [HIO₃] in cm⁻³, [ION] = total ion concentration in cm⁻³, T in K
- The temperature factor `1.40×10⁻⁴⁶ · exp(29900/T)` ≈ 1.0 at T ≈ 284 K

### Temperature clamps

From Zhao 2024 Methods:
> *"We did not consider the temperature dependence of ion-induced nucleation below +10°C. The temperature dependence above +10°C was assumed to be the same as that of neutral nucleation."*

| | T clamp | Reason |
|---|---------|--------|
| Neutral | max(T, 263 K) = −10°C | He et al. 2021 data: NPF rate increases minimally below −10°C; neutral clusters already highly stable |
| Ion-induced | max(T, 283 K) = +10°C | Ion-induced nucleation efficient at all T ≤ 283 K; no T-dependence below +10°C |

### Spot-check values ([ION] = 700 cm⁻³)

| T (K) | [HIO₃] (cm⁻³) | J_neutral | J_ion | Total |
|-------|---------------|-----------|-------|-------|
| 273 | 10⁶ | 3.2×10⁻⁵ | 1.0×10⁻³ | 1.1×10⁻³ |
| 273 | 10⁷ | 0.54 | 0.31 | 0.85 |
| 273 | 10⁸ | 9155 | 95 | 9250 |

### Full expression implemented
```python
f_T_n = 1.40e-46 * exp(29900 / max(T, 263))
f_T_i = 1.40e-46 * exp(29900 / max(T, 283))
J_n   = 2.57e-32 * [HIO₃]^4.23 * f_T_n
J_ion = 1.28e-18 * [HIO₃]^2.48 * (ionc/700) * f_T_i
```

Function: `zhao2024_iodine_oxoacid_rate(temp, hio3, ionc)` in `tomas_jax/physics/nucleation.py`.

---

## Input Requirements

The Zhao 2024 scheme requires additional gas-phase precursor inputs beyond the existing `ricco_dunne` scheme:

| Input | Symbol | Units | Used by |
|-------|--------|-------|---------|
| `ulvoc` | ULVOCs/ELVOCs (O:C > 0.4) | molec cm⁻³ | Mech 6, 7, 8 |
| `dma` | Dimethylamine | molec cm⁻³ | Mech 9 |
| `hio3` | Iodic acid HIO₃ | molec cm⁻³ | Mech 10, 11 |
| `hno3` | Nitric acid HNO₃ | molec cm⁻³ | Mech 5 |
| `nh3_conc` | Ammonia NH₃ | molec cm⁻³ | Mech 2, 4, 5 (existing) |
| `fion` | Ion-pair production rate | pairs cm⁻³ s⁻¹ | Mech 3, 4, 7, 11 (existing) |

Note: `ulvoc` and `org_conc` (existing Riccobono input) are kept separate — `org_conc` is only used by the `ricco_dunne` scheme; `ulvoc` is only used by `zhao2024`.

## Per-Mechanism Enable Masks

Each mechanism can be individually enabled/disabled via `enable_masks`, a tuple of 11 floats (0.0 or 1.0):
```
(bn, tn, bi, ti, syn, porg_n, porg_i, org_sa, amine, iod_n, iod_i)
```
Default: all 1.0 (all enabled). Using float masks avoids JAX recompilation.

Predefined constants:
- `ZHAO2024_ALL_ENABLED` — all 11 mechanisms on
- `ZHAO2024_DUNNE_ONLY` — only mechanisms 1–4 (Dunne 2016)

---

## Driver Function

`zhao2024_nucleation_step()` in `tomas_jax/physics/nucleation.py` is the full 11-mechanism driver. It:

1. Converts gas-phase H₂SO₄ from Gc to molec/cm³
2. Computes steady-state ion concentration from `fion` via `_compute_ionc()`
3. Calls all 11 mechanism functions
4. Sums rates with per-mechanism enable masks × `fn_scale`
5. Computes dN, dM_so4 (90%), dM_org (10%) per cluster
6. Applies gas clamping (same logic as existing `nucleation_step`)
7. Updates Nk[0], Mk[0, SO4], Mk[0, org], Gc[SO4]

### Usage

```python
from tomas_jax.physics.nucleation import zhao2024_nucleation_step, ZHAO2024_ALL_ENABLED

Nk_new, Mk_new, Gc_new = zhao2024_nucleation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
    org_conc=0.0, nh3_conc=1e8, fion=3.0,
    hno3=1e8, ulvoc=1e7, dma=1e8, hio3=0.0,
    enable_masks=ZHAO2024_ALL_ENABLED, fn_scale=1.0,
)
```

### Integration with `make_step()`

```python
from tomas_jax.solvers.condensation import make_step

step_fn = make_step(['nucleation', 'coagulation', 'condensation'],
                    nucl_scheme='zhao2024')
Nk, Mk, Gc = step_fn(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                      org_conc=0.0, nh3_conc=1e8, fion=3.0,
                      hno3=1e8, ulvoc=1e7, dma=1e8, hio3=0.0,
                      enable_masks=ZHAO2024_ALL_ENABLED)
```

### CLI

```bash
python run_box_model.py --nucl-scheme zhao2024 --make-step
```
