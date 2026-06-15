# Marianna SAI Dilution Case — Plan & Specification

> Status: **in development** on branch `feat/marianna-dilution` (based on `dev`).
> This document is the canonical record of the simulation plan, the chosen
> parameters, the unit conversions, and the caveats / future-run alternatives.

## 1. Context & goal

Run a **10-day** TOMAS-JAX box-model simulation of a concentrated stratospheric
SO₂ plume that dilutes according to an **explicit, analytic volume-expansion
parameterization** `V(t)/V₀`. The plume starts from an observed aerosol size
distribution and includes **ion-induced nucleation**. The objective is to mimic
observed plume evolution and produce diagnostic plots: a dilution-trend signal
(inert tracer / volume), gas-phase SO₂+H₂SO₄ time series, a banana plot, and
size / surface-area / volume distributions at several snapshot times.

All required **physics already exists on `dev`**:
- `tomas_jax/physics/dilution.py` — first-order relaxation `C(t+dt)=Cbg+(C−Cbg)·exp(−kdil·dt)`.
- `tomas_jax/solvers/condensation.py:make_step([... ,'dilution'])` — composable step.
- `tomas_jax/physics/nucleation.py:dunne_nucleation_rate(...)` — ion nucleation via `fion`.

The **experiment harness** (analytic V(t)→kdil, tabulated initial size dist,
banana / size-dist plotting) is built fresh here as
`experimental_case/run_marianna_dilution.py`, reusing the digitize→bin helper
ported from the `feat/experimental-sai-dilution` branch
(`experimental_case/background_aerosol_distribution.py`).

## 2. Initial conditions

| Param | Value | Implementation notes |
|---|---|---|
| Temperature T | **210 K** | `TEMP = 210.0` |
| Pressure P | **55 hPa** | `PRES = 5500.0` Pa |
| Initial volume V₀ | 10 m × 10 m × 30 km = **3.0×10⁶ m³** | Informational. A box model evolves *concentrations*; only the **ratio** `V(t)/V₀` enters (via `kdil`). V₀ matters only for total-mass bookkeeping (see §6). |
| Initial SO₂ | **2.9×10⁹ ppt = 2900 ppm** (literal, as specified) | `Gc[SRTSO2]` seeded from **5.50×10¹⁵ molec/cm³**. ⚠ In V₀=3×10⁶ m³ this is **1.76 t** SO₂ — *not* mass-equivalent to the previous 10 t plume (see §6). |
| OH | **5×10⁵ molec/cm³**, constant | `oh_conc = 5e5` |
| [H₂O] | **4 ppm** | Set `rh` so the model's internal `x_h2o = rh·e_sat(T)/P = 4×10⁻⁶`. SO₂ chemistry derives H₂O from `rh` via the Buck formula. |
| Initial plume H₂SO₄ | **1×10⁵ molec/cm³** | `Gc[SRTSO4]` seed at t=0. (Future runs: consider starting at 0 — see §7.) |
| Ion-pair production | **30 pairs/cm³/s** | `fion = 30.0` (Dunne 2016 ion nucleation). |
| Initial size dist | **red circles (obs, 220–230 ppbv)** | Digitized from the provided graph **in STP**, then converted **STP→ambient (×0.0706)** for the model. Both STP and ambient are plotted for cross-check. |
| NH₃, organics | 0 | `nh3_conc = 0`, `org_conc = 0` (binary nucleation only). |
| Duration | **10 days = 240 h** | Multi-resolution time stepping. |

### STP → ambient conversion
The graph y-axis is `dN/dlogDp (cm⁻³ STP)`. The model runs at ambient
210 K / 55 hPa, which is far less dense than STP, so the **actual ambient**
number concentration is lower:

```
n_air(210 K, 55 hPa)      = 1.897×10¹⁸ molec/cm³
n_air(STP: 273.15 K, 1013 hPa) = 2.687×10¹⁹ molec/cm³   (Loschmidt)
factor = n_ambient / n_STP = 0.0706
dN/dlogDp(ambient) = 0.0706 × dN/dlogDp(STP)
```

## 3. Dilution parameterization V(t)/V₀

Piecewise, continuous at t = 10⁴ s (note `10⁴^0.8 = 1584.9 ≈ 1585`):

```
V(t)/V₀ = t^0.8                                    0 < t ≤ 10⁴ s
V(t)/V₀ = 1585 · exp{ 8.89×10⁻⁹ · (t − 10⁴)^(3/2) }   t > 10⁴ s
```

- The exponent argument is **(t − 10⁴)** (read from `(t − t⁴)` in the source,
  which would be negative/complex; `(t − 10⁴)` gives continuity and growth).
- 10-day growth ≈ **1.77×10⁶×**.
- Per-step dilution rate: `kdil[i] = ln(V(t_i+dt)/V(t_i)) / dt`.
- **Inert tracer** integrated as `tracer ·= exp(−kdil·dt)` (background 0) → tracks
  `V₀/V(t)`, isolating the pure dilution signal.

### Dilution background `Cbg` (what species relax toward)
The plume entrains ambient stratospheric air as it expands:
- `Nk_bg, Mk_bg` = the red-circles distribution (ambient) — entrains background aerosol.
- `Gc_bg[SRTSO4]` = **0** (no background H₂SO₄ — system relaxes to equilibrium).
- `Gc_bg[SRTSO2]` = **0.01 ppb** (≈1.90×10⁷ molec/cm³ at 210 K / 55 hPa).

## 4. Time stepping (multi-resolution)
Captures the fast early expansion (`V/V₀ = t^0.8` rises steeply):
- dt = 1 s   for 0 – 20 min
- dt = 10 s  for 20 min – 4 h
- dt = 60 s  for 4 h – 240 h

## 5. Process step
```python
step_fn = make_step(
    ['so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'],
    cond_method='ppm_jit', nucl_scheme='ricco_dunne')
# per step kwargs: oh_conc=5e5, fion=30, nh3_conc=0, org_conc=0,
#                  kdil=kdil_i, Nk_bg, Mk_bg, Gc_bg, dt=dt_i
```

Ion nucleation (Dunne 2016): `fion` → ion concentration via steady-state
recombination `ionc = sqrt(fion/α_ion)`, feeding binary-ion `Jbi` and
ternary-ion `Jti`. With NH₃ = 0, only **binary-neutral (Jbn)** + **binary-ion
(Jbi)** fire. A standalone pre-check (`check_ion_nucleation.py`) confirms
`Jbi > 0` and `J(fion=30) > J(fion=0)` before the full run.

## 6. Mass-equivalence check (important caveat)

| Interpretation | Concentration | Total SO₂ in V₀ = 3×10⁶ m³ |
|---|---|---|
| **2.9×10⁹ ppt (chosen)** | 2900 ppm = 5.50×10¹⁵ molec/cm³ | **1.76 t** |
| Mass-equivalent to prior 10 t plume | 16 517 ppm = 1.65×10¹⁰ ppt = 3.13×10¹⁶ molec/cm³ | 10.0 t |
| Previous reference case | 10.82 ppm = 1.995×10¹³ molec/cm³ | 10.0 t (in 4.71×10⁹ m³) |

The chosen 2.9×10⁹ ppt is used **as specified**, but it represents only ~0.18×
the previous 10-tonne release because the new plume volume (3×10⁶ m³) is ~1570×
smaller than the previous one (4.71×10⁹ m³).

## 7. Future-run alternatives to explore
- (a) Start the **initial plume H₂SO₄ at 0** instead of 1×10⁵ (let all H₂SO₄ form in-situ).
- (b) Use a **mass-equivalent 10 t** SO₂ release (1.65×10¹⁰ ppt) instead of 2.9×10⁹ ppt.
- (c) Set **background SO₂ = 0** instead of 0.01 ppb.
- (d) Sensitivity to ion-pair rate, OH level, and background aerosol distribution.

## 8. Outputs
1. **Dilution trend** — inert tracer and `V(t)/V₀` vs time (absolute + ratio).
2. **Gas time series** — SO₂ and H₂SO₄ (molec/cm³) vs time, full 240 h.
3. **Banana plot** — dN/dlogDp(Dp, t), log color scale.
4. **Size distributions** at **12 h, 24 h, 48 h, 72 h, 168 h, 240 h**, each as
   **dN/dlogDp, dA/dlogDp (area), dV/dlogDp (volume)**, in **both log-y and linear-y**.
5. **N_total** vs time.
- All time series saved to NPZ for re-plotting (`--plot-only`).

## 9. Files
- `experimental_case/run_marianna_dilution.py` — main script (new).
- `experimental_case/background_aerosol_distribution.py` — digitize→bin helper (ported; red-circles dist added).
- `experimental_case/check_ion_nucleation.py` — ion-nucleation pre-check (new).
- `experimental_case/MARIANNA_DILUTION_NOTES.md` — concise run notes (points to this doc).
- Reused on `dev`: `solvers/condensation.py:make_step`, `physics/dilution.py`,
  `physics/nucleation.py`, `physics/so2_chemistry.py`, `core/config.py`,
  `experimental_case/plume_concentration.py`.

## 10. Usage (once implemented)
```bash
uv run python -m experimental_case.check_ion_nucleation       # pre-check
uv run python -m experimental_case.run_marianna_dilution      # run + plots
uv run python -m experimental_case.run_marianna_dilution --plot-only
```
