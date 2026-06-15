# Marianna Dilution — Run Notes

Concise operational notes for `run_marianna_dilution.py`. Full specification and
rationale: **`docs/marianna_dilution.md`**.

## How to run
```bash
# 1. Confirm ion nucleation is active (must pass before the full run)
uv run python -m experimental_case.check_ion_nucleation

# 2. Confirm the digitized initial size distribution (STP vs ambient)
uv run python -m experimental_case.background_aerosol_distribution --redcircles
#    -> results/redcircles_crosscheck.png
uv run python -m experimental_case.background_aerosol_distribution --figure-style
#    -> results/redcircles_figure_style.png  (Fig. S4 framing, STP + ambient;
#       STP panel overlays directly on the original screenshot)

# 3. Full 10-day run + plots (~25 s)
uv run python -m experimental_case.run_marianna_dilution --scenario 1

# Short test / re-plot
uv run python -m experimental_case.run_marianna_dilution --scenario 1 --hours 12
uv run python -m experimental_case.run_marianna_dilution --scenario 1 --plot-only
```
Outputs: `results/marianna/<scenario_slug>/data.npz` + figures in
`.../figs/`: **parameters** (input table), dilution_trend, gas_timeseries,
banana_dN/dA/dV, sizedist_dN/dA/dV (+ t=0 reference), ntotal.

## Scenarios
Defined in the `SCENARIOS` registry in `run_marianna_dilution.py`. To add one,
append a `ScenarioConfig(id=..., name=..., <overrides>)`; only override fields
that differ from scenario 1's defaults.

| id | name |
|----|------|
| 1 | Low Latitude, High Altitude, Clean Stratosphere |

## Chosen parameters (this run)
| Param | Value | Notes |
|---|---|---|
| T, P | 210 K, 55 hPa | |
| [H2O] | 4 ppm | applied as `rh = 0.0172` (back-solved via Buck) |
| OH | 5×10⁵ molec/cm³ | constant |
| Initial SO2 | 2.9×10⁹ ppt = 2900 ppm = 5.50×10¹⁵ molec/cm³ | **only 1.76 t in V0=3×10⁶ m³** ⚠ |
| Initial H2SO4 | 1×10⁵ molec/cm³ | |
| Ion-pair prod. | fion = 30 pairs/cm³/s | binary-ion nucleation active (ion conc 7410 cm⁻³) |
| Initial aerosol | red-circles obs (220–230 ppbv), STP→ambient | N ≈ 3.06 /cm³ ambient (43.4 /cm³ STP) |
| NH3, organics | 0 | binary nucleation only |
| V(t)/V0 | t^0.8 (clamped ≥1), then 1585·exp{8.89e-9 (t−1e4)^1.5} | grows ×1.77×10⁶ over 10 d |
| Dilution bg | aerosol = red-circles ambient; SO2 = 0.01 ppb; H2SO4 = 0 | |
| Duration | 240 h (10 d), dt = 1/10/60 s | |

## Result summary (240 h run)
- Validation: inert tracer_final = 5.67×10⁻⁷ = 1/V(240h) = V0/V(t) (dilution wiring OK).
- N_total: peaks ~4×10⁶/cm³ (d1, intense nucleation) → 2.4×10⁴/cm³ (d10).
- SO2: 5.5×10¹⁵ → 6.9×10⁹ (dilution + oxidation); H2SO4 gas peaks ~1.4×10⁷ (d4).
- Aerosol grows from nucleation mode (~2 nm) to a ~150–200 nm accumulation mode.

## ⚠ Caveats & future-run alternatives to explore
1. **SO2 mass**: 2.9×10⁹ ppt is used *as specified* but represents only **1.76 t**
   SO2 in V0 = 3×10⁶ m³, **not** mass-equivalent to the previous 10 t / 4.71×10⁹ m³
   plume. For a mass-equivalent 10 t release use **1.65×10¹⁰ ppt** (16 517 ppm =
   3.13×10¹⁶ molec/cm³). See `docs/marianna_dilution.md` §6.
2. **Initial H2SO4 = 0**: this run seeds 1×10⁵; a future run could start at 0 and
   let all H2SO4 form in-situ from SO2 oxidation.
3. **Background SO2 = 0**: this run uses 0.01 ppb; consider 0.
4. **V(t)/V0 reading**: formula 6 read as `(t − 10⁴)` (the source `(t − t⁴)` would be
   complex; `(t − 10⁴)` is continuous at t=10⁴ s). `V/V0` clamped ≥ 1 to remove the
   `t → 0` singularity. Confirm this matches the intended parameterization.
5. **Size-distribution digitization**: red-circles `_DATA3` is an eyeball digitization
   of Fig. S4. Verify against `redcircles_crosscheck.png`; replace `_DATA3` in
   `background_aerosol_distribution.py` if corrections are needed, then re-run.
6. **STP vs ambient**: the figure is in cm⁻³ STP; the model uses ambient
   (×0.0706 at 210 K/55 hPa). Both are plotted in the cross-check.
7. Sensitivity sweeps worth exploring: ion-pair rate, OH level, background aerosol.
