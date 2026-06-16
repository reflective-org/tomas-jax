# Marianna SAI Dilution Case — Plan & Specification

> Status: **in development** on branch `feat/marianna-dilution` (based on `dev`).
> This document is the canonical record of the simulation plan, the chosen
> parameters, the unit conversions, and the caveats / future-run alternatives.

## Scenarios

Simulations are a **3 × 5 matrix** = 3 atmospheric **baselines** (B1/B2/B3) ×
5 **dilution regimes** (D1–D5) = 15 runs, generated in
`experimental_case/run_marianna_dilution.py` (`BASELINES` × `DILUTIONS` →
`SCENARIOS`). Each run is a `ScenarioConfig`; outputs go to
`results/marianna/<run_id>/` (NPZ + `figs/`, incl. a `parameters.png` table).
**All runs use the digitized red-circles distribution for both initial and
entrained-background aerosol** (per-baseline background-aerosol specs ignored),
scaled STP→ambient per each baseline's T,P. Duration: **336 h (14 days)**.

### Baselines (only these vary)
| id | name | T[K] | P[hPa] | C0 SO2[ppt] | OH[cm⁻³] | H2O[ppm] | H2SO4₀[cm⁻³] | ion[/cm³/s] |
|----|------|------|--------|-------------|----------|----------|--------------|-------------|
| B1 | Low Lat/High Alt, Clean | 210 | 55  | 2.9e9 | 5e5   | 4   | 1e5 | 30 |
| B2 | High Lat/Low Alt, Clean | 210 | 120 | 1.3e9 | 2e5   | 4   | 5e4 | 40 |
| B3 | Low Lat/High Alt, Geoeng| 213 | 55  | 2.9e9 | 3.5e5 | 5.5 | 4e5 | 30 |

B1 == the original "scenario 1"; **B1-D2 reproduces it exactly**.

### Dilution regimes — V(t)/V0 (shared early `t^0.8` clamped ≥1, 1<t<1e4 s)
| id | late branch | V/V0 @ 10 d |
|----|-------------|-------------|
| D1 Low Kz    | `1585·exp{2.811e-9 (t−1e4)^1.5}` | 1.5e4 |
| D2 Med Kz    | `1585·exp{8.89e-9 (t−1e4)^1.5}`  | 1.8e6 (= scenario 1) |
| D3 High Kz   | `1585·exp{2.811e-8 (t−1e4)^1.5}` | 6.8e12 |
| D4 Burst     | 4-piece (t^0.8; D1-exp to 1.728e5 s; 1906·exp{2.811e-7 …} burst to 2.238e5 s; 4.83e4·exp{2.811e-9 …}) | 2.0e5 |
| D5 Very High | `1585·exp{1.33e-7 (t−1e4)^1.5}`  | 6.1e48 (→ full dilution to bg) |

Dilution is stored as **piecewise segments** (`('power',p)` / `('exp',A,k,t0,q)`)
evaluated by `V_ratio`. **D4-vs-D2 check**: V/V0 matches to 1.000 at 7 days
(the brief's claim), then diverges (D4 → 0.4 % of D2 by 14 d).

### Run
```bash
uv run python -m experimental_case.run_marianna_dilution --scenario B1-D2   # one
uv run python -m experimental_case.run_marianna_dilution --baseline B1       # one row
uv run python -m experimental_case.run_marianna_dilution --all               # all 15 (40-bin)
uv run python -m experimental_case.compare_matrix                            # 3×6 figures (40-bin)

# 80-bin high-resolution set (sqrt2 grid), 11-way parallel, data-only:
printf '%s\n' B1-D1 B1-D2 ... B3-D5 | xargs -P 11 -I{} sh -c \
  'OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \
   uv run python -m experimental_case.run_marianna_dilution --scenario "$1" --nbins 80 --no-plot' _ {}
uv run python -m experimental_case.compare_matrix --nbins 80                 # -> comparison_80bin/
```
Each `compare_matrix` run emits **log-y and linear-y** variants (`--yscale
log|linear|both`, default both; `*_linear.png`). All panels share the x-axis;
y is shared per row (log: 8-decade window; linear: 0→row-max).

### Resolution (40 vs 80 bin)
The matrix is run at both 40 bins (mass-doubling) and **80 bins** (√2 grid,
`make_grid_80bin`; outputs in `<id>_80bin/`). Convergence (B1-D2):
- **Dry mass: converged to 0.1 %** (resolution-independent — good).
- **N_total: 40-bin over-counts by ~1 % (24 h) → ~16 % (240 h)** vs the finer
  80-bin; r_eff differs ~5–7 %. So number/size carry a modest resolution
  sensitivity (40-bin slightly coarse for the coagulation tail); 80-bin is the
  reference. Comparison figures for each resolution: `comparison/` and
  `comparison_80bin/` (`compare_dN/dA/dV/dN_perS.png`).

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
- **Clamp `V/V0 ≥ 1`**: `t^0.8 → 0` as `t → 0`, which is singular (infinite kdil
  on the first step). The plume starts at V0 and only expands, so `V/V0` is
  clamped to ≥ 1; `t^0.8 < 1` only for `t < 1 s`, so the effect is confined to
  the first second.
- **Fast early growth**: `V/V0 ≈ 40` by 100 s, **≈402 by 30 min**, **1585 by
  2.8 h**, then exponential to **≈1.77×10⁶ by 10 days**. Concentrations of inert
  species drop by the same factors (inert tracer ≈ V0/V(t)).
- Per-step dilution rate: `kdil[i] = ln(V(t_i+dt)/V(t_i)) / dt`.
- **Inert tracer** integrated as `tracer ·= exp(−kdil·dt)` (background 0) → tracks
  `V₀/V(t)`, isolating the pure dilution signal.

### Dilution background `Cbg` (what species relax toward)
The plume entrains ambient stratospheric air as it expands:
- `Nk_bg, Mk_bg` = the red-circles distribution (ambient) — entrains background aerosol.
- `Gc_bg[SRTSO4]` = **0** (no background H₂SO₄ — system relaxes to equilibrium).
- `Gc_bg[SRTSO2]` = **0.01 ppb** (≈1.90×10⁷ molec/cm³ at 210 K / 55 hPa).

## 4. Time stepping (multi-resolution)
Per-scenario `dt_schedule` (a `ScenarioConfig` field). Scenario 1 default —
fine head to resolve the early operator-split stress, coarsening as dilution
drops SO₂:
- dt = **0.01 s** for 0 – 2 min   (resolves the sub-second H₂SO₄ transient)
- dt = **0.1 s**  for 2 – 20 min
- dt = **10 s**   for 20 min – 4 h
- dt = **60 s**   for 4 h – 240 h
(38,280 steps, ~50 s wall.)

### Timestep convergence (why this is safe — and why a fine head)
The huge SO₂ makes H₂SO₄ production ≈ 1.8×10⁹ molec/cm³/s, so H₂SO₄ turns over
in **~0.01–0.06 s** — far faster than a 1 s step. A convergence test (dt over
the whole 0–1200 s window at 1 / 0.2 / 0.05 s) showed:
- **N_total, dry mass, and the per-bin size distribution are converged at dt=1 s**
  (0.0 % difference by 12 h); the only transient is a ≤4.5 % per-bin wiggle in
  the **first hour** that coagulation erases by 12 h.
- The instantaneous **peak H₂SO₄ is dt-sensitive** (sampling of the sub-second
  transient): 1.16×10⁸ at dt=1 s → 3.68×10⁸ at dt=0.05 s. It does **not** propagate
  to the particles (nucleation is gas-clamped + adaptively sub-stepped; chemistry
  & condensation are analytic).

The fine head is therefore *conservative insurance* (accurate early-time
diagnostics + robustness for harder scenarios), not a correctness requirement
for the aerosol outputs. **Re-check convergence per scenario** when OH or SO₂
(hence H₂SO₄ production) increase substantially.

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

### Radiative-forcing diagnostics (`rf_efficiency.py`) — choice (c)
Box-honest, geometry-free RF metrics (the box is 0-D = the plume only; absolute
W/m² needs a global-spreading assumption the box does not contain):
- `rf_efficiency.png` — **MSE [m²/g]** and **per-gram forcing efficiency [W/g, all-sky]**
  vs time. These are ratios (scattering / dry mass), so they are
  **dilution-invariant** — pure size-distribution physics. "W/g" = global-mean
  forcing power per gram of aerosol (optically thin). All-sky ≈ −85 to −95 W/g,
  matching Pinatubo (~−89 W/g) — a validation.
- `rf_and_radius.png` — global-mean RF contribution of one plume (× N for a
  campaign), **effective radius** `r_eff = ⟨r³⟩/⟨r²⟩`, and **r_eff / r_opt**
  where `r_opt ≈ 224 nm` (Dp 448 nm) maximizes per-mass forcing (∝ β·Q_sca/r),
  computed **solar-broadband** (matches the literature/codebase ~400-440 nm band).
- `sulfur_normalized.png` — **dilution-normalized** metrics (per total sulfur
  [SO2 + gas H2SO4 + particulate SO4], which is conserved by chemistry and lost
  only to dilution): the gas→particle **conversion fraction** and the **forcing
  per unit sulfur [W/g-S]** (≈ dRF/dS in the optically-thin regime). Unlike the
  per-aerosol efficiency, this rises monotonically (captures conversion).
- **Key results (scenario 1):** (1) `r_eff/r_opt < 1` throughout (peak ~0.63 at
  ~4.6 d) — particles stay **undersized**; forcing is limited by *under-growth*,
  not over-coagulation. (2) Only ~**26 % of injected sulfur converts to aerosol**
  by day 10 (low OH + dilution strip SO2 before it oxidizes); forcing-per-S is
  still rising at day 10.
- **2 wk (336 h) is beyond the 10-day run** — extend `max_hours` to capture it.

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
