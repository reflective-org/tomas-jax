# SO2 + OH Gas-Phase Chemistry

## Overview

The SO2 chemistry module implements the gas-phase oxidation of SO2 by OH radicals, the dominant pathway for atmospheric sulfuric acid (H2SO4) production:

```
SO2 + OH (+M) → HOSO2 → ... → H2SO4
```

This replaces the previous constant H2SO4 production rate with a physically-based parameterization using the Sun et al. (2022) Troe formalism.

## Rate Constant: Sun et al. (2022) Method 1

The rate constant k1(T,p) uses the Troe fall-off formula with H2O enhancement from Table 2 of Sun et al. (2022):

### Troe Formula (Eq. 3)

```
k1(T,p) = k0_eff × [M] × k∞ / (k0_eff × [M] + k∞) × F
```

### Parameters

**N2 bath gas:**
- k0_N2 = 3.03×10⁻³¹ × (T/300)⁻⁴·¹⁰ cm⁶ molec⁻² s⁻¹
- k∞ = 2.00×10⁻¹² cm³ molec⁻¹ s⁻¹ (T-independent)
- Fc = 0.58

**H2O enhancement (Eqs. 5-6):**
- k0_H2O = 1.65×10⁻³⁰ × (T/300)⁻⁴·⁹⁰ cm⁶ molec⁻² s⁻¹
- k0_eff = x_N2 × k0_N2 + x_H2O × k0_H2O
- Same Fc = 0.58

**Broadening factor (Eq. 4):**
```
log₁₀(F) = log₁₀(Fc) / (1 + [log₁₀(k0_eff×[M]/k∞) / N]²)
N = 0.75 - 1.27 × log₁₀(Fc)
```

### Reference Value

At 298 K, 1 atm (dry):
- k1 = 1.05×10⁻¹² cm³ molec⁻¹ s⁻¹
- SO2 lifetime at [OH] = 10⁶ molec/cm³: ~11 days

## Implementation

### Module: `tomas_jax/physics/so2_chemistry.py`

4 JIT-compilable functions:

| Function | Purpose |
|----------|---------|
| `calc_k1_so2_oh(temp, pres, rh)` | Troe rate constant with H2O enhancement |
| `calc_solar_zenith_angle(lat, doy, hour_utc, lon)` | cos(SZA) for diurnal cycle |
| `calc_oh_concentration(oh_const, cos_sza, use_diurnal)` | Constant or diurnal OH |
| `so2_oxidation_step(Gc, temp, pres, boxvol, dt, oh_conc, rh)` | One timestep of SO2→H2SO4 |

### Chemistry Step

Pseudo-first-order analytical solution per timestep:
```
SO2(t+dt) = SO2(t) × exp(-k1 × [OH] × dt)
ΔH2SO4 = ΔSO2 × (MW_H2SO4 / MW_SO2)
```

Sulfur is exactly conserved (in moles).

### Gas Array Extension

`Gc` extended from 43 to 44 elements:
- `Gc[0]` = H2SO4 (SO4)
- `Gc[1-42]` = organics, NH4
- `Gc[43]` = SO2 (new, `SRTSO2 = 43`)

### OH Modes

- **Constant** (`use_diurnal=0.0`): OH = oh_const
- **Diurnal** (`use_diurnal=1.0`): OH = oh_const × max(0, cos(SZA))
  - Nighttime: OH = 0 → no SO2 oxidation
  - SZA computed from latitude, longitude, day of year, hour UTC

## Usage

### CLI

```bash
# SO2 chemistry with constant OH
python run_box_model.py --so2-init 5e10 --oh-conc 1e6 --make-step

# With SO2 emissions
python run_box_model.py --so2-init 5e10 --so2-emission 1e7 --oh-conc 1e6 --make-step

# Diurnal OH cycle
python run_box_model.py --so2-init 5e10 --oh-conc 2e6 --oh-diurnal --lat 45 --make-step

# Legacy mode (no SO2 flags = constant H2SO4 prod, backward compatible)
python run_box_model.py
```

### Python API

```python
from tomas_jax.solvers.condensation import make_step
import jax

# With SO2 chemistry
step_fn = jax.jit(make_step(
    processes=['so2_chemistry', 'nucleation', 'coagulation', 'condensation'],
    cond_method='ppm_jit',
))

Nk, Mk, Gc = step_fn(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    oh_conc=1e6, org_conc=1e7, nh3_conc=1e9, fion=3.0,
)
```

### Direct Function Call

```python
from tomas_jax.physics.so2_chemistry import calc_k1_so2_oh, so2_oxidation_step

# Rate constant
k1 = calc_k1_so2_oh(298.0, 101325.0, rh=0.5)

# One timestep
Gc_new = so2_oxidation_step(Gc, temp, pres, boxvol, dt, oh_conc=1e6, rh=0.5)
```

## Operator Splitting Order

With SO2 chemistry enabled:
1. SO2 emissions (add to Gc[SRTSO2])
2. **SO2 + OH → H2SO4** (so2_chemistry)
3. Nucleation
4. Coagulation
5. H2SO4 condensation

## Validation

Validation script reproduces Sun et al. (2022) figures:

```bash
python -m benchmarks.python.validate_so2_chemistry
# Output: benchmarks/results/so2_chemistry/ (6 figures)
```

### Figures

| Figure | Description |
|--------|-------------|
| A | k1 vs [M] at 5 temperatures (cf. Fig. 9) |
| B | k1 at 298K: Sun vs IUPAC vs NASA (cf. Fig. 10) |
| C | N2 vs H2O bath gas fall-off (cf. Fig. 12) |
| D | k1 vs altitude (standard atmosphere) |
| E | SO2 lifetime vs [OH] |
| F | H2O enhancement map (cf. Fig. 13) |

## Tests

28 tests in `tests/test_so2_chemistry.py`:
- Troe rate constant: value range, P/T dependence, H2O enhancement, JIT
- Solar zenith angle: equator/equinox, midnight, polar
- OH concentration: constant vs diurnal modes
- SO2 oxidation: conservation, edge cases, scan compatibility

## Reference

Sun, W., et al.: Kinetics of OH + SO2 + M: The key reaction in atmospheric SO2-to-sulfate conversion. *Atmos. Chem. Phys.*, 22, 4969-4984, https://doi.org/10.5194/acp-22-4969-2022, 2022.

Supplemental information: `references/so2_chemistry/acp-22-4969-2022-supplement.pdf`
