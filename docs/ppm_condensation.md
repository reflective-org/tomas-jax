# PPM Condensation Algorithm

## 1. Introduction

The Piecewise Parabolic Method (PPM) is an alternative to the TFL (Tzivion-Feingold-Levin) semi-Lagrangian scheme for condensational growth in TOMAS. Key advantages:

- **3rd-order accuracy** (vs 1st-order top-hat in TFL)
- **JIT-compilable** via `jax.lax.fori_loop` (no Python for-loops)
- **Less numerical diffusion** — preserves sharper peaks and narrower distributions
- **CFL-based substepping** — automatically adaptive, no mass-fraction heuristic

| Property | TFL (current) | PPM (new) |
|----------|--------------|-----------|
| Scheme | Semi-Lagrangian moving-center | Eulerian flux-form advection |
| Coordinate | Mass space (linear) | Log-mass space (xi = ln m) |
| Accuracy | 1st order (top-hat) | 3rd order (parabolic) |
| Substepping | ezcond nsteps loop | CFL-based automatic (C_max=0.8) |
| JIT | No (Python for-loops) | Yes (`jax.lax.fori_loop`) |
| dmdt_int | Scalar Python | Vectorized JAX |

## 2. Mathematical Formulation

### 2.1 Log-Mass Coordinate

TOMAS uses mass-doubling bins with boundaries `xk[k]`, so the natural coordinate is:

```
xi = ln(m)
Delta_xi = ln(xk[k+1]) - ln(xk[k]) = ln(2)
```

The number density in log-mass space is:

```
n(xi) = dN/dxi
```

with cell-average `n_bar[k] = Nk[k] / Delta_xi`.

### 2.2 PPM Reconstruction

Within each cell, we reconstruct a parabola:

```
q(eta) = q_L + eta * (Delta_q + q_6 * (1 - eta))
```

where `eta = (xi - xi_L) / Delta_xi` in [0, 1] and:
- `Delta_q = q_R - q_L`
- `q_6 = 6 * (q_bar - (q_L + q_R)/2)`

This ensures integral preservation: `integral_0^1 q(eta) d_eta = q_bar`.

**Edge interpolation** (4th-order):
```
q_{i+1/2} = (7/12)(q_i + q_{i+1}) - (1/12)(q_{i-1} + q_{i+2})
```

**Monotonicity limiting** (van Leer-style): edge values are clipped to the range of adjacent cell averages. **Positivity limiting**: if the parabola goes negative inside the cell, it is reset to a constant.

### 2.3 Colella & Woodward (1984) Flux Computation

For each edge with velocity `u`, the Courant number is `C = u * dt_sub / Delta_xi`.

- **u > 0** (growth): flux from left cell, integrate parabola over `[1-|C|, 1]`
- **u < 0** (evaporation): flux from right cell, integrate over `[0, |C|]`

The integral of the parabola `q(eta)` from `a` to `b` is:
```
I = q_L*(b-a) + 0.5*(Delta_q + q_6)*(b^2 - a^2) - (q_6/3)*(b^3 - a^3)
```

Boundary fluxes are set to zero (closed domain).

## 3. Growth Equation

The analytic growth equation (`dmdt_int`) is shared with TFL, from Stevens et al. (1996, JAS 53(7), Eq. A3):

```
m_h2o = (WR - 1) * m0
x = sqrt(max(0, (2/3)*tau + (m0 + m_h2o)^(2/3)))
m_final = x^3 / WR
```

where `tau` is the growth forcing parameter encoding condensation rate and timestep.

## 4. Algorithm Steps

### 4.1 Entry Point: `ppm_condensation_step`

```
Input: Nk, Mk, xk, TAU, cspecies, dt, icomp_nodiag
```

1. **Freeze WR** at step entry: `WR = compute_wr(Mk, icomp_nodiag)`
2. **CFL substep count**: `n_sub = ceil(max_shift / (C_max * Delta_xi))` where `max_shift = max|ln(m_final/xk)|`
3. **For each substep** (via `jax.lax.fori_loop`):
   - Compute edge velocities (`compute_edge_velocity`)
   - PPM reconstruct number density
   - Compute number flux (`advect_number`) and advect Nk
   - Compute analytical dry mass flux (`ppm_mass_flux`) using mass-weighted integrals
   - Transport all 44 species proportionally (`species_flux` + `advect_totals`) — vectorized, no loops
4. **Positivity limiter**: `Nk = max(Nk, 0)`, `Mk = max(Mk, 0)`

### 4.2 Upwind-Consistent Edge Velocity

At each bin edge, we compute the velocity using the donor-side TAU and WR:
- For growth (u > 0): use left bin's TAU/WR
- For evaporation (u < 0): use right bin's TAU/WR

```
u_edge = ln(m_final / m_edge) / dt_sub
```

where `m_final` is clipped to `[xk[0], xk[-1]]` (zero velocity at boundaries).

## 5. Species Transport

### 5.1 Number: PPM Advection (3rd-order)

Number density is reconstructed with PPM and advected using the Colella & Woodward flux computation.

### 5.2 Analytical Mass-Weighted Dry Mass Flux

The total dry mass flux uses exact analytical integrals of `m(η)*n(η)` over PPM departure regions, where:
- `m(η) = m_L * exp(a*η)` is the dry mass at position η within a bin (mass-doubling: 2x variation across bin)
- `n(η) = n_L + b*η - n_6*η²` is the PPM number density parabola

This requires evaluating antiderivatives `∫ η^k * exp(aη) dη` for k=0,1,2, precomputed as `_INV_A`, `_INV_A2`, `_INV_A3` where `a = ln(2)`.

Key functions in `condensation_ppm.py`:
- `_mass_antideriv(eta)` — evaluates antiderivatives at a point
- `_integrate_mass_parabola_right/left()` — mass integrals over right/left departure regions
- `ppm_mass_flux()` — computes `F_M_dry` at each edge

### 5.3 All Species: Vectorized Proportional Transport

All 44 species (including condensing species) are transported proportionally to the analytical dry mass flux:
```
F_M_all = F_M_dry[:, None] * donor_ratios
```

where `donor_ratios = Mk[donor] / M_dry_analytical[donor]` and `M_dry_analytical` comes from `dry_mass_from_ppm_number()` (not tracked mass sums). This ensures exact conservation when Courant number C=1.

The condensed mass for the condensing species is added **after** PPM transport in ezcond, not inside the PPM step.

### 5.4 Why Analytical Normalization Matters

Using `M_dry_analytical` (from the PPM number parabola integral) instead of tracked `M_dry = sum(Mk[:, :ICOMP_NODIAG])` is critical. The PPM number reconstruction and mass-weighted flux are computed from the same parabola, so dividing by the analytical integral gives exact ratio=1 when no growth occurs. Using tracked mass creates a mismatch that causes systematic mass creation and N loss via MNFIX correction.

## 6. Analytical Dry Mass from PPM Number

The total dry mass per bin is computed from the PPM number reconstruction using the exact integral:

```
M_dry[i] = Delta_xi * m_L * (n_L * I0 + b * I1 - n_6 * I2)
```

where:
- `m_L = xk[i]` is the left-edge mass
- `b = (n_R - n_L) + n_6`
- `I0, I1, I2` are precomputed moment integrals of `eta^k * exp(a*eta)` over [0,1]

## 7. ezcond_ppm Driver

Two variants exist:
- `ezcond_ppm.py` — numpy wrapper (slow, `method='ppm'`)
- `ezcond_ppm_jax.py` — pure-JAX, JIT-compilable (`method='ppm_jit'`)

Both follow the same logic:

1. Compute condensation sink and sinkfrac
2. If CS too small: dump mass in first bin
3. Compute TAU for **full mcond** (not subdivided — PPM handles its own CFL substepping)
4. Call `ppm_condensation_step` with `dt=1.0` — transports all species proportionally
5. Add condensed mass to condensing species bins: `Mk[:, spec] += mcond * sinkfrac`
6. MNFIX for mass-number consistency

## 8. Usage

```python
from tomas_jax.solvers.condensation import condensation_step

# Use PPM JIT condensation (recommended for PPM — fast)
Nk, Mk, Gc = condensation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    method='ppm_jit'
)

# Use TFL JIT condensation (recommended for Fortran-matching — fast)
Nk, Mk, Gc = condensation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    method='tfl_jit'
)

# Scan-fused time loops (eliminates Python dispatch overhead):
from tomas_jax.solvers.condensation import run_condensation_scan, run_condensation_scan_tfl
# PPM scan:
Nk, Mk, Gc, N_history = run_condensation_scan(Nk, Mk, Gc, xk, ..., n_steps=1440)
# TFL scan:
Nk, Mk, Gc, N_history = run_condensation_scan_tfl(Nk, Mk, Gc, xk, ..., n_steps=1440)
```

Or from the command line:
```bash
python run_box_model.py --method ppm_jit
```

## 9. References

- Colella, P. & Woodward, P.R. (1984). "The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations." *J. Comput. Phys.*, 54, 174-201.
- Stevens, B., Feingold, G., & Cotton, W.R. (1996). "A critique of one- and two-dimensional models of the boundary layer containing a stratocumulus cloud." *J. Atmos. Sci.*, 53(7), A3.
- Tzivion, S., Feingold, G., & Levin, Z. (1989). "The evolution of raindrop spectra. Part II: Collisional collection/breakup and evaporation in a rainshaft." *J. Atmos. Sci.*, 46(21), 3312-3327.
