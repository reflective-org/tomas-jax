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
   - Compute number flux and advect Nk
   - Advect passive species with consistent transport
   - Compute total dry mass from updated Nk (exact integral)
   - Set condensing species by closure
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

### 5.2 Passive Species: Consistent Transport

Non-condensing species mass moves with particles:
```
F_M_j[edge] = F_N[edge] * (M_j / N)[donor]
```

where the donor cell is determined by the velocity direction.

### 5.3 Condensing Species: Closure

```
M_cs = M_dry_total - sum(M_passive_dry)
```

where `M_dry_total` is computed from the exact integral of the PPM number reconstruction times the exponential mass coordinate.

### 5.4 Water/Diagnostics

Water and diagnostic species are transported passively (move with particles).

## 6. Dry Mass Closure

The total dry mass per bin is computed from the PPM number reconstruction using the exact integral:

```
M_dry[i] = Delta_xi * m_L * (n_L * I0 + b * I1 - n_6 * I2)
```

where:
- `m_L = xk[i]` is the left-edge mass
- `b = (n_R - n_L) + n_6`
- `I0, I1, I2` are precomputed moment integrals of `eta^k * exp(a*eta)` over [0,1]

## 7. ezcond_ppm Driver

The PPM-aware ezcond driver (`ezcond_ppm.py`) mirrors the TFL ezcond driver:

1. MNFIX input
2. Compute condensation sink and sinkfrac
3. If CS too small: dump mass in first bin
4. Compute TAU for **full mcond** (not subdivided by nsteps — PPM handles its own CFL substepping)
5. Call `ppm_condensation_step` with `dt=1.0`
6. Mass conservation correction (ratio-based rescaling)

## 8. Usage

```python
from tomas_jax.solvers.condensation import condensation_step

# Use PPM condensation
Nk, Mk, Gc = condensation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    method='ppm'
)

# Use TFL condensation (default)
Nk, Mk, Gc = condensation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    method='tfl'
)
```

Or from the command line:
```bash
python run_box_model.py --method ppm
```

## 9. References

- Colella, P. & Woodward, P.R. (1984). "The Piecewise Parabolic Method (PPM) for Gas-Dynamical Simulations." *J. Comput. Phys.*, 54, 174-201.
- Stevens, B., Feingold, G., & Cotton, W.R. (1996). "A critique of one- and two-dimensional models of the boundary layer containing a stratocumulus cloud." *J. Atmos. Sci.*, 53(7), A3.
- Tzivion, S., Feingold, G., & Levin, Z. (1989). "The evolution of raindrop spectra. Part II: Collisional collection/breakup and evaporation in a rainshaft." *J. Atmos. Sci.*, 46(21), 3312-3327.
