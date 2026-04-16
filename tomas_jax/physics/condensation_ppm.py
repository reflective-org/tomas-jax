"""Piecewise Parabolic Method (PPM) Condensation for TOMAS-JAX.

Eulerian advection scheme in log-mass coordinate (xi = ln m).

GUIDING INVARIANTS (enforced by tests):
1. State variables are bin totals (integrals): Nk[k], Mk[k,j]
2. Work in log-mass coordinate: xi = log(xk), delta_xi = log(2)
3. PPM reconstructs cell-average densities: n_bar = Nk / delta_xi
4. Transport is conservative, flux-form, substepped by CFL
5. ALL species (including condensing) transported with upwind flux
   Condensed mass added AFTER transport in ezcond_ppm driver
6. WR frozen for phase 1 (computed once at entry)
7. Edge velocity is upwind-consistent, uses TAU_sub = TAU / n_sub
8. Clipping to [xk[0], xk[-1]] defines boundary behavior

References:
    - Colella & Woodward (1984) "The Piecewise Parabolic Method (PPM)"
    - Stevens, Feingold, Cotton (1996) for dmdt_int growth equation
"""
import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial

from ..core.config import ICOMP, ICOMP_NODIAG

# =============================================================================
# CONSTANTS
# =============================================================================

# Default grid spacing in log-mass coordinate (mass-doubling bins)
DELTA_XI_DEFAULT = jnp.log(2.0)

# Numerical safety constants
EPSN = 1.0e-30  # For ratios involving Nk (prevent division by zero)
EPSM = 0.0      # For mass (don't invent mass)

# Backward-compatible alias
DELTA_XI = DELTA_XI_DEFAULT


def _compute_moment_integrals(a):
    """Compute moment integrals I0, I1, I2 for mass-weighted PPM integrals.

    I_k = integral_0^1 eta^k * exp(a*eta) d_eta

    Args:
        a: Log-mass grid spacing (delta_xi = ln(doubling_factor))

    Returns:
        I0, I1, I2: Moment integrals
    """
    ea = jnp.exp(a)
    I0 = jnp.expm1(a) / a
    I1 = (ea * (a - 1.0) + 1.0) / (a * a)
    I2 = (ea * (a * a - 2.0 * a + 2.0) - 2.0) / (a ** 3)
    return I0, I1, I2


def _mass_antideriv_a(eta, a):
    """Antiderivatives of eta^k * exp(a*eta) for k=0,1,2.

    Parameterized by grid spacing a = delta_xi.
    """
    inv_a = 1.0 / a
    inv_a2 = inv_a * inv_a
    inv_a3 = inv_a2 * inv_a
    ea_eta = jnp.exp(a * eta)
    A0 = ea_eta * inv_a
    A1 = ea_eta * (eta * inv_a - inv_a2)
    A2 = ea_eta * (eta * eta * inv_a - 2.0 * eta * inv_a2 + 2.0 * inv_a3)
    return A0, A1, A2


# Precomputed values for default mass-doubling grid (backward compat)
_A = DELTA_XI_DEFAULT
_EA = jnp.exp(_A)
I0, I1, I2 = _compute_moment_integrals(_A)
_INV_A = 1.0 / _A
_INV_A2 = _INV_A * _INV_A
_INV_A3 = _INV_A2 * _INV_A


# =============================================================================
# 1.2 DMDT_INT - Analytic Growth Solution
# =============================================================================

def dmdt_int(
    m0: jnp.ndarray,
    tau: jnp.ndarray,
    wr: jnp.ndarray
) -> jnp.ndarray:
    """Analytic solution to particle growth equation in mass space.

    Port of dmdt_int.f - Stevens et al. 1996, JAS 53(7), Equation A3.

    Args:
        m0: Initial dry mass [kg], shape (...)
        tau: Growth forcing (per-step, includes dt), shape (...)
        wr: Wet/dry mass ratio, shape (...)

    Returns:
        m_final: Final dry mass after growth [kg], same shape as m0

    Note:
        tau > 0 -> growth (condensation)
        tau < 0 -> shrinkage (evaporation)
    """
    # Water mass (dmdt_int.f line 33)
    m_h2o = (wr - 1.0) * m0

    # Wet mass raised to 2/3 power (line 34, L0=0)
    x = jnp.power(m0 + m_h2o, 2.0/3.0)

    # Apply growth forcing and take square root (line 35)
    # C = 2/3 from parameter statement
    x = jnp.sqrt(jnp.maximum(0.0, (2.0/3.0) * tau + x))

    # Final dry mass (line 37 - keeps WR constant)
    m_final = x * x * x / wr

    # Numerical guards (lines 42-43)
    # For condensation (tau > 0), mass must not decrease
    m_final = jnp.where((tau > 0.0) & (m_final < m0), m0, m_final)
    # For evaporation (tau < 0), mass must not increase
    m_final = jnp.where((tau < 0.0) & (m_final > m0), m0, m_final)

    return m_final


# =============================================================================
# 1.3 FROZEN WR COMPUTATION
# =============================================================================

def compute_wr(
    Mk: jnp.ndarray,
    icomp_nodiag: int
) -> jnp.ndarray:
    """Compute wet/dry mass ratio, frozen at step entry.

    Args:
        Mk: Mass per bin per species [kg], shape (nbins, icomp)
        icomp_nodiag: Number of dry (non-diagnostic) species

    Returns:
        WR: Wet/dry ratio per bin, shape (nbins,)
    """
    # Dry mass = sum of non-diagnostic species
    Mk_dry = jnp.sum(Mk[:, :icomp_nodiag], axis=1)

    # Wet mass = sum of all species (includes water if present)
    Mk_wet = jnp.sum(Mk, axis=1)

    # Ratio with safety for empty bins
    WR = Mk_wet / jnp.maximum(Mk_dry, EPSN)

    # WR should be >= 1 (wet >= dry)
    WR = jnp.maximum(WR, 1.0)

    return WR


# =============================================================================
# 1.4 EDGE VELOCITY (UPWIND-CONSISTENT)
# =============================================================================

def compute_edge_velocity(
    xk: jnp.ndarray,
    TAU_sub: jnp.ndarray,
    WR: jnp.ndarray,
    dt_sub: float
) -> jnp.ndarray:
    """Compute velocity at bin edges using upwind-consistent approach.

    At face k (between bins k-1 and k):
    - Compute m_final using donor-side TAU/WR
    - Donor = left bin for growth (u>0), right bin for evaporation (u<0)
    - Clip m_final to grid bounds (matches Fortran upper cap)

    Args:
        xk: Bin mass boundaries [kg], shape (nbins+1,)
        TAU_sub: Growth forcing for this substep, shape (nbins,)
        WR: Wet/dry ratio per bin (frozen), shape (nbins,)
        dt_sub: Substep size [s]

    Returns:
        u_edges: Velocity in xi-space [1/s], shape (nbins+1,)
    """
    nbins = TAU_sub.shape[0]

    # Extend TAU/WR to edges using boundary values
    # Left donor values (for edges 0..nbins)
    TAU_left = jnp.concatenate([TAU_sub[:1], TAU_sub])  # Edge 0 uses bin 0
    WR_left = jnp.concatenate([WR[:1], WR])

    # Right donor values
    TAU_right = jnp.concatenate([TAU_sub, TAU_sub[-1:]])  # Edge nbins uses bin nbins-1
    WR_right = jnp.concatenate([WR, WR[-1:]])

    # Compute final mass using each donor side
    m_final_left = dmdt_int(xk, TAU_left, WR_left)
    m_final_right = dmdt_int(xk, TAU_right, WR_right)

    # Clip to grid bounds (matches Fortran YU cap, gives zero velocity at boundaries)
    m_final_left = jnp.clip(m_final_left, xk[0], xk[-1])
    m_final_right = jnp.clip(m_final_right, xk[0], xk[-1])

    # Velocity from each side: u = (ln(m_final) - ln(m_edge)) / dt_sub
    u_left = jnp.log(m_final_left / xk) / dt_sub
    u_right = jnp.log(m_final_right / xk) / dt_sub

    # Upwind selection: use left for growth (u>0), right for evaporation (u<0)
    u_avg = 0.5 * (u_left + u_right)
    u_edges = jnp.where(u_avg >= 0, u_left, u_right)

    return u_edges


# =============================================================================
# 1.5 PPM RECONSTRUCTION
# =============================================================================

def ppm_reconstruct(
    q_bar: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """PPM reconstruction on uniform log-mass grid.

    Computes parabolic representation within each cell:
        q(eta) = q_L + eta*(dq + q_6*(1-eta))

    where eta = (xi - xi_L) / dxi in [0, 1], and:
        dq = q_R - q_L
        q_6 = 6*(q_bar - (q_L + q_R)/2)

    This ensures integral_0^1 q(eta) d_eta = q_bar (integral preserving).

    Args:
        q_bar: Cell-average density, shape (nbins,)

    Returns:
        q_L: Left edge value, shape (nbins,)
        q_R: Right edge value, shape (nbins,)
        q_6: Curvature coefficient, shape (nbins,)
    """
    nbins = q_bar.shape[0]

    # Step 1: Fourth-order interface interpolation
    # q_{i+1/2} = (7/12)(q_i + q_{i+1}) - (1/12)(q_{i-1} + q_{i+2})
    # Pad with reflected/constant boundary values
    q_ext = jnp.concatenate([
        q_bar[:1], q_bar[:1],  # Left boundary: repeat first value
        q_bar,
        q_bar[-1:], q_bar[-1:]  # Right boundary: repeat last value
    ])

    # Compute edge values (nbins+1 edges)
    # Edge k sits between original cells k-1 and k
    # In extended array, original cell i is at index i+2
    # So edge k uses extended indices: k, k+1, k+2, k+3
    q_edges = (
        (7.0/12.0) * (q_ext[1:-2] + q_ext[2:-1]) -
        (1.0/12.0) * (q_ext[:-3] + q_ext[3:])
    )

    # Step 2: Monotonicity limiting on edge values
    q_edges = _monotonicity_limit_edges(q_edges, q_bar)

    # Step 3: Extract left/right edge values for each cell
    q_L = q_edges[:-1]  # Left edge of each cell
    q_R = q_edges[1:]   # Right edge of each cell

    # Step 4: Compute q_6 to preserve cell average
    # From: integral_0^1 [q_L + eta(dq + q_6(1-eta))] d_eta = q_bar
    # We get: q_L + 0.5*dq + q_6/6 = q_bar
    # So: q_6 = 6*(q_bar - q_L - 0.5*dq) = 6*(q_bar - 0.5*(q_L + q_R))
    q_6 = 6.0 * (q_bar - 0.5 * (q_L + q_R))

    # Step 5: Limit parabola to prevent new extrema inside cell
    q_L, q_R, q_6 = _limit_parabola(q_L, q_R, q_6, q_bar)

    return q_L, q_R, q_6


def _monotonicity_limit_edges(
    q_edges: jnp.ndarray,
    q_bar: jnp.ndarray
) -> jnp.ndarray:
    """Apply monotonicity limiting to edge values.

    Ensures edge values don't create new extrema relative to cell averages.
    Uses van Leer-style limiting.

    Args:
        q_edges: Interpolated edge values, shape (nbins+1,)
        q_bar: Cell averages, shape (nbins,)

    Returns:
        q_edges_limited: Limited edge values, shape (nbins+1,)
    """
    # For each edge, limit to range of adjacent cell averages
    # Edge k is between cells k-1 and k
    # Pad q_bar for boundary edges
    q_bar_ext = jnp.concatenate([q_bar[:1], q_bar, q_bar[-1:]])

    # Cell averages to left and right of each edge
    q_left = q_bar_ext[:-1]   # Cell to left of edge k
    q_right = q_bar_ext[1:]   # Cell to right of edge k

    # Limit edge values to [min, max] of adjacent cells
    q_min = jnp.minimum(q_left, q_right)
    q_max = jnp.maximum(q_left, q_right)

    q_edges_limited = jnp.clip(q_edges, q_min, q_max)

    return q_edges_limited


def _limit_parabola(
    q_L: jnp.ndarray,
    q_R: jnp.ndarray,
    q_6: jnp.ndarray,
    q_bar: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Limit parabola coefficients to prevent new extrema and ensure positivity.

    Robust PPM limiting following Colella & Woodward (1984):
    1. If q_bar is not between q_L and q_R, reset to constant
    2. Check for interior extremum and flatten if value is outside bounds
    3. Ensure positivity: q(eta) >= 0 for all eta in [0,1]

    The parabola is: q(eta) = q_L + eta*(dq + q_6*(1-eta))
    where dq = q_R - q_L, and eta in [0,1].

    Args:
        q_L, q_R, q_6: Parabola coefficients
        q_bar: Cell averages (must be non-negative for density)

    Returns:
        Limited (q_L, q_R, q_6) that guarantees:
        - No new extrema inside [0,1] that exceed neighbor bounds
        - q(eta) >= 0 for all eta in [0,1] (positivity)
    """
    dq = q_R - q_L

    # Case 1: If q_bar is not between q_L and q_R, reset to constant
    out_of_range = ((q_bar - q_L) * (q_bar - q_R)) > 0
    q_L = jnp.where(out_of_range, q_bar, q_L)
    q_R = jnp.where(out_of_range, q_bar, q_R)
    q_6 = jnp.where(out_of_range, 0.0, q_6)
    dq = q_R - q_L  # Update dq after potential change

    # Case 2: Check for interior extremum
    # Derivative: q'(eta) = dq + q_6 - 2*q_6*eta = 0
    # Extremum at: eta* = (dq + q_6) / (2*q_6) when q_6 != 0

    # Safe computation of eta*
    q_6_safe = jnp.where(jnp.abs(q_6) > 1e-30, q_6, 1.0)
    eta_star = (dq + q_6) / (2.0 * q_6_safe)

    # Check if extremum is inside (0,1)
    has_interior_extremum = (jnp.abs(q_6) > 1e-30) & (eta_star > 0.0) & (eta_star < 1.0)

    # Compute value at extremum: q(eta*) = q_L + eta*(dq + q_6*(1-eta))
    q_at_extremum = q_L + eta_star * (dq + q_6 * (1.0 - eta_star))

    # Bounds from edges and cell average
    q_min_bound = jnp.minimum(jnp.minimum(q_L, q_R), q_bar)
    q_max_bound = jnp.maximum(jnp.maximum(q_L, q_R), q_bar)

    # Extremum is problematic if outside bounds
    extremum_too_low = has_interior_extremum & (q_at_extremum < q_min_bound)
    extremum_too_high = has_interior_extremum & (q_at_extremum > q_max_bound)
    problematic_extremum = extremum_too_low | extremum_too_high

    # Fix: flatten to linear (q_6 = 0) when problematic
    q_6 = jnp.where(problematic_extremum, 0.0, q_6)

    # Case 3: Positivity limiting
    # First ensure edge values are non-negative
    q_L = jnp.maximum(q_L, 0.0)
    q_R = jnp.maximum(q_R, 0.0)

    # Recompute dq and q_6 after positivity clamp on edges
    dq = q_R - q_L
    # q_6 must satisfy: q_bar = (q_L + q_R)/2 + q_6/6
    q_6 = 6.0 * (q_bar - 0.5 * (q_L + q_R))

    # Now check interior minimum again with new q_6
    q_6_safe = jnp.where(jnp.abs(q_6) > 1e-30, q_6, 1.0)
    eta_star = (dq + q_6) / (2.0 * q_6_safe)
    has_interior_extremum = (jnp.abs(q_6) > 1e-30) & (eta_star > 0.0) & (eta_star < 1.0)

    # Value at interior extremum
    q_at_extremum = q_L + eta_star * (dq + q_6 * (1.0 - eta_star))

    # If interior minimum is negative, flatten to constant
    negative_interior = has_interior_extremum & (q_at_extremum < 0.0)

    # When negative interior, fall back to constant reconstruction
    q_L = jnp.where(negative_interior, q_bar, q_L)
    q_R = jnp.where(negative_interior, q_bar, q_R)
    q_6 = jnp.where(negative_interior, 0.0, q_6)

    return q_L, q_R, q_6


# =============================================================================
# 1.6 PPM FLUX COMPUTATION
# =============================================================================

def ppm_flux(
    q_L: jnp.ndarray,
    q_R: jnp.ndarray,
    q_6: jnp.ndarray,
    u_edges: jnp.ndarray,
    dt_sub: float,
    delta_xi: float = DELTA_XI
) -> jnp.ndarray:
    """Compute flux at each bin edge using PPM (departure-point method).

    For each edge, find departure point and integrate parabola
    from departure point to edge.

    Args:
        q_L, q_R, q_6: PPM coefficients for each cell, shape (nbins,)
        u_edges: Velocity at each edge [1/s], shape (nbins+1,)
        dt_sub: Substep size [s]
        delta_xi: Grid spacing (= ln(2))

    Returns:
        F: Flux at each edge [quantity/s], shape (nbins+1,)
    """
    nbins = q_L.shape[0]

    # Courant number at each edge
    C = u_edges * dt_sub / delta_xi  # Signed Courant number

    # For u > 0: flux comes from left cell, integrate over [1-|C|, 1]
    # For u < 0: flux comes from right cell, integrate over [0, |C|]

    # Pad coefficients for boundary access
    # Edge 0 has no left cell, edge nbins has no right cell
    q_L_ext = jnp.concatenate([jnp.zeros(1), q_L])  # Pad left with 0
    q_R_ext = jnp.concatenate([jnp.zeros(1), q_R])
    q_6_ext = jnp.concatenate([jnp.zeros(1), q_6])

    q_L_right = jnp.concatenate([q_L, jnp.zeros(1)])  # Pad right with 0
    q_R_right = jnp.concatenate([q_R, jnp.zeros(1)])
    q_6_right = jnp.concatenate([q_6, jnp.zeros(1)])

    # Compute flux for positive velocity (from left cell)
    # Integral of q(eta) from (1-|C|) to 1
    F_pos = _integrate_parabola_right(q_L_ext, q_R_ext, q_6_ext, jnp.abs(C))

    # Compute flux for negative velocity (from right cell)
    # Integral of q(eta) from 0 to |C|
    F_neg = _integrate_parabola_left(q_L_right, q_R_right, q_6_right, jnp.abs(C))

    # Select based on velocity sign
    # F has units of [quantity] (integral), divide by dt_sub to get [quantity/s]
    F = jnp.where(u_edges >= 0, F_pos, -F_neg) * delta_xi / dt_sub

    # Boundary conditions: no flux at domain edges
    F = F.at[0].set(0.0)
    F = F.at[-1].set(0.0)

    return F


def _integrate_parabola_right(
    q_L: jnp.ndarray,
    q_R: jnp.ndarray,
    q_6: jnp.ndarray,
    C: jnp.ndarray
) -> jnp.ndarray:
    """Integrate parabola over [1-C, 1] (right portion of cell).

    integral_{1-C}^{1} [q_L + eta(dq + q_6(1-eta))] d_eta
    where dq = q_R - q_L.
    """
    dq = q_R - q_L

    def antideriv(eta):
        return q_L * eta + 0.5 * (dq + q_6) * eta**2 - (q_6 / 3.0) * eta**3

    # Clip C to [0, 1] for safety
    C_safe = jnp.clip(C, 0.0, 1.0)

    integral = antideriv(1.0) - antideriv(1.0 - C_safe)

    return integral


def _integrate_parabola_left(
    q_L: jnp.ndarray,
    q_R: jnp.ndarray,
    q_6: jnp.ndarray,
    C: jnp.ndarray
) -> jnp.ndarray:
    """Integrate parabola over [0, C] (left portion of cell).

    integral_{0}^{C} [q_L + eta(dq + q_6(1-eta))] d_eta
    """
    dq = q_R - q_L

    def antideriv(eta):
        return q_L * eta + 0.5 * (dq + q_6) * eta**2 - (q_6 / 3.0) * eta**3

    C_safe = jnp.clip(C, 0.0, 1.0)

    integral = antideriv(C_safe) - antideriv(0.0)

    return integral


# =============================================================================
# 1.6b MASS-WEIGHTED FLUX (ANALYTICAL INTEGRAL)
# =============================================================================

def _mass_antideriv(eta):
    """Antiderivatives for default mass-doubling grid. See _mass_antideriv_a."""
    return _mass_antideriv_a(eta, _A)


def _integrate_mass_parabola_right(
    n_L: jnp.ndarray,
    n_R: jnp.ndarray,
    n_6: jnp.ndarray,
    m_L: jnp.ndarray,
    C: jnp.ndarray,
    delta_xi: float = DELTA_XI_DEFAULT
) -> jnp.ndarray:
    """Mass-weighted integral over right departure region [1-C, 1].

    Computes: delta_xi * m_L * integral_{1-C}^{1} exp(a*eta) * n(eta) d_eta

    where n(eta) = n_L + eta*(b) - n_6*eta^2, b = (n_R - n_L) + n_6,
    and m(eta) = m_L * exp(a*eta) is the dry mass at position eta.
    """
    b = (n_R - n_L) + n_6
    C_safe = jnp.clip(C, 0.0, 1.0)

    A0_hi, A1_hi, A2_hi = _mass_antideriv_a(1.0, delta_xi)
    A0_lo, A1_lo, A2_lo = _mass_antideriv_a(1.0 - C_safe, delta_xi)

    integral = (n_L * (A0_hi - A0_lo)
                + b * (A1_hi - A1_lo)
                - n_6 * (A2_hi - A2_lo))

    return delta_xi * m_L * integral


def _integrate_mass_parabola_left(
    n_L: jnp.ndarray,
    n_R: jnp.ndarray,
    n_6: jnp.ndarray,
    m_L: jnp.ndarray,
    C: jnp.ndarray,
    delta_xi: float = DELTA_XI_DEFAULT
) -> jnp.ndarray:
    """Mass-weighted integral over left departure region [0, C].

    Computes: delta_xi * m_L * integral_{0}^{C} exp(a*eta) * n(eta) d_eta
    """
    b = (n_R - n_L) + n_6
    C_safe = jnp.clip(C, 0.0, 1.0)

    A0_hi, A1_hi, A2_hi = _mass_antideriv_a(C_safe, delta_xi)
    A0_lo, A1_lo, A2_lo = _mass_antideriv_a(0.0, delta_xi)

    integral = (n_L * (A0_hi - A0_lo)
                + b * (A1_hi - A1_lo)
                - n_6 * (A2_hi - A2_lo))

    return delta_xi * m_L * integral


def ppm_mass_flux(
    n_L: jnp.ndarray,
    n_R: jnp.ndarray,
    n_6: jnp.ndarray,
    xk: jnp.ndarray,
    u_edges: jnp.ndarray,
    dt_sub: float,
    delta_xi: float = DELTA_XI
) -> jnp.ndarray:
    """Compute dry mass flux at each edge using analytical mass-weighted integrals.

    Instead of F_M = F_N * r_avg (which assigns average mass to all particles),
    this integrates m(eta)*n(eta) over the departure region, correctly accounting
    for the 2x mass variation across each bin.

    Args:
        n_L, n_R, n_6: PPM coefficients for number density, shape (nbins,)
        xk: Bin mass boundaries [kg], shape (nbins+1,)
        u_edges: Velocity at each edge [1/s], shape (nbins+1,)
        dt_sub: Substep size [s]
        delta_xi: Grid spacing (= ln(2))

    Returns:
        F_M_dry: Dry mass flux at each edge [kg/s], shape (nbins+1,)
    """
    # Courant number at each edge
    C = u_edges * dt_sub / delta_xi

    # --- Positive velocity: flux from left cell (k-1) ---
    # Pad coefficients: edge 0 has no left cell
    n_L_ext = jnp.concatenate([jnp.zeros(1), n_L])
    n_R_ext = jnp.concatenate([jnp.zeros(1), n_R])
    n_6_ext = jnp.concatenate([jnp.zeros(1), n_6])
    # m_L of donor cell (k-1): xk[k-1]
    m_L_pos = jnp.concatenate([xk[:1], xk[:-1]])

    F_M_pos = _integrate_mass_parabola_right(
        n_L_ext, n_R_ext, n_6_ext, m_L_pos, jnp.abs(C), delta_xi)

    # --- Negative velocity: flux from right cell (k) ---
    # Pad coefficients: edge nbins has no right cell
    n_L_right = jnp.concatenate([n_L, jnp.zeros(1)])
    n_R_right = jnp.concatenate([n_R, jnp.zeros(1)])
    n_6_right = jnp.concatenate([n_6, jnp.zeros(1)])
    # m_L of donor cell (k): xk[k]
    m_L_neg = jnp.concatenate([xk[:-1], xk[-2:-1]])

    F_M_neg = _integrate_mass_parabola_left(
        n_L_right, n_R_right, n_6_right, m_L_neg, jnp.abs(C), delta_xi)

    # Select based on velocity sign (mass integral already includes delta_xi)
    F_M = jnp.where(u_edges >= 0, F_M_pos, -F_M_neg) / dt_sub

    # Boundary conditions
    F_M = F_M.at[0].set(0.0)
    F_M = F_M.at[-1].set(0.0)

    return F_M


# =============================================================================
# 1.7 CONSERVATIVE UPDATE FROM FLUX DIVERGENCE
# =============================================================================

def advect_totals(
    Q: jnp.ndarray,
    F: jnp.ndarray,
    dt_sub: float
) -> jnp.ndarray:
    """Apply flux divergence to update conserved totals.

    Q_next = Q - dt_sub * (F[1:] - F[:-1])

    Args:
        Q: Conserved totals, shape (nbins,)
        F: Fluxes at edges [quantity/s], shape (nbins+1,)
        dt_sub: Substep size [s]

    Returns:
        Q_next: Updated totals, shape (nbins,)
    """
    # Net flux out of each cell
    div_F = F[1:] - F[:-1]

    Q_next = Q - dt_sub * div_F

    return Q_next


# =============================================================================
# 1.8 SPECIES TRANSPORT VIA WELL-MIXED RATIO
# =============================================================================

def species_flux(
    F_M_dry: jnp.ndarray,
    Mk: jnp.ndarray,
    M_dry_analytical: jnp.ndarray,
    u_edges: jnp.ndarray
) -> jnp.ndarray:
    """Compute mass flux for ALL species simultaneously using broadcasting.

    Uses the analytical dry mass flux (from ppm_mass_flux) normalized by
    the analytical total dry mass (from dry_mass_from_ppm_number):

        F_M[k,j] = F_M_dry[k] * (Mk[donor,j] / M_dry_analytical[donor])

    Fully vectorized — no loops over species.

    Args:
        F_M_dry: Dry mass flux at edges [kg/s], shape (nbins+1,)
        Mk: Mass of ALL species per bin [kg], shape (nbins, ncomp)
        M_dry_analytical: Analytical dry mass per bin [kg], shape (nbins,)
        u_edges: Velocity at edges (for upwind direction), shape (nbins+1,)

    Returns:
        F_M_all: Mass flux at edges for all species [kg/s], shape (nbins+1, ncomp)
    """
    nbins = Mk.shape[0]

    # Composition ratio for all species: (nbins, ncomp)
    M_dry_2d = M_dry_analytical[:, None]
    # Safe division: clamp denominator first to avoid NaN in reverse-mode AD,
    # then zero out bins with no mass (jnp.where evaluates both branches).
    ratio = Mk / jnp.maximum(M_dry_2d, 1e-30)
    ratio = jnp.where(M_dry_2d > 1e-30, ratio, 0.0)

    # Donor bin indices (upwind)
    idx = jnp.arange(nbins + 1)
    donor_left = jnp.clip(idx - 1, 0, nbins - 1)
    donor_right = jnp.clip(idx, 0, nbins - 1)
    donor = jnp.where(u_edges >= 0, donor_left, donor_right)

    # Extract donor ratios: (nbins+1, ncomp)
    donor_ratios = ratio[donor]

    # Multiply 1D dry flux by 2D donor ratios: (nbins+1, 1) * (nbins+1, ncomp)
    F_M_all = F_M_dry[:, None] * donor_ratios

    return F_M_all


# =============================================================================
# 1.9 DRY MASS FROM PPM NUMBER (CLOSED FORM)
# =============================================================================

def dry_mass_from_ppm_number(
    n_L: jnp.ndarray,
    n_R: jnp.ndarray,
    n_6: jnp.ndarray,
    xk: jnp.ndarray,
    delta_xi: float = DELTA_XI
) -> jnp.ndarray:
    """Compute total dry mass per bin from PPM number distribution.

    Exact integral: M_dry[i] = integral m(xi) n(xi) d_xi over bin i

    With xi = xi_L + eta*a, m = m_L*exp(a*eta), and
    n(eta) = n_L + b*eta - n_6*eta^2 where b = (n_R - n_L) + n_6

    M_dry = a * m_L * (n_L*I_0 + b*I_1 - n_6*I_2)

    where I_0, I_1, I_2 are precomputed moment integrals.

    Args:
        n_L, n_R, n_6: PPM coefficients for number density, shape (nbins,)
        xk: Bin mass boundaries [kg], shape (nbins+1,)
        delta_xi: Grid spacing = ln(2)

    Returns:
        M_dry: Total dry mass per bin [kg], shape (nbins,)
    """
    # Parabola coefficient b = dn + n_6
    dn = n_R - n_L
    b = dn + n_6

    # Left-edge mass for each bin
    m_L = xk[:-1]

    # Compute moment integrals for actual grid spacing
    I0_a, I1_a, I2_a = _compute_moment_integrals(delta_xi)

    M_dry = delta_xi * m_L * (n_L * I0_a + b * I1_a - n_6 * I2_a)

    return M_dry


def dry_mass_from_ppm_number_quadrature(
    n_L: jnp.ndarray,
    n_R: jnp.ndarray,
    n_6: jnp.ndarray,
    xk: jnp.ndarray,
    delta_xi: float = DELTA_XI
) -> jnp.ndarray:
    """Compute dry mass using 3-point Gauss quadrature (for validation).

    Same result as closed form, but useful for testing.
    """
    # Gauss-Legendre 3-point quadrature on [0,1]
    eta_quad = jnp.array([0.1127016654, 0.5, 0.8872983346])
    w_quad = jnp.array([5.0/18.0, 8.0/18.0, 5.0/18.0])

    xi_L = jnp.log(xk[:-1])

    M_dry = jnp.zeros_like(n_L)
    for q in range(3):
        eta = eta_quad[q]
        w = w_quad[q]

        # xi and m at quadrature point
        xi_q = xi_L + eta * delta_xi
        m_q = jnp.exp(xi_q)

        # n at quadrature point (from PPM parabola)
        dn = n_R - n_L
        n_q = n_L + eta * (dn + n_6 * (1.0 - eta))

        # Contribution: w * m(xi) * n(xi) * dxi
        M_dry = M_dry + w * m_q * n_q * delta_xi

    return M_dry


# =============================================================================
# 2.1 CFL SUBSTEP COUNT
# =============================================================================

def compute_substeps(
    xk: jnp.ndarray,
    TAU: jnp.ndarray,
    WR: jnp.ndarray,
    dt: float,
    C_max: float = 0.8
) -> int:
    """Compute number of substeps needed for CFL stability.

    Based on worst-case shift in xi from the analytic growth map.

    Args:
        xk: Bin mass boundaries [kg], shape (nbins+1,)
        TAU: Growth forcing (full step), shape (nbins,)
        WR: Wet/dry ratio per bin, shape (nbins,)
        dt: Full timestep [s]
        C_max: Maximum allowed Courant number

    Returns:
        n_sub: Number of substeps (>= 1)
    """
    nbins = TAU.shape[0]

    # Compute growth at each edge using adjacent bin parameters
    TAU_edges = jnp.concatenate([TAU[:1], 0.5*(TAU[:-1]+TAU[1:]), TAU[-1:]])
    WR_edges = jnp.concatenate([WR[:1], 0.5*(WR[:-1]+WR[1:]), WR[-1:]])

    # Final mass at each edge
    m_final = dmdt_int(xk, TAU_edges, WR_edges)

    # Shift in xi = ln(m_final / xk)
    # Clip to avoid log of non-positive
    m_final_safe = jnp.maximum(m_final, xk[0] * 0.1)
    delta_xi_max = jnp.max(jnp.abs(jnp.log(m_final_safe / xk)))

    # Number of substeps (use actual grid spacing)
    actual_delta_xi = jnp.log(xk[1] / xk[0])
    n_sub = jnp.ceil(delta_xi_max / (C_max * actual_delta_xi)).astype(int)
    n_sub = jnp.maximum(n_sub, 1)

    return n_sub


# =============================================================================
# 2.2 MAIN PPM CONDENSATION STEPPER
# =============================================================================

def ppm_condensation_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    TAU: jnp.ndarray,
    cspecies: int,
    dt: float,
    icomp_nodiag: int,
    C_max: float = 0.8
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """PPM condensation step - Eulerian advection in log-mass space.

    Algorithm:
    1. Freeze WR at step entry (match Fortran)
    2. Compute substep count from CFL
    3. Precompute loop-invariant quantities (edge velocities, moment integrals)
    4. For each substep:
       a. Reconstruct number density (PPM)
       b. Compute number flux and advect Nk
       c. Transport ALL species with upwind flux (including cspecies)
       d. Apply positivity limiter
    5. Condensed mass is added AFTER transport by ezcond_ppm driver

    Args:
        Nk: Number per bin [#], shape (nbins,)
        Mk: Mass per bin per species [kg], shape (nbins, icomp)
        xk: Bin mass boundaries [kg], shape (nbins+1,)
        TAU: Growth forcing (per full step), shape (nbins,)
        cspecies: Index of condensing species
        dt: Full timestep [s]
        icomp_nodiag: Number of dry (non-diagnostic) species
        C_max: Maximum Courant number for substepping

    Returns:
        Nk_new: Updated number per bin [#], shape (nbins,)
        Mk_new: Updated mass per bin [kg], shape (nbins, icomp)
    """
    nbins = Nk.shape[0]
    ncomp = Mk.shape[1]

    # Derive delta_xi from actual grid spacing
    delta_xi = jnp.log(xk[1] / xk[0])

    # 1. Freeze WR
    WR = compute_wr(Mk, icomp_nodiag)

    # 2. Compute substep count
    n_sub = compute_substeps(xk, TAU, WR, dt, C_max)
    dt_sub = dt / n_sub
    TAU_sub = TAU / n_sub

    # 3. Precompute loop-invariant quantities (constant across all substeps)
    u_edges = compute_edge_velocity(xk, TAU_sub, WR, dt_sub)
    I0_a, I1_a, I2_a = _compute_moment_integrals(delta_xi)

    # 4. Substep loop using lax.fori_loop
    def substep_body(i, carry):
        Nk_curr, Mk_curr = carry

        # 4a. Reconstruct number density
        n_bar = Nk_curr / delta_xi
        n_L, n_R, n_6 = ppm_reconstruct(n_bar)

        # 4b. Compute number flux and advect
        F_N = ppm_flux(n_L, n_R, n_6, u_edges, dt_sub, delta_xi)
        Nk_next = advect_totals(Nk_curr, F_N, dt_sub)

        # 4c. Compute analytical dry mass flux (mass-weighted integral)
        F_M_dry = ppm_mass_flux(n_L, n_R, n_6, xk, u_edges, dt_sub, delta_xi)

        # 4d. Transport ALL species — vectorized, no loops
        # Inline dry_mass_from_ppm_number with precomputed moment integrals
        dn = n_R - n_L
        b = dn + n_6
        m_L = xk[:-1]
        M_dry_analytical = delta_xi * m_L * (n_L * I0_a + b * I1_a - n_6 * I2_a)
        F_M_all = species_flux(F_M_dry, Mk_curr, M_dry_analytical, u_edges)
        Mk_next = advect_totals(Mk_curr, F_M_all, dt_sub)

        # 4e. Positivity clamp
        Nk_next = jnp.maximum(Nk_next, 0.0)
        Mk_next = jnp.maximum(Mk_next, 0.0)

        return (Nk_next, Mk_next)

    # Run substep loop
    Nk_new, Mk_new = jax.lax.fori_loop(0, n_sub, substep_body, (Nk, Mk))

    # 4. Positivity limiter
    Nk_new = jnp.maximum(Nk_new, 0.0)
    Mk_new = jnp.maximum(Mk_new, 0.0)

    return Nk_new, Mk_new


# =============================================================================
# NUMBER-ONLY STEPPER (FOR INITIAL VALIDATION)
# =============================================================================

def ppm_advect_number_only(
    Nk: jnp.ndarray,
    xk: jnp.ndarray,
    TAU: jnp.ndarray,
    WR: jnp.ndarray,
    dt: float,
    C_max: float = 0.8
) -> jnp.ndarray:
    """PPM advection of number distribution only (for validation).

    Simpler version without species transport - use this first to
    validate PPM translation and conservation before adding complexity.

    Args:
        Nk: Number per bin [#], shape (nbins,)
        xk: Bin mass boundaries [kg], shape (nbins+1,)
        TAU: Growth forcing (per full step), shape (nbins,)
        WR: Wet/dry ratio per bin (frozen), shape (nbins,)
        dt: Full timestep [s]
        C_max: Maximum Courant number

    Returns:
        Nk_new: Updated number per bin [#], shape (nbins,)
    """
    # Compute substep count
    n_sub = compute_substeps(xk, TAU, WR, dt, C_max)
    dt_sub = dt / n_sub
    TAU_sub = TAU / n_sub

    # Precompute loop-invariant edge velocities
    u_edges = compute_edge_velocity(xk, TAU_sub, WR, dt_sub)

    def substep_body(i, Nk_curr):
        # Reconstruct number density
        n_bar = Nk_curr / DELTA_XI
        n_L, n_R, n_6 = ppm_reconstruct(n_bar)

        # Compute flux and advect
        F_N = ppm_flux(n_L, n_R, n_6, u_edges, dt_sub, DELTA_XI)
        Nk_next = advect_totals(Nk_curr, F_N, dt_sub)

        return Nk_next

    # Run substep loop
    Nk_new = jax.lax.fori_loop(0, n_sub, substep_body, Nk)

    # Positivity
    Nk_new = jnp.maximum(Nk_new, 0.0)

    return Nk_new
