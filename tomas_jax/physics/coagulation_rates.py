"""Coagulation rate calculations (dNdt, dMdt) with float64 precision.

Port of multicoag.f lines 220-340 (TFL algorithm).
Refactored for full JAX vectorization (removing lax.scan).

References:
    Tzivion, Feingold, and Levin (1987) "An Efficient Numerical Solution
    to the Stochastic Collection Equation", J. Atmos. Sci., 44, 3139-3149.
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Optional, Tuple

from ..core.config import ICOMP_NODIAG

NEPS = 1.0e-3


def compute_zeta(xk: jnp.ndarray) -> float:
    """Compute closure parameter ξ̄_p from bin boundaries.

    Tzivion et al. (1987), Eq. (B10):  ξ̄_p = 0.5 * [1 + (p+1)²/(4p)]
    where p = xk[1]/xk[0] is the bin mass ratio.

    For p=2 (standard TOMAS): ξ̄ = 1.0625
    For p=√2 (72 bins):       ξ̄ ≈ 1.0152
    For p=2^{1/4} (144 bins): ξ̄ ≈ 1.0038
    """
    p = xk[1] / xk[0]
    return 0.5 * (1.0 + (p + 1.0) ** 2 / (4.0 * p))


def _preprocess_concentrations(
    Nk: jnp.ndarray, Mk: jnp.ndarray, xk: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Clean concentrations: handle empty bins and add diagnostic mass.
    
    Equivalent to multicoag.f lines 109-118.
    """
    mask_empty = Nk < NEPS
    
    # Set Nk to Neps for empty bins
    Nk_safe = jnp.where(mask_empty, NEPS, Nk)

    # Set all Mk to 0 for empty bins, then add tiny SO4 mass (index 0)
    Mk_safe = jnp.where(mask_empty[:, None], 0.0, Mk)
    # Use geometric mean sqrt(xk[k]*xk[k+1]) — always inside bin for any p
    xk_geo = jnp.sqrt(xk[:-1] * xk[1:])
    Mk_safe = Mk_safe.at[:, 0].set(
        jnp.where(mask_empty, NEPS * xk_geo, Mk_safe[:, 0])
    )
    
    return Nk_safe, Mk_safe


def calc_xbar_phi_eff(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = ICOMP_NODIAG
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate xbar, phi, and eff for TFL algorithm.
    
    Ref: multicoag.f lines 223-239
    """
    # Average mass per bin
    xbar = jnp.sum(Mk[:, :icomp_nodiag], axis=1) / Nk

    # Bin boundaries for calculation
    xk_lo = xk[:-1]
    xk_hi = xk[1:]

    # TFL equations 13a, 13b — generalized for arbitrary bin ratio p
    # Fortran hardcodes p=2; for general p: eff,phi ∝ 1/(p-1)
    ratio = xbar / xk_lo
    factor = 2.0 * Nk / xk_lo
    p = xk_hi / xk_lo  # bin mass ratio (2 for standard, √2 for 72 bins, etc.)

    eff = factor * (p - ratio) / (p - 1.0)
    phi = factor * (ratio - 1.0) / (p - 1.0)

    # Constraints (equation 15)
    # Case 1: xbar < xk_lo
    mask_low = xbar < xk_lo
    eff = jnp.where(mask_low, factor, eff)
    phi = jnp.where(mask_low, 0.0, phi)

    # Case 2: xbar > xk_hi
    mask_high = xbar > xk_hi
    phi = jnp.where(mask_high, factor, phi)
    eff = jnp.where(mask_high, 0.0, eff)

    return xbar, phi, eff


def calc_coagulation_rates(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    kij: Optional[jnp.ndarray],
    xk: jnp.ndarray,
    icomp_nodiag: int = ICOMP_NODIAG,
    kij_parts: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate coagulation rates dNdt and dMdt using TFL algorithm.

    This implementation is fully vectorized. It replaces the sequential loop
    (lax.scan) with triangular matrix multiplications and array shifting.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        kij: Coagulation kernel [s⁻¹], shape (ibins, ibins). May be None
            when kij_parts is given.
        xk: Bin boundaries [kg], shape (ibins+1,)
        icomp_nodiag: Number of non-diagnostic species.
        kij_parts: Optional precomputed (tril(kij,-1), triu(kij,1),
            diag(kij)). Pass when calling repeatedly with a frozen kernel
            (e.g. Euler substep loops) so the triangular masks are not
            rebuilt every call. When given, kij itself is IGNORED — do
            not pass parts from a stale kernel alongside a fresh kij.

    Returns:
        dNdt: Rate of change of number, shape (ibins,)
        dMdt: Rate of change of mass, shape (ibins, icomp)
        dM_overflow: Mass rate overflowing the top bin, shape (icomp,).
            This is mass that would enter a hypothetical bin above the grid.
            Physically expected for finite bin grids; accumulate over time
            for mass budget closure: M(0) = M(t) + integral(dM_overflow).
    """
    # 1. Preprocess Inputs
    Nk_safe, Mk_safe = _preprocess_concentrations(Nk, Mk, xk)
    xbar, phi, eff = calc_xbar_phi_eff(Nk_safe, Mk_safe, xk, icomp_nodiag)

    # Closure parameter — adapts to bin width (Tzivion 1987, Eq. B10)
    zeta = compute_zeta(xk)
    zeta3 = zeta ** 3

    # Slices for calculations
    Mk_nodiag = Mk_safe[:, :icomp_nodiag]  # (nbins, icomp_nodiag)
    xk_mid = xk[:-1]
    xk_next = xk[1:]

    # 2. Calculate Summation Terms (The "Loop" replacement)
    # Create triangular masks for the kernel
    if kij_parts is None:
        if kij is None:
            raise ValueError("Provide either kij or kij_parts")
        kij_lower = jnp.tril(kij, k=-1)
        kij_upper = jnp.triu(kij, k=1)
        kij_diag = jnp.diag(kij)
    else:
        kij_lower, kij_upper, kij_diag = kij_parts

    # Calculate "k1" terms (Sum over i < k), one GEMM instead of three so
    # kij_lower is read once. Column blocks: [M, M*xbar, M*xbar^2].
    # Shapes: (nbins, nbins) @ (nbins, 3*icomp) -> (nbins, 3*icomp)
    nnd = Mk_nodiag.shape[1]
    k1_all = kij_lower @ jnp.concatenate(
        [Mk_nodiag,
         Mk_nodiag * xbar[:, None],
         Mk_nodiag * (xbar[:, None] ** 2)],
        axis=1,
    )
    k1m_vec = k1_all[:, :nnd]
    k1mx_vec = k1_all[:, nnd:2 * nnd]
    k1mx2_vec = k1_all[:, 2 * nnd:]

    # Sum over components for the Number equation (dNdt uses totals)
    k1mtot = jnp.sum(k1m_vec, axis=1)    # (nbins,)
    k1mxtot = jnp.sum(k1mx_vec, axis=1)  # (nbins,)
    
    # Calculate "in" term (Sum over i > k)
    in_term = (kij_upper @ Nk_safe) # (nbins,)

    # 3. Calculate terms for Current Bin (k)
    # dNdt parts for current k
    dNdt_curr = (
        -kij_diag * Nk_safe**2
        - phi * k1mtot
        - zeta * (eff - phi) / (2.0 * xk_mid) * k1mxtot
        - Nk_safe * in_term
    )

    # dMdt parts for current k (broadcasting over components)
    dMdt_curr = (
        Nk_safe[:, None] * k1m_vec
        - kij_diag[:, None] * Nk_safe[:, None] * Mk_nodiag
        - Mk_nodiag * in_term[:, None]
        - phi[:, None] * xk_next[:, None] * k1m_vec
        - 0.5 * zeta * eff[:, None] * k1mx_vec
        + zeta3 * (phi[:, None] - eff[:, None]) / (2.0 * xk_mid[:, None]) * k1mx2_vec
    )

    # 4. Calculate terms for Previous Bin (k-1)
    # Helper to shift array: [a, b, c] -> [fill_value, a, b]
    def shift_right(arr, fill_value=0.0):
        return jnp.concatenate([
            jnp.full_like(arr[:1], fill_value), 
            arr[:-1]
        ], axis=0)

    # Terms derived from index k-1
    Nk_prev = shift_right(Nk_safe)
    phi_prev = shift_right(phi)
    eff_prev = shift_right(eff)
    
    # CRITICAL FIX: Fill xk_prev with 1.0 (or any non-zero safe value).
    # This prevents division by zero at k=0, where numerators are 0.0 anyway.
    xk_mid_prev = shift_right(xk_mid, fill_value=1.0)
    
    # k1 sums shifted become the "sk2" sums from the original code
    sk2mtot = shift_right(k1mtot)
    sk2mxtot = shift_right(k1mxtot)
    sk2m_vec = shift_right(k1m_vec)
    sk2mx_vec = shift_right(k1mx_vec)
    sk2mx2_vec = shift_right(k1mx2_vec)

    # Retrieve kij(k-1, k-1)
    kij_diag_prev = shift_right(kij_diag)
    Mk_prev = shift_right(Mk_nodiag)

    # dNdt contribution from k-1
    dNdt_prev = (
        0.5 * kij_diag_prev * Nk_prev**2
        + phi_prev * sk2mtot
        + zeta * (eff_prev - phi_prev) / (2.0 * xk_mid_prev) * sk2mxtot
    )

    # dMdt contribution from k-1
    # Note: The second term uses xk[k] (xk_mid), not xk[k-1]
    dMdt_prev = (
        kij_diag_prev[:, None] * Nk_prev[:, None] * Mk_prev
        + phi_prev[:, None] * xk_mid[:, None] * sk2m_vec
        + 0.5 * zeta * eff_prev[:, None] * sk2mx_vec
        - zeta3 * (phi_prev[:, None] - eff_prev[:, None]) / (2.0 * xk_mid_prev[:, None]) * sk2mx2_vec
    )

    # 5. Top-bin overflow: mass that would enter a hypothetical bin above the grid.
    # This is the dMdt_prev contribution from bin[-1] that shift_right drops.
    # Uses xk[-1] (upper boundary of last bin) as the "xk_mid" for the overflow bin.
    dM_overflow_nodiag = (
        kij_diag[-1] * Nk_safe[-1] * Mk_nodiag[-1]
        + phi[-1] * xk[-1] * k1m_vec[-1]
        + 0.5 * zeta * eff[-1] * k1mx_vec[-1]
        - zeta3 * (phi[-1] - eff[-1]) / (2.0 * xk_mid[-1]) * k1mx2_vec[-1]
    )  # shape: (icomp_nodiag,)
    dM_overflow = jnp.zeros(Mk.shape[1])
    dM_overflow = dM_overflow.at[:icomp_nodiag].set(dM_overflow_nodiag)

    # 6. Combine and Return
    dNdt = dNdt_curr + dNdt_prev

    # Combine dMdt for non-diagnostic species
    dMdt_nodiag = dMdt_curr + dMdt_prev

    # Create full dMdt array (fill diagnostic species with 0)
    dMdt = jnp.zeros_like(Mk)
    dMdt = dMdt.at[:, :icomp_nodiag].set(dMdt_nodiag)

    return dNdt, dMdt, dM_overflow