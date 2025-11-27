"""Coagulation rate calculations (dNdt, dMdt) with float64 precision.

Port of multicoag.f lines 220-340 (TFL algorithm).
Refactored for full JAX vectorization (removing lax.scan).

References:
    Tzivion, Feingold, and Levin (1987) "An Efficient Numerical Solution
    to the Stochastic Collection Equation", J. Atmos. Sci., 44, 3139-3149.
"""
import jax
# Ensure we use 64-bit precision for physics stability
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Tuple

# Algorithm parameter (multicoag.f line 98)
ZETA = 1.0625
NEPS = 1.0e-3


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
    Mk_safe = Mk_safe.at[:, 0].set(
        jnp.where(mask_empty, NEPS * 1.4 * xk[:-1], Mk_safe[:, 0])
    )
    
    return Nk_safe, Mk_safe


def calc_xbar_phi_eff(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate xbar, phi, and eff for TFL algorithm.
    
    Ref: multicoag.f lines 223-239
    """
    # Average mass per bin
    xbar = jnp.sum(Mk[:, :icomp_nodiag], axis=1) / Nk

    # Bin boundaries for calculation
    xk_lo = xk[:-1]
    xk_hi = xk[1:]

    # TFL equations 13a, 13b
    ratio = xbar / xk_lo
    factor = 2.0 * Nk / xk_lo
    
    eff = factor * (2.0 - ratio)
    phi = factor * (ratio - 1.0)

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
    kij: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate coagulation rates dNdt and dMdt using TFL algorithm.

    This implementation is fully vectorized. It replaces the sequential loop
    (lax.scan) with triangular matrix multiplications and array shifting.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        kij: Coagulation kernel [s⁻¹], shape (ibins, ibins)
        xk: Bin boundaries [kg], shape (ibins+1,)
        icomp_nodiag: Number of non-diagnostic species.

    Returns:
        dNdt: Rate of change of number, shape (ibins,)
        dMdt: Rate of change of mass, shape (ibins, icomp)
    """
    # 1. Preprocess Inputs
    Nk_safe, Mk_safe = _preprocess_concentrations(Nk, Mk, xk)
    xbar, phi, eff = calc_xbar_phi_eff(Nk_safe, Mk_safe, xk, icomp_nodiag)
    
    # Slices for calculations
    Mk_nodiag = Mk_safe[:, :icomp_nodiag]  # (nbins, icomp_nodiag)
    xk_mid = xk[:-1]
    xk_next = xk[1:]

    # 2. Calculate Summation Terms (The "Loop" replacement)
    # Create triangular masks for the kernel
    kij_lower = jnp.tril(kij, k=-1) 
    kij_upper = jnp.triu(kij, k=1)

    # Calculate "k1" terms (Sum over i < k)
    # Shapes: (nbins, nbins) @ (nbins, icomp) -> (nbins, icomp)
    k1m_vec = kij_lower @ Mk_nodiag 
    k1mx_vec = kij_lower @ (Mk_nodiag * xbar[:, None])
    k1mx2_vec = kij_lower @ (Mk_nodiag * (xbar[:, None]**2))

    # Sum over components for the Number equation (dNdt uses totals)
    k1mtot = jnp.sum(k1m_vec, axis=1)    # (nbins,)
    k1mxtot = jnp.sum(k1mx_vec, axis=1)  # (nbins,)
    
    # Calculate "in" term (Sum over i > k)
    in_term = (kij_upper @ Nk_safe) # (nbins,)

    # 3. Calculate terms for Current Bin (k)
    kij_diag = jnp.diag(kij)
    
    # dNdt parts for current k
    dNdt_curr = (
        -kij_diag * Nk_safe**2
        - phi * k1mtot
        - ZETA * (eff - phi) / (2.0 * xk_mid) * k1mxtot
        - Nk_safe * in_term
    )
    
    # dMdt parts for current k (broadcasting over components)
    dMdt_curr = (
        Nk_safe[:, None] * k1m_vec
        - kij_diag[:, None] * Nk_safe[:, None] * Mk_nodiag
        - Mk_nodiag * in_term[:, None]
        - phi[:, None] * xk_next[:, None] * k1m_vec
        - 0.5 * ZETA * eff[:, None] * k1mx_vec
        + ZETA**3 * (phi[:, None] - eff[:, None]) / (2.0 * xk_mid[:, None]) * k1mx2_vec
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
        + ZETA * (eff_prev - phi_prev) / (2.0 * xk_mid_prev) * sk2mxtot
    )
    
    # dMdt contribution from k-1
    # Note: The second term uses xk[k] (xk_mid), not xk[k-1]
    dMdt_prev = (
        kij_diag_prev[:, None] * Nk_prev[:, None] * Mk_prev
        + phi_prev[:, None] * xk_mid[:, None] * sk2m_vec 
        + 0.5 * ZETA * eff_prev[:, None] * sk2mx_vec
        - ZETA**3 * (phi_prev[:, None] - eff_prev[:, None]) / (2.0 * xk_mid_prev[:, None]) * sk2mx2_vec
    )

    # 5. Combine and Return
    dNdt = dNdt_curr + dNdt_prev
    
    # Combine dMdt for non-diagnostic species
    dMdt_nodiag = dMdt_curr + dMdt_prev
    
    # Create full dMdt array (fill diagnostic species with 0)
    dMdt = jnp.zeros_like(Mk)
    dMdt = dMdt.at[:, :icomp_nodiag].set(dMdt_nodiag)

    return dNdt, dMdt