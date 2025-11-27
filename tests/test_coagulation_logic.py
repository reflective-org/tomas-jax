import jax
# Enable x64 immediately to ensure precision matches
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import time
from typing import Tuple

# ==========================================
# 1. The Original Implementation (Baseline)
# ==========================================
def calc_coagulation_rates_scan(Nk, Mk, kij, xk, icomp_nodiag=42):
    """Legacy implementation using lax.scan."""
    nbins = Nk.shape[0]
    ncomp = Mk.shape[1]
    ZETA = 1.0625
    Neps = 1.0e-3 

    # Preprocessing
    mask_empty = Nk < Neps
    Nk = jnp.where(mask_empty, Neps, Nk)
    Mk_processed = jnp.where(mask_empty[:, None], 0.0, Mk)
    Mk_processed = Mk_processed.at[:, 0].set(
        jnp.where(mask_empty, Neps * 1.4 * xk[:-1], Mk_processed[:, 0])
    )

    # Helper for xbar/phi/eff (Compact version for the test)
    xbar = jnp.sum(Mk_processed[:, :icomp_nodiag], axis=1) / Nk
    eff = 2.0 * Nk / xk[:-1] * (2.0 - xbar / xk[:-1])
    phi = 2.0 * Nk / xk[:-1] * (xbar / xk[:-1] - 1.0)
    mask_low = xbar < xk[:-1]
    eff = jnp.where(mask_low, 2.0 * Nk / xk[:-1], eff)
    phi = jnp.where(mask_low, 0.0, phi)
    mask_high = xbar > xk[1:]
    phi = jnp.where(mask_high, 2.0 * Nk / xk[:-1], phi)
    eff = jnp.where(mask_high, 0.0, eff)

    # Scan logic
    def process_bin(carry, k):
        sk2mtot, sk2mxtot, sk2m, sk2mx, sk2mx2 = carry
        i_indices = jnp.arange(nbins)
        mask_lower = i_indices < k
        kij_k = kij[k, :] * mask_lower
        Mk_all = Mk_processed[:, :icomp_nodiag]

        k1m = jnp.sum(kij_k[:, None] * Mk_all, axis=0)
        k1mx = jnp.sum(kij_k[:, None] * Mk_all * xbar[:, None], axis=0)
        k1mx2 = jnp.sum(kij_k[:, None] * Mk_all * (xbar[:, None]**2), axis=0)
        k1mtot = jnp.sum(k1m)
        k1mxtot = jnp.sum(k1mx)

        mask_upper = i_indices > k
        in_term = jnp.sum(Nk * kij[k, :] * mask_upper)

        dNdt_k = (
            -kij[k, k] * Nk[k]**2
            - phi[k] * k1mtot
            - ZETA * (eff[k] - phi[k]) / (2.0 * xk[k]) * k1mxtot
            - Nk[k] * in_term
        )

        km1_contrib = (
            0.5 * kij[jnp.maximum(k-1, 0), jnp.maximum(k-1, 0)] * Nk[jnp.maximum(k-1, 0)]**2
            + phi[jnp.maximum(k-1, 0)] * sk2mtot
            + ZETA * (eff[jnp.maximum(k-1, 0)] - phi[jnp.maximum(k-1, 0)]) / (2.0 * xk[jnp.maximum(k-1, 0)]) * sk2mxtot
        )
        dNdt_k = jnp.where(k > 0, dNdt_k + km1_contrib, dNdt_k)

        j_indices = jnp.arange(icomp_nodiag)
        dMdt_kj = (
            Nk[k] * k1m
            - kij[k, k] * Nk[k] * Mk_processed[k, :icomp_nodiag]
            - Mk_processed[k, :icomp_nodiag] * in_term
            - phi[k] * xk[k+1] * k1m
            - 0.5 * ZETA * eff[k] * k1mx
            + ZETA**3 * (phi[k] - eff[k]) / (2.0 * xk[k]) * k1mx2
        )

        km1_contrib_M = (
            kij[jnp.maximum(k-1, 0), jnp.maximum(k-1, 0)] * Nk[jnp.maximum(k-1, 0)] * Mk_processed[jnp.maximum(k-1, 0), :icomp_nodiag]
            + phi[jnp.maximum(k-1, 0)] * xk[k] * sk2m
            + 0.5 * ZETA * eff[jnp.maximum(k-1, 0)] * sk2mx
            - ZETA**3 * (phi[jnp.maximum(k-1, 0)] - eff[jnp.maximum(k-1, 0)]) / (2.0 * xk[jnp.maximum(k-1, 0)]) * sk2mx2
        )
        dMdt_kj = jnp.where(k > 0, dMdt_kj + km1_contrib_M, dMdt_kj)
        dMdt_k = jnp.zeros(ncomp, dtype=jnp.float64)
        dMdt_k = dMdt_k.at[:icomp_nodiag].set(dMdt_kj)

        return (k1mtot, k1mxtot, k1m, k1mx, k1mx2), (dNdt_k, dMdt_k)

    init_carry = (0.0, 0.0, jnp.zeros(icomp_nodiag), jnp.zeros(icomp_nodiag), jnp.zeros(icomp_nodiag))
    _, (dNdt, dMdt) = jax.lax.scan(process_bin, init_carry, jnp.arange(nbins))
    return dNdt, dMdt

# ==========================================
# 2. The New Implementation (Vectorized)
# ==========================================
def calc_coagulation_rates_vectorized(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    kij: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Optimized implementation using matrix operations.
    
    Corrected to handle boundary conditions at k=0 without NaNs.
    """
    nbins = Nk.shape[0]
    ZETA = 1.0625
    NEPS = 1.0e-3
    
    # 1. Preprocess
    mask_empty = Nk < NEPS
    Nk_safe = jnp.where(mask_empty, NEPS, Nk)
    Mk_safe = jnp.where(mask_empty[:, None], 0.0, Mk)
    Mk_safe = Mk_safe.at[:, 0].set(jnp.where(mask_empty, NEPS * 1.4 * xk[:-1], Mk_safe[:, 0]))
    
    # 2. xbar, phi, eff
    # Calculate averages
    xbar = jnp.sum(Mk_safe[:, :icomp_nodiag], axis=1) / Nk_safe
    xk_lo, xk_hi = xk[:-1], xk[1:]
    
    ratio = xbar / xk_lo
    factor = 2.0 * Nk_safe / xk_lo
    
    eff = factor * (2.0 - ratio)
    phi = factor * (ratio - 1.0)
    
    # Apply constraints
    mask_low = xbar < xk_lo
    eff = jnp.where(mask_low, factor, eff)
    phi = jnp.where(mask_low, 0.0, phi)
    
    mask_high = xbar > xk_hi
    phi = jnp.where(mask_high, factor, phi)
    eff = jnp.where(mask_high, 0.0, eff)
    
    # 3. Vectorized Sums (Lower and Upper triangles)
    Mk_nodiag = Mk_safe[:, :icomp_nodiag]
    kij_lower = jnp.tril(kij, k=-1)
    kij_upper = jnp.triu(kij, k=1)
    
    # Lower triangle (i < k)
    k1m_vec = kij_lower @ Mk_nodiag
    k1mx_vec = kij_lower @ (Mk_nodiag * xbar[:, None])
    k1mx2_vec = kij_lower @ (Mk_nodiag * (xbar[:, None]**2))
    
    k1mtot = jnp.sum(k1m_vec, axis=1)
    k1mxtot = jnp.sum(k1mx_vec, axis=1)
    
    # Upper triangle (i > k)
    in_term = kij_upper @ Nk_safe
    kij_diag = jnp.diag(kij)

    # 4. Current Bin Terms (k)
    dNdt_curr = (
        -kij_diag * Nk_safe**2
        - phi * k1mtot 
        - ZETA * (eff - phi) / (2.0 * xk_lo) * k1mxtot
        - Nk_safe * in_term
    )
    
    dMdt_curr = (
        Nk_safe[:, None] * k1m_vec
        - kij_diag[:, None] * Nk_safe[:, None] * Mk_nodiag 
        - Mk_nodiag * in_term[:, None]
        - phi[:, None] * xk_hi[:, None] * k1m_vec 
        - 0.5 * ZETA * eff[:, None] * k1mx_vec 
        + ZETA**3 * (phi[:, None] - eff[:, None]) / (2.0 * xk_lo[:, None]) * k1mx2_vec
    )

    # 5. Previous Bin Terms (k-1)
    # Helper to shift right.
    def shift(arr, fill_value=0.0):
        return jnp.concatenate([jnp.full_like(arr[:1], fill_value), arr[:-1]], axis=0)
    
    # Standard shifts (fill with 0.0)
    Nk_prev = shift(Nk_safe)
    phi_prev = shift(phi)
    eff_prev = shift(eff)
    
    sk2mtot = shift(k1mtot)
    sk2mxtot = shift(k1mxtot)
    
    sk2m_vec = shift(k1m_vec)
    sk2mx_vec = shift(k1mx_vec)
    sk2mx2_vec = shift(k1mx2_vec)
    
    kij_diag_prev = shift(kij_diag)
    Mk_prev = shift(Mk_nodiag)

    # CRITICAL FIX: Shift xk with 1.0 (safe value) to avoid 0/0 NaN at index 0
    # The numerator terms (eff_prev, phi_prev) are 0.0 at index 0, so 0.0/1.0 = 0.0
    xk_prev = shift(xk_lo, fill_value=1.0) 

    dNdt_prev = (
        0.5 * kij_diag_prev * Nk_prev**2
        + phi_prev * sk2mtot 
        + ZETA * (eff_prev - phi_prev) / (2.0 * xk_prev) * sk2mxtot
    )
    
    dMdt_prev = (
        kij_diag_prev[:, None] * Nk_prev[:, None] * Mk_prev
        + phi_prev[:, None] * xk_lo[:, None] * sk2m_vec # Note: uses current xk_lo (multicoag.f logic)
        + 0.5 * ZETA * eff_prev[:, None] * sk2mx_vec 
        - ZETA**3 * (phi_prev[:, None] - eff_prev[:, None]) / (2.0 * xk_prev[:, None]) * sk2mx2_vec
    )

    # 6. Combine
    dNdt = dNdt_curr + dNdt_prev
    
    dMdt = jnp.zeros_like(Mk)
    dMdt = dMdt.at[:, :icomp_nodiag].set(dMdt_curr + dMdt_prev)

    return dNdt, dMdt
    
# ==========================================
# 3. Test Runner
# ==========================================
def test_equivalence():
    print("--- Starting Coagulation Rate Unit Test ---")
    
    # 1. Generate Dummy Data
    nbins = 30
    ncomp = 5 # 3 species + 2 diagnostic
    icomp_nodiag = 3
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Bin boundaries (Geometric spacing)
    xk = jnp.logspace(-15, -9, nbins + 1, dtype=jnp.float64)
    
    # Number Concentration (Random Lognormal-ish)
    Nk = jax.random.uniform(k1, (nbins,), dtype=jnp.float64, minval=0.0, maxval=100.0)
    # Make some bins empty to test Neps logic
    Nk = Nk.at[0].set(1e-10) 
    Nk = Nk.at[15].set(1e-10)

    # Mass Concentration (Random, scaling with Nk)
    Mk = jax.random.uniform(k2, (nbins, ncomp), dtype=jnp.float64) * Nk[:, None] * xk[:-1, None]

    # Coagulation Kernel (Symmetric random matrix)
    # In physics, Kij ~ (xi + xj)^3 approx. Let's make a symmetric random one.
    kij_raw = jax.random.uniform(k3, (nbins, nbins), dtype=jnp.float64) * 1e-10
    kij = (kij_raw + kij_raw.T) / 2.0
    
    print(f"Test Configuration: {nbins} bins, {ncomp} components")

    # 2. Run Implementations
    print("Running Original (Scan)...")
    start = time.time()
    # JIT compile to be fair, though first run includes compile time
    scan_fn = jax.jit(calc_coagulation_rates_scan, static_argnames=['icomp_nodiag'])
    dNdt_scan, dMdt_scan = scan_fn(Nk, Mk, kij, xk, icomp_nodiag)
    # Block to ensure execution finished
    dNdt_scan.block_until_ready()
    print(f"Scan finished in {time.time() - start:.4f}s")

    print("Running New (Vectorized)...")
    start = time.time()
    vec_fn = jax.jit(calc_coagulation_rates_vectorized, static_argnames=['icomp_nodiag'])
    dNdt_vec, dMdt_vec = vec_fn(Nk, Mk, kij, xk, icomp_nodiag)
    dNdt_vec.block_until_ready()
    print(f"Vectorized finished in {time.time() - start:.4f}s")

    # 3. Assertions
    # We use a strict tolerance because we are in float64 and the logic should be identical
    print("\nVerifying dNdt (Number Rate)...")
    try:
        np.testing.assert_allclose(dNdt_scan, dNdt_vec, rtol=1e-10, atol=1e-14)
        print("✅ dNdt Match Success")
    except AssertionError as e:
        print("❌ dNdt Mismatch!")
        print(e)
        return

    print("Verifying dMdt (Mass Rate)...")
    try:
        np.testing.assert_allclose(dMdt_scan, dMdt_vec, rtol=1e-10, atol=1e-14)
        print("✅ dMdt Match Success")
    except AssertionError as e:
        print("❌ dMdt Mismatch!")
        print(e)
        return

    print("\n🎉 ALL TESTS PASSED. The vectorized implementation is mathematically equivalent.")

if __name__ == "__main__":
    test_equivalence()