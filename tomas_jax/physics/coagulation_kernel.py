"""Coagulation kernel calculation with float64 precision.

Port of multicoag.f lines 194-205.
Refactored to use JAX broadcasting for high-performance XLA compilation.

References:
    Seinfeld & Pandis (S&P) "Atmospheric Chemistry and Physics"
    - Equation 12.46: Brownian coagulation coefficient
    - Equation 12.50: Non-continuum correction factor (beta)
    - Equation 12.51: Knudsen number
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Union

def calc_coagulation_kernel(
    Dpk: jnp.ndarray,
    Dk: jnp.ndarray,
    ck: jnp.ndarray,
    boxvol: Union[float, jnp.ndarray]
) -> jnp.ndarray:
    """Calculate coagulation kernel matrix for all bin pairs.

    Implements Brownian coagulation with non-continuum correction (Fuchs).
    Uses implicit broadcasting (O(1) memory overhead) instead of meshgrid.

    Args:
        Dpk: Particle diameters [m], shape (ibins,)
        Dk: Diffusivities [m²/s], shape (ibins,)
        ck: Mean thermal speeds [m/s], shape (ibins,)
        boxvol: Grid cell volume [cm³] (Scalar)

    Returns:
        kij: Coagulation kernel [s⁻¹], shape (ibins, ibins)
             Normalized by grid cell volume for use with total particle number.
    """
    # 1. Broadcasting Setup
    # Reshape arrays to (N, 1) and (1, N) to trigger implicit broadcasting
    # i represents rows (receiving bin), j represents columns (colliding bin)
    Dpk_i, Dpk_j = Dpk[:, None], Dpk[None, :]
    Dk_i, Dk_j = Dk[:, None], Dk[None, :]
    ck_i, ck_j = ck[:, None], ck[None, :]

    # 2. Physics Calculations (Seinfeld & Pandis Ch 12)
    
    # Denominator for Kn (Mean thermal speed of the pair)
    # S&P 12.51 denominator: sqrt(ci^2 + cj^2)
    # Use jnp.square for efficiency over jnp.power
    c_pair = jnp.sqrt(jnp.square(ck_i) + jnp.square(ck_j))
    
    # Knudsen number
    # Dpk > 0 guaranteed by properties.py (mp clamped to 1e-25 for empty bins).
    Kn = (4.0 * (Dk_i + Dk_j)) / (c_pair * (Dpk_i + Dpk_j))

    # Beta correction factor (Fuchs form)
    # S&P 12.50
    beta = (1.0 + Kn) / (1.0 + 2.0 * Kn * (1.0 + Kn))

    # Uncorrected Brownian kernel (Continuum limit)
    # S&P 12.46 (without beta)
    K_continuum = 2.0 * jnp.pi * (Dpk_i + Dpk_j) * (Dk_i + Dk_j)
    
    # Final Kernel [m³/s]
    kij_phys = K_continuum * beta

    # 3. Unit Conversion
    # Convert m³/s -> cm³/s (factor 1e6)
    # Divide by boxvol [cm³] to get frequency [s⁻¹]
    kij = kij_phys * 1.0e6 / boxvol

    return kij


def calc_single_kernel_element(
    Dpk_i: float,
    Dpk_j: float,
    Dk_i: float,
    Dk_j: float,
    ck_i: float,
    ck_j: float,
    boxvol: float
) -> float:
    """Calculate single coagulation kernel element for bins i and j.
    
    Scalar implementation for unit testing and verification.
    Mathematically identical to the vectorized version.
    """
    # S&P 12.51
    # OPTIMIZATION: Updated to use jnp.square to match vectorized version
    c_pair = jnp.sqrt(jnp.square(ck_i) + jnp.square(ck_j))
    
    Kn = (4.0 * (Dk_i + Dk_j)) / (c_pair * (Dpk_i + Dpk_j))

    # S&P 12.50
    beta = (1.0 + Kn) / (1.0 + 2.0 * Kn * (1.0 + Kn))

    # S&P 12.46
    kij_phys = 2.0 * jnp.pi * (Dpk_i + Dpk_j) * (Dk_i + Dk_j) * beta

    return kij_phys * 1.0e6 / boxvol