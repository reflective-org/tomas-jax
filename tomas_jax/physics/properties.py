"""Particle property calculations (diameter, diffusivity, thermal speed).

Port of multicoag.f lines 120-150.
Optimized for JAX JIT compilation (removed Scipy dependencies).

References:
    Seinfeld & Pandis (S&P) "Atmospheric Chemistry and Physics"
    Phillips (1975) - Flux matching approximation for Diffusivity
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Tuple, Union

# Import optimized density calculator
from .density import calc_density

# Import constants
from ..core.config import PI, KB, R_GAS, MOLAR_MASS_AIR

def calc_air_properties(
    temp: Union[float, jnp.ndarray], 
    pres: Union[float, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate air viscosity and mean free path.

    Args:
        temp: Temperature [K]
        pres: Pressure [Pa]

    Returns:
        mu: Air viscosity [kg/m/s]
        mfp: Mean free path of air molecule [m]
    """
    # Air viscosity (Sutherland's law approximation used in TOMAS)
    # multicoag.f line 122
    mu = 2.5277e-7 * jnp.power(temp, 0.75302)

    # Mean free path (S&P eqn 8.6)
    # multicoag.f line 123: mfp = 2*mu / (pres * sqrt(8*M_air/(pi*R*T)))
    # This equals 2*mu / (rho_air * c_bar_air) where rho = pres*M/(R*T)
    mfp = 2.0 * mu / (pres * jnp.sqrt(8.0 * MOLAR_MASS_AIR / (PI * R_GAS * temp)))

    return mu, mfp


def calc_diffusivity_vectorized(
    temp: Union[float, jnp.ndarray], 
    mu: Union[float, jnp.ndarray], 
    Dpk: jnp.ndarray, 
    mfp: Union[float, jnp.ndarray]
) -> jnp.ndarray:
    """Calculate particle diffusivity with non-continuum correction.

    Uses the Phillips (1975) flux-matching approximation for the 
    Cunningham slip correction, as used in the original Fortran code.
    
    Args:
        temp: Temperature [K]
        mu: Viscosity [kg/m/s]
        Dpk: Particle Diameter [m] (nbins,)
        mfp: Mean Free Path [m]

    Returns:
        Dk: Diffusivity [m²/s] (nbins,)
    """
    # Knudsen number
    # Add epsilon to Dpk to prevent division by zero for empty bins
    Kn = 2.0 * mfp / (Dpk + 1e-30)

    # Optimized polynomial evaluation (Phillips approximation)
    # multicoag.f lines 147-148
    
    # Numerator: 5 + 4Kn + 6Kn^2 + 18Kn^3 (Horner form)
    num = 5.0 + Kn * (4.0 + Kn * (6.0 + 18.0 * Kn))

    # Denominator: 5 - Kn + (8 + pi)Kn^2
    den = 5.0 - Kn + (8.0 + PI) * Kn * Kn
    
    # Slip correction factor
    cc = num / den

    # Stokes-Einstein Diffusivity
    Dk = (KB * temp * cc) / (3.0 * PI * mu * (Dpk + 1e-30))
    
    return Dk


def calc_particle_properties(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate particle diameter, diffusivity, and thermal speed.

    FULLY VECTORIZED: Operates on entire arrays at once.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        temp: Temperature [K]
        pres: Pressure [Pa]

    Returns:
        Dpk: Particle diameter [m], shape (ibins,)
        Dk: Diffusivity [m²/s], shape (ibins,)
        ck: Mean thermal speed [m/s], shape (ibins,)
    """
    # 1. Calculate Air Properties (Scalars or 0-D arrays)
    mu, mfp = calc_air_properties(temp, pres)

    # 2. Calculate Aerosol Density (Vectorized via density.py)
    # We pass the full Mk matrix. The density module handles indices.
    rho_p = calc_density(Mk)

    # 3. Calculate Average Mass per Particle (mp)
    # Sum all species masses for each bin
    total_mass_conc = jnp.sum(Mk, axis=1)
    
    # Safety: Clamp denominator before division to prevent NaN in XLA select
    # (jnp.where evaluates both branches; 0/0 produces NaN that poisons gradients)
    safe_Nk = jnp.maximum(Nk, 1e-20)
    mp = jnp.where(Nk > 1e-20, total_mass_conc / safe_Nk, 1e-25)

    # 4. Calculate Diameter (Dpk)
    # Volume = mp / rho
    # Dpk = (6 * Vol / pi)^(1/3)
    vol = mp / rho_p
    Dpk = jnp.cbrt(vol * (6.0 / PI))

    # 5. Calculate Thermal Speed (ck)
    # ck = sqrt(8 kT / pi mp)
    ck = jnp.sqrt(8.0 * KB * temp / (PI * mp))

    # 6. Calculate Diffusivity (Dk)
    Dk = calc_diffusivity_vectorized(temp, mu, Dpk, mfp)

    return Dpk, Dk, ck