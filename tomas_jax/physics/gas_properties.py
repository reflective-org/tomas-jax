"""Gas-phase property calculations for TOMAS-JAX.

Exact port of getCondSink.f and gasdiff.f parameterizations for computing
gas diffusivity, mean molecular speed, mean free path, Knudsen number,
and the Fuchs-Sutugin transition-regime correction factor.

All functions default to H2SO4 properties but accept arbitrary molecular
weight and diffusion volume parameters for other condensing species.

WARNING: The mean free path formula (mfp = 2*Di/ms) is TOMAS-specific
and differs from the standard kinetic theory formula. This is deliberate
for consistency with the Dahneke correction factor used in getCondSink.f.

References:
    - gasdiff.f (Fuller-Schettler-Giddings method, Perry's Handbook)
    - getCondSink.f lines 77-84 (Jeff Pierce, May 2007)
    - Seinfeld & Pandis, "Atmospheric Chemistry and Physics", eqns 9.2, 11.35
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Union

from ..core.config import PI, R_GAS, MW_H2SO4, SV_H2SO4

# Air properties (from gasdiff.f)
MW_AIR = 28.9   # Molecular weight of air [g/mol]
SV_AIR = 20.1   # Atomic diffusion volume of air


def calc_air_viscosity(temp: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """Dynamic viscosity of air (getCondSink.f line 78).

    Args:
        temp: Temperature [K]

    Returns:
        mu: Dynamic viscosity [kg/m s]
    """
    return 2.5277e-7 * temp**0.75302


def calc_mean_molecular_speed(
    temp: Union[float, jnp.ndarray],
    molecular_weight: float = MW_H2SO4
) -> Union[float, jnp.ndarray]:
    """Mean molecular speed (S&P eqn 9.2, getCondSink.f line 79).

    Args:
        temp: Temperature [K]
        molecular_weight: Molecular weight [g/mol]

    Returns:
        ms: Mean molecular speed [m/s]
    """
    mw_kg = molecular_weight / 1000.0
    return jnp.sqrt(8.0 * R_GAS * temp / (PI * mw_kg))


def calc_gas_diffusivity(
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray],
    molecular_weight: float = MW_H2SO4,
    diffusion_volume: float = SV_H2SO4
) -> Union[float, jnp.ndarray]:
    """Gas diffusivity via Fuller-Schettler-Giddings (gasdiff.f lines 39-41).

    Args:
        temp: Temperature [K]
        pres: Pressure [Pa]
        molecular_weight: Molecular weight of gas [g/mol]
        diffusion_volume: Sum of atomic diffusion volumes

    Returns:
        Di: Gas diffusivity [m^2/s]
    """
    mw = molecular_weight
    mwf = jnp.sqrt((mw + MW_AIR) / (mw * MW_AIR))
    Svf = jnp.square(jnp.cbrt(diffusion_volume) + jnp.cbrt(SV_AIR))
    Di = 1.0e-7 * jnp.power(temp, 1.75) * mwf / (pres * 1.0e-5) / Svf
    return Di


def calc_mean_free_path(
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray],
    molecular_weight: float = MW_H2SO4,
    diffusion_volume: float = SV_H2SO4
) -> Union[float, jnp.ndarray]:
    """Mean free path using TOMAS-specific formula mfp = 2*Di/ms (getCondSink.f line 84).

    NOT the standard kinetic theory formula.

    Args:
        temp: Temperature [K]
        pres: Pressure [Pa]
        molecular_weight: Molecular weight [g/mol]
        diffusion_volume: Diffusion volume parameter

    Returns:
        mfp: Mean free path [m]
    """
    Di = calc_gas_diffusivity(temp, pres, molecular_weight, diffusion_volume)
    ms = calc_mean_molecular_speed(temp, molecular_weight)
    return 2.0 * Di / ms


def calc_knudsen_number(
    particle_diameter: jnp.ndarray,
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray],
    molecular_weight: float = MW_H2SO4,
    diffusion_volume: float = SV_H2SO4
) -> jnp.ndarray:
    """Knudsen number Kn = 2*mfp/Dp (getCondSink.f line 111).

    Args:
        particle_diameter: Particle diameter [m]
        temp: Temperature [K]
        pres: Pressure [Pa]
        molecular_weight: Molecular weight [g/mol]
        diffusion_volume: Diffusion volume parameter

    Returns:
        Kn: Knudsen number (dimensionless)
    """
    mfp = calc_mean_free_path(temp, pres, molecular_weight, diffusion_volume)
    return 2.0 * mfp / particle_diameter


def calc_fuchs_sutugin_correction(
    knudsen_number: jnp.ndarray,
    alpha: float = 1.0
) -> jnp.ndarray:
    """Fuchs-Sutugin (Dahneke) correction factor (getCondSink.f line 112).

    beta = (1+Kn)/(1+2*Kn*(1+Kn)/alpha)

    Args:
        knudsen_number: Kn (dimensionless)
        alpha: Accommodation coefficient, default 1.0

    Returns:
        beta: Correction factor (dimensionless)
    """
    Kn = knudsen_number
    return (1.0 + Kn) / (1.0 + 2.0 * Kn * (1.0 + Kn) / alpha)


def calc_h2so4_properties(
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray]
) -> tuple:
    """Convenience: all gas properties for H2SO4.

    Returns:
        (Di, ms, mfp, mu)
    """
    Di = calc_gas_diffusivity(temp, pres, MW_H2SO4, SV_H2SO4)
    ms = calc_mean_molecular_speed(temp, MW_H2SO4)
    mfp = calc_mean_free_path(temp, pres, MW_H2SO4, SV_H2SO4)
    mu = calc_air_viscosity(temp)
    return Di, ms, mfp, mu
