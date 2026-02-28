"""Condensation sink calculations for TOMAS-JAX.

Exact port of getCondSink.f (lines 1-135).

The condensation sink CS [s^-1] is the pseudo-first-order rate constant
for gas-phase loss to the aerosol population. It depends on gas diffusivity,
particle diameters, number concentrations, and the Fuchs-Sutugin correction
for non-continuum effects.

    CS = 2 * pi * Di * sum_k(Dpk * Nk * beta_k) / (boxvol * 1e-6)

The per-bin sink fraction ``sinkfrac`` distributes condensing mass across
bins in proportion to each bin's contribution to the total sink.

References:
    - getCondSink.f (Jeff Pierce, May 2007)
    - Seinfeld & Pandis eqn 11.35
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Tuple, Union

from .gas_properties import (
    calc_gas_diffusivity,
    calc_knudsen_number,
    calc_fuchs_sutugin_correction,
)
from .properties import calc_particle_properties
from ..core.config import PI, MW_H2SO4, SV_H2SO4


def calc_condensation_sink(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray],
    boxvol: Union[float, jnp.ndarray],
    molecular_weight: float = MW_H2SO4,
    diffusion_volume: float = SV_H2SO4,
    accommodation_coeff: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate condensation sink from aerosol size distribution.

    Exact port of getCondSink.f lines 77-135.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        molecular_weight: MW of condensing species [g/mol]
        diffusion_volume: Diffusion volume parameter
        accommodation_coeff: alpha (default 1.0)

    Returns:
        CS: Condensation sink [s^-1]
        sinkfrac: Fraction of CS from each bin, shape (ibins,)
    """
    Di = calc_gas_diffusivity(temp, pres, molecular_weight, diffusion_volume)
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres)

    Kn = calc_knudsen_number(
        Dpk, temp, pres, molecular_weight, diffusion_volume
    )
    beta = calc_fuchs_sutugin_correction(Kn, accommodation_coeff)

    # Guard: bins with Dpk=0 produce Kn=inf -> beta=NaN.
    # Zero out their contribution (no diameter = no condensation).
    safe_beta = jnp.where(Dpk > 0.0, beta, 0.0)
    sink_contributions = Dpk * Nk * safe_beta
    CS_sum = jnp.sum(sink_contributions)

    # CS = 2*pi*Di*sum / (boxvol*1e-6)  [boxvol cm^3 -> m^3]
    CS = 2.0 * PI * Di * CS_sum / (boxvol * 1e-6)

    sinkfrac = jnp.where(
        CS_sum > 1e-30,
        sink_contributions / CS_sum,
        jnp.zeros_like(Dpk)
    )

    return CS, sinkfrac
