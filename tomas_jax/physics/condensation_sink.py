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
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Tuple, Union

from .gas_properties import (
    calc_gas_diffusivity,
    calc_mean_free_path,
    calc_fuchs_sutugin_correction,
)
from .density import calc_density
from ..core.config import PI, MW_H2SO4, SV_H2SO4, ICOMP_NODIAG

# Fortran getCondSink.f: parameter(Neps=1.0d10)
NEPS_CONDSINK = 1.0e10


def calc_condensation_sink(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray],
    boxvol: Union[float, jnp.ndarray],
    molecular_weight: float = MW_H2SO4,
    diffusion_volume: float = SV_H2SO4,
    accommodation_coeff: float = 1.0,
    xk: jnp.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate condensation sink from aerosol size distribution.

    Exact port of getCondSink.f lines 77-135.

    For bins with Nk > 1e10: uses actual density and mp from composition.
    For bins with Nk <= 1e10: uses default density=1500 and mp=1.4*xk[k],
    matching Fortran getCondSink.f parameter(Neps=1.0d10).

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        molecular_weight: MW of condensing species [g/mol]
        diffusion_volume: Diffusion volume parameter
        accommodation_coeff: alpha (default 1.0)
        xk: Bin boundaries [kg], shape (ibins+1,). If None, uses default 36-bin grid.

    Returns:
        CS: Condensation sink [s^-1]
        sinkfrac: Fraction of CS from each bin, shape (ibins,)
    """
    if xk is None:
        from ..core.config import xk_boundaries
        xk = xk_boundaries()

    Di = calc_gas_diffusivity(temp, pres, molecular_weight, diffusion_volume)
    mfp = calc_mean_free_path(temp, pres, molecular_weight, diffusion_volume)

    # --- Compute Dpk following Fortran getCondSink.f lines 89-113 ---
    # For bins with Nk > Neps: actual density and mp
    # For bins with Nk <= Neps: default density=1500, mp=1.4*xk[k]
    has_particles = Nk > NEPS_CONDSINK

    # Actual properties (for populated bins)
    Mktot = jnp.sum(Mk, axis=1)  # total wet mass per bin
    mp_actual = Mktot / jnp.maximum(Nk, 1e-30)
    density_actual = calc_density(Mk)

    # Default properties (for sparse bins)
    mp_default = 1.4 * xk[:-1]
    density_default = 1500.0

    mp = jnp.where(has_particles, mp_actual, mp_default)
    density = jnp.where(has_particles, density_actual, density_default)

    # Dpk = ((mp/density) * (6/pi))^(1/3)
    Dpk = jnp.cbrt(mp / density * (6.0 / PI))

    # Kn = 2*mfp/Dpk (getCondSink.f line 111)
    Kn = 2.0 * mfp / jnp.maximum(Dpk, 1e-30)

    # beta = (1+Kn)/(1+2*Kn*(1+Kn)/alpha) (getCondSink.f line 112)
    beta = calc_fuchs_sutugin_correction(Kn, accommodation_coeff)

    # Guard: bins with Dpk=0 produce Kn=inf -> beta=NaN.
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
