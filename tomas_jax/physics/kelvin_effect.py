"""Kelvin effect correction for organic condensation in TOMAS-JAX.

The Kelvin effect increases the equilibrium vapor pressure over small
particles due to surface curvature. For organics:

    Ke(k) = exp(4 × σ × MW / (R × T × ρ × Dp(k)))

where σ is surface tension, MW is molecular weight, ρ is density, and
Dp is particle diameter.

Ke >= 1 always; approaches 1 for large particles (Dp >> nm).

**Open question**: Whether Kelvin effect should also apply to inorganic
H2SO4 condensation. Fortran soacond.f applies it only to organics.
The Fortran ezcond.f (H2SO4 path) does NOT include Kelvin. Some models
include it for H2SO4 on sub-3nm clusters. For now, implement for
organics only and flag as future investigation.

References:
    - soacond.f line 335 (Jeff Pierce, September 2010)
    - Seinfeld & Pandis (2016), eqn 9.40
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ..core.config import PI, R_GAS


def calc_kelvin_factor(Dpk, sigma, mw, rho, temp):
    """Kelvin correction factor for each size bin.

    Ke = exp(4 × σ × MW / (R × T × ρ × Dp))

    Matches soacond.f line 335:
        scalefactor = exp((storg * 0.001 * mw(j)) / (8.314 * temp * 1200. * Dpk(k)))

    Note: The Fortran formula uses σ in mN/m (hence ×0.001), MW in g/mol,
    ρ in kg/m³, and Dp in m. The factor 4/(R×T×ρ×Dp) arises from the
    molar volume Vm = MW/ρ and the 4σVm/(R×T×Dp) form of Kelvin.

    Args:
        Dpk: Particle diameters [m], shape (nbins,)
        sigma: Surface tension [N/m] (e.g. 0.025 for organics)
        mw: Molecular weight [g/mol] (e.g. 200 for SOA)
        rho: Organic density [kg/m³] (e.g. 1200)
        temp: Temperature [K]

    Returns:
        Ke: Kelvin correction factors, shape (nbins,). Always >= 1.0.
    """
    mw_kg = mw * 1.0e-3  # g/mol → kg/mol
    exponent = 4.0 * sigma * mw_kg / (R_GAS * temp * rho * jnp.maximum(Dpk, 1.0e-30))
    Ke = jnp.exp(exponent)
    return jnp.maximum(Ke, 1.0)


def calc_diameters_from_xk(xk, Nk, Mk, icomp_nodiag):
    """Compute wet particle diameters from mass and number.

    For bins with particles: Dp = (6 × mp / (π × ρ_eff))^(1/3)
    where mp = Mtot/Nk and ρ_eff is from total wet composition.

    For sparse bins: uses geometric mean of bin boundaries with
    default density = 1500 kg/m³.

    Args:
        xk: Bin boundaries [kg], shape (nbins+1,)
        Nk: Number concentration [#/grid cell], shape (nbins,)
        Mk: Mass concentration [kg/grid cell], shape (nbins, icomp)
        icomp_nodiag: Number of non-diagnostic species

    Returns:
        Dpk: Particle diameters [m], shape (nbins,)
    """
    from .density import calc_density

    NEPS = 1.0e10
    has_particles = Nk > NEPS

    Mktot = jnp.sum(Mk, axis=1)
    mp_actual = Mktot / jnp.maximum(Nk, 1.0e-30)
    density_actual = calc_density(Mk)

    mp_default = 1.4 * xk[:-1]
    density_default = 1500.0

    mp = jnp.where(has_particles, mp_actual, mp_default)
    density = jnp.where(has_particles, density_actual, density_default)

    Dpk = jnp.cbrt(mp / density * (6.0 / PI))
    return Dpk
