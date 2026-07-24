"""Binary H2SO4/H2O aerosol density (Tang 1997), batched.

This is exactly the NH4=organics=0 limit of physics/density.calc_density
(aerodens.f): with no ammonium the ammonium-sulfate ratio is zero, all
sulfate is speciated as H2SO4, and the mixture density collapses to the
Tang sulfuric-acid polynomial evaluated at

    x = 100 * (mSO4 * 98/96) / (mSO4 + mH2O)

(the Fortran convention: H2SO4-basis solute mass over raw SO4+H2O total).
Kept unclipped so this function is bit-identical to the full model's
limit, including x slightly above 100 for dry bins.

Reference:
    Tang, I. N. (1997). JGR, 102, 1883-1893.
"""
import jax.numpy as jnp

from .config import MW_H2SO4, MW_SO4, SRTSO4, SRTH2O


def calc_density(Mk: jnp.ndarray) -> jnp.ndarray:
    """Density of binary H2SO4/H2O particles.

    Args:
        Mk: Mass [kg/grid cell], shape (..., nbins, 2) [SO4, H2O]

    Returns:
        Density [kg/m^3], shape (..., nbins)
    """
    mso4 = Mk[..., SRTSO4]
    mh2o = Mk[..., SRTH2O]
    mtot = mso4 + mh2o
    mtot_safe = jnp.maximum(mtot, 1e-30)

    x = 100.0 * (mso4 * (MW_H2SO4 / MW_SO4)) / mtot_safe

    # Tang (1997) sulfuric acid, Horner form (matches physics/density.py ds0)
    ds0 = 0.9971 + x * (
        7.367e-3 + x * (-4.934e-5 + x * (1.754e-6 + x * -1.104e-8))
    )
    density = ds0 * 1000.0

    # Fortran aerodens.f empty-bin fallback
    return jnp.where(mtot < 1e-15, 1000.0, density)
