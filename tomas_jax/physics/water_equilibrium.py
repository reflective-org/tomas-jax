"""Water equilibrium calculations for TOMAS-JAX.

Exact port of ezwatereqm.f, waterso4.f, and waternacl.f.

Calculates equilibrium water uptake by aerosol particles based on
relative humidity using piecewise polynomial fits to ISORROPIA
thermodynamic model results at 273 K.

Assumptions:
    - Sulfate particles treated as ammonium bisulfate (NH4HSO4),
      converted from SO4 mass with a factor of 1.2.
    - Organic aerosol assumed to have the same water uptake as sulfate.
    - Sea salt water uptake is computed but currently disabled in TOMAS.
    - Instantaneous equilibrium (no kinetic limitations).

References:
    - ezwatereqm.f (Peter Adams, March 2000)
    - waterso4.f (Peter Adams, November 2001) - ammonium bisulfate fits
    - waternacl.f (Peter Adams, November 2001) - sea salt fits
    - ISORROPIA thermodynamic model (Nenes et al., 1998)
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Union

from ..core.config import SRTSO4, SRTORG1, SRTH2O, IORG


def water_uptake_sulfate(rh_percent: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """Wet/dry mass ratio for ammonium bisulfate aerosol (waterso4.f).

    Piecewise polynomial fit based on ISORROPIA results at 273 K.

    Args:
        rh_percent: Relative humidity [%], range 0-100

    Returns:
        wr: Wet/dry mass ratio (1.0 = dry)
    """
    rh = jnp.clip(rh_percent, 1.0, 99.0)

    wr_96 = (0.7540688 * rh**3 - 218.5647 * rh**2
             + 21118.19 * rh - 6.801999e5)
    wr_91 = 8.517e-2 * rh**2 - 15.388 * rh + 698.25
    wr_81 = 8.2696e-3 * rh**2 - 1.3076 * rh + 53.697
    wr_61 = 9.3562e-4 * rh**2 - 0.10427 * rh + 4.3155
    wr_41 = 1.9149e-4 * rh**2 - 8.8619e-3 * rh + 1.2535
    wr_low = 5.1337e-5 * rh**2 + 2.6266e-3 * rh + 1.0149

    wr = jnp.where(
        rh > 96.0, wr_96,
        jnp.where(rh > 91.0, wr_91,
        jnp.where(rh > 81.0, wr_81,
        jnp.where(rh > 61.0, wr_61,
        jnp.where(rh > 41.0, wr_41, wr_low)))))

    return jnp.clip(wr, 1.0, 30.0)


def water_uptake_seasalt(rh_percent: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """Wet/dry mass ratio for sea salt aerosol (waternacl.f).

    Piecewise polynomial fit based on ISORROPIA results at 273 K.

    Args:
        rh_percent: Relative humidity [%], range 0-100

    Returns:
        wr: Wet/dry mass ratio (1.0 = dry)
    """
    rh = jnp.clip(rh_percent, 1.0, 99.0)

    wr_90 = (5.1667642e-2 * rh**3 - 14.153121 * rh**2
             + 1292.8377 * rh - 3.9373536e4)
    wr_80 = (1.0629e-3 * rh**3 - 0.25281 * rh**2
             + 20.171 * rh - 5.3558e2)
    wr_50 = (4.2967e-5 * rh**3 - 7.3654e-3 * rh**2
             + 0.46312 * rh - 7.5731)
    wr_20 = (2.9443e-5 * rh**3 - 2.4739e-3 * rh**2
             + 7.3430e-2 * rh + 1.3727)
    wr_low = 1.17

    wr = jnp.where(
        rh > 90.0, wr_90,
        jnp.where(rh > 80.0, wr_80,
        jnp.where(rh > 50.0, wr_50,
        jnp.where(rh > 20.0, wr_20, wr_low))))

    return jnp.clip(wr, 1.0, 45.0)


def calc_equilibrium_water(
    Mk: jnp.ndarray,
    rh: float,
) -> jnp.ndarray:
    """Calculate equilibrium water content for each aerosol bin.

    Exact port of ezwatereqm.f (Peter Adams, March 2000).

    Args:
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        rh: Relative humidity [fraction 0-1]

    Returns:
        Mk_new: Updated mass with equilibrium water, shape (ibins, icomp)
    """
    rh_percent = jnp.clip(rh * 100.0, 1.0, 99.0)
    wr_so4 = water_uptake_sulfate(rh_percent)
    wr_nacl = water_uptake_seasalt(rh_percent)

    # Sulfate mass converted to NH4HSO4 (factor 1.2)
    so4_mass = Mk[:, SRTSO4] * 1.2

    # Sea salt (disabled in TOMAS)
    nacl_mass = jnp.zeros(Mk.shape[0])

    # Organic mass (same water uptake as sulfate)
    org_mass = jnp.sum(Mk[:, SRTORG1:SRTORG1 + IORG], axis=1)

    # Water = dry_mass * (wr - 1)
    water_mass = ((so4_mass + org_mass) * (wr_so4 - 1.0)
                  + nacl_mass * (wr_nacl - 1.0))

    Mk_new = Mk.at[:, SRTH2O].set(water_mass)
    return Mk_new
