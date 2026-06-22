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
# float64 enforced by core/config.py
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


# =========================================================================
# Tabazadeh et al. (1997) — pure H2SO4/H2O binary water uptake (no NH3)
# =========================================================================
# For a clean (NH3-free) stratospheric sulfate aerosol the equilibrium
# composition is a binary H2SO4/H2O solution, NOT ammonium bisulfate. The
# equilibrium H2SO4 weight percent depends on (T, RH) via the water-vapor-
# pressure parameterization of Tabazadeh et al. (1997), GRL Table 1:
#     ln P_H2O(mb) = a + b/T + c/T^2     (over the solution)
# and the pure-water saturation pressure (their eq. 1). At equilibrium the
# solution vapor pressure equals the ambient partial pressure RH*P_sat(T),
# which fixes the weight percent. Valid T = 185-260 K, wt% = 10-80%.
#
# Reference: Tabazadeh, A., O. B. Toon, S. L. Clegg, P. Hamill (1997),
#   A new parameterization of H2SO4/H2O aerosol composition, GRL 24(15).
_TABZ_WT = jnp.array([10., 15., 20., 25., 30., 35., 40., 45., 50.,
                      55., 60., 65., 70., 75., 80.])
_TABZ_A = jnp.array([19.726, 19.747, 19.761, 19.794, 19.883, 20.078, 20.379,
                     20.637, 20.682, 20.555, 20.405, 20.383, 20.585, 21.169,
                     21.808])
_TABZ_B = jnp.array([-4364.8, -4390.9, -4414.7, -4451.1, -4519.2, -4644.0,
                     -4828.5, -5011.5, -5121.3, -5177.6, -5252.1, -5422.4,
                     -5743.8, -6310.6, -6985.9])
_TABZ_C = jnp.array([-147620., -144690., -142940., -140870., -136500., -127240.,
                     -112550., -98811., -94033., -96984., -100840., -97966.,
                     -83701., -48396., -12170.])
_TABZ_EQ1 = (18.452406985, -3505.1578801, -330918.55082, 12725068.262)


def h2so4_weight_percent(temp, rh):
    """Equilibrium H2SO4 weight percent for a binary H2SO4/H2O droplet (JIT).

    Tabazadeh et al. (1997). `temp` [K], `rh` [fraction 0-1]. Returns wt% [%],
    clamped to the tabulated range (10-80%)."""
    rh_c = jnp.clip(rh, 1e-4, 0.9999)
    c0, c1, c2, c3 = _TABZ_EQ1
    ln_p_sat = c0 + c1 / temp + c2 / temp**2 + c3 / temp**3
    ln_p_target = jnp.log(rh_c) + ln_p_sat
    # Solution vapor pressure per tabulated wt% (decreasing in wt%).
    ln_p_sol = _TABZ_A + _TABZ_B / temp + _TABZ_C / temp**2
    # interp requires increasing xp -> negate (both sides) so wt% increases.
    return jnp.interp(-ln_p_target, -ln_p_sol, _TABZ_WT)


def calc_equilibrium_water_h2so4(
    Mk: jnp.ndarray,
    rh: float,
    temp: float,
) -> jnp.ndarray:
    """Equilibrium water for a PURE H2SO4/H2O aerosol (Tabazadeh 1997, JIT).

    Drop-in alternative to `calc_equilibrium_water` for clean (NH3-free) cases.
    The wet/dry mass ratio is `wr = 100/wt%` (solution mass per unit dry acid);
    water_mass = dry_acid_mass * (wr - 1), set into the H2O column.

    Args:
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp).
        rh: Relative humidity [fraction 0-1].
        temp: Temperature [K].

    Returns:
        Mk_new: Updated mass with binary-solution equilibrium water.

    Note: the dry-acid basis is `Mk[:,SRTSO4]` (+ organics, treated like acid).
    Whether that column carries SO4 (96) or H2SO4 (98) is ambiguous in TOMAS
    (~2% on the dry basis); we use it directly, matching the ISORROPIA path's
    convention of taking Mk[:,SRTSO4] as the dry basis.
    """
    wt = h2so4_weight_percent(temp, rh)
    wr = 100.0 / wt
    dry_acid = Mk[:, SRTSO4] + jnp.sum(Mk[:, SRTORG1:SRTORG1 + IORG], axis=1)
    water_mass = dry_acid * (wr - 1.0)
    return Mk.at[:, SRTH2O].set(water_mass)


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
