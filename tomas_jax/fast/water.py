"""Tabazadeh (1997) H2SO4/H2O equilibrium water uptake, pure JAX.

Replaces the ISORROPIA ammonium-bisulfate fit (physics/water_equilibrium.py)
for the binary-sulfate reduced model. The equilibrium weight percent is
found where the solution water vapor pressure (Table 1 of Tabazadeh et al.
1997, GRL 24, 1931-1934) matches the ambient partial pressure
(RH/100 * P_sat).

Validity and boundary handling:
  - Fitted range T = 185-260 K, wt% = 10-80. Outside the T range the
    Clausius-Clapeyron-like row form ln P = a + b/T + c/T^2 is smoothly
    EXTRAPOLATED (not clamped): clamping T at 260 K would bias the entire
    troposphere of a global run, while the extrapolation to ~300 K agrees
    with 25 C water-activity data to ~2 wt% (e.g. ~43 wt% at RH=50%).
  - RH is clipped to [1e-3, 99] %: RH->0 returns the 80 wt% table ceiling
    (as the numpy reference did); RH>=100 returns the 10 wt% table floor,
    capping water at ~9x the acid mass (analogous to the ISORROPIA fit's
    wr<=30 cap; the reference's 0 wt% would give infinite water).

All functions broadcast over a leading cell axis.
"""
import jax.numpy as jnp

from .config import MW_H2SO4, MW_SO4, SRTSO4, SRTH2O

# Tabazadeh et al. (1997) Table 1: P_H2O(mb) = exp[a + b/T + c/T^2]
# Columns: (weight_percent, a, b, c). Copied from physics/radiative_forcing.py.
_TABAZADEH_TABLE1 = jnp.array([
    [10, 19.726, -4364.8, -147620],
    [15, 19.747, -4390.9, -144690],
    [20, 19.761, -4414.7, -142940],
    [25, 19.794, -4451.1, -140870],
    [30, 19.883, -4519.2, -136500],
    [35, 20.078, -4644.0, -127240],
    [40, 20.379, -4828.5, -112550],
    [45, 20.637, -5011.5, -98811],
    [50, 20.682, -5121.3, -94033],
    [55, 20.555, -5177.6, -96984],
    [60, 20.405, -5252.1, -100840],
    [65, 20.383, -5422.4, -97966],
    [70, 20.585, -5743.8, -83701],
    [75, 21.169, -6310.6, -48396],
    [80, 21.808, -6985.9, -12170],
], dtype=jnp.float64)

# Saturation vapor pressure of pure water [mbar]: ln P = c0 + c1/T + c2/T^2 + c3/T^3
_TABAZADEH_EQ1 = (18.452406985, -3505.1578801, -330918.55082, 12725068.262)


def h2so4_equilibrium_wt(temp, rh_percent):
    """Equilibrium H2SO4 weight percent of a binary H2SO4/H2O droplet.

    Pure-JAX, batched port of radiative_forcing.h2so4_equilibrium_wt.

    Args:
        temp: Temperature [K], scalar or shape (C,)
        rh_percent: Relative humidity [%], scalar or shape (C,)

    Returns:
        wt_percent in [10, 80], same shape as the broadcast inputs.
    """
    temp = jnp.asarray(temp, dtype=jnp.float64)
    rh = jnp.clip(jnp.asarray(rh_percent, dtype=jnp.float64), 1e-3, 99.0)

    c0, c1, c2, c3 = _TABAZADEH_EQ1
    ln_p_sat = c0 + c1 / temp + c2 / temp**2 + c3 / temp**3
    # Interpolation coordinate: -ln P increases with concentration
    x = -(jnp.log(rh / 100.0) + ln_p_sat)

    wt_tab = _TABAZADEH_TABLE1[:, 0]
    T = temp[..., None]
    xp = -(_TABAZADEH_TABLE1[:, 1]
           + _TABAZADEH_TABLE1[:, 2] / T
           + _TABAZADEH_TABLE1[:, 3] / T**2)  # (..., 15), increasing

    # Piecewise-linear interpolation with end clamping (== jnp.interp,
    # but supports a per-cell xp table).
    n = wt_tab.shape[0]
    i = jnp.clip(jnp.sum(xp < x[..., None], axis=-1), 1, n - 1)
    x0 = jnp.take_along_axis(xp, (i - 1)[..., None], axis=-1)[..., 0]
    x1 = jnp.take_along_axis(xp, i[..., None], axis=-1)[..., 0]
    frac = jnp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return wt_tab[i - 1] + frac * (wt_tab[i] - wt_tab[i - 1])


def equilibrium_water(Mk, temp, rh):
    """Set the diagnostic water column to Tabazadeh equilibrium.

    water = m_acid * (100/wt - 1), with m_acid = mSO4 * 98/96 (wt% is
    defined on H2SO4 mass; the H2/96 is folded into the water diagnostic
    so wet mass = m_acid * 100/wt exactly).

    Args:
        Mk: Mass [kg/grid cell], shape (..., nbins, 2)
        temp: [K], scalar or (C,)
        rh: fraction 0-1, scalar or (C,)

    Returns:
        Mk with column SRTH2O replaced.
    """
    wt = h2so4_equilibrium_wt(temp, jnp.asarray(rh) * 100.0)
    acid = Mk[..., SRTSO4] * (MW_H2SO4 / MW_SO4)
    water = acid * (100.0 / wt[..., None] - 1.0)
    return Mk.at[..., SRTH2O].set(water)
