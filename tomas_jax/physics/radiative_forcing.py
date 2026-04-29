"""Radiative forcing calculation for TOMAS-JAX aerosol distributions.

Computes the direct shortwave radiative forcing (RF) of the aerosol
size distribution using the Chylek & Wong (1995) simplified equation,
following the implementation in Pierce et al. (2010).

The calculation pipeline:
    1. Mie scattering: Qsca, gsca for each TOMAS bin (Bohren & Huffman)
    2. Upscatter fraction: Wiscombe & Grams (1976) eqn 22
    3. Optical depth: from number concentration and scattering cross section
    4. RF = -(S₀/4) × T_atm² × (1-α)² × 2β × τ   [Chylek & Wong 1995]

Two usage modes:
    - Global annual average (default): uses representative albedo and solar flux
    - Latitude/month resolved: loops over latitudes and months as in getRF_monthly.py

References:
    Chylek, P. and J. Wong (1995), Effect of absorbing aerosols on global
        radiation budget, GRL, 22, 929-931.
    Pierce, J. R. et al. (2010), Efficient formation of stratospheric aerosol
        for climate engineering, GRL, 37, L18805.
    Wiscombe, W. J. and G. W. Grams (1976), The backscattered fraction in
        two-stream approximations, J. Atmos. Sci., 33, 2440-2451.
    Bohren, C. F. and D. R. Huffman (1983), Absorption and Scattering of
        Light by Small Particles, Wiley-Interscience.

Usage:
    >>> from tomas_jax.physics.radiative_forcing import (
    ...     precompute_mie_properties, compute_rf)
    >>> mie = precompute_mie_properties(state.xk)
    >>> rf_total, rf_per_bin = compute_rf(state.Nk, mie, column_area=1e10)
"""
import numpy as np
from scipy.integrate import quad
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .bhmie import bhmie, bhmie_qsca_jax
from ..core.config import PI, SRTSO4, SRTH2O


# =========================================================================
# Constants
# =========================================================================
SOLAR_CONSTANT = 1361.0     # W/m² (TSI, Kopp & Lean 2011)
TATM_DEFAULT = 0.85         # Atmospheric transmittance above aerosol layer
ALBEDO_GLOBAL_AVG = 0.30    # Global annual average surface albedo
DENSITY_SULFATE = 1770.0    # kg/m³ (ammonium sulfate, default)
WAVELENGTH_DEFAULT = 0.55e-6  # m (550 nm, peak solar-weighted scattering)
REFINDEX_SULFATE = complex(1.4, 1e-8)  # 75% H₂SO₄/H₂O solution

# Pierce et al. (2010) SI parameters for stratospheric aerosol RF
SOLAR_CONSTANT_PIERCE = 1370.0   # W/m² (Pierce et al. 2010)
TATM_STRATOSPHERIC = 1.0         # No attenuation above stratospheric layer
CLOUD_FRACTION_DEFAULT = 0.6     # Global average cloud fraction (Pierce SI)
ALBEDO_SURFACE_CLEARSKY = 0.15   # Clear-sky surface albedo (Pierce SI)
DENSITY_H2SO4_SOLUTION = 1700.0  # kg/m³ (~75 wt% H₂SO₄/H₂O at T=220K, RH=10%)
WAVELENGTH_PIERCE = 0.500e-6     # m (500 nm, Pierce getRF_monthly.py default)
H2SO4_WEIGHT_FRACTION = 0.75     # H₂SO₄ mass fraction in solution droplet (placeholder, use h2so4_equilibrium_wt)

# =========================================================================
# Tabazadeh et al. (1997) H₂SO₄/H₂O binary equilibrium
# =========================================================================
# Table 1: Water vapor pressure over aqueous H₂SO₄
#   P_H₂O(mb) = exp[a + b/T + c/T²]
# Columns: (weight_percent, a, b, c)
_TABAZADEH_TABLE1 = np.array([
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
], dtype=np.float64)

# Equation (1): saturation vapor pressure of pure water [mbar]
#   ln P_H₂O = c0 + c1/T + c2/T² + c3/T³
_TABAZADEH_EQ1 = (18.452406985, -3505.1578801, -330918.55082, 12725068.262)

# H₂SO₄/H₂O solution density [kg/m³] at 25°C (CRC Handbook)
# Columns: (weight_percent, density)
_H2SO4_DENSITY_TABLE = np.array([
    [0,  998],
    [10, 1066],
    [20, 1139],
    [30, 1219],
    [40, 1303],
    [50, 1395],
    [60, 1498],
    [70, 1611],
    [75, 1669],
    [80, 1727],
    [90, 1814],
    [98, 1836],
], dtype=np.float64)


def h2so4_equilibrium_wt(temp, rh_percent):
    """Compute equilibrium H₂SO₄ weight percent for a binary H₂SO₄/H₂O droplet.

    Uses Tabazadeh et al. (1997) GRL Table 1 vapor pressure parameterization.
    Valid for T = 185–260 K, wt% = 10–80%.

    At equilibrium, the water vapor pressure over the solution equals the
    ambient partial pressure: P_solution(wt%, T) = (RH/100) × P_sat(T).

    Args:
        temp: Temperature [K]. Scalar.
        rh_percent: Relative humidity [%]. Scalar.

    Returns:
        wt_percent: H₂SO₄ weight percent at equilibrium [%].

    Reference:
        Tabazadeh, A., O. B. Toon, S. L. Clegg, and P. Hamill (1997),
        A new parameterization of H₂SO₄/H₂O aerosol composition:
        Atmospheric implications, GRL, 24(15), 1931-1934.
    """
    T = float(temp)
    rh = float(rh_percent)

    if rh <= 0:
        return 80.0  # Maximum tabulated concentration
    if rh >= 100:
        return 0.0

    # Pure water saturation vapor pressure (equation 1)
    c0, c1, c2, c3 = _TABAZADEH_EQ1
    ln_p_sat = c0 + c1 / T + c2 / T**2 + c3 / T**3

    # Target: equilibrium water partial pressure
    ln_p_target = np.log(rh / 100.0) + ln_p_sat

    # Solution vapor pressure at each tabulated weight percent
    wt_tab = _TABAZADEH_TABLE1[:, 0]
    a_tab = _TABAZADEH_TABLE1[:, 1]
    b_tab = _TABAZADEH_TABLE1[:, 2]
    c_tab = _TABAZADEH_TABLE1[:, 3]
    ln_p_solution = a_tab + b_tab / T + c_tab / T**2

    # ln_p_solution decreases with increasing wt% (more concentrated → less vapor)
    # Find the wt% where ln_p_solution = ln_p_target by interpolation
    if ln_p_target >= ln_p_solution[0]:
        return wt_tab[0]  # More dilute than 10%
    if ln_p_target <= ln_p_solution[-1]:
        return wt_tab[-1]  # More concentrated than 80%

    # Interpolate (ln_p_solution is monotonically decreasing)
    wt_result = np.interp(-ln_p_target, -ln_p_solution, wt_tab)
    return float(wt_result)


@jax.jit
def h2so4_solution_density(wt_percent):
    """Density of H₂SO₄/H₂O solution at given composition.

    Interpolates CRC Handbook values at 25°C. Stratospheric densities
    (~220 K) are ~2-3% higher; this correction is small compared to
    the composition effect.

    Args:
        wt_percent: H₂SO₄ weight percent [%].

    Returns:
        density: Solution density [kg/m³].
    """
    return jnp.interp(
        wt_percent, _H2SO4_DENSITY_TABLE[:, 0], _H2SO4_DENSITY_TABLE[:, 1]
    )


# Planck spectrum constants
_H_PLANCK = 6.62607015e-34  # J·s
_C_LIGHT = 2.99792458e8     # m/s
_K_BOLTZ = 1.380649e-23     # J/K
_T_SUN = 5778.0             # K (solar effective temperature)

# Latitude-dependent albedo (from Earth observations)
# Used for latitude-resolved RF calculation
# Source: http://www.tak2000.com/data/planets/earth.htm
_ALBEDO_TABLE = np.array([
    # (lat_lower, albedo) — lat_lower is the southern edge of the band
    (-90., 0.70), (-80., 0.64), (-70., 0.51), (-60., 0.42),
    (-50., 0.35), (-40., 0.29), (-30., 0.24), (-20., 0.23),
    (-10., 0.23), (0., 0.25), (10., 0.24), (20., 0.26),
    (30., 0.31), (40., 0.36), (50., 0.41), (60., 0.46),
    (70., 0.57), (80., 0.67),
])


# =========================================================================
# Data containers
# =========================================================================
class MieProperties(NamedTuple):
    """Precomputed Mie scattering properties for each TOMAS bin.

    All arrays have shape (nbins,) where nbins = len(xk) - 1.

    Attributes:
        radii: Geometric-mean particle radius for each bin [m].
        Qsca: Scattering efficiency (dimensionless).
        Qext: Extinction efficiency (dimensionless).
        gsca: Asymmetry parameter <cos(theta)> (dimensionless).
        upscatter_avg: Global annual average upscatter fraction per bin.
        wavelength: Wavelength used [m].
        density: Assumed particle density [kg/m³].
    """
    radii: np.ndarray
    Qsca: np.ndarray
    Qext: np.ndarray
    gsca: np.ndarray
    upscatter_avg: np.ndarray
    wavelength: float
    density: float


# =========================================================================
# Mie precomputation
# =========================================================================
def precompute_mie_properties(
    xk,
    density=DENSITY_SULFATE,
    wavelength=WAVELENGTH_DEFAULT,
    refindex=REFINDEX_SULFATE,
    global_avg_upscatter=True,
    spectral=False,
    n_wavelengths=30,
):
    """Precompute Mie scattering properties for each TOMAS size bin.

    For each bin, computes Qsca, Qext, gsca from Bohren-Huffman Mie theory,
    and optionally the global annual average upscatter fraction.

    When spectral=True, integrates over the solar spectrum (Planck at 5778 K,
    300-2500 nm) to produce effective Qsca and upscatter values. The returned
    MieProperties work transparently with compute_rf() — the spectral
    weighting is baked into the effective values.

    Args:
        xk: Bin boundary masses [kg], shape (nbins+1,). JAX or numpy array.
        density: Particle density [kg/m³]. Default 1770 (ammonium sulfate).
        wavelength: Incident wavelength [m]. Default 550 nm.
            Used only when spectral=False.
        refindex: Complex refractive index. Default 1.4+1e-8j (sulfate).
        global_avg_upscatter: If True, compute globally-averaged upscatter
            fraction. If False, set upscatter_avg to NaN.
        spectral: If True, integrate Mie properties over the solar spectrum
            weighted by Planck irradiance. Implies global_avg_upscatter=True.
        n_wavelengths: Number of wavelength points for spectral integration.

    Returns:
        MieProperties NamedTuple with per-bin scattering properties.
        When spectral=True, Qsca and upscatter_avg are spectrally-weighted
        effective values; wavelength field is set to 0 (indicating broadband).
    """
    xk_np = np.asarray(xk, dtype=np.float64)
    nbins = len(xk_np) - 1

    # Geometric-mean mass for each bin (center of log-space bin)
    m_geo = np.sqrt(xk_np[:-1] * xk_np[1:])

    # Particle radius from mass: m = (4/3) * pi * rho * r^3
    radii = (3.0 * m_geo / (4.0 * PI * density))**(1.0 / 3.0)

    if not spectral:
        # Single-wavelength mode — vectorized via vmap
        size_param = jnp.array(2.0 * PI * radii / wavelength)

        # Batched Mie: vmap over all bins at once
        vmap_mie = jax.vmap(bhmie_qsca_jax, in_axes=(0, None))
        Qext, Qsca, gsca = vmap_mie(size_param, refindex)
        Qext = np.asarray(Qext)
        Qsca = np.asarray(Qsca)
        gsca = np.asarray(gsca)

        if global_avg_upscatter:
            upscatter_avg = _compute_global_avg_upscatter(gsca)
        else:
            upscatter_avg = np.full(nbins, np.nan)

        return MieProperties(
            radii=radii, Qsca=Qsca, Qext=Qext, gsca=gsca,
            upscatter_avg=upscatter_avg, wavelength=wavelength, density=density,
        )

    # --- Spectral integration mode — vectorized via 2D vmap ---
    wavelengths, weights = _solar_spectral_weights(n_wl=n_wavelengths)

    # Build upscatter lookup table (one-time cost)
    g_table, beta_table = _build_upscatter_lookup(n_points=50)

    # 2D size parameter array: (nwav, nbins)
    radii_jax = jnp.array(radii)
    size_params = 2.0 * PI * radii_jax[jnp.newaxis, :] / wavelengths[:, jnp.newaxis]

    # Batched Mie: vmap over bins (inner), then wavelengths (outer)
    vmap_2d = jax.vmap(jax.vmap(bhmie_qsca_jax, (0, None)), (0, None))
    Qext_all, Qsca_all, gsca_all = vmap_2d(size_params, refindex)
    # shapes: (nwav, nbins)

    # Upscatter interpolation for all (wl, bin) pairs
    beta_all = _interpolate_upscatter(
        gsca_all, jnp.asarray(g_table), jnp.asarray(beta_table)
    )

    # Weighted accumulation
    w = weights[:, jnp.newaxis]  # (nwav, 1) for broadcasting
    Qsca_eff = jnp.sum(w * Qsca_all, axis=0)
    Qext_eff = jnp.sum(w * Qext_all, axis=0)
    gsca_eff_raw = jnp.sum(w * Qsca_all * gsca_all, axis=0)
    beta_Qsca_product = jnp.sum(w * beta_all * Qsca_all, axis=0)

    # Normalize gsca by Qsca weight
    gsca_eff = jnp.where(Qsca_eff > 0, gsca_eff_raw / Qsca_eff, 0.0)

    # Effective upscatter: defined so that upscatter_eff × Qsca_eff = beta_Qsca_product
    # This makes compute_rf() give the correct spectrally-integrated result
    upscatter_eff = jnp.where(Qsca_eff > 0, beta_Qsca_product / Qsca_eff, 0.0)

    return MieProperties(
        radii=radii, Qsca=np.asarray(Qsca_eff), Qext=np.asarray(Qext_eff),
        gsca=np.asarray(gsca_eff),
        upscatter_avg=np.asarray(upscatter_eff), wavelength=0.0, density=density,
    )


# =========================================================================
# Gauss-Legendre quadrature (replaces scipy.quad for JIT compatibility)
# =========================================================================
_N_GL = 32
_gl_nodes_np, _gl_weights_np = np.polynomial.legendre.leggauss(_N_GL)
_GL_NODES = jnp.array(_gl_nodes_np)
_GL_WEIGHTS = jnp.array(_gl_weights_np)


@jax.jit
def _upscatter_fraction_gl(g, sza_rad):
    """Upscatter fraction via Gauss-Legendre quadrature (JIT-compilable).

    Replaces scipy.quad with 32-point GL quadrature for Wiscombe & Grams
    (1976) equation 22. Accuracy ~1e-6 vs scipy (sufficient for this
    approximation).

    Args:
        g: Asymmetry parameter (scalar, traced).
        sza_rad: Solar zenith angle [radians] (scalar, traced).

    Returns:
        beta: Upscatter fraction (scalar).
    """
    theta = jnp.clip(sza_rad, 1e-6, jnp.pi / 2.0 - 1e-6)

    # Integral 1: [pi/2-theta, pi/2+theta]
    a1 = jnp.pi / 2.0 - theta
    b1 = jnp.pi / 2.0 + theta
    half1 = (b1 - a1) / 2.0
    mid1 = (a1 + b1) / 2.0
    thpr1 = half1 * _GL_NODES + mid1

    arg1 = jnp.clip(1.0 / (jnp.tan(theta) * jnp.tan(thpr1)), -1.0, 1.0)
    hg1 = (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * jnp.cos(thpr1))**1.5
    f1 = jnp.arccos(arg1) * hg1 * jnp.sin(thpr1)
    int1 = half1 * jnp.sum(_GL_WEIGHTS * f1)

    # Integral 2: [pi/2+theta, pi]
    a2 = jnp.pi / 2.0 + theta
    b2 = jnp.pi
    half2 = (b2 - a2) / 2.0
    mid2 = (a2 + b2) / 2.0
    thpr2 = half2 * _GL_NODES + mid2

    hg2 = (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * jnp.cos(thpr2))**1.5
    f2 = hg2 * jnp.sin(thpr2)
    int2 = half2 * jnp.sum(_GL_WEIGHTS * f2)

    return (1.0 / (2.0 * jnp.pi)) * int1 + 0.5 * int2


@jax.jit
def _avg_solar_power_gl(lat_rad, sda, So):
    """24-hour average solar power via GL quadrature (JIT-compilable).

    Replaces scipy.quad integration of _solar_power over [0, 24] hours.

    Args:
        lat_rad: Latitude [radians] (scalar, traced).
        sda: Solar declination angle [radians] (scalar, traced).
        So: Solar constant [W/m²] (scalar).

    Returns:
        avg_power: 24-hour average solar irradiance [W/m²].
    """
    half = 12.0
    mid = 12.0
    hours = half * _GL_NODES + mid
    ang = jnp.pi / 2.0 - jnp.arcsin(
        jnp.sin(lat_rad) * jnp.sin(sda)
        - jnp.cos(lat_rad) * jnp.cos(sda) * jnp.cos(2.0 * jnp.pi * hours / 24.0)
    )
    power = jnp.maximum(-So * jnp.cos(ang), 0.0)
    return half * jnp.sum(_GL_WEIGHTS * power) / 24.0


# =========================================================================
# Upscatter fraction (Wiscombe & Grams 1976)
# =========================================================================
def _henyey_greenstein(g, cos_theta):
    """Henyey-Greenstein phase function.

    P(cos_theta) = (1 - g²) / (1 + g² - 2g*cos_theta)^(3/2)

    Reference: Perry, eqn 11.23.
    """
    return (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * cos_theta)**1.5


def _upscatter_integrand_1(thpr, theta, g):
    """First integrand for upscatter fraction (Wiscombe & Grams 1976 eqn 22)."""
    return (np.arccos(1.0 / np.tan(theta) / np.tan(thpr))
            * _henyey_greenstein(g, np.cos(thpr))
            * np.sin(thpr))


def _upscatter_integrand_2(thpr, theta, g):
    """Second integrand for upscatter fraction."""
    return _henyey_greenstein(g, np.cos(thpr)) * np.sin(thpr)


def upscatter_fraction(g, sza_rad):
    """Compute upscatter fraction for given asymmetry parameter and zenith angle.

    Uses Wiscombe & Grams (1976) equation 22.

    Args:
        g: Asymmetry parameter (scalar).
        sza_rad: Solar zenith angle [radians]. Must be in (0, pi/2).

    Returns:
        beta: Upscatter fraction (scalar in [0, 1]).
    """
    theta = sza_rad
    if abs(theta) < 1e-6:
        theta = 1e-6
    if abs(theta) >= np.pi / 2.0:
        return 0.0

    int1, _ = quad(_upscatter_integrand_1,
                   np.pi / 2.0 - theta, np.pi / 2.0 + theta,
                   args=(theta, g))
    int2, _ = quad(_upscatter_integrand_2,
                   np.pi / 2.0 + theta, np.pi,
                   args=(theta, g))

    beta = (1.0 / (2.0 * np.pi)) * int1 + 0.5 * int2
    return beta


def _solar_power(hour, lat_rad, sda, So):
    """24-hour solar power at a given latitude and solar declination.

    Args:
        hour: Hour of day [0, 24].
        lat_rad: Latitude [radians].
        sda: Solar declination angle [radians].
        So: Solar constant [W/m²].

    Returns:
        Instantaneous solar irradiance [W/m²] (0 if below horizon).
    """
    ang = np.pi / 2.0 - np.arcsin(
        np.sin(lat_rad) * np.sin(sda)
        - np.cos(lat_rad) * np.cos(sda) * np.cos(2.0 * np.pi * hour / 24.0)
    )
    power = -So * np.cos(ang)
    return max(power, 0.0)


def _get_albedo(lat_deg):
    """Latitude-dependent surface albedo from observations.

    Args:
        lat_deg: Latitude in degrees.

    Returns:
        Surface albedo (scalar).
    """
    for i in range(len(_ALBEDO_TABLE) - 1, -1, -1):
        if lat_deg > _ALBEDO_TABLE[i, 0]:
            return _ALBEDO_TABLE[i, 1]
    return _ALBEDO_TABLE[0, 1]


def _compute_global_avg_upscatter(gsca_array):
    """Compute global annual average upscatter fraction for each bin.

    Averages over 18 latitude bands and 12 months, weighted by
    cos(latitude) × average solar power. Uses Gauss-Legendre quadrature
    and vectorized JAX operations for speed.

    Args:
        gsca_array: Asymmetry parameter per bin, shape (nbins,).

    Returns:
        upscatter_avg: Global annual average upscatter fraction, shape (nbins,).
    """
    gsca_jnp = jnp.asarray(gsca_array)

    # Latitude grid (18 bands, 10° spacing)
    lats_rad = jnp.radians(jnp.arange(-85.0, 90.0, 10.0))
    nlats = lats_rad.shape[0]

    # Monthly solar declination angles
    daymonth = jnp.array([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
    cum_days = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(daymonth[:-1])])
    avg_days = cum_days + daymonth / 2.0
    sda = 0.409 * jnp.cos(2.0 * jnp.pi * (avg_days - 173.0) / 365.0)

    # 24h average solar power for all (lat, month) pairs via GL quadrature
    lat_grid, sda_grid = jnp.meshgrid(lats_rad, sda, indexing='ij')  # (nlats, 12)
    avg_pow = jax.vmap(
        lambda la, sd: _avg_solar_power_gl(la, sd, SOLAR_CONSTANT)
    )(lat_grid.ravel(), sda_grid.ravel()).reshape(nlats, 12)

    # Solar zenith angle approximation for each (lat, month)
    sza_grid = jnp.maximum(jnp.abs(lat_grid - sda_grid), 1e-3)

    # Validity mask: SZA < pi/2 and meaningful solar power
    valid = (sza_grid < jnp.pi / 2.0) & (avg_pow > 1e-6)

    # Weights: cos(lat) × avg_pow × valid
    cos_lat = jnp.cos(lats_rad)
    weights = cos_lat[:, jnp.newaxis] * avg_pow * valid
    weight_total = jnp.sum(weights)

    # Upscatter for all (g, sza) combinations via 2D vmap
    sza_flat = sza_grid.ravel()  # (nlats*12,)
    weights_flat = weights.ravel()

    # vmap: outer over g values (nbins), inner over sza values (nlats*12)
    beta_all = jax.vmap(
        jax.vmap(_upscatter_fraction_gl, (None, 0)), (0, None)
    )(gsca_jnp, sza_flat)  # (nbins, nlats*12)

    # Weighted average per bin
    upscatter_avg = jnp.sum(beta_all * weights_flat[jnp.newaxis, :], axis=1)
    upscatter_avg = upscatter_avg / jnp.maximum(weight_total, 1e-30)

    return np.asarray(upscatter_avg)


# =========================================================================
# Solar spectrum and spectral integration
# =========================================================================
def _planck(wavelength_m, T=_T_SUN):
    """Planck spectral radiance B(λ, T) [W/m²/sr/m].

    Args:
        wavelength_m: Wavelength [m], scalar or array.
        T: Temperature [K].

    Returns:
        Spectral radiance [W/m²/sr/m].
    """
    lam = wavelength_m
    return (2.0 * _H_PLANCK * _C_LIGHT**2 / lam**5) / (
        jnp.exp(_H_PLANCK * _C_LIGHT / (lam * _K_BOLTZ * T)) - 1.0
    )


def _solar_spectral_weights(n_wl=30, wl_min=300e-9, wl_max=2500e-9):
    """Compute normalized solar spectral weights using Planck function.

    Returns wavelengths and weights such that Σ weights = 1.
    Each weight represents the fraction of total solar energy
    in that wavelength band.

    Args:
        n_wl: Number of wavelength points.
        wl_min: Minimum wavelength [m]. Default 300 nm.
        wl_max: Maximum wavelength [m]. Default 2500 nm.

    Returns:
        wavelengths: Wavelength centers [m], shape (n_wl,).
        weights: Normalized spectral weights (sum to 1), shape (n_wl,).
    """
    wavelengths = jnp.linspace(wl_min, wl_max, n_wl)
    dlamb = wavelengths[1] - wavelengths[0]

    B = _planck(wavelengths, _T_SUN)
    raw_weights = B * dlamb
    weights = raw_weights / jnp.sum(raw_weights)

    return wavelengths, weights


def _build_upscatter_lookup(n_points=50):
    """Build lookup table of global-avg upscatter fraction vs asymmetry parameter.

    Precomputes the expensive latitude/month averaging for a grid of g values,
    enabling fast interpolation for spectral integration.

    Args:
        n_points: Number of g values to tabulate.

    Returns:
        g_table: Asymmetry parameter values, shape (n_points,).
        beta_table: Corresponding global-avg upscatter fractions, shape (n_points,).
    """
    g_table = np.linspace(0.0, 0.95, n_points)
    beta_table = _compute_global_avg_upscatter(g_table)
    return g_table, beta_table


@jax.jit
def _interpolate_upscatter(gsca_values, g_table, beta_table):
    """Interpolate upscatter fraction from lookup table.

    Args:
        gsca_values: Asymmetry parameters to look up, any shape.
        g_table: Tabulated g values from _build_upscatter_lookup().
        beta_table: Tabulated upscatter values.

    Returns:
        Interpolated upscatter fractions, same shape as gsca_values.
    """
    return jnp.interp(gsca_values, g_table, beta_table)


# =========================================================================
# Optical depth
# =========================================================================
@jax.jit
def compute_optical_depth(Nk, mie, column_area):
    """Compute aerosol scattering optical depth per bin.

    τ_k = (N_k / A) × π × r_k² × Q_sca,k

    where N_k/A is the column number density [#/m²].

    Args:
        Nk: Number per grid cell [#/cell], shape (nbins,).
        mie: MieProperties from precompute_mie_properties().
        column_area: Horizontal cross-sectional area of the grid cell [m²].
            Relates to boxvol via: column_area = boxvol [m³] / layer_depth [m].

    Returns:
        tau: Scattering optical depth per bin, shape (nbins,).
    """
    N_col = Nk / column_area
    sigma_sca = PI * mie.radii**2 * mie.Qsca
    tau = N_col * sigma_sca
    return tau


# =========================================================================
# Radiative forcing (Chylek & Wong 1995)
# =========================================================================
@jax.jit
def compute_rf(
    Nk,
    mie,
    column_area,
    solar_constant=SOLAR_CONSTANT,
    Tatm=TATM_DEFAULT,
    albedo=ALBEDO_GLOBAL_AVG,
    cloud_fraction=0.0,
):
    """Compute direct shortwave radiative forcing from aerosol scattering.

    Uses the Chylek & Wong (1995) / Pierce et al. (2010) equation:
        RF = -(S₀/4) × T² × (1-A) × (1-R)² × 2β × τ

    where S₀/4 is the global average insolation, T is atmospheric
    transmittance above the aerosol layer, A is cloud fraction, R is
    clear-sky surface albedo, β is upscatter fraction, τ is scattering
    optical depth. The negative sign indicates cooling.

    For stratospheric aerosol, use Pierce SI parameters:
        Tatm=1.0, albedo=0.15, cloud_fraction=0.6

    Args:
        Nk: Number per grid cell [#/cell], shape (nbins,).
        mie: MieProperties from precompute_mie_properties().
        column_area: Horizontal area of the grid cell [m²].
        solar_constant: Total solar irradiance [W/m²]. Default 1361.
        Tatm: Atmospheric transmittance above aerosol layer. Default 0.85.
        albedo: Surface albedo (clear-sky). Default 0.30 (global average).
        cloud_fraction: Fraction of sky covered by clouds [0-1]. Default 0.

    Returns:
        rf_total: Total radiative forcing [W/m²] (negative = cooling).
        rf_per_bin: Per-bin RF contribution [W/m²], shape (nbins,).
    """
    tau = compute_optical_depth(Nk, mie, column_area)

    # Global average incoming solar flux
    avg_solar = solar_constant / 4.0

    # Chylek & Wong 1995 / Pierce et al. 2010 SI eqn
    rf_per_bin = (-avg_solar * Tatm**2 * (1.0 - cloud_fraction)
                  * (1.0 - albedo)**2 * 2.0 * mie.upscatter_avg * tau)

    rf_total = jnp.sum(rf_per_bin)
    return rf_total, rf_per_bin


# =========================================================================
# Diagnostic quantities
# =========================================================================
@jax.jit
def compute_mass_scattering_efficiency(Nk, Mk, mie):
    """Compute mass scattering efficiency [m²/g].

    MSE = total scattering cross section / total dry aerosol mass.
    This is a key intensive property of the size distribution.

    Args:
        Nk: Number per grid cell [#/cell], shape (nbins,).
        Mk: Mass per grid cell [kg/cell], shape (nbins, icomp).
        mie: MieProperties from precompute_mie_properties().

    Returns:
        mse: Mass scattering efficiency [m²/g] (scalar).
        mse_per_bin: Per-bin contribution [m²/g], shape (nbins,).
    """
    # Total scattering cross section per bin [m²/cell]
    sigma_sca = PI * mie.radii**2 * mie.Qsca
    total_sca = Nk * sigma_sca  # [m²/cell]

    # Dry mass per bin (exclude water, species 0 through SRTH2O-1)
    M_dry = jnp.sum(Mk[:, :SRTH2O], axis=1)  # [kg/cell]
    M_dry_total = jnp.sum(M_dry)

    # MSE [m²/g] = total_sca_area [m²] / mass [g]
    mse = jnp.where(M_dry_total > 0,
                     jnp.sum(total_sca) / (M_dry_total * 1e3),
                     0.0)

    M_dry_g = jnp.maximum(M_dry * 1e3, 1e-30)
    mse_per_bin = total_sca / M_dry_g

    return mse, mse_per_bin


@jax.jit
def compute_rf_efficiency(Nk, Mk, mie, Tatm=TATM_DEFAULT, albedo=ALBEDO_GLOBAL_AVG):
    """Compute RF per unit aerosol burden [W/m² per g/m²].

    This intensive property describes how efficiently the current size
    distribution produces radiative forcing per unit dry mass burden.
    Useful for comparing size distributions without specifying geometry.

    RF_eff = -(S₀/4) × T_atm² × (1-α)² × 2 × Σ(β_k × MSE_k × f_k)

    where f_k = M_dry,k / M_dry_total is the mass fraction in bin k.

    Args:
        Nk: Number per grid cell [#/cell], shape (nbins,).
        Mk: Mass per grid cell [kg/cell], shape (nbins, icomp).
        mie: MieProperties from precompute_mie_properties().
        Tatm: Atmospheric transmittance. Default 0.85.
        albedo: Surface albedo. Default 0.30.

    Returns:
        rf_eff: RF efficiency [W/m² per g-aerosol/m²] (negative = cooling).
    """
    mse, _ = compute_mass_scattering_efficiency(Nk, Mk, mie)

    # Weight MSE by upscatter fraction (bin-weighted)
    sigma_sca = PI * mie.radii**2 * mie.Qsca
    total_sca = Nk * sigma_sca
    sca_total = jnp.sum(total_sca)

    # Scattering-weighted average upscatter fraction
    beta_avg = jnp.where(sca_total > 0,
                          jnp.sum(mie.upscatter_avg * total_sca) / sca_total,
                          0.0)

    avg_solar = SOLAR_CONSTANT / 4.0
    rf_eff = -avg_solar * Tatm**2 * (1.0 - albedo)**2 * 2.0 * beta_avg * mse

    return rf_eff


def scattering_efficiency_vs_radius(
    wavelength=WAVELENGTH_DEFAULT,
    refindex=REFINDEX_SULFATE,
    density=DENSITY_SULFATE,
    n_radii=200,
    r_min=1e-9,
    r_max=10e-6,
    solar_constant=SOLAR_CONSTANT,
    Tatm=TATM_DEFAULT,
    albedo=ALBEDO_GLOBAL_AVG,
    cloud_fraction=0.0,
    spectral=False,
    n_wavelengths=30,
):
    """Compute scattering cooling efficiency as a function of particle radius.

    Reproduces Pierce et al. (2010) Figure 1:
    RF per unit mass burden as a function of monodisperse particle size.

    For a monodisperse population with burden B [kg/m²]:
        n_col = B / m_particle  [#/m²]
        τ = n_col × π × r² × Qsca
        RF = -(S₀/4) × T² × (1-A) × (1-R)² × 2β × τ
        RF/B = -(S₀/4) × T² × (1-A) × (1-R)² × 2β × (3Qsca)/(4ρr)

    For Pierce et al. (2010) SI parameters (stratospheric aerosol):
        solar_constant=1370, Tatm=1.0, albedo=0.15, cloud_fraction=0.6,
        density=1700 (wet H₂SO₄/H₂O)

    Args:
        wavelength: Wavelength [m]. Used only when spectral=False.
        refindex: Complex refractive index.
        density: Particle density [kg/m³].
        n_radii: Number of radius points.
        r_min, r_max: Radius range [m].
        solar_constant: Total solar irradiance [W/m²]. Default 1361.
        Tatm: Atmospheric transmittance above aerosol layer.
        albedo: Clear-sky surface albedo.
        cloud_fraction: Fraction of sky covered by clouds [0-1]. Default 0.
        spectral: If True, integrate over solar spectrum (Planck, 300-2500 nm).
        n_wavelengths: Number of wavelength points for spectral integration.

    Returns:
        radii: Particle radii [m], shape (n_radii,).
        rf_per_burden: RF per unit burden [W/m² per kg/m²], shape (n_radii,).
            Negative values indicate cooling.
    """
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)

    avg_solar = solar_constant / 4.0
    prefactor = avg_solar * Tatm**2 * (1.0 - cloud_fraction) * (1.0 - albedo)**2 * 2.0

    if not spectral:
        # Single-wavelength mode — vectorized via vmap
        radii_jax = jnp.array(radii)
        x_arr = 2.0 * PI * radii_jax / wavelength
        Qext_arr, Qsca_arr, gsca_arr = jax.vmap(
            bhmie_qsca_jax, (0, None)
        )(x_arr, refindex)

        # Batch upscatter computation for all radii at once
        beta_arr = _compute_global_avg_upscatter(np.asarray(gsca_arr))

        m_p = (4.0 / 3.0) * PI * density * radii**3
        sigma_sca = PI * radii**2 * np.asarray(Qsca_arr)
        rf_per_burden = -prefactor * beta_arr * sigma_sca / m_p
    else:
        # Spectral integration mode — vectorized via 2D vmap
        wavelengths, weights = _solar_spectral_weights(n_wl=n_wavelengths)
        g_table, beta_table = _build_upscatter_lookup(n_points=50)

        radii_jax = jnp.array(radii)
        # 2D size parameters: (n_wl, n_radii)
        size_params = 2.0 * PI * radii_jax[jnp.newaxis, :] / wavelengths[:, jnp.newaxis]

        vmap_2d = jax.vmap(jax.vmap(bhmie_qsca_jax, (0, None)), (0, None))
        _, Qsca_all, gsca_all = vmap_2d(size_params, refindex)
        # shapes: (n_wl, n_radii)

        beta_all = _interpolate_upscatter(
            gsca_all, jnp.asarray(g_table), jnp.asarray(beta_table)
        )

        # Weighted spectral integration
        w = weights[:, jnp.newaxis]  # (n_wl, 1)
        sigma_sca_all = PI * radii_jax[jnp.newaxis, :]**2 * Qsca_all
        weighted_beta_sigma = jnp.sum(w * beta_all * sigma_sca_all, axis=0)

        m_p = (4.0 / 3.0) * PI * density * radii**3
        rf_per_burden = np.asarray(-prefactor * weighted_beta_sigma / m_p)

    return radii, rf_per_burden


# =========================================================================
# Latitude-resolved RF (full calculation as in getRF_monthly.py)
# =========================================================================
def compute_rf_latitude_resolved(
    Nk,
    mie,
    column_area,
    solar_constant=SOLAR_CONSTANT,
    Tatm=TATM_DEFAULT,
    nlats=18,
):
    """Compute RF resolved by latitude band, then globally average.

    Full calculation matching getRF_monthly.py but simplified for a single
    aerosol layer (box model). Loops over 18 latitude bands × 12 months,
    computes local upscatter fractions and solar fluxes, then area-weights.

    Args:
        Nk: Number per grid cell [#/cell], shape (nbins,).
        mie: MieProperties from precompute_mie_properties().
        column_area: Horizontal area of the grid cell [m²].
        solar_constant: TSI [W/m²].
        Tatm: Atmospheric transmittance.
        nlats: Number of latitude bands.

    Returns:
        rf_global: Globally averaged RF [W/m²] (negative = cooling).
        rf_by_lat: RF at each latitude [W/m²], shape (nlats,).
        lats: Latitude centers [degrees], shape (nlats,).
    """
    tau = compute_optical_depth(Nk, mie, column_area)
    nbins = len(tau)

    # Latitude grid
    lats_deg = np.linspace(-85, 85, nlats)
    lats_rad = np.radians(lats_deg)

    # Monthly solar declination angles
    daymonth = np.array([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
    sda = np.zeros(12)
    tot_days = 0.0
    for m in range(12):
        avg_day = tot_days + daymonth[m] / 2.0
        sda[m] = 0.409 * np.cos(2.0 * np.pi * (avg_day - 173.0) / 365.0)
        tot_days += daymonth[m]

    # Compute RF at each latitude and month
    rf_lat_month = np.zeros((12, nlats))

    for la in range(nlats):
        alb = _get_albedo(lats_deg[la])
        for m in range(12):
            # 24h average solar power
            power, _ = quad(_solar_power, 0.0, 24.0,
                            args=(lats_rad[la], sda[m], solar_constant))
            avg_pow = power / 24.0

            # Solar zenith angle approximation
            sza = abs(lats_rad[la] - sda[m])
            sza = max(sza, 1e-3)

            if sza >= np.pi / 2.0 or avg_pow < 1e-6:
                continue

            # Sum over bins
            for k in range(nbins):
                beta = upscatter_fraction(mie.gsca[k], sza)
                rf_lat_month[m, la] += (
                    -avg_pow * Tatm**2 * (1.0 - alb)**2 * 2.0 * beta * tau[k]
                )

    # Average over months
    rf_by_lat = rf_lat_month.mean(axis=0)

    # Global average (area-weighted by cos(lat))
    cos_lat = np.cos(lats_rad)
    rf_global = np.sum(rf_by_lat * cos_lat) / np.sum(cos_lat)

    return rf_global, rf_by_lat, lats_deg
