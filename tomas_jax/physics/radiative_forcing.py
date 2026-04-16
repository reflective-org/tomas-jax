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

from .bhmie import bhmie
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
    wt = float(wt_percent)
    return float(np.interp(
        wt, _H2SO4_DENSITY_TABLE[:, 0], _H2SO4_DENSITY_TABLE[:, 1]
    ))


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
        # Single-wavelength mode (original behavior)
        size_param = 2.0 * PI * radii / wavelength

        Qsca = np.zeros(nbins)
        Qext = np.zeros(nbins)
        gsca = np.zeros(nbins)

        for k in range(nbins):
            _, _, qext_k, qsca_k, _, gsca_k = bhmie(size_param[k], refindex, 2)
            Qsca[k] = qsca_k
            Qext[k] = qext_k
            gsca[k] = gsca_k

        if global_avg_upscatter:
            upscatter_avg = _compute_global_avg_upscatter(gsca)
        else:
            upscatter_avg = np.full(nbins, np.nan)

        return MieProperties(
            radii=radii, Qsca=Qsca, Qext=Qext, gsca=gsca,
            upscatter_avg=upscatter_avg, wavelength=wavelength, density=density,
        )

    # --- Spectral integration mode ---
    wavelengths, weights = _solar_spectral_weights(n_wl=n_wavelengths)

    # Build upscatter lookup table (one-time cost)
    g_table, beta_table = _build_upscatter_lookup(n_points=50)

    # Accumulate spectrally-weighted quantities per bin
    Qsca_eff = np.zeros(nbins)
    Qext_eff = np.zeros(nbins)
    gsca_eff = np.zeros(nbins)
    # The coupled product: Σ w(λ) × β(g(λ)) × Qsca(λ)
    beta_Qsca_product = np.zeros(nbins)

    for iw, (wl, w) in enumerate(zip(wavelengths, weights)):
        size_param = 2.0 * PI * radii / wl
        for k in range(nbins):
            _, _, qext_k, qsca_k, _, gsca_k = bhmie(size_param[k], refindex, 2)
            beta_k = _interpolate_upscatter(gsca_k, g_table, beta_table)

            Qsca_eff[k] += w * qsca_k
            Qext_eff[k] += w * qext_k
            gsca_eff[k] += w * qsca_k * gsca_k  # Qsca-weighted g
            beta_Qsca_product[k] += w * beta_k * qsca_k

    # Normalize gsca by Qsca weight
    gsca_eff = np.where(Qsca_eff > 0, gsca_eff / Qsca_eff, 0.0)

    # Effective upscatter: defined so that upscatter_eff × Qsca_eff = beta_Qsca_product
    # This makes compute_rf() give the correct spectrally-integrated result
    upscatter_eff = np.where(Qsca_eff > 0, beta_Qsca_product / Qsca_eff, 0.0)

    return MieProperties(
        radii=radii, Qsca=Qsca_eff, Qext=Qext_eff, gsca=gsca_eff,
        upscatter_avg=upscatter_eff, wavelength=0.0, density=density,
    )


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
    cos(latitude) × average solar power.

    Args:
        gsca_array: Asymmetry parameter per bin, shape (nbins,).

    Returns:
        upscatter_avg: Global annual average upscatter fraction, shape (nbins,).
    """
    nbins = len(gsca_array)

    # Latitude grid (18 bands, 10° spacing)
    lats_deg = np.arange(-85, 90, 10.0)  # [-85, -75, ..., 75, 85]
    lats_rad = np.radians(lats_deg)
    nlats = len(lats_deg)

    # Monthly solar declination angles
    daymonth = np.array([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
    sda = np.zeros(12)
    tot_days = 0.0
    for m in range(12):
        avg_day = tot_days + daymonth[m] / 2.0
        sda[m] = 0.409 * np.cos(2.0 * np.pi * (avg_day - 173.0) / 365.0)
        tot_days += daymonth[m]

    # Compute 24h average solar power for each month and latitude
    avg_pow = np.zeros((12, nlats))
    for la in range(nlats):
        for m in range(12):
            power, _ = quad(_solar_power, 0.0, 24.0,
                            args=(lats_rad[la], sda[m], SOLAR_CONSTANT))
            avg_pow[m, la] = power / 24.0

    # Compute upscatter fraction weighted by solar power and cos(lat)
    upscatter_avg = np.zeros(nbins)
    cos_lat = np.cos(lats_rad)

    for k in range(nbins):
        g = gsca_array[k]
        weighted_sum = 0.0
        weight_total = 0.0

        for la in range(nlats):
            for m in range(12):
                if avg_pow[m, la] < 1e-6:
                    continue
                # Approximate solar zenith angle for this lat/month
                # Use the subsolar latitude (declination) to estimate typical SZA
                sza = abs(lats_rad[la] - sda[m])
                sza = max(sza, 1e-3)
                if sza >= np.pi / 2.0:
                    continue

                beta = upscatter_fraction(g, sza)
                w = cos_lat[la] * avg_pow[m, la]
                weighted_sum += beta * w
                weight_total += w

        if weight_total > 0:
            upscatter_avg[k] = weighted_sum / weight_total

    return upscatter_avg


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
        np.exp(_H_PLANCK * _C_LIGHT / (lam * _K_BOLTZ * T)) - 1.0
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
    wavelengths = np.linspace(wl_min, wl_max, n_wl)
    dlamb = wavelengths[1] - wavelengths[0]

    B = _planck(wavelengths, _T_SUN)
    raw_weights = B * dlamb
    weights = raw_weights / np.sum(raw_weights)

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


def _interpolate_upscatter(gsca_values, g_table, beta_table):
    """Interpolate upscatter fraction from lookup table.

    Args:
        gsca_values: Asymmetry parameters to look up, any shape.
        g_table: Tabulated g values from _build_upscatter_lookup().
        beta_table: Tabulated upscatter values.

    Returns:
        Interpolated upscatter fractions, same shape as gsca_values.
    """
    return np.interp(gsca_values, g_table, beta_table)


# =========================================================================
# Optical depth
# =========================================================================
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
    Nk_np = np.asarray(Nk, dtype=np.float64)
    # Column number density [#/m²]
    N_col = Nk_np / column_area
    # Scattering cross section per particle [m²]
    sigma_sca = PI * mie.radii**2 * mie.Qsca
    # Optical depth per bin
    tau = N_col * sigma_sca
    return tau


# =========================================================================
# Radiative forcing (Chylek & Wong 1995)
# =========================================================================
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

    rf_total = np.sum(rf_per_bin)
    return rf_total, rf_per_bin


# =========================================================================
# Diagnostic quantities
# =========================================================================
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
    Nk_np = np.asarray(Nk, dtype=np.float64)
    Mk_np = np.asarray(Mk, dtype=np.float64)

    # Total scattering cross section per bin [m²/cell]
    sigma_sca = PI * mie.radii**2 * mie.Qsca
    total_sca = Nk_np * sigma_sca  # [m²/cell]

    # Dry mass per bin (exclude water, species 0 through SRTH2O-1)
    M_dry = np.sum(Mk_np[:, :SRTH2O], axis=1)  # [kg/cell]
    M_dry_total = np.sum(M_dry)

    # MSE [m²/g] = total_sca_area [m²] / mass [g]
    if M_dry_total > 0:
        mse = np.sum(total_sca) / (M_dry_total * 1e3)  # kg→g
    else:
        mse = 0.0

    M_dry_g = np.maximum(M_dry * 1e3, 1e-30)
    mse_per_bin = total_sca / M_dry_g

    return mse, mse_per_bin


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
    Nk_np = np.asarray(Nk, dtype=np.float64)
    sigma_sca = PI * mie.radii**2 * mie.Qsca
    total_sca = Nk_np * sigma_sca
    sca_total = np.sum(total_sca)

    if sca_total > 0:
        # Scattering-weighted average upscatter fraction
        beta_avg = np.sum(mie.upscatter_avg * total_sca) / sca_total
    else:
        beta_avg = 0.0

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
    rf_per_burden = np.zeros(n_radii)

    avg_solar = solar_constant / 4.0
    prefactor = avg_solar * Tatm**2 * (1.0 - cloud_fraction) * (1.0 - albedo)**2 * 2.0

    if not spectral:
        # Single-wavelength mode
        for i, r in enumerate(radii):
            x = 2.0 * PI * r / wavelength
            _, _, _, qsca, _, gsca = bhmie(x, refindex, 2)
            beta = _compute_global_avg_upscatter(np.array([gsca]))[0]
            m_p = (4.0 / 3.0) * PI * density * r**3
            sigma_sca = PI * r**2 * qsca
            rf_per_burden[i] = -prefactor * beta * sigma_sca / m_p
    else:
        # Spectral integration mode
        wavelengths, weights = _solar_spectral_weights(n_wl=n_wavelengths)
        g_table, beta_table = _build_upscatter_lookup(n_points=50)

        for i, r in enumerate(radii):
            m_p = (4.0 / 3.0) * PI * density * r**3
            # Integrate over wavelengths: Σ w(λ) × β(g(λ)) × σ_sca(λ)
            weighted_beta_sigma = 0.0
            for wl, w in zip(wavelengths, weights):
                x = 2.0 * PI * r / wl
                _, _, _, qsca, _, gsca = bhmie(x, refindex, 2)
                beta = _interpolate_upscatter(gsca, g_table, beta_table)
                sigma_sca = PI * r**2 * qsca
                weighted_beta_sigma += w * beta * sigma_sca

            rf_per_burden[i] = -prefactor * weighted_beta_sigma / m_p

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
