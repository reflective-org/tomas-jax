"""SO2 + OH gas-phase chemistry using Sun et al. (2022) Troe formalism.

Implements the reaction:
    SO2 + OH (+M) -> HOSO2 -> ... -> H2SO4

Rate coefficient k1(T,p) uses the Troe formula with H2O enhancement
from Sun et al. (2022), Table 2, Method 1.

Reference:
    Sun, W., et al.: Kinetics of OH + SO2 + M: The key reaction in
    atmospheric SO2-to-sulfate conversion. Atmos. Chem. Phys., 22,
    4969-4984, https://doi.org/10.5194/acp-22-4969-2022, 2022.

All functions are JIT-compilable.
"""
import jax
import jax.numpy as jnp
from functools import partial

from ..core.config import (
    SRTSO4, SRTSO2, MW_H2SO4, MW_SO2, KB, AVOGADRO,
)


# =========================================================================
# Constants
# =========================================================================

# Sun et al. (2022) Table 2, Method 1 — N2 bath gas
_K0_N2_300 = 3.03e-31   # cm^6 molec^-2 s^-1
_N_K0_N2 = 4.10          # T-exponent for k0_N2
_KINF_300 = 2.00e-12     # cm^3 molec^-1 s^-1 (T-independent, m=0)
_FC = 0.58                # Broadening factor

# Sun et al. (2022) Eqs. 5-6 — H2O bath gas enhancement
_K0_H2O_300 = 1.65e-30  # cm^6 molec^-2 s^-1
_N_K0_H2O = 4.90          # T-exponent for k0_H2O

# Buck (1981) saturation vapor pressure constants
_BUCK_A = 611.21   # Pa
_BUCK_B = 18.678
_BUCK_C = 257.14   # degC
_BUCK_D = 234.5    # degC


# =========================================================================
# 1. Troe rate constant: k1(T, p, rh)
# =========================================================================

@jax.jit
def calc_k1_so2_oh(temp, pres, rh=0.0):
    """Compute k1 for SO2 + OH (+M) using Sun et al. (2022) Troe formula.

    Args:
        temp: Temperature [K]
        pres: Pressure [Pa]
        rh: Relative humidity [fraction 0-1] (for H2O enhancement)

    Returns:
        k1: Rate constant [cm^3 molec^-1 s^-1]
    """
    # Total number density [M] in molec/cm^3
    M_total = pres / (KB * temp) * 1.0e-6  # Pa -> cm^-3

    # Temperature-dependent low-pressure rate constants
    T_ratio = temp / 300.0
    k0_N2 = _K0_N2_300 * jnp.power(T_ratio, -_N_K0_N2)
    k0_H2O = _K0_H2O_300 * jnp.power(T_ratio, -_N_K0_H2O)

    # High-pressure limit (T-independent for Method 1)
    k_inf = _KINF_300

    # H2O number density from RH and Buck formula
    # Buck (1981): e_sat = A * exp((B - T_C/D) * T_C / (T_C + C))
    T_C = temp - 273.15
    e_sat = _BUCK_A * jnp.exp((_BUCK_B - T_C / _BUCK_D) * T_C / (T_C + _BUCK_C))
    # Water vapor partial pressure [Pa]
    p_h2o = rh * e_sat
    # Water number density [molec/cm^3]
    n_h2o = p_h2o / (KB * temp) * 1.0e-6

    # Mole fractions
    x_h2o = n_h2o / jnp.maximum(M_total, 1.0)
    x_n2 = 1.0 - x_h2o

    # Effective low-pressure rate constant (Sun et al. Eq. 5)
    k0_eff = x_n2 * k0_N2 + x_h2o * k0_H2O

    # Troe formula (Eq. 3)
    ratio = k0_eff * M_total / k_inf

    # Broadening factor N (Eq. 4 parameter)
    N = 0.75 - 1.27 * jnp.log10(_FC)

    # log10(F) = log10(Fc) / (1 + (log10(ratio)/N)^2)
    log10_ratio = jnp.log10(jnp.maximum(ratio, 1.0e-300))
    log10_F = jnp.log10(_FC) / (1.0 + (log10_ratio / N) ** 2)
    F = jnp.power(10.0, log10_F)

    # k1 = k0_eff * [M] * k_inf / (k0_eff * [M] + k_inf) * F
    k1 = ratio / (1.0 + ratio) * k_inf * F

    return k1


# =========================================================================
# 2. Solar zenith angle
# =========================================================================

@jax.jit
def calc_solar_zenith_angle(lat, day_of_year, hour_utc, lon=0.0):
    """Compute cosine of solar zenith angle.

    Uses Spencer (1971) declination approximation.

    Args:
        lat: Latitude [degrees]
        day_of_year: Day of year [1-365]
        hour_utc: Hour in UTC [0-24]
        lon: Longitude [degrees, East positive]

    Returns:
        cos_sza: Cosine of solar zenith angle (can be negative = nighttime)
    """
    # Convert to radians
    lat_rad = lat * jnp.pi / 180.0

    # Solar declination (simple approximation)
    decl = -23.45 * jnp.cos(2.0 * jnp.pi * (day_of_year + 10.0) / 365.0)
    decl_rad = decl * jnp.pi / 180.0

    # Local solar time -> hour angle
    # Solar noon at lon=0 is 12 UTC; offset by lon/15 hours
    local_solar_time = hour_utc + lon / 15.0
    hour_angle = (local_solar_time - 12.0) * 15.0 * jnp.pi / 180.0

    # cos(SZA) = sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(h)
    cos_sza = (jnp.sin(lat_rad) * jnp.sin(decl_rad) +
               jnp.cos(lat_rad) * jnp.cos(decl_rad) * jnp.cos(hour_angle))

    return cos_sza


# =========================================================================
# 3. OH concentration (constant or diurnal)
# =========================================================================

@jax.jit
def calc_oh_concentration(oh_const, cos_sza, use_diurnal=0.0):
    """Compute OH concentration, optionally with diurnal cycle.

    Args:
        oh_const: Baseline OH concentration [molec/cm^3]
        cos_sza: Cosine of solar zenith angle
        use_diurnal: 0.0 = constant mode, 1.0 = diurnal mode (float for JIT)

    Returns:
        oh: OH concentration [molec/cm^3]
    """
    # Diurnal: OH proportional to cos(SZA), zero at night
    oh_diurnal = oh_const * jnp.maximum(0.0, cos_sza)

    # Select mode (float mask avoids recompilation)
    oh = jnp.where(use_diurnal > 0.5, oh_diurnal, oh_const)

    return oh


# =========================================================================
# 4. SO2 oxidation step
# =========================================================================

@jax.jit
def so2_oxidation_step(Gc, temp, pres, boxvol, dt, oh_conc, rh=0.0):
    """Apply one timestep of SO2 + OH -> H2SO4 chemistry.

    Uses pseudo-first-order analytical solution:
        SO2(t+dt) = SO2(t) * exp(-k1 * [OH] * dt)
        dH2SO4 = dSO2 * (MW_H2SO4 / MW_SO2)

    Args:
        Gc: Gas-phase concentrations [kg/grid cell], shape (N_GAS_SPECIES,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        dt: Timestep [s]
        oh_conc: OH concentration [molec/cm^3]
        rh: Relative humidity [fraction 0-1]

    Returns:
        Gc_new: Updated gas-phase concentrations
    """
    # Rate constant
    k1 = calc_k1_so2_oh(temp, pres, rh)

    # Convert SO2 from kg/grid cell to molec/cm^3
    so2_kg = Gc[SRTSO2]
    so2_molec_cm3 = so2_kg / boxvol * (AVOGADRO / (MW_SO2 / 1000.0))

    # Pseudo-first-order decay
    decay = jnp.exp(-k1 * oh_conc * dt)
    so2_molec_cm3_new = so2_molec_cm3 * decay

    # SO2 consumed [molec/cm^3]
    d_so2_molec_cm3 = so2_molec_cm3 - so2_molec_cm3_new

    # Convert consumed SO2 back to kg/grid cell
    d_so2_kg = d_so2_molec_cm3 * boxvol * (MW_SO2 / 1000.0) / AVOGADRO

    # H2SO4 produced [kg/grid cell] (MW ratio conversion)
    d_h2so4_kg = d_so2_kg * (MW_H2SO4 / MW_SO2)

    # Guard against negative SO2 (numerical safety)
    d_so2_kg = jnp.minimum(d_so2_kg, so2_kg)
    d_h2so4_kg = d_so2_kg * (MW_H2SO4 / MW_SO2)

    # Update Gc
    Gc_new = Gc.at[SRTSO2].add(-d_so2_kg)
    Gc_new = Gc_new.at[SRTSO4].add(d_h2so4_kg)

    return Gc_new
