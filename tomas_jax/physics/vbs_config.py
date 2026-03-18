"""Volatility Basis Set (VBS) configuration for TOMAS-JAX.

Defines VBS bin properties (C* values, enthalpies, molecular weights),
temperature-corrected saturation concentrations, and unit conversions.

The standard 6-bin VBS spans C* = 0.01 to 1000 µg/m³ at 298K, covering
extremely-low to intermediate volatility organics (Donahue et al. 2006).

VBS bins map to TOMAS organic species indices 1–6 (SRTORG1 + 0..5).
Species 7–41 are reserved for future expansion.

References:
    - Donahue et al. (2006), Environ. Sci. Technol., 40, 2635–2643
    - Epstein et al. (2010), Environ. Sci. Technol., 44, 743–748
    - soacond.f (Jeff Pierce, September 2010)
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import NamedTuple

from ..core.config import R_GAS, KB, SRTORG1


# =========================================================================
# VBS bin definitions
# =========================================================================

N_VBS_BINS = 6

# Reference saturation concentrations at 298K [µg/m³]
CSTAR_REF_UG = jnp.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

# Reference temperature [K]
T_REF = 298.0

# Enthalpy of vaporization [kJ/mol] — Epstein et al. (2010) parameterization
# ΔHvap decreases with increasing C*
DELTA_HVAP_KJ = jnp.array([150.0, 136.0, 122.0, 108.0, 94.0, 80.0])

# Molecular weights [g/mol] — typical for SOA
MW_ORG = jnp.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0])

# Surface tension [N/m] — typical for organic aerosol
SIGMA_ORG = 0.025

# Organic density [kg/m³]
RHO_ORG = 1200.0

# Diffusion volume for organics (Fuller-Schettler-Giddings)
# Typical value for C10-C15 oxygenated organics
SV_ORG = 120.0

# Particle-phase diffusion coefficient [m²/s] (Zaveri et al. 2014)
DBK_DEFAULT = 1.0e-10

# First-order loss rate of species in particle phase [s⁻¹]
KC_DEFAULT = 0.0

# TOMAS species indices for each VBS bin (0-based)
VBS_SPECIES_INDICES = tuple(SRTORG1 + i for i in range(N_VBS_BINS))


class VBSConfig(NamedTuple):
    """VBS configuration — all properties needed for SOA condensation."""
    n_bins: int
    cstar_ref_ug: jnp.ndarray    # C* at T_ref [µg/m³], shape (n_bins,)
    delta_Hvap_kJ: jnp.ndarray   # ΔHvap [kJ/mol], shape (n_bins,)
    mw: jnp.ndarray              # Molecular weight [g/mol], shape (n_bins,)
    sigma: float                  # Surface tension [N/m]
    rho: float                    # Organic density [kg/m³]
    sv: float                     # Diffusion volume
    t_ref: float                  # Reference temperature [K]
    species_indices: tuple        # TOMAS species indices
    Dbk: float = DBK_DEFAULT     # Particle-phase diffusion coeff [m²/s]
    kc: float = KC_DEFAULT       # First-order particle-phase loss [s⁻¹]


DEFAULT_VBS_CONFIG = VBSConfig(
    n_bins=N_VBS_BINS,
    cstar_ref_ug=CSTAR_REF_UG,
    delta_Hvap_kJ=DELTA_HVAP_KJ,
    mw=MW_ORG,
    sigma=SIGMA_ORG,
    rho=RHO_ORG,
    sv=SV_ORG,
    t_ref=T_REF,
    species_indices=VBS_SPECIES_INDICES,
    Dbk=DBK_DEFAULT,
    kc=KC_DEFAULT,
)


# =========================================================================
# Temperature correction
# =========================================================================

def calc_Cstar_T(cstar_ref_ug, delta_Hvap_kJ, temp, t_ref=T_REF):
    """Temperature-corrected C* via Clausius-Clapeyron.

    C*(T) = C*(Tref) × exp(−ΔHvap/R × (1/T − 1/Tref))

    Args:
        cstar_ref_ug: C* at reference T [µg/m³], scalar or array
        delta_Hvap_kJ: Enthalpy of vaporization [kJ/mol]
        temp: Temperature [K]
        t_ref: Reference temperature [K]

    Returns:
        C* at temperature T [µg/m³]
    """
    delta_Hvap_J = delta_Hvap_kJ * 1000.0
    exponent = -(delta_Hvap_J / R_GAS) * (1.0 / temp - 1.0 / t_ref)
    return cstar_ref_ug * jnp.exp(exponent)


# =========================================================================
# Unit conversions
# =========================================================================

def cstar_ug_to_Pa(cstar_ug, mw, temp):
    """Convert C* from µg/m³ to Pa.

    Using ideal gas: P = (C/MW) × R × T
    where C is in kg/m³ = µg/m³ × 1e-9

    Args:
        cstar_ug: Saturation concentration [µg/m³]
        mw: Molecular weight [g/mol]
        temp: Temperature [K]

    Returns:
        C* in [Pa]
    """
    cstar_kg = cstar_ug * 1.0e-9  # µg/m³ → kg/m³
    mw_kg = mw * 1.0e-3           # g/mol → kg/mol
    return (cstar_kg / mw_kg) * R_GAS * temp


def cstar_ug_to_molec_cm3(cstar_ug, mw):
    """Convert C* from µg/m³ to molec/cm³.

    Args:
        cstar_ug: Saturation concentration [µg/m³]
        mw: Molecular weight [g/mol]

    Returns:
        C* in [molec/cm³]
    """
    from ..core.config import AVOGADRO
    cstar_g_cm3 = cstar_ug * 1.0e-12  # µg/m³ → g/cm³
    return cstar_g_cm3 / mw * AVOGADRO
