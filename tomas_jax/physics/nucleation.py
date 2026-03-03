"""Nucleation parameterizations for TOMAS-JAX.

Ports of Fortran subroutines from TRACER_SOM-TOMAS:
  - ricco_nucl.f  (Riccobono 2014 + Yu 2017 temperature correction)
  - dunne_inorg_nucl.f  (Dunne 2016, 4 mechanisms)
  - nucleation.f  (driver)

All functions are pure-JAX and JIT-compilable.
"""
import jax.numpy as jnp

from ..core.config import (
    SRTSO4, SRTORGLAST, NBINS, ICOMP,
    AVOGADRO, PI,
)

# =========================================================================
# Riccobono 2014 constants
# =========================================================================
_KM = 3.27e-21       # nucleation rate constant [cm^6 s^-1]
_DELTA_H = 2.66e-19  # enthalpy change [J]
_KB_NUC = 1.38e-23   # Boltzmann constant [J/K]
_T0 = 278.0          # reference temperature [K]

# =========================================================================
# Dunne 2016 constants (complex T-dependence)
# =========================================================================
_PBN = 3.95;  _UBN = 9.70;   _VBN = 12.6;  _WBN = -0.00707
_PTN = 2.89;  _UTN = 182.0;  _VTN = 1.20;  _WTN = -4.19
_PAN = 8.00;  _AN  = 1.6e-6

_PBI = 3.37;  _UBI = -11.5;  _VBI = 25.5;  _WBI = 0.181
_PTI = 3.14;  _UTI = -23.8;  _VTI = 37.0;  _WTI = 0.227
_PAI = 3.07;  _AI  = 0.00485

# =========================================================================
# Nucleation cluster properties
# =========================================================================
_RNUC_NM = 0.85          # critical cluster radius [nm]
_RNUC_M = _RNUC_NM * 1e-9  # [m]
_RHO_NUC = 1350.0        # cluster density [kg/m^3]
_MNUC = (4.0 / 3.0) * PI * _RNUC_M**3 * _RHO_NUC  # cluster mass [kg]


def ricco_nucleation_rate(temp, h2so4, org):
    """Riccobono 2014 organic nucleation + Yu 2017 T-correction.

    Args:
        temp: Temperature [K] (scalar)
        h2so4: Gas-phase H2SO4 [molec/cm3] (scalar)
        org: Gas-phase oxidized organics [molec/cm3] (scalar)

    Returns:
        fn: Nucleation rate [cm^-3 s^-1]
    """
    Jnotemp = _KM * h2so4**2 * org
    ft = jnp.exp(_DELTA_H / _KB_NUC * (1.0 / temp - 1.0 / _T0))
    fn = Jnotemp * ft
    return fn


def dunne_nucleation_rate(temp, fion, h2so4, nh3, Mair):
    """Dunne 2016 inorganic nucleation (4 mechanisms).

    Args:
        temp: Temperature [K] (scalar)
        fion: Ion-pair production rate [pairs/cm3/s] (scalar)
        h2so4: Gas-phase H2SO4 [molec/cm3] (scalar)
        nh3: Gas-phase NH3 [molec/cm3] (scalar)
        Mair: Air number density [molec/cm3] (scalar)

    Returns:
        (fn_total, Jbn, Jtn, Jbi, Jti) all [cm^-3 s^-1]
    """
    # Scale to Dunne units (1e6 molec/cm3)
    cna = h2so4 * 1e-6
    nh3_s = nh3 * 1e-6

    # Ion concentration from steady-state recombination
    alpha_ion = (6e-8 * jnp.sqrt(300.0 / temp)
                 + 6e-26 * Mair * (300.0 / temp)**4)
    ionc = jnp.sqrt(fion / alpha_ion)

    # Rate constants (complex T-dependence)
    t1000 = temp / 1000.0
    kbn = jnp.exp(_UBN - jnp.exp(_VBN * (t1000 - _WBN)))
    ktn = jnp.exp(_UTN - jnp.exp(_VTN * (t1000 - _WTN)))
    kbi = jnp.exp(_UBI - jnp.exp(_VBI * (t1000 - _WBI)))
    kti = jnp.exp(_UTI - jnp.exp(_VTI * (t1000 - _WTI)))

    # Ternary f-values (with nh3 threshold)
    ffn = jnp.where(
        nh3_s > 1e-10,
        nh3_s * cna**_PTN / (_AN + cna**_PTN / nh3_s**_PAN),
        0.0,
    )
    ffi = jnp.where(
        nh3_s > 1e-10,
        nh3_s * cna**_PTI / (_AI + cna**_PTI / nh3_s**_PAI),
        0.0,
    )

    # Four nucleation mechanisms
    Jbn = kbn * cna**_PBN
    Jtn = ktn * ffn
    Jbi = kbi * ionc * cna**_PBI
    Jti = kti * ionc * ffi

    fn_total = Jbn + Jtn + Jbi + Jti
    return fn_total, Jbn, Jtn, Jbi, Jti


def nucleation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
    org_conc, nh3_conc, fion,
    enable_organic=1.0, enable_inorganic=1.0, fn_scale=1.0,
):
    """Apply nucleation to aerosol state (JIT-compilable).

    Args:
        Nk: Number per bin [#/grid cell], shape (NBINS,)
        Mk: Mass per bin [kg/grid cell], shape (NBINS, ICOMP)
        Gc: Gas concentrations [kg/grid cell], shape (N_GAS_SPECIES,)
        xk: Bin boundaries [kg], shape (NBINS+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        dt: Timestep [s]
        org_conc: Organic vapor concentration [molec/cm3]
        nh3_conc: NH3 concentration [molec/cm3]
        fion: Ion-pair production rate [pairs/cm3/s]
        enable_organic: 1.0 to enable Riccobono, 0.0 to disable
        enable_inorganic: 1.0 to enable Dunne, 0.0 to disable
        fn_scale: Multiplicative scaling factor for total nucleation rate

    Returns:
        (Nk_new, Mk_new, Gc_new)
    """
    # Convert gas-phase H2SO4: kg/grid_cell -> molec/cm3
    h2so4 = Gc[SRTSO4] / boxvol * 1000.0 / 98.0 * AVOGADRO

    # Air number density [molec/cm3]
    Mair = 2.69e19 * 273.15 / temp * pres / 101325.0

    # Organic nucleation (Riccobono 2014)
    fn_org = ricco_nucleation_rate(temp, h2so4, org_conc)

    # Inorganic nucleation (Dunne 2016)
    fn_inorg, _, _, _, _ = dunne_nucleation_rate(
        temp, fion, h2so4, nh3_conc, Mair
    )

    # Total rate with enable masks and scaling
    fn = (fn_org * enable_organic + fn_inorg * enable_inorganic) * fn_scale

    # Number of new particles
    dN = fn * boxvol * dt

    # Mass added to bin 0
    dM_total = fn * _MNUC * boxvol * dt
    dM_so4 = 0.9 * dM_total
    dM_org = 0.1 * dM_total

    # Check if gas can supply the SO4 mass
    # Fortran depletes Gc by the SO4 mass added (without 98/96 MW correction)
    gc_so4 = Gc[SRTSO4]

    # Gas depletion = SO4 mass added to bin (Fortran: Gcf = Gci - (Mkf-mold))
    gas_depleted = dM_so4

    # Clamp: if we'd deplete more gas than available
    need_clamp = gas_depleted > gc_so4

    # Clamped values (Fortran: use all available gas, convert H2SO4→SO4)
    dM_so4_clamped = gc_so4 * (96.0 / 98.0)
    dN_clamped = dM_so4_clamped / _MNUC

    # Clamp organic mass proportionally when gas supply limits nucleation.
    # Without this, dM_org uses the full (enormous) fn while dN is clamped
    # to a tiny value, creating particles with avg mass >> bin boundary.
    dM_org_clamped = dN_clamped * 0.1 * _MNUC

    # Select clamped or normal path
    dN_final = jnp.where(need_clamp, dN_clamped, dN)
    dM_so4_final = jnp.where(need_clamp, dM_so4_clamped, dM_so4)
    dM_org_final = jnp.where(need_clamp, dM_org_clamped, dM_org)
    gas_depleted_final = jnp.where(need_clamp, gc_so4, gas_depleted)

    # Only modify if fn > 0
    has_nuc = fn > 0.0

    Nk_new = Nk.at[0].add(jnp.where(has_nuc, dN_final, 0.0))
    Mk_new = Mk.at[0, SRTSO4].add(jnp.where(has_nuc, dM_so4_final, 0.0))
    Mk_new = Mk_new.at[0, SRTORGLAST].add(jnp.where(has_nuc, dM_org_final, 0.0))
    Gc_new = Gc.at[SRTSO4].add(-jnp.where(has_nuc, gas_depleted_final, 0.0))

    return Nk_new, Mk_new, Gc_new
