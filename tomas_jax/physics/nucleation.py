"""Nucleation parameterizations for TOMAS-JAX.

Ports of Fortran subroutines from TRACER_SOM-TOMAS:
  - ricco_nucl.f  (Riccobono 2014 + Yu 2017 temperature correction)
  - dunne_inorg_nucl.f  (Dunne 2016, 4 mechanisms)
  - nucleation.f  (driver)

All functions are pure-JAX and JIT-compilable.
"""
import jax.numpy as jnp

from ..core.config import (
    SRTSO4, SRTORGLAST, ICOMP,
    AVOGADRO, PI,
)

# =========================================================================
# Riccobono 2014 constants
# Riccobono et al. (2014), Science 344:717-721, Eq. 1, Table 1
# Temperature correction: Yu et al. (2017), ACP 17:4997-5005, Eq. 3
# =========================================================================
_KM = 3.27e-21       # nucleation rate constant [cm^6 s^-1]
_DELTA_H = 2.66e-19  # enthalpy change [J] (Yu 2017 T-correction)
_KB_NUC = 1.38e-23   # Boltzmann constant [J/K]
_T0 = 278.0          # reference temperature [K]

# =========================================================================
# Dunne 2016 constants (complex T-dependence)
# Dunne et al. (2016), Science 354:1119-1124, Table S1
# Binary neutral (BN), ternary neutral (TN), actinometric neutral (AN),
# binary ion (BI), ternary ion (TI), actinometric ion (AI)
# =========================================================================
_PBN = 3.95;  _UBN = 9.70;   _VBN = 12.6;  _WBN = -0.00707
_PTN = 2.89;  _UTN = 182.0;  _VTN = 1.20;  _WTN = -4.19
_PAN = 8.00;  _AN  = 1.6e-6

_PBI = 3.37;  _UBI = -11.5;  _VBI = 25.5;  _WBI = 0.181
_PTI = 3.14;  _UTI = -23.8;  _VTI = 37.0;  _WTI = 0.227
_PAI = 3.07;  _AI  = 0.00485

# =========================================================================
# Nucleation cluster properties
# Kulmala et al. (2013): r ~ 0.85 nm for sulfuric acid clusters
# Density 1350 kg/m3: intermediate between H2SO4 (1830) and organic (1000)
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


# =========================================================================
# Zhao et al. 2024 — Mechanism 5: H2SO4-HNO3-NH3 synergistic
# =========================================================================
# Source: Wang et al. 2022 (Nature 605, 483-489), eq. 9 + Zhao 2024 cutoff.
#
# Wang 2022 eq. 9 (power-law fit to CLOUD data at 223 K):
#   J_1.7 = 2.9e-98 * exp(14000/T) * [H2SO4]^3 * [HNO3]^2 * [NH3]^4
#
# Zhao 2024 Methods: "applied only to temperatures <238 K; at higher
# temperatures, the NPF rate was set to zero and smooth transition was
# implemented near 238 K to avoid abrupt change."
# Implementation decision: logistic sigmoid with width=2 K (see
# docs/zhao2024_nucleation.md and docs/figures/nucleation/ for rationale).

_J_SYN_PREFACTOR = 2.9e-98  # [cm^9 s^-1] (Wang 2022 eq.9)
_E_SYN_OVER_R = 14000.0     # activation energy / R [K]
_T_SYN_CUTOFF = 238.0       # Zhao 2024 temperature cutoff [K]
_T_SYN_WIDTH = 2.0          # sigmoid width [K]; not in Zhao 2024, chosen for smoothness


def zhao2024_synergistic_hno3_rate(temp, h2so4, hno3, nh3):
    """H2SO4-HNO3-NH3 synergistic nucleation (Wang 2022 + Zhao 2024 cutoff).

    J = 2.9e-98 * exp(14000/T) * [H2SO4]^3 * [HNO3]^2 * [NH3]^4
        * sigmoid(T < 238 K, width=2 K)

    Args:
        temp:  Temperature [K]
        h2so4: H2SO4 concentration [molec cm^-3]
        hno3:  HNO3 concentration [molec cm^-3]
        nh3:   NH3 concentration [molec cm^-3]

    Returns:
        J [cm^-3 s^-1]
    """
    J_base = (_J_SYN_PREFACTOR
              * jnp.exp(_E_SYN_OVER_R / temp)
              * h2so4**3 * hno3**2 * nh3**4)
    f_cut = 1.0 / (1.0 + jnp.exp((temp - _T_SYN_CUTOFF) / _T_SYN_WIDTH))
    return J_base * f_cut


# =========================================================================
# Zhao et al. 2024 — Mechanisms 6–7: Pure-organic neutral and ion-induced
# =========================================================================
# Source: Kirkby et al. 2016 (Nature 533, 521-526), eq. 4 (Methods).
# Temperature dependence: Zhao 2024 Methods ("NPF module with 11 nucleation
# mechanisms"): J_T = J_278K * exp(-(T - 278) / 13), a factor of 2.15x
# per 10 K temperature decrease, from quantum chemistry calculations and
# the buffering effect of volatility shift with temperature.
#
# Kirkby 2016 Methods eq. 4:
#   J_n   = a1 * c^(a2 + a3/c)
#   J_ion = 2 * [n±] * a4 * c^(a4 + a5/c)
#
# where c = [HOM] in 10^6 cm^-3.  [n±] unit: NOT stated in Kirkby 2016;
# back-calculated from Fig. 3 as 10^3 cm^-3 — see docs/future_features.md.
# Zhao 2024 substitutes ULVOC (O:C > 0.4) for total HOM with the same
# functional form and Kirkby 2016 constants.
# Note: a4 appears both as a multiplicative prefactor AND in the exponent
# — this is the empirical form as published in Kirkby 2016 eq. 4.

_K16_A1 = 0.04001   # neutral prefactor [cm^-3 s^-1 at c=1 in 10^6 cm^-3]
_K16_A2 = 1.848     # neutral HOM power-law exponent
_K16_A3 = 0.001366  # neutral low-concentration correction
_K16_A4 = 1.566     # ion-induced coefficient (prefactor and exponent)
_K16_A5 = 0.1863    # ion-induced low-concentration correction
_K16_T_REF = 278.0  # Zhao 2024 temperature reference [K]
_K16_T_SCALE = 13.0 # Zhao 2024 temperature e-folding scale [K]


def kirkby2016_pure_organic_rate(temp, ulvoc, ionc):
    """Pure-organic neutral and ion-induced nucleation (Kirkby 2016 + Zhao 2024).

    Mechanisms 6 (neutral) and 7 (ion-induced) of the Zhao 2024 scheme.

    J_n   = a1 * c^(a2 + a3/c) * exp(-(T - 278) / 13)
    J_ion = 2 * (ionc/1e3) * a4 * c^(a4 + a5/c) * exp(-(T - 278) / 13)

    where c = ulvoc in 10^6 cm^-3 units.

    Args:
        temp:  Temperature [K]
        ulvoc: ULVOC concentration [molec cm^-3] (O:C > 0.4 fraction)
        ionc:  Small-ion concentration [cm^-3]

    Returns:
        (J_neutral, J_ion_induced) both [cm^-3 s^-1]
    """
    c = jnp.maximum(ulvoc * 1e-6, 1e-30)   # [10^6 cm^-3], guarded against 0
    n_pm = ionc * 1e-3                       # [10^3 cm^-3] (Kirkby 2016 unit)
    f_T = jnp.exp(-(temp - _K16_T_REF) / _K16_T_SCALE)

    J_n = _K16_A1 * c ** (_K16_A2 + _K16_A3 / c) * f_T
    J_ion = 2.0 * n_pm * _K16_A4 * c ** (_K16_A4 + _K16_A5 / c) * f_T

    # Zero out when ulvoc = 0 (avoids tiny nonzero from the safe floor)
    has_ulvoc = ulvoc > 0.0
    return jnp.where(has_ulvoc, J_n, 0.0), jnp.where(has_ulvoc, J_ion, 0.0)


# =========================================================================
# Zhao et al. 2024 — Mechanism 8: Organic–H2SO4 nucleation
# =========================================================================
# Source: Zhao 2024 Methods, refitted from Lehtipalo et al. 2018
# (Sci. Adv. 4, eaau5363) CLOUD data.
#
# Zhao 2024 tried two equation forms and adopted the bilinear one (R²=0.80):
#   J_org_sa = k * [H2SO4] * [ORG]   with k = 1.85e-14 cm^6 s^-1
#
# where [H2SO4] and [ORG] are in cm^-3.  [ORG] in Zhao 2024 is non-nitrate
# HOM dimer (cm^-3) from Lehtipalo 2018; in this implementation we use
# ulvoc as a proxy (O:C > 0.4 fraction).  Strictly, Zhao 2024 uses
# ULVOCs + ELVOCs (C* < 3e-5 ug/m3) for this mechanism vs. ULVOC-only
# (C* < 3e-9 ug/m3) for mechanisms 6-7 — see docs/future_features.md.
#
# Same temperature dependence as mechanisms 6-7 (Zhao 2024 Methods):
#   J_T = J_278K * exp(-(T - 278) / 13)

_ORG_SA_K = 1.85e-14    # bilinear rate constant [cm^6 s^-1] (Zhao 2024)
# reuses _K16_T_REF = 278.0 K and _K16_T_SCALE = 13.0 K from above


def zhao2024_organic_h2so4_rate(temp, h2so4, ulvoc):
    """Organic–H2SO4 nucleation (Zhao 2024, mechanism 8).

    Refitted from Lehtipalo et al. 2018 CLOUD data using bilinear form:
        J = 1.85e-14 * [H2SO4] * [ORG] * exp(-(T - 278) / 13)

    Args:
        temp:  Temperature [K]
        h2so4: H2SO4 concentration [molec cm^-3]
        ulvoc: ULVOC concentration [molec cm^-3] (proxy for HOM dimer)

    Returns:
        J [cm^-3 s^-1]
    """
    f_T = jnp.exp(-(temp - _K16_T_REF) / _K16_T_SCALE)
    return _ORG_SA_K * h2so4 * ulvoc * f_T


# =========================================================================
# Zhao et al. 2024 — Mechanism 9: Amine–H2SO4 nucleation
# =========================================================================
# Source: Zhao 2024 Methods, based on Almeida et al. 2013 (Nature 502, 359)
# + Kürten et al. 2018 (ACP 18, 845) x10 correction for self-coagulation
# + Cai et al. 2021 temperature-dependence from cluster kinetics.
#
# Base rate at 278 K (from Hanson et al. 2017, eq. 11 in Kürten 2018):
#   J_278K = exp(-129 + 16200/278) * [H2SO4]^3 * [DMA]^1.5
#          ≈ 2.19e-31 * [H2SO4]^3 * [DMA]^1.5    [cm^-3 s^-1]
#
# Hanson (2017) showed this agrees with the Kürten-corrected (x10) CLOUD
# data at 278 K, 40 pptv DMA, [H2SO4] = 1e6-3e7 cm^-3.  Recommended
# validity: DMA = 2-16 pptv, but Kürten 2018 showed rates remain near
# the kinetic limit even at 1 pptv DMA.
#
# Temperature dependence (Zhao 2024 Methods, from Cai et al. 2021):
#   f_T = 1.576*exp(-((T-250.6)/23.18)^2)
#       + 0.6956*exp(-((T-273.1)/13.01)^2)
#   f_T(278K) ≈ 0.994 ≈ 1.0  (reference temperature confirmed)
#
# Sum of two Gaussians: captures the non-Arrhenius T-dependence of
# amine-H2SO4 cluster kinetics (sharp drop above ~310 K where clusters
# become unstable, and gradual decrease below ~240 K).
#
# Note: Zhao 2024 main text says "scaling factor of 10 to the
# parameterization of Almeida et al." The Hanson formula at 278 K gives
# rates equivalent to Almeida x10 (Kürten 2018, Fig. 1).  The exact
# base-rate formula is not given in the Zhao 2024 main text (it is in
# their supplementary materials).

import math as _math
_AMINE_J_REF = _math.exp(-129.0 + 16200.0 / 278.0)  # ≈ 2.19e-31 [cm^{3(3)+1.5(1.5)} s^-1]
_AMINE_SA_EXP = 3.0     # H2SO4 power (Hanson 2017)
_AMINE_DMA_EXP = 1.5    # DMA power (Hanson 2017)
# Cai et al. 2021 temperature function (sum of two Gaussians)
_AMINE_G1_AMP = 1.576;   _AMINE_G1_MU = 250.6;  _AMINE_G1_SIG = 23.18
_AMINE_G2_AMP = 0.6956;  _AMINE_G2_MU = 273.1;  _AMINE_G2_SIG = 13.01


def zhao2024_amine_h2so4_rate(temp, h2so4, dma):
    """Amine–H2SO4 nucleation (Zhao 2024, mechanism 9).

    J = exp(-129 + 16200/278) * [H2SO4]^3 * [DMA]^1.5 * f_T(Cai 2021)

    f_T = 1.576*exp(-((T-250.6)/23.18)^2) + 0.6956*exp(-((T-273.1)/13.01)^2)

    Args:
        temp:  Temperature [K]
        h2so4: H2SO4 concentration [molec cm^-3]
        dma:   Dimethylamine concentration [molec cm^-3]

    Returns:
        J [cm^-3 s^-1]
    """
    f_T = (_AMINE_G1_AMP * jnp.exp(-((temp - _AMINE_G1_MU) / _AMINE_G1_SIG)**2)
         + _AMINE_G2_AMP * jnp.exp(-((temp - _AMINE_G2_MU) / _AMINE_G2_SIG)**2))
    return _AMINE_J_REF * h2so4**_AMINE_SA_EXP * dma**_AMINE_DMA_EXP * f_T


# =========================================================================
# Zhao et al. 2024 — Mechanisms 10–11: Iodine oxoacids nucleation
# =========================================================================
# Source: Zhao 2024 Methods, derived by fitting CLOUD experimental data
# from He et al. 2021 (Science 371, 589-595).
#
# Neutral (mechanism 10):
#   J_n = 2.57e-32 * [HIO3]^4.23
#       * 1.40e-46 * exp(29900 / max(T, 263))
#
# Ion-induced (mechanism 11):
#   J_ion = 1.28e-18 * [HIO3]^2.48 * ([ION] / 700)
#         * 1.40e-46 * exp(29900 / max(T, 283))
#
# [HIO3] in cm^-3, [ION] = total ion concentration in cm^-3, T in K.
#
# Temperature factor: Arrhenius form exp(29900/T), clamped at Tmin to
# prevent divergence.  For neutral: Tmin=263K (clamp at -10°C because
# He et al. 2021 only covered +10°C to -10°C, and quantum chemistry
# shows neutral clusters are already stable at -10°C).  For ion-induced:
# Tmin=283K (no T-dependence below +10°C because ions are already
# efficient at that temperature; He et al. 2021).
#
# Zhao 2024: "we did not consider the temperature dependence of ion-
# induced nucleation below +10°C.  The temperature dependence above
# +10°C was assumed to be the same as that of neutral nucleation."

_IOD_N_A = 2.57e-32    # neutral HIO3 prefactor [cm^{-3-4.23*3} s^-1]
_IOD_N_P = 4.23        # neutral HIO3 exponent
_IOD_I_A = 1.28e-18    # ion-induced HIO3 prefactor
_IOD_I_P = 2.48        # ion-induced HIO3 exponent
_IOD_I_ION_REF = 700.0 # ion reference concentration [cm^-3]
_IOD_T_A = 1.40e-46    # temperature Arrhenius prefactor
_IOD_T_E = 29900.0     # temperature activation energy / R [K]
_IOD_T_MIN_N = 263.0   # neutral Tmin clamp [K] (−10°C)
_IOD_T_MIN_I = 283.0   # ion-induced Tmin clamp [K] (+10°C)


def zhao2024_iodine_oxoacid_rate(temp, hio3, ionc):
    """Iodine oxoacids neutral and ion-induced nucleation (Zhao 2024).

    Mechanisms 10 (neutral) and 11 (ion-induced) from He et al. 2021 CLOUD data.

    J_n   = 2.57e-32 * [HIO3]^4.23 * 1.40e-46 * exp(29900/max(T, 263))
    J_ion = 1.28e-18 * [HIO3]^2.48 * (ionc/700) * 1.40e-46 * exp(29900/max(T, 283))

    Args:
        temp:  Temperature [K]
        hio3:  HIO3 (iodic acid) concentration [molec cm^-3]
        ionc:  Total ion concentration [cm^-3]

    Returns:
        (J_neutral, J_ion_induced) both [cm^-3 s^-1]
    """
    f_T_n = _IOD_T_A * jnp.exp(_IOD_T_E / jnp.maximum(temp, _IOD_T_MIN_N))
    f_T_i = _IOD_T_A * jnp.exp(_IOD_T_E / jnp.maximum(temp, _IOD_T_MIN_I))

    J_n = _IOD_N_A * hio3**_IOD_N_P * f_T_n
    J_ion = _IOD_I_A * hio3**_IOD_I_P * (ionc / _IOD_I_ION_REF) * f_T_i

    return J_n, J_ion


# =========================================================================
# Default enable masks for Zhao 2024 (all 11 mechanisms enabled)
# =========================================================================
# Order: (bn, tn, bi, ti, syn, porg_n, porg_i, org_sa, amine, iod_n, iod_i)
ZHAO2024_ALL_ENABLED = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
ZHAO2024_DUNNE_ONLY = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _compute_ionc(fion, temp, Mair):
    """Steady-state ion concentration from ion-pair production rate.

    Same recombination formula used by Dunne 2016.

    Args:
        fion: Ion-pair production rate [pairs/cm3/s]
        temp: Temperature [K]
        Mair: Air number density [molec/cm3]

    Returns:
        ionc: Small-ion concentration [cm^-3]
    """
    alpha_ion = (6e-8 * jnp.sqrt(300.0 / temp)
                 + 6e-26 * Mair * (300.0 / temp)**4)
    return jnp.sqrt(fion / alpha_ion)


def zhao2024_nucleation_rate(
    Gc, temp, pres, boxvol,
    org_conc, nh3_conc, fion,
    hno3=0.0, ulvoc=0.0, dma=0.0, hio3=0.0,
    enable_masks=ZHAO2024_ALL_ENABLED, fn_scale=1.0,
):
    """Zhao 2024 11-mechanism total nucleation rate [cm^-3 s^-1] without state mutation.

    Sums per-mechanism rates with enable_masks and applies fn_scale to the total.
    Mechanism order in the returned per-mechanism array matches enable_masks:
      (bn, tn, bi, ti, syn, porg_n, porg_i, org_sa, amine, iod_n, iod_i)

    Args:
        Gc: Gas concentrations [kg/grid cell], shape (N_GAS_SPECIES,)
        temp, pres, boxvol: Environment scalars
        org_conc: Organic vapor [molec/cm3] (kept for API symmetry; unused by Zhao)
        nh3_conc: NH3 concentration [molec/cm3] (mechs tn, ti, syn)
        fion: Ion-pair production rate [pairs/cm3/s] (mechs bi, ti, porg_i, iod_i)
        hno3: HNO3 [molec/cm3] (mech syn)
        ulvoc: ULVOC [molec/cm3] (mechs porg_n, porg_i, org_sa)
        dma: Dimethylamine [molec/cm3] (mech amine)
        hio3: HIO3 iodic acid [molec/cm3] (mechs iod_n, iod_i)
        enable_masks: Tuple of 11 floats (0.0/1.0) per mechanism
        fn_scale: Multiplicative scaling factor on the total

    Returns:
        (fn_total, fn_per_mechanism): scalar [cm^-3 s^-1] and shape-(11,) array
        where the per-mechanism values already include enable_masks but NOT fn_scale.
        fn_total is sum(fn_per_mechanism) * fn_scale.
    """
    m_bn, m_tn, m_bi, m_ti, m_syn, m_porg_n, m_porg_i, m_org_sa, m_amine, m_iod_n, m_iod_i = enable_masks

    # Convert gas-phase H2SO4: kg/grid_cell -> molec/cm3
    h2so4 = Gc[SRTSO4] / boxvol * 1000.0 / 98.0 * AVOGADRO

    # Air number density [molec/cm3]
    Mair = 2.69e19 * 273.15 / temp * pres / 101325.0

    # Ion concentration (shared by mechs bi, ti, porg_i, iod_i)
    ionc = _compute_ionc(fion, temp, Mair)

    _, Jbn, Jtn, Jbi, Jti = dunne_nucleation_rate(temp, fion, h2so4, nh3_conc, Mair)
    J_syn = zhao2024_synergistic_hno3_rate(temp, h2so4, hno3, nh3_conc)
    J_porg_n, J_porg_i = kirkby2016_pure_organic_rate(temp, ulvoc, ionc)
    J_org_sa = zhao2024_organic_h2so4_rate(temp, h2so4, ulvoc)
    J_amine = zhao2024_amine_h2so4_rate(temp, h2so4, dma)
    J_iod_n, J_iod_i = zhao2024_iodine_oxoacid_rate(temp, hio3, ionc)

    fn_per_mech = jnp.array([
        Jbn * m_bn,
        Jtn * m_tn,
        Jbi * m_bi,
        Jti * m_ti,
        J_syn * m_syn,
        J_porg_n * m_porg_n,
        J_porg_i * m_porg_i,
        J_org_sa * m_org_sa,
        J_amine * m_amine,
        J_iod_n * m_iod_n,
        J_iod_i * m_iod_i,
    ])
    fn_total = jnp.sum(fn_per_mech) * fn_scale
    return fn_total, fn_per_mech


def zhao2024_nucleation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
    org_conc, nh3_conc, fion,
    hno3=0.0, ulvoc=0.0, dma=0.0, hio3=0.0,
    enable_masks=ZHAO2024_ALL_ENABLED, fn_scale=1.0,
):
    """Apply Zhao 2024 11-mechanism nucleation to aerosol state (JIT-compilable).

    Mechanisms 1-4: Dunne 2016 (H2SO4 ± NH3 ± ions)
    Mechanism 5:    H2SO4-HNO3-NH3 synergistic (Wang 2022 + Zhao 2024)
    Mechanisms 6-7: Pure-organic neutral/ion (Kirkby 2016 + Zhao 2024)
    Mechanism 8:    Organic-H2SO4 (Lehtipalo 2018 refit, Zhao 2024)
    Mechanism 9:    Amine-H2SO4 (Hanson/Kürten/Cai, Zhao 2024)
    Mechanisms 10-11: Iodine oxoacids neutral/ion (He 2021, Zhao 2024)

    Delegates rate computation to ``zhao2024_nucleation_rate`` and applies
    the same gas-clamping logic as ``nucleation_step``.

    Args:
        Nk, Mk, Gc, xk: Aerosol state arrays (same as nucleation_step)
        temp, pres, boxvol, dt: Environment and timestep
        org_conc, nh3_conc, fion, hno3, ulvoc, dma, hio3: Gas concentrations.
            See ``zhao2024_nucleation_rate`` for units and per-mechanism use.
        enable_masks: Tuple of 11 floats (0.0/1.0) to enable/disable each mechanism
        fn_scale: Multiplicative scaling factor for total nucleation rate

    Returns:
        (Nk_new, Mk_new, Gc_new)
    """
    fn, _ = zhao2024_nucleation_rate(
        Gc, temp, pres, boxvol,
        org_conc, nh3_conc, fion,
        hno3=hno3, ulvoc=ulvoc, dma=dma, hio3=hio3,
        enable_masks=enable_masks, fn_scale=fn_scale,
    )

    # Number of new particles
    dN = fn * boxvol * dt

    # Mass added to bin 0 (same cluster properties as existing scheme)
    dM_total = fn * _MNUC * boxvol * dt
    dM_so4 = 0.9 * dM_total
    dM_org = 0.1 * dM_total

    # Gas clamping (same logic as nucleation_step)
    gc_so4 = Gc[SRTSO4]
    gas_depleted = dM_so4
    need_clamp = gas_depleted > gc_so4

    dM_so4_clamped = gc_so4 * (96.0 / 98.0)
    dN_clamped = dM_so4_clamped / _MNUC
    dM_org_clamped = dN_clamped * 0.1 * _MNUC

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


def ricco_dunne_nucleation_rate(
                              Gc, temp, pres, boxvol,
                              org_conc, nh3_conc, fion,
                              enable_organic=1.0, enable_inorganic=1.0, fn_scale=1.0):
    """Riccobono 2014 + Dunne 2016 total nucleation rate [cm^-3 s^-1].

    Pure rate calculation, no state mutation. Used by adaptive sub-stepping
    and by the API diagnostic snapshots.
    """
    h2so4 = Gc[SRTSO4] / boxvol * 1000.0 / 98.0 * AVOGADRO
    Mair = 2.69e19 * 273.15 / temp * pres / 101325.0

    fn_org = ricco_nucleation_rate(temp, h2so4, org_conc)
    fn_inorg, _, _, _, _ = dunne_nucleation_rate(temp, fion, h2so4, nh3_conc, Mair)

    return (fn_org * enable_organic + fn_inorg * enable_inorganic) * fn_scale


# Backward-compatible alias. Prefer ``ricco_dunne_nucleation_rate`` in new code.
estimate_nucleation_rate = ricco_dunne_nucleation_rate


def compute_nucleation_substeps(fn, boxvol, dt, N_total,
                                 max_frac=0.5, max_substeps=20):
    """Compute nucleation sub-step count.

    Criterion: limit dN per substep to max_frac * N_total.

    Args:
        fn: Nucleation rate [cm^-3 s^-1]
        boxvol: Grid cell volume [cm^3]
        dt: Full timestep [s]
        N_total: Current total particle number [#/cell]
        max_frac: Max fractional increase per substep (default 0.5 = 50%)
        max_substeps: Hard cap on number of substeps

    Returns:
        n_sub: int (JAX traced value, >= 1)
    """
    dN_full = fn * boxvol * dt
    n_sub = jnp.where(
        N_total > 1.0,
        jnp.ceil(dN_full / (max_frac * N_total)),
        1.0
    ).astype(int)
    return jnp.clip(n_sub, 1, max_substeps)


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
