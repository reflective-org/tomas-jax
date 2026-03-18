"""Condensation sink calculations for TOMAS-JAX.

Exact port of getCondSink.f (lines 1-135).

The condensation sink CS [s^-1] is the pseudo-first-order rate constant
for gas-phase loss to the aerosol population. It depends on gas diffusivity,
particle diameters, number concentrations, and the Fuchs-Sutugin correction
for non-continuum effects.

    CS = 2 * pi * Di * sum_k(Dpk * Nk * beta_k) / (boxvol * 1e-6)

The per-bin sink fraction ``sinkfrac`` distributes condensing mass across
bins in proportion to each bin's contribution to the total sink.

References:
    - getCondSink.f (Jeff Pierce, May 2007)
    - Seinfeld & Pandis eqn 11.35
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Tuple, Union

from .gas_properties import (
    calc_gas_diffusivity,
    calc_mean_free_path,
    calc_fuchs_sutugin_correction,
)
from .density import calc_density
from ..core.config import PI, MW_H2SO4, SV_H2SO4, ICOMP_NODIAG

# Fortran getCondSink.f: parameter(Neps=1.0d10)
NEPS_CONDSINK = 1.0e10


def calc_condensation_sink(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray],
    boxvol: Union[float, jnp.ndarray],
    molecular_weight: float = MW_H2SO4,
    diffusion_volume: float = SV_H2SO4,
    accommodation_coeff: float = 1.0,
    xk: jnp.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate condensation sink from aerosol size distribution.

    Exact port of getCondSink.f lines 77-135.

    For bins with Nk > 1e10: uses actual density and mp from composition.
    For bins with Nk <= 1e10: uses default density=1500 and mp=1.4*xk[k],
    matching Fortran getCondSink.f parameter(Neps=1.0d10).

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        molecular_weight: MW of condensing species [g/mol]
        diffusion_volume: Diffusion volume parameter
        accommodation_coeff: alpha (default 1.0)
        xk: Bin boundaries [kg], shape (ibins+1,). If None, uses default 36-bin grid.

    Returns:
        CS: Condensation sink [s^-1]
        sinkfrac: Fraction of CS from each bin, shape (ibins,)
    """
    if xk is None:
        from ..core.config import xk_boundaries
        xk = xk_boundaries()

    Di = calc_gas_diffusivity(temp, pres, molecular_weight, diffusion_volume)
    mfp = calc_mean_free_path(temp, pres, molecular_weight, diffusion_volume)

    # --- Compute Dpk following Fortran getCondSink.f lines 89-113 ---
    # For bins with Nk > Neps: actual density and mp
    # For bins with Nk <= Neps: default density=1500, mp=1.4*xk[k]
    has_particles = Nk > NEPS_CONDSINK

    # Actual properties (for populated bins)
    Mktot = jnp.sum(Mk, axis=1)  # total wet mass per bin
    mp_actual = Mktot / jnp.maximum(Nk, 1e-30)
    density_actual = calc_density(Mk)

    # Default properties (for sparse bins)
    mp_default = 1.4 * xk[:-1]
    density_default = 1500.0

    mp = jnp.where(has_particles, mp_actual, mp_default)
    density = jnp.where(has_particles, density_actual, density_default)

    # Dpk = ((mp/density) * (6/pi))^(1/3)
    Dpk = jnp.cbrt(mp / density * (6.0 / PI))

    # Kn = 2*mfp/Dpk (getCondSink.f line 111)
    Kn = 2.0 * mfp / jnp.maximum(Dpk, 1e-30)

    # beta = (1+Kn)/(1+2*Kn*(1+Kn)/alpha) (getCondSink.f line 112)
    beta = calc_fuchs_sutugin_correction(Kn, accommodation_coeff)

    # Guard: bins with Dpk=0 produce Kn=inf -> beta=NaN.
    safe_beta = jnp.where(Dpk > 0.0, beta, 0.0)
    sink_contributions = Dpk * Nk * safe_beta
    CS_sum = jnp.sum(sink_contributions)

    # CS = 2*pi*Di*sum / (boxvol*1e-6)  [boxvol cm^3 -> m^3]
    CS = 2.0 * PI * Di * CS_sum / (boxvol * 1e-6)

    sinkfrac = jnp.where(
        CS_sum > 1e-30,
        sink_contributions / CS_sum,
        jnp.zeros_like(Dpk)
    )

    return CS, sinkfrac


def calc_organic_condensation_sink(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray],
    boxvol: Union[float, jnp.ndarray],
    molecular_weight: float,
    diffusion_volume: float,
    accommodation_coeff: float,
    xk: jnp.ndarray,
    cstar_ug: Union[float, jnp.ndarray],
    Dbk: float = 1.0e-10,
    kc: float = 0.0,
    use_gasside_only: bool = False,
) -> tuple:
    """Organic condensation sink following Zaveri et al. (2014).

    Implements the two-film theory for kinetic SOA partitioning with
    gas-phase diffusion, interfacial accommodation, particle-phase bulk
    diffusion, and particle-phase reaction.

    Two approximations (Zaveri et al. 2014, Section 3.1):
      - Approximation 1 (kc >= 0.01 s⁻¹): gas-side k_g only, with Q
        correction applied to driving force by the caller.
      - Approximation 2 (kc < 0.01 s⁻¹): two-film K_g that captures
        particle-phase resistance in the mass transfer coefficient.

    When use_gasside_only=True, returns gas-side-only CS (for Approx 1).
    When use_gasside_only=False (default), returns two-film CS (Approx 2).

    Args:
        Nk: Number concentration [#/grid cell], shape (nbins,)
        Mk: Mass concentration [kg/grid cell], shape (nbins, icomp)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm³]
        molecular_weight: MW of condensing species [g/mol]
        diffusion_volume: Diffusion volume for Fuller-Schettler-Giddings
        accommodation_coeff: Mass accommodation coefficient
        xk: Bin boundaries [kg], shape (nbins+1,)
        cstar_ug: Temperature-corrected C* [µg/m³]
        Dbk: Particle-phase diffusion coefficient [m²/s]
        kc: First-order loss rate in particle phase [s⁻¹]
        use_gasside_only: If True, return gas-side CS (Approximation 1).

    Returns:
        CS: Condensation sink [s⁻¹]
        sinkfrac: Per-bin fraction, shape (nbins,)
        Q: Quasi-steady-state parameter per bin, shape (nbins,).
           Q = avg/surface particle-phase concentration ratio (Eq. 8).
           Q = 1 for liquid (well-mixed), Q < 1 for semisolid.
    """
    # Gas diffusivity and mean free path
    Di = calc_gas_diffusivity(temp, pres, molecular_weight, diffusion_volume)
    mfp = calc_mean_free_path(temp, pres, molecular_weight, diffusion_volume)

    # Dpk and density (same logic as calc_condensation_sink)
    has_particles = Nk > NEPS_CONDSINK

    Mktot = jnp.sum(Mk, axis=1)
    mp_actual = Mktot / jnp.maximum(Nk, 1e-30)
    density_actual = calc_density(Mk)

    mp_default = 1.4 * xk[:-1]
    density_default = 1500.0

    mp = jnp.where(has_particles, mp_actual, mp_default)
    density = jnp.where(has_particles, density_actual, density_default)

    Dpk = jnp.cbrt(mp / density * (6.0 / PI))
    Rpk = Dpk / 2.0

    # Knudsen number
    Kn = 2.0 * mfp / jnp.maximum(Dpk, 1e-30)

    # Fuchs-Sutugin correction — Zaveri et al. (2014) Eq. 14
    # Coefficient 0.283 from Fuchs & Sutugin (1971), confirmed by
    # Seinfeld & Pandis (2006, Table 12.1). Note: Fortran soacond.f
    # uses 0.238 (transcription error); paper clearly states 0.283.
    alpha = accommodation_coeff
    FC = (0.75 * alpha * (1.0 + Kn)
          / (Kn * (1.0 + Kn) + 0.283 * alpha * Kn + 0.75 * alpha))

    # Gas-side mass transfer coefficient [m/s] — Eq. 13
    safe_Rp = jnp.maximum(Rpk, 1e-30)
    kgk = Di / safe_Rp * FC

    # Particle-side mass transfer coefficient [m/s] — Eq. 23-24
    # When kc=0: kpk = 5*Dbk/Rp (well-mixed, Eq. 24)
    # When kc>0: diffuso-reactive with coth (Eq. 23)
    kpk_simple = 5.0 * Dbk / safe_Rp

    # Diffuso-reactive parameter q — Eq. 4
    # q = Rp * sqrt(kc / Db) = ratio of particle radius to reacto-diffusive length
    qk = safe_Rp * jnp.sqrt(jnp.maximum(kc, 0.0) / jnp.maximum(Dbk, 1e-30))
    safe_qk = jnp.maximum(qk, 1e-30)
    coth_qk = 1.0 / jnp.tanh(jnp.maximum(safe_qk, 1e-10))

    # Q = average / surface concentration ratio at quasi-steady state — Eq. 8
    # Q → 1 for liquid (q → 0), Q → 0 for semisolid with fast reaction
    Qkk_raw = 3.0 * (safe_qk * coth_qk - 1.0) / safe_qk**2
    Qkk = jnp.where(qk > 1e-6, jnp.minimum(Qkk_raw, 0.99), 1.0)

    # Particle-side k_p — Eq. 23
    kpk_reactive = Dbk / safe_Rp * (safe_qk * coth_qk - 1.0) / (1.0 - Qkk)
    kpk = jnp.where(kc > 0.0, kpk_reactive, kpk_simple)

    # --- Choose CS formulation ---
    if use_gasside_only:
        # Approximation 1 (Eq. 29): gas-side only.
        # Particle-phase effect captured by Q in the driving force.
        # Effective coefficient = Rp × kgk (consistent units with Kgkk [m²/s])
        Keff = safe_Rp * kgk
    else:
        # Approximation 2 (Eq. 31): two-film K_g — Eq. 20
        # Kgkk = Rp / (1/kgk + 1/kpk × C*/density/1e9)
        cstar_weight = cstar_ug / density / 1.0e9
        Keff = safe_Rp / (1.0 / jnp.maximum(kgk, 1e-30)
                          + cstar_weight / jnp.maximum(kpk, 1e-30))

    # Guard empty bins
    Keff = jnp.where(Dpk > 0.0, Keff, 0.0)

    # CS = 2π × Σ(Dpk × Nk × Keff) / boxvol_m3
    sink_contributions = Dpk * Nk * Keff
    CS_sum = jnp.sum(sink_contributions)
    CS = 2.0 * PI * CS_sum / (boxvol * 1e-6)

    sinkfrac = jnp.where(
        CS_sum > 1e-30,
        sink_contributions / CS_sum,
        jnp.zeros_like(Dpk),
    )

    return CS, sinkfrac, Qkk


def calc_organic_condensation_sink_batch(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    temp: Union[float, jnp.ndarray],
    pres: Union[float, jnp.ndarray],
    boxvol: Union[float, jnp.ndarray],
    molecular_weights: jnp.ndarray,
    diffusion_volume: float,
    accommodation_coeff: float,
    xk: jnp.ndarray,
    cstar_ug_arr: jnp.ndarray,
    Dbk: float = 1.0e-10,
    kc: float = 0.0,
    use_gasside_only: bool = False,
) -> tuple:
    """Batch organic condensation sink for all VBS species at once.

    Vectorized version of calc_organic_condensation_sink that computes
    CS, sinkfrac, Q for n_vbs species simultaneously via broadcasting.

    Args:
        molecular_weights: (n_vbs,) array of MW [g/mol]
        diffusion_volume: Diffusion volume (scalar, shared across species)
        cstar_ug_arr: (n_vbs,) array of C* [µg/m³]
        ... other args same as single-species version

    Returns:
        CS: (n_vbs,) condensation sink per species [s⁻¹]
        sinkfrac: (nbins, n_vbs) per-bin fractions
        Q: (nbins,) quasi-steady-state parameter (shared across species)
    """
    # Gas diffusivity and MFP per species: (n_vbs,)
    Di = calc_gas_diffusivity(temp, pres, molecular_weights, diffusion_volume)
    mfp = calc_mean_free_path(temp, pres, molecular_weights, diffusion_volume)

    # Shared particle properties: (nbins,)
    has_particles = Nk > NEPS_CONDSINK
    Mktot = jnp.sum(Mk, axis=1)
    mp_actual = Mktot / jnp.maximum(Nk, 1e-30)
    density_actual = calc_density(Mk)
    mp_default = 1.4 * xk[:-1]
    density_default = 1500.0
    mp = jnp.where(has_particles, mp_actual, mp_default)
    density = jnp.where(has_particles, density_actual, density_default)
    Dpk = jnp.cbrt(mp / density * (6.0 / PI))
    Rpk = Dpk / 2.0
    safe_Rp = jnp.maximum(Rpk, 1e-30)

    # Knudsen number per species: (nbins, n_vbs)
    Kn = 2.0 * mfp[None, :] / jnp.maximum(Dpk[:, None], 1e-30)

    # Fuchs-Sutugin correction (Zaveri Eq. 14): (nbins, n_vbs)
    alpha = accommodation_coeff
    FC = (0.75 * alpha * (1.0 + Kn)
          / (Kn * (1.0 + Kn) + 0.283 * alpha * Kn + 0.75 * alpha))

    # Gas-side mass transfer coefficient: (nbins, n_vbs)
    kgk = Di[None, :] / safe_Rp[:, None] * FC

    # Particle-side (shared across species): (nbins,)
    kpk_simple = 5.0 * Dbk / safe_Rp
    qk = safe_Rp * jnp.sqrt(jnp.maximum(kc, 0.0) / jnp.maximum(Dbk, 1e-30))
    safe_qk = jnp.maximum(qk, 1e-30)
    coth_qk = 1.0 / jnp.tanh(jnp.maximum(safe_qk, 1e-10))
    Qkk_raw = 3.0 * (safe_qk * coth_qk - 1.0) / safe_qk**2
    Qkk = jnp.where(qk > 1e-6, jnp.minimum(Qkk_raw, 0.99), 1.0)
    kpk_reactive = Dbk / safe_Rp * (safe_qk * coth_qk - 1.0) / (1.0 - Qkk)
    kpk = jnp.where(kc > 0.0, kpk_reactive, kpk_simple)

    # Effective mass transfer coefficient: (nbins, n_vbs)
    if use_gasside_only:
        Keff = safe_Rp[:, None] * kgk
    else:
        cstar_weight = cstar_ug_arr[None, :] / density[:, None] / 1.0e9
        Keff = safe_Rp[:, None] / (
            1.0 / jnp.maximum(kgk, 1e-30)
            + cstar_weight / jnp.maximum(kpk[:, None], 1e-30)
        )

    Keff = jnp.where(Dpk[:, None] > 0.0, Keff, 0.0)

    # CS and sinkfrac per species
    sink_contributions = Dpk[:, None] * Nk[:, None] * Keff  # (nbins, n_vbs)
    CS_sum = jnp.sum(sink_contributions, axis=0)  # (n_vbs,)
    CS = 2.0 * PI * CS_sum / (boxvol * 1e-6)

    sinkfrac = jnp.where(
        CS_sum[None, :] > 1e-30,
        sink_contributions / CS_sum[None, :],
        0.0,
    )

    return CS, sinkfrac, Qkk
