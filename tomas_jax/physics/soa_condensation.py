"""SOA condensation driver for TOMAS-JAX using the VBS framework.

Implements kinetic condensation/evaporation of semi-volatile organic
species with equilibrium correction, following the approach of soacond.f
(Jeff Pierce, September 2010) but with cleaner modular architecture.

For each VBS species j, per timestep:
    1. Compute C*(T) via Clausius-Clapeyron
    2. Compute Kelvin factors (size-dependent vapor pressure correction)
    3. Compute condensation sink CS_j (species-specific diffusivity)
    4. Compute per-bin driving force dp(k) = pamb - psat(k)
    5. Analytical gas depletion toward equilibrium:
       Gc(t) = Gc_eq + (Gc_0 - Gc_eq) × exp(-CS × t)
    6. Distribute mass to bins by dp-weighted sink fraction
    7. Equilibrium correction prevents per-bin overshoot (fodc relaxation)
    8. Update particle and gas concentrations

The species loop is Python-level (unrolled at JAX trace time, 6 iterations
for the standard VBS). MNFIX is NOT called internally — the caller
(make_step or condensation_step) handles it.

References:
    - soacond.f (Jeff Pierce, September 2010)
    - Donahue et al. (2006), Environ. Sci. Technol., 40, 2635–2643
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ..core.config import R_GAS, ICOMP_NODIAG, SRTORG1, SRTORGLAST
from .vbs_config import (
    DEFAULT_VBS_CONFIG, VBSConfig, calc_Cstar_T, cstar_ug_to_Pa,
)
from .kelvin_effect import calc_kelvin_factor, calc_diameters_from_xk
from .vbs_driving_force import (
    calc_ambient_pressure, calc_equilibrium_pressure,
    calc_driving_force, calc_equilibrium_mass,
)
from .condensation_sink import calc_condensation_sink


def soa_condensation_step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                          vbs_config=DEFAULT_VBS_CONFIG):
    """SOA condensation/evaporation step for all VBS species.

    Iterates over VBS species (Python loop, unrolled at trace time)
    applying kinetic condensation with equilibrium correction.

    Args:
        Nk: Number concentration [#/grid cell], shape (nbins,)
        Mk: Mass concentration [kg/grid cell], shape (nbins, icomp)
        Gc: Gas-phase concentrations [kg/grid cell], shape (N_GAS_SPECIES,)
        xk: Bin boundaries [kg], shape (nbins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm³]
        rh: Relative humidity [fraction 0-1]
        alpha: Accommodation coefficient
        dt: Timestep [s]
        vbs_config: VBSConfig NamedTuple with bin properties

    Returns:
        (Nk, Mk, Gc): Updated arrays (Nk unchanged, Mk and Gc modified)
    """
    # Pre-compute particle diameters (shared across VBS species)
    Dpk = calc_diameters_from_xk(xk, Nk, Mk, ICOMP_NODIAG)

    for j in range(vbs_config.n_bins):
        spec_idx = vbs_config.species_indices[j]

        # Temperature-corrected C*
        cstar_ug = calc_Cstar_T(
            vbs_config.cstar_ref_ug[j],
            vbs_config.delta_Hvap_kJ[j],
            temp,
            vbs_config.t_ref,
        )
        cstar_Pa = cstar_ug_to_Pa(cstar_ug, vbs_config.mw[j], temp)

        # Kelvin factors for this species
        Ke = calc_kelvin_factor(
            Dpk, vbs_config.sigma,
            vbs_config.mw[j], vbs_config.rho, temp,
        )

        Nk, Mk, Gc = _condense_one_species(
            Nk, Mk, Gc, xk, temp, pres, boxvol, alpha, dt,
            spec_idx, cstar_Pa, vbs_config.mw[j], vbs_config.sv, Ke,
        )

    return Nk, Mk, Gc


def _condense_one_species(Nk, Mk, Gc, xk, temp, pres, boxvol, alpha, dt,
                          spec_idx, cstar_Pa, mw, sv, Ke):
    """Condense/evaporate one VBS species using kinetic + equilibrium.

    Uses analytical gas depletion toward equilibrium:
        Gc(t) = Gc_eq + (Gc_0 - Gc_eq) × exp(-CS × t)

    Mass distributed to bins proportional to dp-weighted sink fraction.
    Per-bin equilibrium correction prevents overshoot.

    Args:
        spec_idx: TOMAS species index (e.g. 1 for first organic)
        cstar_Pa: Temperature-corrected C* [Pa]
        mw: Molecular weight [g/mol] (JAX scalar)
        sv: Diffusion volume parameter
        Ke: Kelvin correction factors, shape (nbins,)

    Returns:
        (Nk, Mk, Gc): Updated arrays
    """
    mw_kg = mw * 1.0e-3
    boxvol_m3 = boxvol * 1.0e-6

    # --- 1. Condensation sink (species-specific) ---
    CS, sinkfrac = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        molecular_weight=mw, diffusion_volume=sv,
        accommodation_coeff=alpha, xk=xk,
    )

    # --- 2. Driving force per bin ---
    pamb = calc_ambient_pressure(Gc[spec_idx], mw, temp, pres, boxvol)
    Mtot_org = jnp.sum(Mk[:, SRTORG1:SRTORGLAST + 1], axis=1)
    psat = calc_equilibrium_pressure(cstar_Pa, Mk[:, spec_idx], Mtot_org, Ke)
    dp = calc_driving_force(pamb, psat)

    # --- 3. Equilibrium gas concentration ---
    # psat_eff = sink-weighted mean equilibrium pressure
    psat_eff = jnp.sum(sinkfrac * psat)
    Gc_eq = psat_eff * boxvol_m3 * mw_kg / (R_GAS * temp)

    # --- 4. Analytical gas depletion ---
    # mcond = (Gc_0 - Gc_eq) × (1 - exp(-CS × dt))
    # Positive = net condensation, negative = net evaporation
    gas_depl = jnp.where(CS > 1.0e-30,
                         1.0 - jnp.exp(-CS * dt),
                         CS * dt)
    mcond = (Gc[spec_idx] - Gc_eq) * gas_depl

    # --- 5. Distribute mass to bins ---
    # Each bin receives mass proportional to sinkfrac(k) × dp(k)
    dp_weight = sinkfrac * dp
    dp_weight_sum = jnp.sum(dp_weight)

    mass_per_bin = jnp.where(
        jnp.abs(dp_weight_sum) > 1.0e-30,
        dp_weight / dp_weight_sum * mcond,
        0.0,
    )

    # --- 6. Equilibrium correction (soacond.f lines 456-469) ---
    maddEQ = calc_equilibrium_mass(
        pamb, cstar_Pa, Mtot_org, Ke, Mk[:, spec_idx], Nk,
    )
    massEQ_per_bin = maddEQ * jnp.where(Nk > 1.0e-10, Nk, 0.0)

    # Overshoot detection: kinetic exceeds equilibrium in the same direction
    same_sign = jnp.sign(mass_per_bin) == jnp.sign(massEQ_per_bin)
    overshoots = same_sign & (jnp.abs(mass_per_bin) > jnp.abs(massEQ_per_bin))
    has_eq = jnp.abs(massEQ_per_bin) > 1.0e-30

    # Exponential relaxation: fodc = |kinetic/equilibrium| / dt
    # corrected = equilibrium × (1 - exp(-fodc × dt))
    safe_massEQ = jnp.where(has_eq, massEQ_per_bin, 1.0)
    fodc = jnp.abs(mass_per_bin / safe_massEQ) / jnp.maximum(dt, 1.0e-30)
    mass_relaxed = massEQ_per_bin * (1.0 - jnp.exp(-fodc * dt))

    mass_corrected = jnp.where(overshoots & has_eq, mass_relaxed, mass_per_bin)

    # --- 7. Global conservation ---
    total_mass_change = jnp.sum(mass_corrected)

    # Can't evaporate more than exists in particles
    total_species_in_particles = jnp.sum(Mk[:, spec_idx])
    total_mass_change = jnp.maximum(total_mass_change, -total_species_in_particles)

    # Can't condense more than available gas
    total_mass_change = jnp.minimum(total_mass_change,
                                     jnp.maximum(Gc[spec_idx], 0.0))

    # Scale per-bin distribution to match clamped total
    raw_sum = jnp.sum(mass_corrected)
    scale = jnp.where(jnp.abs(raw_sum) > 1.0e-30,
                       total_mass_change / raw_sum, 0.0)
    mass_change = mass_corrected * scale

    # --- 8. Apply updates ---
    Gc = Gc.at[spec_idx].add(-total_mass_change)
    Mk = Mk.at[:, spec_idx].add(mass_change)

    # Floor: no negative particle mass
    Mk = Mk.at[:, spec_idx].set(jnp.maximum(Mk[:, spec_idx], 0.0))

    return Nk, Mk, Gc
