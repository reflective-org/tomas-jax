"""SOA condensation driver for TOMAS-JAX using the VBS framework.

Implements kinetic condensation/evaporation of semi-volatile organic
species with equilibrium correction, following the approach of soacond.f
(Jeff Pierce, September 2010) but with PPM redistribution instead of
Fortran's top-hat (tmcond).

For each VBS species j, per timestep:
    1. Compute C*(T) via Clausius-Clapeyron
    2. Compute Kelvin factors (size-dependent vapor pressure correction)
    3. Compute condensation sink CS_j (species-specific diffusivity)
    4. Compute per-bin driving force dp(k) = pamb - psat(k)
    5. Analytical gas depletion toward equilibrium:
       Gc(t) = Gc_eq + (Gc_0 - Gc_eq) × exp(-CS × t)
    6. Distribute mass to bins by dp-weighted sink fraction
    7. Equilibrium correction prevents per-bin overshoot (fodc relaxation)
    8. Compute tau from per-bin mass addition (Stevens et al. 1996)
    9. PPM transport: redistribute particles across bins (conservative)
   10. Add condensed mass after transport, update gas

The species loop is Python-level (unrolled at JAX trace time, 6 iterations
for the standard VBS). MNFIX is NOT called internally — the caller
(make_step or condensation_step) handles it.

The Fortran soacond.f computes tau and calls tmcond() (top-hat redistribution).
This implementation uses PPM (Piecewise Parabolic Method) instead, giving
smoother, more accurate redistribution. When use_ppm=False, falls back to
direct mass addition (no redistribution).

References:
    - soacond.f (Jeff Pierce, September 2010)
    - Donahue et al. (2006), Environ. Sci. Technol., 40, 2635–2643
    - Colella & Woodward (1984) for PPM advection
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ..core.config import R_GAS, ICOMP, ICOMP_NODIAG, SRTORG1, SRTORGLAST
from .vbs_config import (
    DEFAULT_VBS_CONFIG, VBSConfig, calc_Cstar_T, cstar_ug_to_Pa,
)
from .kelvin_effect import calc_kelvin_factor, calc_diameters_from_xk
from .vbs_driving_force import (
    calc_ambient_pressure, calc_equilibrium_pressure,
    calc_driving_force, calc_equilibrium_mass,
)
from .condensation_sink import calc_organic_condensation_sink
from .condensation_ppm import ppm_condensation_step, compute_wr


def soa_condensation_step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                          vbs_config=DEFAULT_VBS_CONFIG, use_ppm=True):
    """SOA condensation/evaporation step for all VBS species.

    Iterates over VBS species (Python loop, unrolled at trace time)
    applying kinetic condensation with equilibrium correction.

    When use_ppm=True (default), uses PPM redistribution to move particles
    across bin boundaries based on growth/shrinkage, matching Fortran's
    tmcond approach but with smoother PPM advection.

    When use_ppm=False, uses direct mass addition (no redistribution).

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
        use_ppm: If True (default), use PPM redistribution. If False,
                 direct mass addition only.

    Returns:
        (Nk, Mk, Gc): Updated arrays (Nk modified when use_ppm=True)
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
            spec_idx, cstar_Pa, cstar_ug, vbs_config.mw[j], vbs_config.sv,
            Ke, vbs_config.Dbk, vbs_config.kc, use_ppm,
        )

    return Nk, Mk, Gc


def _condense_one_species(Nk, Mk, Gc, xk, temp, pres, boxvol, alpha, dt,
                          spec_idx, cstar_Pa, cstar_ug, mw, sv, Ke,
                          Dbk=1e-10, kc=0.0, use_ppm=True):
    """Condense/evaporate one VBS species (Zaveri et al. 2014).

    Implements the full kinetic gas-particle partitioning framework from
    Zaveri et al. (2014) ACP 14, 5153-5181, including:
    - Gas-phase diffusion + Fuchs-Sutugin correction
    - Particle-phase bulk diffusion (D_b) and reaction (k_c)
    - Two approximations selected by k_c threshold:
        Approx 1 (k_c >= 0.01 s⁻¹): gas-side CS, Q-modified driving force
        Approx 2 (k_c < 0.01 s⁻¹): two-film K_g CS, standard driving force
    - Particle-phase reaction loss: -k_c × M_species × dt

    Then applies PPM redistribution (if use_ppm=True) or direct mass addition.

    Args:
        spec_idx: TOMAS species index (e.g. 1 for first organic)
        cstar_Pa: Temperature-corrected C* [Pa]
        cstar_ug: Temperature-corrected C* [µg/m³]
        mw: Molecular weight [g/mol] (JAX scalar)
        sv: Diffusion volume parameter
        Ke: Kelvin correction factors, shape (nbins,)
        Dbk: Particle-phase diffusion coefficient [m²/s]
        kc: First-order loss rate in particle phase [s⁻¹]
        use_ppm: Use PPM redistribution (Python bool, resolved at trace time)

    Returns:
        (Nk, Mk, Gc): Updated arrays
    """
    mw_kg = mw * 1.0e-3
    boxvol_m3 = boxvol * 1.0e-6

    # --- 1. Organic condensation sink (Zaveri et al. 2014) ---
    # Select approximation based on k_c (Python-level dispatch at trace time):
    #   Approx 1 (kc >= 0.01): gas-side only CS, Q applied to driving force
    #   Approx 2 (kc < 0.01):  two-film K_g CS, standard driving force
    use_approx1 = (kc >= 0.01)
    CS, sinkfrac, Q = calc_organic_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        molecular_weight=mw, diffusion_volume=sv,
        accommodation_coeff=alpha, xk=xk,
        cstar_ug=cstar_ug, Dbk=Dbk, kc=kc,
        use_gasside_only=use_approx1,
    )

    # --- 2. Driving force per bin ---
    pamb = calc_ambient_pressure(Gc[spec_idx], mw, temp, pres, boxvol)
    Mtot_org = jnp.sum(Mk[:, SRTORG1:SRTORGLAST + 1], axis=1)
    psat = calc_equilibrium_pressure(cstar_Pa, Mk[:, spec_idx], Mtot_org, Ke)

    # Zaveri et al. (2014) driving force:
    #   Approx 1 (Eq. 29): dp = pamb - psat/Q  (Q corrects for non-uniform
    #     particle-phase concentration in semisolid particles)
    #   Approx 2 (Eq. 31): dp = pamb - psat    (particle-phase resistance
    #     captured in K_g instead)
    # For kc=0, Q=1 and both are identical.
    if use_approx1:
        safe_Q = jnp.maximum(Q, 1.0e-10)
        psat_eff_bins = psat / safe_Q
    else:
        psat_eff_bins = psat

    dp = calc_driving_force(pamb, psat_eff_bins)

    # --- 3. Equilibrium gas concentration ---
    # psat_eff = sink-weighted mean equilibrium pressure
    psat_eff = jnp.sum(sinkfrac * psat_eff_bins)
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

    # --- 6. Equilibrium correction (soacond.f fodc scheme) ---
    # Use Q-modified equilibrium for Approx 1 (consistent with driving force)
    if use_approx1:
        maddEQ = calc_equilibrium_mass(
            pamb, cstar_Pa, Mtot_org, Ke, Mk[:, spec_idx], Nk, Q=Q,
        )
    else:
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

    # --- 8. PPM redistribution (or direct mass addition) ---
    # Track particle mass before to ensure exact gas-particle conservation
    Mk_spec_before = jnp.sum(Mk[:, spec_idx])

    if use_ppm:
        # Compute tau from per-particle mass addition
        # tau = 1.5 * ((m_wet + madd*WR)^(2/3) - m_wet^(2/3))
        safe_Nk = jnp.maximum(Nk, 1e-30)
        madd_pp = mass_change / safe_Nk       # per-particle mass addition

        mpw = jnp.sum(Mk, axis=1)             # total wet mass per bin
        mpw_pp = mpw / safe_Nk                 # per-particle wet mass

        WR = compute_wr(Mk, ICOMP_NODIAG)

        tdt = 2.0 / 3.0
        inner = mpw_pp + madd_pp * WR
        tau = jnp.where(
            (Nk > 1e-10) & (inner > 0.0),
            1.5 * (jnp.power(inner, tdt) - jnp.power(mpw_pp, tdt)),
            0.0,
        )

        # PPM transport: conservative redistribution of Nk and Mk
        Nk, Mk = ppm_condensation_step(
            Nk, Mk, xk, tau, spec_idx, dt=1.0,
            icomp_nodiag=ICOMP_NODIAG,
        )

    # Add condensed mass to (possibly redistributed) bins
    Mk = Mk.at[:, spec_idx].add(mass_change)

    # Floor: no negative particle mass
    Mk = Mk.at[:, spec_idx].set(jnp.maximum(Mk[:, spec_idx], 0.0))

    # Update gas: only condensation affects gas (not particle-phase reaction)
    # Zaveri Eq. 30/32: dCg/dt = -Σ[condensation rate] (no k_c term)
    actual_particle_change = jnp.sum(Mk[:, spec_idx]) - Mk_spec_before
    Gc = Gc.at[spec_idx].add(-actual_particle_change)

    # --- 9. Particle-phase reaction loss (Zaveri Eq. 19/21: -k_c × Ā_i) ---
    # First-order decay of volatile species in particle phase.
    # Absorbed organic reacts to form nonvolatile products (oligomers,
    # organic salts, etc.). The lost mass remains in the particle as
    # total mass but is no longer volatile (cannot re-evaporate).
    # Note: gas is NOT affected — reaction is internal to particle.
    if kc > 0.0:
        reaction_loss = Mk[:, spec_idx] * (1.0 - jnp.exp(-kc * dt))
        Mk = Mk.at[:, spec_idx].add(-reaction_loss)
        Mk = Mk.at[:, spec_idx].set(jnp.maximum(Mk[:, spec_idx], 0.0))
        # TODO: Track reaction products in a designated nonvolatile organic
        # species. For now, the lost volatile mass is removed from the
        # volatile species but not added elsewhere (total particle mass
        # decreases). To conserve total organic mass, add reaction_loss
        # to a nonvolatile species index.

    return Nk, Mk, Gc
