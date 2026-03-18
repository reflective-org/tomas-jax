"""SOA condensation driver for TOMAS-JAX using the VBS framework.

Three switchable solvers for organic gas-particle partitioning:

  solver='sequential' (default):
    Sequential Python loop over 6 VBS species (Gauss-Seidel ordering).
    Includes adaptive sub-stepping when CS*dt is large.
    Matches existing behavior and Fortran comparison.

  solver='coupled':
    All 6 species solved simultaneously via vectorized operations.
    Fixed-point iteration (n_iter=3) resolves Raoult coupling.
    Adaptive sub-stepping for accuracy at high CS.
    No PPM redistribution — uses direct mass addition + MNFIX.

Select via ``soa_condensation_step(..., solver='coupled')`` or pass
``soa_solver='coupled'`` to ``make_step()``.

References:
    - soacond.f (Jeff Pierce, September 2010)
    - Donahue et al. (2006), Environ. Sci. Technol., 40, 2635-2643
    - Zaveri et al. (2014), ACP 14, 5153-5181
    - Colella & Woodward (1984) for PPM advection
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ..core.config import (
    R_GAS, ICOMP, ICOMP_NODIAG, SRTORG1, SRTORGLAST, MOLAR_MASS_AIR,
)
from .vbs_config import (
    DEFAULT_VBS_CONFIG, VBSConfig, calc_Cstar_T, cstar_ug_to_Pa,
)
from .kelvin_effect import (
    calc_kelvin_factor, calc_kelvin_factor_batch, calc_diameters_from_xk,
)
from .vbs_driving_force import (
    calc_ambient_pressure, calc_equilibrium_pressure,
    calc_driving_force, calc_equilibrium_mass,
)
from .condensation_sink import (
    calc_organic_condensation_sink, calc_organic_condensation_sink_batch,
)
from .condensation_ppm import ppm_condensation_step, compute_wr


# Default max sub-steps for adaptive sub-stepping
MAX_SOA_SUBSTEPS = 10

# Default fixed-point iterations for coupled solver
N_COUPLED_ITER = 3


def soa_condensation_step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                          vbs_config=DEFAULT_VBS_CONFIG, use_ppm=True,
                          solver='sequential',
                          max_soa_substeps=MAX_SOA_SUBSTEPS,
                          n_iter=N_COUPLED_ITER):
    """SOA condensation/evaporation step for all VBS species.

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
        use_ppm: If True (default), use PPM redistribution (sequential only).
        solver: 'sequential' (default) or 'coupled'
        max_soa_substeps: Max adaptive sub-steps (default 10)
        n_iter: Fixed-point iterations for coupled solver (default 3)

    Returns:
        (Nk, Mk, Gc): Updated arrays
    """
    if solver == 'sequential':
        return _soa_sequential(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
            vbs_config, use_ppm, max_soa_substeps,
        )
    elif solver == 'coupled':
        return _soa_coupled(
            Nk, Mk, Gc, xk, temp, pres, boxvol, alpha, dt,
            vbs_config, max_soa_substeps, n_iter,
        )
    else:
        raise ValueError(
            f"Unknown SOA solver '{solver}'. Valid: 'sequential', 'coupled'"
        )


# =========================================================================
# Sequential solver (backward-compatible, with adaptive sub-stepping)
# =========================================================================

def _soa_sequential(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                    vbs_config, use_ppm, max_soa_substeps):
    """Sequential solver: Python loop over VBS species with sub-stepping."""
    Dpk = calc_diameters_from_xk(xk, Nk, Mk, ICOMP_NODIAG)

    for j in range(vbs_config.n_bins):
        spec_idx = vbs_config.species_indices[j]

        cstar_ug = calc_Cstar_T(
            vbs_config.cstar_ref_ug[j],
            vbs_config.delta_Hvap_kJ[j],
            temp,
            vbs_config.t_ref,
        )
        cstar_Pa = cstar_ug_to_Pa(cstar_ug, vbs_config.mw[j], temp)

        Ke = calc_kelvin_factor(
            Dpk, vbs_config.sigma,
            vbs_config.mw[j], vbs_config.rho, temp,
        )

        Nk, Mk, Gc = _condense_one_species(
            Nk, Mk, Gc, xk, temp, pres, boxvol, alpha, dt,
            spec_idx, cstar_Pa, cstar_ug, vbs_config.mw[j], vbs_config.sv,
            Ke, vbs_config.Dbk, vbs_config.kc, use_ppm,
            max_soa_substeps,
        )

    return Nk, Mk, Gc


def _condense_one_species(Nk, Mk, Gc, xk, temp, pres, boxvol, alpha, dt,
                          spec_idx, cstar_Pa, cstar_ug, mw, sv, Ke,
                          Dbk=1e-10, kc=0.0, use_ppm=True,
                          max_soa_substeps=MAX_SOA_SUBSTEPS):
    """Condense/evaporate one VBS species with adaptive sub-stepping.

    CS, sinkfrac, Q, Ke are frozen per species step. The kinetic exchange
    (driving force → gas depletion → mass distribution → eq correction)
    is sub-stepped when exp(-CS*dt) < 0.01 (Fortran soacond.f approach).
    """
    mw_kg = mw * 1.0e-3
    boxvol_m3 = boxvol * 1.0e-6

    # --- 1. Condensation sink (frozen per species step) ---
    use_approx1 = (kc >= 0.01)
    CS, sinkfrac, Q = calc_organic_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        molecular_weight=mw, diffusion_volume=sv,
        accommodation_coeff=alpha, xk=xk,
        cstar_ug=cstar_ug, Dbk=Dbk, kc=kc,
        use_gasside_only=use_approx1,
    )

    # --- 2. Adaptive sub-stepping ---
    # n_sub = ceil(CS * dt / ln(100)), ensures exp(-CS*dt_sub) > 0.01
    n_sub = jnp.where(
        CS > 1.0e-30,
        jnp.ceil(CS * dt / jnp.log(100.0)).astype(jnp.int32),
        1,
    )
    n_sub = jnp.clip(n_sub, 1, max_soa_substeps)
    dt_sub = dt / n_sub

    # --- 3. Sub-stepped kinetic exchange ---
    Mk_spec_before = jnp.sum(Mk[:, spec_idx])

    def _kinetic_body(i, carry):
        Nk_s, Mk_s, Gc_s = carry

        # Driving force
        pamb = calc_ambient_pressure(Gc_s[spec_idx], mw, temp, pres, boxvol)
        Mtot_org = jnp.sum(Mk_s[:, SRTORG1:SRTORGLAST + 1], axis=1)
        psat = calc_equilibrium_pressure(cstar_Pa, Mk_s[:, spec_idx], Mtot_org, Ke)

        if use_approx1:
            psat_eff_bins = psat / jnp.maximum(Q, 1.0e-10)
        else:
            psat_eff_bins = psat

        dp = calc_driving_force(pamb, psat_eff_bins)

        # Equilibrium gas
        psat_eff = jnp.sum(sinkfrac * psat_eff_bins)
        Gc_eq = psat_eff * boxvol_m3 * mw_kg / (R_GAS * temp)

        # Analytical gas depletion
        gas_depl = jnp.where(CS > 1.0e-30,
                             1.0 - jnp.exp(-CS * dt_sub),
                             CS * dt_sub)
        mcond = (Gc_s[spec_idx] - Gc_eq) * gas_depl

        # Distribute mass
        dp_weight = sinkfrac * dp
        dp_weight_sum = jnp.sum(dp_weight)
        mass_per_bin = jnp.where(
            jnp.abs(dp_weight_sum) > 1.0e-30,
            dp_weight / dp_weight_sum * mcond,
            0.0,
        )

        # Equilibrium correction (fodc scheme)
        if use_approx1:
            maddEQ = calc_equilibrium_mass(
                pamb, cstar_Pa, Mtot_org, Ke, Mk_s[:, spec_idx], Nk_s, Q=Q)
        else:
            maddEQ = calc_equilibrium_mass(
                pamb, cstar_Pa, Mtot_org, Ke, Mk_s[:, spec_idx], Nk_s)
        massEQ_per_bin = maddEQ * jnp.where(Nk_s > 1.0e-10, Nk_s, 0.0)

        same_sign = jnp.sign(mass_per_bin) == jnp.sign(massEQ_per_bin)
        overshoots = same_sign & (jnp.abs(mass_per_bin) > jnp.abs(massEQ_per_bin))
        has_eq = jnp.abs(massEQ_per_bin) > 1.0e-30
        safe_massEQ = jnp.where(has_eq, massEQ_per_bin, 1.0)
        fodc = jnp.abs(mass_per_bin / safe_massEQ) / jnp.maximum(dt_sub, 1.0e-30)
        mass_relaxed = massEQ_per_bin * (1.0 - jnp.exp(-fodc * dt_sub))
        mass_corrected = jnp.where(overshoots & has_eq, mass_relaxed, mass_per_bin)

        # Conservation clamps
        total_mass_change = jnp.sum(mass_corrected)
        total_mass_change = jnp.maximum(total_mass_change,
                                         -jnp.sum(Mk_s[:, spec_idx]))
        total_mass_change = jnp.minimum(total_mass_change,
                                         jnp.maximum(Gc_s[spec_idx], 0.0))
        raw_sum = jnp.sum(mass_corrected)
        scale = jnp.where(jnp.abs(raw_sum) > 1.0e-30,
                           total_mass_change / raw_sum, 0.0)
        mass_change = mass_corrected * scale

        # PPM redistribution (within each sub-step)
        Mk_sub_before = jnp.sum(Mk_s[:, spec_idx])

        if use_ppm:
            safe_Nk = jnp.maximum(Nk_s, 1e-30)
            madd_pp = mass_change / safe_Nk
            mpw = jnp.sum(Mk_s, axis=1)
            mpw_pp = mpw / safe_Nk
            WR = compute_wr(Mk_s, ICOMP_NODIAG)
            tdt = 2.0 / 3.0
            inner = mpw_pp + madd_pp * WR
            tau = jnp.where(
                (Nk_s > 1e-10) & (inner > 0.0),
                1.5 * (jnp.power(inner, tdt) - jnp.power(mpw_pp, tdt)),
                0.0,
            )
            Nk_s, Mk_s = ppm_condensation_step(
                Nk_s, Mk_s, xk, tau, spec_idx, dt=1.0,
                icomp_nodiag=ICOMP_NODIAG,
            )

        # Add mass and update gas
        Mk_s = Mk_s.at[:, spec_idx].add(mass_change)
        Mk_s = Mk_s.at[:, spec_idx].set(jnp.maximum(Mk_s[:, spec_idx], 0.0))
        actual_change = jnp.sum(Mk_s[:, spec_idx]) - Mk_sub_before
        Gc_s = Gc_s.at[spec_idx].add(-actual_change)

        return (Nk_s, Mk_s, Gc_s)

    Nk, Mk, Gc = jax.lax.fori_loop(0, n_sub, _kinetic_body, (Nk, Mk, Gc))

    # --- 4. Particle-phase reaction loss ---
    if kc > 0.0:
        reaction_loss = Mk[:, spec_idx] * (1.0 - jnp.exp(-kc * dt))
        Mk = Mk.at[:, spec_idx].add(-reaction_loss)
        Mk = Mk.at[:, spec_idx].set(jnp.maximum(Mk[:, spec_idx], 0.0))

    return Nk, Mk, Gc


# =========================================================================
# Coupled solver (vectorized, all species simultaneously)
# =========================================================================

def _soa_coupled(Nk, Mk, Gc, xk, temp, pres, boxvol, alpha, dt,
                 vbs_config, max_soa_substeps, n_iter):
    """Coupled solver: all VBS species simultaneously with Raoult iteration.

    Uses vectorized batch functions for CS, Kelvin, and driving force.
    Fixed-point iteration resolves the Raoult coupling (Mtot_org denominator).
    Adaptive sub-stepping when max(CS)*dt is large.
    No PPM redistribution — direct mass addition.
    """
    n_vbs = vbs_config.n_bins
    s0 = SRTORG1
    s1 = SRTORG1 + n_vbs

    # --- Pre-compute (frozen per operator-split step) ---
    Dpk = calc_diameters_from_xk(xk, Nk, Mk, ICOMP_NODIAG)

    # Temperature-corrected C* for all species: (n_vbs,)
    cstar_ug = calc_Cstar_T(
        vbs_config.cstar_ref_ug, vbs_config.delta_Hvap_kJ,
        temp, vbs_config.t_ref,
    )
    cstar_Pa = cstar_ug_to_Pa(cstar_ug, vbs_config.mw, temp)

    # Kelvin factors: (nbins, n_vbs)
    Ke = calc_kelvin_factor_batch(
        Dpk, vbs_config.sigma, vbs_config.mw, vbs_config.rho, temp,
    )

    # Condensation sink: CS(n_vbs,), sinkfrac(nbins, n_vbs), Q(nbins,)
    use_approx1 = (vbs_config.kc >= 0.01)
    CS, sinkfrac, Q = calc_organic_condensation_sink_batch(
        Nk, Mk, temp, pres, boxvol,
        molecular_weights=vbs_config.mw,
        diffusion_volume=vbs_config.sv,
        accommodation_coeff=alpha, xk=xk,
        cstar_ug_arr=cstar_ug,
        Dbk=vbs_config.Dbk, kc=vbs_config.kc,
        use_gasside_only=use_approx1,
    )

    # --- Adaptive sub-stepping ---
    max_CS = jnp.max(CS)
    n_sub = jnp.where(
        max_CS > 1.0e-30,
        jnp.ceil(max_CS * dt / jnp.log(100.0)).astype(jnp.int32),
        1,
    )
    n_sub = jnp.clip(n_sub, 1, max_soa_substeps)
    dt_sub = dt / n_sub

    # Pre-computed scalars
    mw_kg = vbs_config.mw * 1.0e-3  # (n_vbs,)
    boxvol_m3 = boxvol * 1.0e-6
    boxmass = pres * boxvol_m3 * MOLAR_MASS_AIR / (R_GAS * temp)
    n_air = boxmass / MOLAR_MASS_AIR

    # Q correction for driving force
    if use_approx1:
        Q_corr = jnp.maximum(Q, 1.0e-10)  # (nbins,)
    # else: no correction needed (handled below)

    def _substep_body(i, carry):
        Nk_s, Mk_s, Gc_s = carry

        def _iter_body(j, Mk_trial):
            # Raoult coupling: Mtot_org from tentative Mk
            Mtot_org = jnp.sum(Mk_trial[:, SRTORG1:SRTORGLAST + 1], axis=1)
            safe_Mtot = jnp.maximum(Mtot_org, 1.0e-30)

            # pamb from Gc_s (fixed within sub-step): (n_vbs,)
            Gc_spec = Gc_s[s0:s1]
            n_gas = Gc_spec / mw_kg
            pamb = jnp.where(n_air > 1.0e-30, (n_gas / n_air) * pres, 0.0)
            pamb = jnp.maximum(pamb, 0.0)

            # psat from tentative Mk: (nbins, n_vbs)
            Mk_spec = Mk_trial[:, s0:s1]
            psat = jnp.where(
                Mtot_org[:, None] > 1.0e-30,
                cstar_Pa[None, :] * (Mk_spec / safe_Mtot[:, None]) * Ke,
                0.0,
            )

            # Effective psat (with Q correction for Approx 1)
            if use_approx1:
                psat_eff_bins = psat / Q_corr[:, None]
            else:
                psat_eff_bins = psat

            # Driving force: (nbins, n_vbs)
            dp = pamb[None, :] - psat_eff_bins

            # Sink-weighted mean psat → equilibrium gas: (n_vbs,)
            psat_eff = jnp.sum(sinkfrac * psat_eff_bins, axis=0)
            Gc_eq = psat_eff * boxvol_m3 * mw_kg / (R_GAS * temp)

            # Analytical gas depletion: (n_vbs,)
            gas_depl = jnp.where(CS > 1.0e-30,
                                  1.0 - jnp.exp(-CS * dt_sub),
                                  CS * dt_sub)
            mcond = (Gc_spec - Gc_eq) * gas_depl

            # Distribute to bins: (nbins, n_vbs)
            dp_weight = sinkfrac * dp
            dp_weight_sum = jnp.sum(dp_weight, axis=0)  # (n_vbs,)
            mass_per_bin = jnp.where(
                jnp.abs(dp_weight_sum)[None, :] > 1.0e-30,
                dp_weight / dp_weight_sum[None, :] * mcond[None, :],
                0.0,
            )

            # Equilibrium correction (vectorized fodc): (nbins, n_vbs)
            if use_approx1:
                Q_factor = Q_corr[:, None]
            else:
                Q_factor = 1.0
            masseqm = jnp.where(
                (Mtot_org[:, None] > 1.0e-30) & (cstar_Pa[None, :] > 1.0e-30),
                pamb[None, :] * safe_Mtot[:, None] * Q_factor
                / (cstar_Pa[None, :] * Ke),
                0.0,
            )
            Nk_safe = jnp.maximum(Nk_s, 1.0e-30)
            maddEQ_pp = jnp.where(
                Nk_s[:, None] > 1.0e-10,
                (masseqm - Mk_spec) / Nk_safe[:, None],
                0.0,
            )
            massEQ_per_bin = maddEQ_pp * jnp.where(
                Nk_s[:, None] > 1.0e-10, Nk_s[:, None], 0.0,
            )

            same_sign = jnp.sign(mass_per_bin) == jnp.sign(massEQ_per_bin)
            overshoots = same_sign & (
                jnp.abs(mass_per_bin) > jnp.abs(massEQ_per_bin))
            has_eq = jnp.abs(massEQ_per_bin) > 1.0e-30
            safe_massEQ = jnp.where(has_eq, massEQ_per_bin, 1.0)
            fodc = (jnp.abs(mass_per_bin / safe_massEQ)
                    / jnp.maximum(dt_sub, 1.0e-30))
            mass_relaxed = massEQ_per_bin * (1.0 - jnp.exp(-fodc * dt_sub))
            mass_corrected = jnp.where(
                overshoots & has_eq, mass_relaxed, mass_per_bin)

            # Conservation clamps per species: (n_vbs,)
            total_change = jnp.sum(mass_corrected, axis=0)
            Mk_start_spec = Mk_s[:, s0:s1]
            total_in_particles = jnp.sum(Mk_start_spec, axis=0)
            total_change = jnp.maximum(total_change, -total_in_particles)
            total_change = jnp.minimum(total_change,
                                        jnp.maximum(Gc_spec, 0.0))

            raw_sum = jnp.sum(mass_corrected, axis=0)
            scale = jnp.where(jnp.abs(raw_sum) > 1.0e-30,
                               total_change / raw_sum, 0.0)
            mass_change = mass_corrected * scale[None, :]  # (nbins, n_vbs)

            # Tentative Mk update for next iteration
            Mk_new = Mk_s.at[:, s0:s1].set(
                Mk_start_spec + mass_change
            )
            Mk_new = Mk_new.at[:, s0:s1].set(
                jnp.maximum(Mk_new[:, s0:s1], 0.0)
            )
            return Mk_new

        # Run fixed-point iterations
        Mk_converged = jax.lax.fori_loop(0, n_iter, _iter_body, Mk_s)

        # Final mass change → update Gc
        mass_change_final = Mk_converged[:, s0:s1] - Mk_s[:, s0:s1]
        gas_change = jnp.sum(mass_change_final, axis=0)  # (n_vbs,)
        Gc_new = Gc_s.at[s0:s1].add(-gas_change)

        return (Nk_s, Mk_converged, Gc_new)

    # Run sub-steps
    Nk, Mk, Gc = jax.lax.fori_loop(0, n_sub, _substep_body, (Nk, Mk, Gc))

    # Particle-phase reaction loss (shared kc for all species)
    if vbs_config.kc > 0.0:
        for j in range(n_vbs):
            idx = vbs_config.species_indices[j]
            reaction_loss = Mk[:, idx] * (1.0 - jnp.exp(-vbs_config.kc * dt))
            Mk = Mk.at[:, idx].add(-reaction_loss)
            Mk = Mk.at[:, idx].set(jnp.maximum(Mk[:, idx], 0.0))

    return Nk, Mk, Gc
