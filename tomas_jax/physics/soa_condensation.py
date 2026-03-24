"""SOA condensation driver for TOMAS-JAX using the VBS framework.

Three switchable solvers for organic gas-particle partitioning:

  solver='sequential' (default):
    Sequential Python loop over 6 VBS species (Gauss-Seidel ordering).
    Faithfully matches Fortran soacond.f algorithm:
    - CS recomputed at each adaptive sub-step (not frozen)
    - m^(2/3) geometric growth via atau formula
    - fodc equilibrium correction (Fortran-matching)
    - Per-bin evaporation clamping
    - Switchable redistribution: TFL (top-hat), PPM, or direct mass addition

  solver='coupled':
    All 6 species solved simultaneously via vectorized operations.
    Fixed-point iteration (n_iter=3) resolves Raoult coupling.
    Adaptive sub-stepping for accuracy at high CS.
    Uses direct mass addition + MNFIX (redistribution not applicable).

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
    PI, R_GAS, ICOMP, ICOMP_NODIAG, SRTORG1, SRTORGLAST, MOLAR_MASS_AIR,
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
from .condensation_tfl_jax import tmcond_jax
from .density import calc_density
from .water_equilibrium import calc_equilibrium_water
from ..core.mnfix_jax import mnfix_jax


# Default max sub-steps for adaptive sub-stepping
MAX_SOA_SUBSTEPS = 10

# Default fixed-point iterations for coupled solver
N_COUPLED_ITER = 3

# Minimum Nk to participate in SOA condensation.
# Matches Fortran soacond.f parameter(Neps=1.0d-5).
# Ghost bins (Nk=1e-3 from preprocessing) have negligible atau
# because mc^(2/3) dominates the growth integral, so including them
# does not cause mass accumulation issues.
NEPS_SOA = 1.0e-5

# Fortran soacond.f line 138: parameter(R=8.314).
# Even a 0.006% per-step bias from the exact R_GAS = 8.314462618
# compounds catastrophically through MNFIX bin redistribution over
# 8640 steps (24h), creating 100% errors in the size distribution.
# The sequential solver uses this truncated value for Fortran matching.
_R_SOA = 8.314


def soa_condensation_step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                          vbs_config=DEFAULT_VBS_CONFIG, redistribution='tfl',
                          solver='sequential',
                          max_soa_substeps=MAX_SOA_SUBSTEPS,
                          n_iter=N_COUPLED_ITER,
                          **kwargs):
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
        redistribution: Bin redistribution method for sequential solver:
            'tfl' (default) — TFL Lagrangian remapping (Fortran-matching)
            'ppm' — PPM advection (smoother, no TFL artifacts)
            'direct' — Direct mass addition only (no redistribution)
        solver: 'sequential' (default) or 'coupled'
        max_soa_substeps: Max adaptive sub-steps (default 10)
        n_iter: Fixed-point iterations for coupled solver (default 3)

    Returns:
        (Nk, Mk, Gc): Updated arrays
    """
    # Backward compatibility: old use_ppm= kwarg
    if 'use_ppm' in kwargs:
        redistribution = 'tfl' if kwargs['use_ppm'] else 'direct'

    _valid_redist = {'tfl', 'ppm', 'direct'}
    if redistribution not in _valid_redist:
        raise ValueError(
            f"Unknown redistribution '{redistribution}'. "
            f"Valid: {sorted(_valid_redist)}")

    if solver == 'sequential':
        return _soa_sequential(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
            vbs_config, redistribution, max_soa_substeps,
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
# Sequential solver — Fortran-faithful soacond.f algorithm
# =========================================================================

def _soa_sequential(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                    vbs_config, redistribution, max_soa_substeps):
    """Sequential solver: Python loop over VBS species with sub-stepping.

    Matches Fortran soacond.f: CS recomputed each sub-step, atau formula
    for m^(2/3) geometric growth, fodc equilibrium correction.

    Density is computed from the start-of-timestep Mk (matching Fortran's
    use of the COMMON block Mk rather than the working copy Mkf).
    """
    # Save initial Mk for density computation (Fortran soacond.f uses
    # the COMMON block Mk, not the working copy Mkf — see lines 221-227).
    Mk_init = Mk

    for j in range(vbs_config.n_bins):
        spec_idx = vbs_config.species_indices[j]

        cstar_ug = calc_Cstar_T(
            vbs_config.cstar_ref_ug[j],
            vbs_config.delta_Hvap_kJ[j],
            temp,
            vbs_config.t_ref,
            r_gas=_R_SOA,
        )
        cstar_Pa = cstar_ug_to_Pa(cstar_ug, vbs_config.mw[j], temp,
                                   r_gas=_R_SOA)

        Nk, Mk, Gc = _condense_one_species(
            Nk, Mk, Gc, xk, temp, pres, boxvol, alpha, dt,
            spec_idx, cstar_Pa, cstar_ug,
            vbs_config.mw[j], vbs_config.sv,
            vbs_config.sigma, vbs_config.rho,
            vbs_config.Dbk, vbs_config.kc, redistribution,
            max_soa_substeps, Mk_init,
        )

        # Update water equilibrium after each species (Fortran soacond.f
        # calls ezwatereqm(Mkf) after each species' tmcond + gas update).
        # Water uptake at RH>0 makes particles larger, increasing CS and
        # growth rates for subsequent species.
        Mk = calc_equilibrium_water(Mk, rh)

    return Nk, Mk, Gc


def _condense_one_species(Nk, Mk, Gc, xk, temp, pres, boxvol, alpha, dt,
                          spec_idx, cstar_Pa, cstar_ug, mw, sv,
                          sigma, rho_org,
                          Dbk=1e-10, kc=0.0, redistribution='tfl',
                          max_soa_substeps=MAX_SOA_SUBSTEPS,
                          Mk_init=None):
    """Condense/evaporate one VBS species — Fortran soacond.f algorithm.

    Faithfully matches Fortran:
    1. CS and all quantities recomputed at each adaptive sub-step
    2. atau formula for m^(2/3) geometric growth integral
    3. madd from atau inversion: ((atau/1.5 + mc^tdt)^(1/tdt) - mc) / WR
    4. fodc equilibrium correction (always applied, not just overshoot)
    5. atau as tau for bin redistribution (TFL or PPM)
    6. Per-bin evaporation clamping

    Args:
        spec_idx: TOMAS species index for this VBS bin
        cstar_Pa: Temperature-corrected C* [Pa]
        cstar_ug: Temperature-corrected C* [µg/m³]
        mw: Molecular weight [g/mol]
        sv: Diffusion volume for FSG
        sigma: Surface tension [N/m]
        rho_org: Organic density [kg/m³]
        Dbk: Particle-phase diffusion coefficient [m²/s]
        kc: First-order particle-phase loss rate [s⁻¹]
        redistribution: 'tfl' (TFL remapping), 'ppm' (PPM advection),
                        or 'direct' (mass addition only)
        max_soa_substeps: Maximum number of adaptive sub-steps
        Mk_init: Initial Mk for density computation (Fortran COMMON block).
                 If None, uses working copy Mk_s (non-Fortran behavior).
    """
    # Use initial Mk for density if provided (Fortran soacond.f lines 221-227)
    if Mk_init is None:
        Mk_init = Mk

    # Precompute density from initial Mk — constant across all substeps.
    # Fortran soacond.f uses COMMON block Mk for density computation
    # (lines 221-227), NOT the working copy Mkf.
    density_init = calc_density(Mk_init)

    mw_kg = mw * 1.0e-3
    boxvol_m3 = boxvol * 1.0e-6
    tdt = 2.0 / 3.0
    use_approx1 = (kc >= 0.01)

    # Fortran's tj = 2*pi*MW_kg/(R*T)  (soacond.f line 191)
    tj = 2.0 * PI * mw_kg / (_R_SOA * temp)

    def _substep_body(i, carry):
        Nk_s, Mk_s, Gc_s, time_elapsed = carry

        remaining = dt - time_elapsed
        # Skip if no time remaining (JAX can't early-exit, so use masking)
        active = remaining > 1.0e-10

        # --- 0. MNFIX at start of each substep (Fortran soacond.f line 201) ---
        Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG)

        # --- 1. Recompute CS from current state (Fortran goto 10) ---
        # Use Fortran soacond.f Neps=1e-5 (not getCondSink's 1e10)
        CS, sinkfrac, Q = calc_organic_condensation_sink(
            Nk_s, Mk_s, temp, pres, boxvol,
            molecular_weight=mw, diffusion_volume=sv,
            accommodation_coeff=alpha, xk=xk,
            cstar_ug=cstar_ug, Dbk=Dbk, kc=kc,
            use_gasside_only=use_approx1,
            density_override=density_init, neps=1.0e-5,
            r_gas=_R_SOA,
        )

        # --- 2. Adaptive timestep (soacond.f lines 392-405) ---
        # Fortran: if exp(-sK*cdt) < 0.01, set cdt = ln(100)/(2*sK).
        # The factor of 2 is a safety margin: targets exp(-sK*cdt)=0.1
        # (gas drops to 10% per substep, not 1%).
        cdt = jnp.where(
            CS > 1.0e-30,
            jnp.minimum(remaining, jnp.log(100.0) / (2.0 * CS)),
            remaining,
        )
        cdt = jnp.minimum(cdt, 900.0)
        cdt = jnp.maximum(cdt, 0.0)

        # --- 3. Per-bin quantities ---
        # WR = wet/dry mass ratio (soacond.f lines 211-218)
        Mktot = jnp.sum(Mk_s, axis=1)
        Mkdry = jnp.sum(Mk_s[:, :ICOMP_NODIAG], axis=1)
        WR = jnp.where(Mkdry > 1.0e-30, Mktot / Mkdry, 1.0)

        # mc = total wet mass per particle (soacond.f line 229)
        safe_Nk = jnp.maximum(Nk_s, 1.0e-30)
        mc = Mktot / safe_Nk

        # Per-bin density from initial Mk (Fortran COMMON block behavior).
        density = density_init

        # Dpk: compute directly matching Fortran soacond.f lines 229-237.
        # Fortran uses Neps=1e-5 for property thresholds (NOT 1e10 as in
        # getCondSink for H2SO4).  calc_diameters_from_xk uses 1e10
        # which wrongly gives defaults for most bins in low-N scenarios.
        NEPS_FORT = 1.0e-5
        has_part = Nk_s > NEPS_FORT
        mp_real = Mktot / safe_Nk
        mp_def = 1.4 * xk[:-1]
        rho_def = 1500.0
        mp_use = jnp.where(has_part, mp_real, mp_def)
        rho_use = jnp.where(has_part, density, rho_def)
        # Fortran soacond.f line 238: Dpk = ((mp/density)*(6/pi))**(0.333)
        # Uses 0.333 (truncated), not 1/3 (exact). This ~1.7% difference
        # in Dpk creates a ~1.3% per-step atau bias that compounds over
        # hundreds of timesteps, causing catastrophic bimodal artifacts.
        # Match Fortran exactly to prevent divergence.
        Dpk = jnp.power(mp_use / rho_use * (6.0 / PI), 0.333)

        # Kelvin factor from current Dpk
        Ke = calc_kelvin_factor(Dpk, sigma, mw, rho_org, temp,
                                r_gas=_R_SOA)

        # --- 4. Driving force per bin (soacond.f lines 329-358) ---
        pamb = calc_ambient_pressure(Gc_s[spec_idx], mw, temp, pres, boxvol,
                                     r_gas=_R_SOA)
        Mtot_org = jnp.sum(Mk_s[:, SRTORG1:SRTORGLAST + 1], axis=1)
        psat = calc_equilibrium_pressure(
            cstar_Pa, Mk_s[:, spec_idx], Mtot_org, Ke)

        psat_eff = jnp.where(use_approx1,
                             psat / jnp.maximum(Q, 1e-10), psat)
        dp = pamb - psat_eff

        # --- 5. Equilibrium mass per particle (soacond.f lines 368-382) ---
        masseqm = jnp.where(
            (Mtot_org > 1.0e-30) & (cstar_Pa > 1.0e-30),
            pamb * Mtot_org / (cstar_Pa * Ke),
            0.0,
        )
        maddEQ = jnp.where(
            Nk_s > NEPS_SOA,
            (masseqm - Mk_s[:, spec_idx]) / safe_Nk,
            0.0,
        )

        # --- 6. atau formula (soacond.f lines 419-426) ---
        # atau(k) = tj * tk(k) * dp(k) / sK * (1-exp(-sK*cdt)) * WR(k)
        # where tk(k) = (6/pi/density)^(1/3) * Kgkk(k)
        # and sK = 2*pi * sum(Dpk*Nk*Kgkk) / boxvol_m3
        #
        # Back-compute Kgkk from sinkfrac:
        #   sinkfrac(k) = Dpk(k)*Nk(k)*Kgkk(k) / sum(Dpk*Nk*Kgkk)
        #   sum(Dpk*Nk*Kgkk) = CS * boxvol_m3 / (2*pi)
        #   Kgkk(k) = sinkfrac(k) * CS * boxvol_m3 / (2*pi * Dpk(k) * Nk(k))
        safe_Dpk = jnp.maximum(Dpk, 1.0e-30)
        Kgkk = jnp.where(
            (Dpk > 0.0) & (Nk_s > 0.0),
            sinkfrac * CS * boxvol_m3 / (2.0 * PI * safe_Dpk * safe_Nk),
            0.0,
        )

        # tk(k) = (6/pi/density)^(1/3) * Kgkk(k)
        safe_density = jnp.maximum(density, 1.0)
        geo_factor = jnp.cbrt(6.0 / (PI * safe_density))
        tk = geo_factor * Kgkk

        # gas_depl_over_cs = (1-exp(-CS*cdt)) / CS
        gas_depl_over_cs = jnp.where(
            CS > 1.0e-30,
            (1.0 - jnp.exp(-CS * cdt)) / CS,
            cdt,
        )

        # atau(k) = tj * tk(k) * dp(k) * gas_depl_over_cs * WR(k)
        atau = jnp.where(
            Nk_s > NEPS_SOA,
            tj * tk * dp * gas_depl_over_cs * WR,
            0.0,
        )

        # --- 7. madd from atau (m^(2/3) inversion, soacond.f lines 436-441) ---
        # madd(k) = ((atau(k)/1.5 + mc^tdt)^(1/tdt) - mc) / WR(k)
        mc_tdt = jnp.power(jnp.maximum(mc, 0.0), tdt)
        inner = atau / 1.5 + mc_tdt
        madd = jnp.where(
            (inner >= 0.0) & (Nk_s > NEPS_SOA),
            (jnp.power(jnp.maximum(inner, 0.0), 1.0 / tdt) - mc)
            / jnp.maximum(WR, 1.0e-30),
            maddEQ,
        )

        # --- 8. fodc equilibrium correction (soacond.f lines 466-478) ---
        # Fortran applies fodc when madd/maddEQ > 1e-30 (same sign, ratio > 0)
        # fodc = madd / (cdt * maddEQ) = 1/(cdt * maddEQ/madd)
        # madd_corrected = maddEQ * (1 - exp(-fodc * cdt))
        safe_maddEQ = jnp.where(
            jnp.abs(maddEQ) > 1.0e-30, maddEQ, 1.0)
        madd_ratio = madd / safe_maddEQ
        has_fodc = (madd_ratio > 1.0e-30) & (jnp.abs(maddEQ) > 1.0e-30)

        fodc = jnp.where(
            has_fodc,
            jnp.abs(madd / safe_maddEQ) / jnp.maximum(cdt, 1.0e-30),
            0.0,
        )
        madd_relaxed = maddEQ * (1.0 - jnp.exp(-fodc * cdt))
        madd_corrected = jnp.where(has_fodc, madd_relaxed, madd)

        # Recompute atau from corrected madd (soacond.f lines 475-476)
        # atau_new = 1.5 * ((mc + madd*WR)^(2/3) - mc^(2/3))
        mc_plus_madd = jnp.maximum(mc + madd_corrected * WR, 0.0)
        atau_corrected = jnp.where(
            has_fodc,
            1.5 * (jnp.power(mc_plus_madd, tdt) - mc_tdt),
            atau,
        )

        # Zero out bins with Nk < Neps
        madd_final = jnp.where(Nk_s > NEPS_SOA, madd_corrected, 0.0)
        atau_final = jnp.where(Nk_s > NEPS_SOA, atau_corrected, 0.0)

        # Per-bin evaporation clamp: never remove more than species mass
        madd_final = jnp.maximum(
            madd_final, -Mk_s[:, spec_idx] / safe_Nk)

        # Mass change per bin = madd * Nk
        mass_change = madd_final * Nk_s

        # --- 9. Apply mass change + redistribution ---
        Mk_sub_before = jnp.sum(Mk_s[:, spec_idx])

        if redistribution == 'tfl':
            # TFL redistribution — matches Fortran soacond.f which calls
            # tmcond(atau, xk, Mk, Nk, j, zeros).  tmcond is Lagrangian
            # (top-hat remapping) which preserves sharp features.
            Nk_s_new, Mk_s_new = tmcond_jax(
                atau_final, xk, Mk_s, Nk_s, spec_idx, ICOMP_NODIAG,
            )
        elif redistribution == 'ppm':
            # PPM redistribution — smoother than TFL, avoids bin-boundary
            # oscillation artifacts.  Pattern matches ezcond_ppm_jax:
            # 1) PPM transport moves particles to correct bins
            # 2) Condensed mass added after transport
            Nk_s_new, Mk_s_new = ppm_condensation_step(
                Nk_s, Mk_s, xk, atau_final, spec_idx,
                dt=1.0, icomp_nodiag=ICOMP_NODIAG,
            )
            Mk_s_new = Mk_s_new.at[:, spec_idx].add(mass_change)
            Mk_s_new = Mk_s_new.at[:, spec_idx].set(
                jnp.maximum(Mk_s_new[:, spec_idx], 0.0))
        else:  # 'direct'
            Nk_s_new = Nk_s
            Mk_s_new = Mk_s.at[:, spec_idx].add(mass_change)
            Mk_s_new = Mk_s_new.at[:, spec_idx].set(
                jnp.maximum(Mk_s_new[:, spec_idx], 0.0))

        # Update gas from actual mass change
        actual_change = jnp.sum(Mk_s_new[:, spec_idx]) - Mk_sub_before
        Gc_s_new = Gc_s.at[spec_idx].add(-actual_change)

        # Advance time
        new_time = time_elapsed + cdt

        # Apply updates only if active (remaining > 0)
        Nk_out = jnp.where(active, Nk_s_new, Nk_s)
        Mk_out = jnp.where(active, Mk_s_new, Mk_s)
        Gc_out = jnp.where(active, Gc_s_new, Gc_s)
        time_out = jnp.where(active, new_time, time_elapsed)

        return (Nk_out, Mk_out, Gc_out, time_out)

    Nk, Mk, Gc, _ = jax.lax.fori_loop(
        0, max_soa_substeps, _substep_body,
        (Nk, Mk, Gc, jnp.float64(0.0)),
    )

    # Particle-phase reaction loss
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

    Uses analytical gas depletion + sinkfrac-weighted bin distribution:
    1. CS/Ke/Dpk recomputed at each adaptive sub-step (not frozen)
    2. Analytical gas depletion: Gc(t) = Gc_eq + (Gc_0 - Gc_eq)*exp(-CS*t)
    3. Sinkfrac-weighted dp distribution to bins
    4. fodc equilibrium correction (always applied when ratio > 1e-30)
    5. Per-bin evaporation clamping
    6. Fixed-point iteration (n_iter) resolves Raoult coupling (Mtot_org)

    Note: does NOT use atau m^(2/3) inversion (that requires sequential
    Gauss-Seidel ordering). Uses linear distribution with proper fodc
    relaxation instead.
    """
    n_vbs = vbs_config.n_bins
    s0 = SRTORG1
    s1 = SRTORG1 + n_vbs

    # Temperature-corrected C* for all species: (n_vbs,)
    cstar_ug = calc_Cstar_T(
        vbs_config.cstar_ref_ug, vbs_config.delta_Hvap_kJ,
        temp, vbs_config.t_ref,
    )
    cstar_Pa = cstar_ug_to_Pa(cstar_ug, vbs_config.mw, temp)

    use_approx1 = (vbs_config.kc >= 0.01)
    mw_kg = vbs_config.mw * 1.0e-3  # (n_vbs,)
    boxvol_m3 = boxvol * 1.0e-6

    # These scalars don't change with particle state
    boxmass = pres * boxvol_m3 * MOLAR_MASS_AIR / (R_GAS * temp)
    n_air = boxmass / MOLAR_MASS_AIR

    def _substep_body(i, carry):
        Nk_s, Mk_s, Gc_s, time_elapsed = carry

        remaining = dt - time_elapsed
        active = remaining > 1.0e-10

        # --- 1. Recompute CS, Ke, Dpk from current state ---
        Dpk = calc_diameters_from_xk(xk, Nk_s, Mk_s, ICOMP_NODIAG)
        Ke = calc_kelvin_factor_batch(
            Dpk, vbs_config.sigma, vbs_config.mw, vbs_config.rho, temp,
        )
        CS, sinkfrac, Q = calc_organic_condensation_sink_batch(
            Nk_s, Mk_s, temp, pres, boxvol,
            molecular_weights=vbs_config.mw,
            diffusion_volume=vbs_config.sv,
            accommodation_coeff=alpha, xk=xk,
            cstar_ug_arr=cstar_ug,
            Dbk=vbs_config.Dbk, kc=vbs_config.kc,
            use_gasside_only=use_approx1,
        )

        # --- 2. Adaptive timestep (match Fortran factor-of-2 safety) ---
        max_CS = jnp.max(CS)
        cdt = jnp.where(
            max_CS > 1.0e-30,
            jnp.minimum(remaining, jnp.log(100.0) / (2.0 * max_CS)),
            remaining,
        )
        cdt = jnp.clip(cdt, 0.0, 900.0)

        safe_Nk = jnp.maximum(Nk_s, 1.0e-30)

        # Q correction factor: (nbins,)
        if use_approx1:
            Q_corr = jnp.maximum(Q, 1.0e-10)
        else:
            Q_corr = jnp.ones_like(Q)

        def _iter_body(j, Mk_trial):
            # Raoult coupling: Mtot_org from tentative Mk
            Mtot_org = jnp.sum(
                Mk_trial[:, SRTORG1:SRTORGLAST + 1], axis=1)
            safe_Mtot = jnp.maximum(Mtot_org, 1.0e-30)

            # pamb from Gc_s (fixed within sub-step): (n_vbs,)
            Gc_spec = Gc_s[s0:s1]
            n_gas = Gc_spec / mw_kg
            pamb = jnp.maximum(
                jnp.where(n_air > 1.0e-30, (n_gas / n_air) * pres, 0.0),
                0.0,
            )

            # psat from tentative Mk: (nbins, n_vbs)
            Mk_spec = Mk_trial[:, s0:s1]
            psat = jnp.where(
                Mtot_org[:, None] > 1.0e-30,
                cstar_Pa[None, :] * (Mk_spec / safe_Mtot[:, None]) * Ke,
                0.0,
            )

            # Effective psat with Q correction
            psat_eff = psat / Q_corr[:, None]

            # Driving force: (nbins, n_vbs)
            dp = pamb[None, :] - psat_eff

            # Sink-weighted mean psat → equilibrium gas: (n_vbs,)
            psat_eff_mean = jnp.sum(sinkfrac * psat_eff, axis=0)
            Gc_eq = psat_eff_mean * boxvol_m3 * mw_kg / (R_GAS * temp)

            # Analytical gas depletion: (n_vbs,)
            gas_depl = jnp.where(
                CS > 1.0e-30,
                1.0 - jnp.exp(-CS * cdt),
                CS * cdt,
            )
            mcond = (Gc_spec - Gc_eq) * gas_depl  # total mass to condense

            # Distribute to bins by sinkfrac × dp: (nbins, n_vbs)
            dp_weight = sinkfrac * dp
            dp_weight_sum = jnp.sum(dp_weight, axis=0)  # (n_vbs,)
            mass_per_bin = jnp.where(
                jnp.abs(dp_weight_sum)[None, :] > 1.0e-30,
                dp_weight / dp_weight_sum[None, :] * mcond[None, :],
                0.0,
            )

            # --- Equilibrium mass (no Q, pure thermo): (nbins, n_vbs) ---
            masseqm = jnp.where(
                (Mtot_org[:, None] > 1.0e-30)
                & (cstar_Pa[None, :] > 1.0e-30),
                pamb[None, :] * safe_Mtot[:, None]
                / (cstar_Pa[None, :] * Ke),
                0.0,
            )
            Mk_start_spec = Mk_s[:, s0:s1]
            massEQ_per_bin = jnp.where(
                Nk_s[:, None] > 1.0e-10,
                masseqm - Mk_spec,
                0.0,
            )

            # --- fodc (always apply when ratio > 1e-30): (nbins, n_vbs) ---
            safe_massEQ = jnp.where(
                jnp.abs(massEQ_per_bin) > 1.0e-30, massEQ_per_bin, 1.0)
            mass_ratio = mass_per_bin / safe_massEQ
            has_fodc = ((mass_ratio > 1.0e-30)
                        & (jnp.abs(massEQ_per_bin) > 1.0e-30))

            fodc = jnp.where(
                has_fodc,
                jnp.abs(mass_per_bin / safe_massEQ)
                / jnp.maximum(cdt, 1.0e-30),
                0.0,
            )
            mass_relaxed = massEQ_per_bin * (1.0 - jnp.exp(-fodc * cdt))
            mass_corrected = jnp.where(has_fodc, mass_relaxed, mass_per_bin)

            # Per-bin evaporation clamp
            mass_corrected = jnp.maximum(mass_corrected, -Mk_start_spec)

            # Conservation clamps per species: (n_vbs,)
            total_change = jnp.sum(mass_corrected, axis=0)
            total_in_particles = jnp.sum(Mk_start_spec, axis=0)
            total_change = jnp.maximum(total_change, -total_in_particles)
            total_change = jnp.minimum(
                total_change, jnp.maximum(Gc_spec, 0.0))

            raw_sum = jnp.sum(mass_corrected, axis=0)
            scale = jnp.where(
                jnp.abs(raw_sum) > 1.0e-30, total_change / raw_sum, 0.0)
            mass_change = mass_corrected * scale[None, :]

            # Tentative Mk update for next Raoult iteration
            Mk_new = Mk_s.at[:, s0:s1].set(
                Mk_start_spec + mass_change)
            Mk_new = Mk_new.at[:, s0:s1].set(
                jnp.maximum(Mk_new[:, s0:s1], 0.0))
            return Mk_new

        # Run fixed-point Raoult iterations
        Mk_converged = jax.lax.fori_loop(0, n_iter, _iter_body, Mk_s)

        # MNFIX after mass addition to keep particles within bin bounds.
        # Without PPM redistribution, large mass addition can push avg mass
        # far beyond bin boundaries. MNFIX inside the sub-step prevents
        # catastrophic accumulation before the next sub-step's CS recompute.
        Nk_fixed, Mk_fixed = mnfix_jax(Nk_s, Mk_converged, xk, ICOMP_NODIAG)

        # Final mass change → update Gc
        mass_change_final = Mk_fixed[:, s0:s1] - Mk_s[:, s0:s1]
        gas_change = jnp.sum(mass_change_final, axis=0)  # (n_vbs,)
        Gc_new = Gc_s.at[s0:s1].add(-gas_change)

        # Advance time
        new_time = time_elapsed + cdt

        # Apply updates only if active (time remaining)
        Nk_out = jnp.where(active, Nk_fixed, Nk_s)
        Mk_out = jnp.where(active, Mk_fixed, Mk_s)
        Gc_out = jnp.where(active, Gc_new, Gc_s)
        time_out = jnp.where(active, new_time, time_elapsed)

        return (Nk_out, Mk_out, Gc_out, time_out)

    # Run adaptive sub-steps
    Nk, Mk, Gc, _ = jax.lax.fori_loop(
        0, max_soa_substeps, _substep_body,
        (Nk, Mk, Gc, jnp.float64(0.0)),
    )

    # Particle-phase reaction loss (shared kc for all species)
    if vbs_config.kc > 0.0:
        for j in range(n_vbs):
            idx = vbs_config.species_indices[j]
            reaction_loss = Mk[:, idx] * (1.0 - jnp.exp(-vbs_config.kc * dt))
            Mk = Mk.at[:, idx].add(-reaction_loss)
            Mk = Mk.at[:, idx].set(jnp.maximum(Mk[:, idx], 0.0))

    return Nk, Mk, Gc
