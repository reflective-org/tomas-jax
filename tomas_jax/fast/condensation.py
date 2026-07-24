"""Batched H2SO4 PPM condensation with a global-max capped CFL substep loop.

Recomposes solvers/condensation._condensation_step_core + ezcond_ppm_jax +
ppm_condensation_step for a leading cell axis:

  - The dump / PPM / no-op ``lax.cond`` branches of ezcond_ppm_jax become
    ``jnp.where`` masks: under batching every branch runs for the whole
    batch anyway, and masks avoid per-cell traced-predicate loops.
  - The per-cell CFL substep count (condensation_ppm.compute_substeps, up
    to ~200 in nucleation bursts) becomes ONE shared loop of
    ``n_glob = min(max_over_cells(n_sub), n_sub_cap)`` substeps. Cells that
    need fewer substeps simply run at a smaller Courant number — strictly
    MORE accurate, never less; only the capped case reduces accuracy, so a
    ``cap_hit`` diagnostic is returned and asserted in benchmarks.
  - NH3 equilibrium is dropped (no ammonia in the reduced model); the
    ISORROPIA water fit is replaced by Tabazadeh 1997 (fast/water.py).
  - No eznh3eqm, no per-call re-JIT: caller composes and jits.

The PPM substep math itself is reused verbatim from physics/condensation_ppm
via jax.vmap.
"""
import jax
import jax.numpy as jnp

from ..physics.condensation_ppm import (
    _compute_moment_integrals,
    advect_totals,
    compute_edge_velocity,
    compute_substeps,
    compute_wr,
    ppm_flux,
    ppm_mass_flux,
    ppm_reconstruct,
    species_flux,
)
from ..physics.gas_properties import (
    calc_fuchs_sutugin_correction,
    calc_gas_diffusivity,
    calc_mean_free_path,
)
from .config import (
    CS_EPS,
    GH2SO4,
    ICOMP_NODIAG,
    MW_H2SO4,
    MW_SO4,
    PI,
    SRTSO4,
    SV_H2SO4,
)
from .density import calc_density
from .mnfix import mnfix

# Fortran getCondSink.f: parameter(Neps=1.0d10)
_NEPS_CONDSINK = 1.0e10
_TDT = 2.0 / 3.0


def _condensation_sink_cell(Nk, Mk, temp, pres, boxvol, xk, alpha):
    """Per-cell copy of physics/condensation_sink.calc_condensation_sink
    with the binary H2SO4/H2O density."""
    Di = calc_gas_diffusivity(temp, pres, MW_H2SO4, SV_H2SO4)
    mfp = calc_mean_free_path(temp, pres, MW_H2SO4, SV_H2SO4)

    has_particles = Nk > _NEPS_CONDSINK
    mp_actual = jnp.sum(Mk, axis=-1) / jnp.maximum(Nk, 1e-30)
    mp = jnp.where(has_particles, mp_actual, 1.4 * xk[:-1])
    density = jnp.where(has_particles, calc_density(Mk), 1500.0)

    Dpk = jnp.cbrt(mp / density * (6.0 / PI))
    Kn = 2.0 * mfp / jnp.maximum(Dpk, 1e-30)
    beta = calc_fuchs_sutugin_correction(Kn, alpha)
    safe_beta = jnp.where(Dpk > 0.0, beta, 0.0)

    sink = Dpk * Nk * safe_beta
    CS_sum = jnp.sum(sink)
    CS = 2.0 * PI * Di * CS_sum / (boxvol * 1e-6)
    sinkfrac = jnp.where(CS_sum > 1e-30, sink / CS_sum, jnp.zeros_like(Dpk))
    return CS, sinkfrac


# When True, H2SO4 kg leaving the gas deposits as SO4 kg * 96/98 (the H2/
# 96 goes to the water diagnostic at the next equilibrium call), so the
# total-S budget closes exactly. The full model (and Fortran) move kg 1:1,
# creating sulfur at the 2% level of the condensed flux — set False only
# for bit-comparison runs against the full model.
S_CONSERVING_CONDENSATION = True


def condensation_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, dt, alpha=1.0, n_sub_cap=40
):
    """Batched condensation: MNFIX -> CS -> gas depletion -> PPM.

    Water equilibrium is NOT applied here — the step composer applies
    Tabazadeh water where the pipeline needs it (before coagulation and
    after condensation).

    Args:
        Nk: (C, nbins); Mk: (C, nbins, 2); Gc: (C, 2)
        xk: (nbins+1,); temp/pres/boxvol: (C,); dt: [s]
        alpha: accommodation coefficient (scalar)
        n_sub_cap: static cap on the shared CFL substep count

    Returns:
        (Nk, Mk, Gc, cap_hit) — cap_hit is a bool scalar diagnostic: the
        uncapped batch-max substep requirement exceeded n_sub_cap.
    """
    Nk, Mk = mnfix(Nk, Mk, xk, ICOMP_NODIAG)

    CS, sinkfrac = jax.vmap(
        _condensation_sink_cell, in_axes=(0, 0, 0, 0, 0, None, None)
    )(Nk, Mk, temp, pres, boxvol, xk, alpha)

    # --- Gas depletion over dt (exponential CS decay) ---
    gc_so4 = Gc[..., GH2SO4]
    active = (CS > CS_EPS) & (gc_so4 > 0.0)
    mcond_gas = jnp.where(active, gc_so4 * (1.0 - jnp.exp(-CS * dt)), 0.0)
    # Aerosol-side deposit (see S_CONSERVING_CONDENSATION)
    mw_aer = (MW_SO4 / MW_H2SO4) if S_CONSERVING_CONDENSATION else 1.0
    mcond = mcond_gas * mw_aer

    # --- TAU per bin (ezcond_ppm_jax cond_path, batched) ---
    totsinkfrac = jnp.sum(sinkfrac, axis=-1)
    mpo = jnp.sum(Mk[..., :ICOMP_NODIAG], axis=-1)
    mpw = jnp.sum(Mk, axis=-1)
    WR_ez = jnp.where(mpo > 0.0, mpw / jnp.maximum(mpo, 1e-30), 1.0)

    safe_Nk = jnp.maximum(Nk, 1e-30)
    safe_tsf = jnp.maximum(totsinkfrac, 1e-30)[..., None]
    maddp = mcond[..., None] * sinkfrac / safe_tsf / safe_Nk
    mpw_pp = mpw / safe_Nk
    tau = jnp.where(
        (Nk > 0.0) & (totsinkfrac > 0.0)[..., None],
        1.5 * (jnp.power(mpw_pp + maddp * WR_ez, _TDT)
               - jnp.power(mpw_pp, _TDT)),
        0.0,
    )

    # PPM applies where the condensed mass is significant (ezcond rule:
    # mcond > tot_s * 1e-12; with mcond >= 0 the simple-add branch of the
    # original is unreachable, and the no-op branch keeps Nk/Mk while the
    # gas depletion above still stands — faithful to the full model).
    tot_s = jnp.sum(Mk[..., SRTSO4], axis=-1)
    use_ppm = active & (mcond > tot_s * 1e-12)

    # --- Shared CFL substep count: batch-max with static cap ---
    WR_ppm = jax.vmap(compute_wr, in_axes=(0, None))(Mk, ICOMP_NODIAG)
    n_cells = jax.vmap(compute_substeps, in_axes=(None, 0, 0, None))(
        xk, tau, WR_ppm, 1.0
    )
    n_raw_max = jnp.max(jnp.where(use_ppm, n_cells, 1))
    n_glob = jnp.clip(n_raw_max, 1, n_sub_cap)
    cap_hit = n_raw_max > n_sub_cap

    dt_sub = 1.0 / n_glob
    TAU_sub = tau / n_glob
    u_edges = jax.vmap(compute_edge_velocity, in_axes=(None, 0, 0, None))(
        xk, TAU_sub, WR_ppm, dt_sub
    )

    delta_xi = jnp.log(xk[1] / xk[0])
    I0_a, I1_a, I2_a = _compute_moment_integrals(delta_xi)

    def cell_substep(Nk_c, Mk_c, u_e):
        # Verbatim ppm_condensation_step substep body (per cell)
        n_bar = Nk_c / delta_xi
        n_L, n_R, n_6 = ppm_reconstruct(n_bar)
        F_N = ppm_flux(n_L, n_R, n_6, u_e, dt_sub, delta_xi)
        Nk_n = advect_totals(Nk_c, F_N, dt_sub)

        F_M_dry = ppm_mass_flux(n_L, n_R, n_6, xk, u_e, dt_sub, delta_xi)
        b = (n_R - n_L) + n_6
        M_dry = delta_xi * xk[:-1] * (n_L * I0_a + b * I1_a - n_6 * I2_a)
        F_M_all = species_flux(F_M_dry, Mk_c, M_dry, u_e)
        Mk_n = advect_totals(Mk_c, F_M_all, dt_sub)

        return jnp.maximum(Nk_n, 0.0), jnp.maximum(Mk_n, 0.0)

    def body(_, carry):
        Nk_c, Mk_c = carry
        return jax.vmap(cell_substep)(Nk_c, Mk_c, u_edges)

    Nk_ppm, Mk_ppm = jax.lax.fori_loop(0, n_glob, body, (Nk, Mk))

    # Condensed mass added after transport, sink-fraction weighted — but
    # only onto bins that still hold particles. Strong growth can advect a
    # donor bin's entire population upward within the step; depositing its
    # sink share into the now-empty bin strands mass that the next MNFIX
    # phase-1 reset would destroy (worst case ~2%/step of a cell's mass at
    # dt=360 s; the full model has the same leak, smaller at dt=60 s).
    # Redistributing to surviving bins keeps gas->aerosol transfer exact.
    alive = Nk_ppm > 1e-5  # MNFIX phase-1 NEPS threshold
    sf_alive = jnp.where(alive, sinkfrac, 0.0)
    sf_norm = jnp.maximum(jnp.sum(sf_alive, axis=-1), 1e-30)[..., None]
    mass_add = mcond[..., None] * sf_alive / sf_norm
    Mk_ppm = Mk_ppm.at[..., SRTSO4].add(mass_add)

    # --- Dump path: CS negligible but gas present -> all gas to bin 0 ---
    should_dump = (CS <= CS_EPS) & (gc_so4 > 0.0)
    dump_M = gc_so4 * mw_aer
    dump_N = dump_M / jnp.sqrt(xk[0] * xk[1])

    # --- Select per cell: dump / ppm / unchanged ---
    Nk_out = jnp.where(
        should_dump[..., None],
        Nk.at[..., 0].add(dump_N),
        jnp.where(use_ppm[..., None], Nk_ppm, Nk),
    )
    Mk_out = jnp.where(
        should_dump[..., None, None],
        Mk.at[..., 0, SRTSO4].add(dump_M),
        jnp.where(use_ppm[..., None, None], Mk_ppm, Mk),
    )
    gc_out = jnp.where(should_dump, 0.0, gc_so4 - mcond_gas)
    Gc_out = Gc.at[..., GH2SO4].set(gc_out)

    # NOTE: no trailing MNFIX here — the step composer applies water
    # equilibrium first and then MNFIX (full-model order: ezcond -> water
    # -> mnfix). Standalone callers should do the same.
    return Nk_out, Mk_out, Gc_out, cap_hit
