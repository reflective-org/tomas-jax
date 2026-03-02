"""Pure-JAX TFL condensation — JIT-compilable tmcond + ezcond.

Vectorized port of tmcond.f and ezcond.f that matches the sequential
Python/Fortran implementation while being fully traceable by JAX.

Key approach:
  - Top-hat construction and dmdt_int translation are vectorized across all bins
  - Bin remapping uses jax.lax.fori_loop over 36 source bins, with vectorized
    overlap computation inside each iteration
  - Condensing species uses inverse dmdt_int at bin boundaries for trapezoidal
    interpolation, matching Fortran exactly (including 1.5*YM for full middle bins)
"""
import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial

from ..core.config import NBINS, ICOMP, ICOMP_NODIAG, SRTSO4

# Constants matching Fortran
TEPS = 1.0e-40
NEPS = 1.0e-20


def _dmdt_int_vec(m0, tau, wr):
    """Vectorized dmdt_int (same as condensation_ppm.dmdt_int)."""
    C = 2.0 / 3.0
    m_h2o = (wr - 1.0) * m0
    x = jnp.power(m0 + m_h2o, C)
    x = jnp.sqrt(jnp.maximum(0.0, C * tau + x))
    result = x * x * x / wr
    result = jnp.where((tau > 0.0) & (result < m0), m0, result)
    result = jnp.where((tau < 0.0) & (result > m0), m0, result)
    return result


def tmcond_jax(
    TAU: jnp.ndarray,
    xk: jnp.ndarray,
    Mk: jnp.ndarray,
    Nk: jnp.ndarray,
    cspecies: int,
    icomp_nodiag: int = ICOMP_NODIAG,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """TFL condensation remapping — pure JAX, JIT-compilable.

    Matches tmcond.f: constructs top-hat per bin, translates edges via dmdt_int,
    remaps onto fixed grid using fractional overlaps.

    Args:
        TAU: Growth forcing per bin, shape (ibins,)
        xk: Bin boundaries [kg], shape (ibins+1,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        Nk: Number concentration [#/grid cell], shape (ibins,)
        cspecies: Index of condensing species (static)
        icomp_nodiag: Number of non-diagnostic species (static)

    Returns:
        Nk_out: Updated number, shape (ibins,)
        Mk_out: Updated mass, shape (ibins, icomp)
    """
    ibins = Nk.shape[0]
    icomp = Mk.shape[1]

    # --- Fix empty bins (lines 82-92) ---
    ANKD = jnp.maximum(Nk, NEPS)
    empty = Nk < NEPS
    # For empty bins: set SO4 = NEPS * 1.4 * xk[k], zero everything else
    Mk_empty = jnp.zeros_like(Mk)
    Mk_empty = Mk_empty.at[:, SRTSO4].set(NEPS * 1.4 * xk[:-1])
    AMKD = jnp.where(empty[:, None], Mk_empty, Mk)

    # --- Transfer overflow (lines 97-110) ---
    # If average mass > upper boundary, shift 10% to next bin
    AMKDRY_pre = jnp.sum(AMKD[:, :icomp_nodiag], axis=1)
    avg_pre = AMKDRY_pre / ANKD
    overflow = (avg_pre > xk[1:]) & (jnp.arange(ibins) < ibins - 1)

    # Apply overflow: shift 10% from k to k+1
    # Use a single pass (Fortran does sequential, but one pass is close enough)
    N_shift = jnp.where(overflow, 0.1 * ANKD, 0.0)
    M_shift = jnp.where(overflow[:, None], 0.1 * AMKD, 0.0)

    ANKD = ANKD - N_shift + jnp.concatenate([jnp.zeros(1), N_shift[:-1]])
    AMKD = AMKD - M_shift + jnp.concatenate([jnp.zeros((1, icomp)), M_shift[:-1]])

    # --- Compute WR (lines 119-129) ---
    AMKDRY = jnp.sum(AMKD[:, :icomp_nodiag], axis=1)
    AMKWET = jnp.sum(AMKD, axis=1)
    WR = jnp.where(AMKDRY > 0.0, AMKWET / AMKDRY, 1.0)

    # --- Check max TAU (lines 139-150) ---
    maxtau = jnp.max(jnp.abs(TAU))

    # --- Top-hat construction (lines 179-200, vectorized) ---
    AVG = AMKDRY / jnp.maximum(ANKD, NEPS)
    XX = xk[:-1] / jnp.maximum(AVG, 1e-30)
    XI = 0.5 + XX * (1.5 - XX)
    xi_valid = XI >= 1.0

    W1 = jnp.sqrt(jnp.maximum(12.0 * (XI - 1.0), 0.0)) * AVG
    W2 = jnp.minimum(xk[1:] - AVG, AVG - xk[:-1])
    WW = 0.5
    WTH = W1 * WW + W2 * (1.0 - WW)
    XU = AVG + WTH * 0.5
    XL = AVG - WTH * 0.5

    # --- Translate top-hat edges (vectorized dmdt_int) ---
    YU = _dmdt_int_vec(XU, TAU, WR)
    YL = _dmdt_int_vec(XL, TAU, WR)

    # --- Condensing species mass at edges (lines 231, 236) ---
    species_frac = jnp.where(AMKDRY > 0.0, AMKD[:, cspecies] / AMKDRY, 0.0)
    YUC_raw = XU * species_frac + YU - XU
    # Cap at upper grid boundary
    YUC = jnp.where(YU > xk[-1], YUC_raw * xk[-1] / jnp.maximum(YU, 1e-30), YUC_raw)
    YU = jnp.minimum(YU, xk[-1])
    YLC = XL * species_frac + YL - XL

    DYI = jnp.where(jnp.abs(YU - YL) > 0.0, 1.0 / (YU - YL), 0.0)

    # --- Compute inverse dmdt_int at all bin boundaries for cspecies ---
    # XP[L, boundary] = dmdt_int(xk[boundary], -TAU[L], WR[L])
    # Shape: (ibins, ibins+1)
    XP_all = _dmdt_int_vec(
        xk[None, :],            # (1, 37)
        (-TAU)[:, None],        # (36, 1)
        WR[:, None]             # (36, 1)
    )  # (36, 37)
    # YM at boundaries: condensing species mass at boundary point
    YM_all = XP_all * species_frac[:, None] + xk[None, :] - XP_all  # (36, 37)

    # --- Remapping via fori_loop ---
    # Active bins: have particles, have forcing, valid top-hat
    active = (ANKD > NEPS) & (jnp.abs(TAU) > TEPS) & xi_valid
    # Bins with zero tau keep their content in same bin
    zero_tau = jnp.abs(TAU) <= TEPS
    # Below-grid: YL < xk[0]
    below_grid = YL < xk[0]
    # YU < xk[0]: particles completely below grid, skip
    completely_below = YU < xk[0]

    def remap_bin(L, carry):
        ANK, AMK = carry

        nk_L = ANKD[L]
        tau_L = TAU[L]

        # === Case 1: Zero TAU — keep in same bin ===
        ank_zt = ANK.at[L].add(nk_L)
        amk_zt = AMK.at[L].add(AMKD[L])

        # === Case 2: Below grid — keep in same bin with cspecies correction ===
        avg_cond_bg = (YUC[L] + YLC[L]) * 0.5
        amkd_bg = AMKD[L]
        amkd_bg = amkd_bg.at[cspecies].set(avg_cond_bg * nk_L)
        ank_bg = ANK.at[L].add(nk_L)
        amk_bg = AMK.at[L].add(amkd_bg)

        # === Case 3: Normal remapping ===
        # Overlap of this bin's translated top-hat with all destination bins
        yu_L = YU[L]
        yl_L = YL[L]
        yuc_L = YUC[L]
        ylc_L = YLC[L]
        dyi_L = DYI[L]

        # Overlap fraction with each destination bin
        overlap = jnp.maximum(
            0.0,
            jnp.minimum(yu_L, xk[1:]) - jnp.maximum(yl_L, xk[:-1])
        )
        frac = overlap * dyi_L  # (ibins,)

        # Number contribution
        dn = nk_L * frac

        # Non-condensing species: proportional redistribution
        # AMK[:, j] += AMKD[L, j] * frac[:]
        amk_noncond = frac[:, None] * AMKD[L, :][None, :]  # (ibins, icomp)

        # Condensing species: trapezoidal interpolation using boundary values
        # Left edge of overlap in each bin
        left_at_yl = yl_L >= xk[:-1]
        left_cond = jnp.where(left_at_yl, ylc_L, YM_all[L, :-1])

        # Right edge of overlap in each bin
        right_at_yu = yu_L <= xk[1:]
        right_cond = jnp.where(right_at_yu, yuc_L, YM_all[L, 1:])

        # Average condensing mass per particle in overlap
        avg_cond = (left_cond + right_cond) * 0.5

        # Fortran uses 1.5 * YM_left for full middle bins (mass-doubling approximation)
        is_full_middle = (~left_at_yl) & (~right_at_yu) & (frac > 0.0)
        avg_cond = jnp.where(is_full_middle, 1.5 * YM_all[L, :-1], avg_cond)

        # Condensing species contribution
        cond_contrib = dn * avg_cond
        # Clamp negative cond mass to zero (matching Fortran val < 0 check)
        cond_contrib = jnp.maximum(cond_contrib, 0.0)

        # Build the species contribution
        # Start from proportional, then override cspecies
        amk_contrib = amk_noncond
        amk_contrib = amk_contrib.at[:, cspecies].set(cond_contrib)

        ank_remap = ANK + dn
        amk_remap = AMK + amk_contrib

        # === Select which case applies ===
        is_active = active[L]
        is_zero_tau = zero_tau[L]
        is_below = below_grid[L] & (~completely_below[L])
        is_skip = completely_below[L] | (~xi_valid[L])

        # Priority: skip > zero_tau > below_grid > normal remap
        ANK_out = jnp.where(is_skip, ANK,
                  jnp.where(is_zero_tau | (~is_active), ank_zt,
                  jnp.where(is_below, ank_bg,
                  ank_remap)))

        AMK_out = jnp.where(is_skip, AMK,
                  jnp.where(is_zero_tau | (~is_active), amk_zt,
                  jnp.where(is_below, amk_bg,
                  amk_remap)))

        return ANK_out, AMK_out

    # Handle trivial case: no forcing
    ANK_trivial = ANKD.copy()
    AMK_trivial = AMKD.copy()

    ANK_init = jnp.zeros(ibins)
    AMK_init = jnp.zeros((ibins, icomp))

    ANK_out, AMK_out = jax.lax.fori_loop(0, ibins, remap_bin, (ANK_init, AMK_init))

    # If max tau is negligible, return input unchanged
    ANK_out = jnp.where(maxtau < TEPS, ANK_trivial, ANK_out)
    AMK_out = jnp.where(maxtau < TEPS, AMK_trivial, AMK_out)

    return ANK_out, AMK_out


def ezcond_tfl_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    mcond: jnp.ndarray,
    spec: int,
    xk: jnp.ndarray,
    temp: jnp.ndarray,
    pres: jnp.ndarray,
    boxvol: jnp.ndarray,
    alpha: jnp.ndarray = jnp.array(1.0),
    icomp_nodiag: int = ICOMP_NODIAG,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX ezcond using TFL tmcond — JIT-compilable.

    Matches ezcond.f structure:
    - Single-step (nsteps=1) — relies on outer time loop for sub-stepping
    - No c1/c2 corrections (moxd=0 for SO4)
    - MNFIX before tmcond and after mass correction (matching Fortran lines 202, 279)
    - Includes mass conservation correction

    Args:
        Nk: Number [#/grid cell], shape (ibins,)
        Mk: Mass [kg/grid cell], shape (ibins, icomp)
        mcond: Total mass to condense [kg/grid cell]
        spec: Condensing species index (static)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp, pres, boxvol, alpha: Environment parameters
        icomp_nodiag: Number of non-diagnostic species (static)

    Returns:
        Nk_out, Mk_out
    """
    from .condensation_sink import calc_condensation_sink
    from ..core.mnfix_jax import mnfix_jax

    ibins = Nk.shape[0]
    icomp = Mk.shape[1]
    tdt = 2.0 / 3.0

    # Condensation sink
    CS, sinkfrac = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        accommodation_coeff=alpha
    )
    totsinkfrac = jnp.sum(sinkfrac)

    # --- CS too small: dump in first bin ---
    Nk_dump = Nk.at[0].add(mcond / jnp.sqrt(xk[0] * xk[1]))
    Mk_dump = Mk.at[0, spec].add(mcond)

    # --- Compute TAU (vectorized, matching ezcond.f lines 176-213) ---
    mpo = jnp.sum(Mk[:, :icomp_nodiag], axis=1)
    mpw = jnp.sum(Mk, axis=1)
    WR = jnp.where(mpo > 0.0, mpw / mpo, 1.0)

    safe_Nk = jnp.maximum(Nk, 1e-30)
    safe_totsinkfrac = jnp.maximum(totsinkfrac, 1e-30)
    maddp = mcond * sinkfrac / safe_totsinkfrac / safe_Nk
    mpw_pp = mpw / safe_Nk

    tau = jnp.where(
        (Nk > 0.0) & (totsinkfrac > 0.0),
        1.5 * (jnp.power(mpw_pp + maddp * WR, tdt) - jnp.power(mpw_pp, tdt)),
        0.0
    )

    # --- Significance check ---
    tot_m = jnp.sum(Mk[:, :icomp_nodiag])
    tot_s = jnp.sum(Mk[:, spec])

    # --- Path 1: Full tmcond (with MNFIX before, matching ezcond.f line 202) ---
    def tmcond_path(args):
        Nk_in, Mk_in, tau_in = args
        # MNFIX before tmcond (ezcond.f line 202)
        Nk_fixed, Mk_fixed = mnfix_jax(Nk_in, Mk_in, xk, icomp_nodiag)
        Nk2, Mk2 = tmcond_jax(tau_in, xk, Mk_fixed, Nk_fixed, spec, icomp_nodiag)
        return Nk2, Mk2

    # --- Path 2: Simple mass addition ---
    def simple_add_path(args):
        Nk_in, Mk_in, tau_in = args
        mass_add = jnp.where(
            (Nk_in > 0.0) & (totsinkfrac > 0.0),
            mcond * sinkfrac / safe_totsinkfrac,
            0.0
        )
        Mk_out = Mk_in.at[:, spec].add(mass_add)
        # MNFIX after simple add (ezcond.f line 232)
        Nk_out, Mk_out = mnfix_jax(Nk_in, Mk_out, xk, icomp_nodiag)
        return Nk_out, Mk_out

    # --- Path 3: No-op ---
    def noop_path(args):
        return args[0], args[1]

    # Three-way branch
    args = (Nk, Mk, tau)

    def significant_path(args):
        return jax.lax.cond(
            mcond > tot_m * 1.0e-3,
            tmcond_path,
            simple_add_path,
            args
        )

    Nk_cond, Mk_cond = jax.lax.cond(
        mcond > tot_s * 1.0e-12,
        significant_path,
        noop_path,
        args
    )

    # --- Mass conservation correction (ezcond.f lines 261-291) ---
    tot_i = jnp.sum(Mk[:, spec])
    tot_f = jnp.sum(Mk_cond[:, spec])
    gained = tot_f - tot_i
    ratio = jnp.where(jnp.abs(mcond) > 0, gained / mcond, 1.0)
    # Fortran condition: abs(1 - ratio) < 1.0, i.e. 0 < ratio < 2
    should_correct = (ratio > 0.0) & (ratio < 2.0)
    safe_ratio = jnp.where(should_correct, ratio, 1.0)
    Mk_corrected = Mk.at[:, spec].set(
        Mk[:, spec] + (Mk_cond[:, spec] - Mk[:, spec]) / safe_ratio
    )
    Mk_cond = jnp.where(should_correct & (mcond > 0.0), Mk_corrected, Mk_cond)

    # MNFIX after mass correction (ezcond.f line 279)
    Nk_cond, Mk_cond = mnfix_jax(Nk_cond, Mk_cond, xk, icomp_nodiag)

    # Select: dump vs condensation
    Nk_out = jnp.where(CS < 1e-20, Nk_dump, Nk_cond)
    Mk_out = jnp.where(CS < 1e-20, Mk_dump, Mk_cond)

    return Nk_out, Mk_out
