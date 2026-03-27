"""Pure JAX implementation of MNFIX — Fortran-faithful partial-transfer algorithm.

Port of mnfix.f (Peter Adams, September 2000).

When average mass per particle drifts outside a bin's boundaries, this routine
performs a **partial transfer**: it splits the bin's population so that some
particles stay (at the geometric mean mass) and the excess moves to the
appropriate neighbor bin. This conserves both total number and total mass.

Key difference from a naive "move-all" approach:
  - Fortran/this code: partial split → smooth gradual redistribution
  - Naive move-all: entire bin dumps to neighbor → oscillation/pulses

Uses ``jax.lax.fori_loop`` for sequential bin processing, matching the Fortran
in-place modification order.

References:
    - mnfix.f (Peter Adams, September 2000)
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Tuple


def mnfix_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Correct mass/number drift using Fortran-faithful partial transfer.

    Port of mnfix.f: when avg mass exceeds bin boundaries, split the bin
    population so the remainder sits at the geometric mean and the excess
    moves to the appropriate neighbor bin.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        xk: Bin boundaries [kg], shape (ibins+1,)
        icomp_nodiag: Number of non-diagnostic species.

    Returns:
        Nk_new: Corrected number concentration
        Mk_new: Corrected mass concentration
    """
    nbins = Nk.shape[0]
    icomp = Mk.shape[1]

    # Constants matching Fortran mnfix.f
    EPS = 1.0e-40
    NEPS = 1.0e-5  # Fortran: parameter(Neps=1E-5)

    # Enforce physical positivity
    Nk = jnp.maximum(Nk, 0.0)
    Mk = jnp.maximum(Mk, 0.0)

    # =========================================================================
    # Phase 1: Fix empty/near-empty bins (mnfix.f lines 56-72)
    # If Nk < Neps: zero out dry mass, set Nk=Neps, Mk_so4 = Neps*sqrt(xk*xk+1)
    # =========================================================================
    def fix_empty_bin(k, state):
        Nk_s, Mk_s = state
        is_empty = Nk_s[k] < NEPS

        # For empty bins: zero all dry mass, set SO4 to tiny value
        xk_geo = jnp.sqrt(xk[k] * xk[k + 1])
        Mk_empty = jnp.zeros(icomp)
        Mk_empty = Mk_empty.at[0].set(NEPS * xk_geo)  # SO4 = Neps * geometric mean

        Nk_s = jnp.where(is_empty, Nk_s.at[k].set(NEPS), Nk_s)
        Mk_s = jnp.where(is_empty, Mk_s.at[k].set(Mk_empty), Mk_s)
        return Nk_s, Mk_s

    Nk, Mk = jax.lax.fori_loop(0, nbins, fix_empty_bin, (Nk, Mk))

    # =========================================================================
    # Phase 2: Fix extreme out-of-range (mnfix.f lines 75-94)
    # avg > xk[ibins+1]: trim mass
    # avg < xk[0]: trim number
    # =========================================================================
    def fix_extreme(k, state):
        Nk_s, Mk_s = state

        drymass = jnp.sum(Mk_s[k, :icomp_nodiag])
        avg = drymass / (Nk_s[k] + EPS)

        # avg > xk[ibins+1] → trim mass
        too_high = avg > xk[nbins]
        mshift = Nk_s[k] * xk[nbins] / 1.2
        scale = mshift / (drymass + EPS)
        Mk_trimmed = Mk_s[k] * scale
        Mk_s = jnp.where(too_high, Mk_s.at[k].set(Mk_trimmed), Mk_s)

        # Recompute drymass after potential trim
        drymass2 = jnp.where(too_high, jnp.sum(Mk_trimmed[:icomp_nodiag]), drymass)

        # avg < xk[0] → trim number
        too_low = drymass2 / (Nk_s[k] + EPS) < xk[0]
        new_nk = drymass2 / (xk[0] * 1.2)
        Nk_s = jnp.where(too_low, Nk_s.at[k].set(new_nk), Nk_s)

        return Nk_s, Mk_s

    Nk, Mk = jax.lax.fori_loop(0, nbins, fix_extreme, (Nk, Mk))

    # =========================================================================
    # Phase 3: Partial transfer for out-of-range bins (mnfix.f lines 96-156)
    # This is the CRITICAL part that differs from the old "move-all" approach.
    #
    # When avg > xk[k+1]:
    #   xold = sqrt(xk[k]*xk[k+1])  (geometric mean of current bin)
    #   xnew = xk[kk+1] / 1.1       (target mass in destination bin kk)
    #   nshift = (drymass - xold*number) / (xnew - xold)
    #   Particles staying in bin k get avg mass = xold
    #   Particles moving to bin kk get avg mass = xnew
    #
    # Similarly for avg < xk[k] (downward shift).
    # =========================================================================
    def fix_drift(k, state):
        Nk_s, Mk_s = state

        drymass = jnp.sum(Mk_s[k, :icomp_nodiag])
        number = Nk_s[k]
        avg = drymass / (number + EPS)

        # --- Handle N=0 edge case (Fortran lines 103-110) ---
        avg = jnp.where(
            (number == 0.0) & (drymass <= 0.0),
            jnp.sqrt(xk[k] * xk[k + 1]),  # use geometric mean
            avg
        )

        xk_lo = xk[k]
        xk_hi = xk[k + 1]
        xold = jnp.sqrt(xk_lo * xk_hi)  # geometric mean of current bin

        # Composition fractions
        fj = Mk_s[k] / (drymass + EPS)

        # ===== UPWARD SHIFT (avg > xk[k+1]) =====
        needs_up = (avg > xk_hi) & (k < nbins - 1)

        # Find target bin kk such that xk[kk+1]/1.1 > avg (Fortran goto loop).
        # General formula for xk[j] = xk[0] * p^j where p = xk[1]/xk[0]:
        #   xk[kk+1] > avg*1.1 => p^(kk+1) > avg*1.1/xk[0]
        #   kk = ceil(log_p(avg*1.1/xk[0])) - 1
        log_p = jnp.log(xk[1] / xk[0])  # log of doubling factor
        ratio_up = avg * 1.1 / (xk[0] + EPS)
        kk_up_raw = jnp.ceil(jnp.log(jnp.maximum(ratio_up, 1.0)) / log_p).astype(jnp.int32) - 1
        kk_up = jnp.clip(kk_up_raw, k + 1, nbins - 1)
        xnew_up = xk[jnp.minimum(kk_up + 1, nbins)] / 1.1

        nshift_up = (drymass - xold * number) / (xnew_up - xold + EPS)
        nshift_up = jnp.maximum(nshift_up, 0.0)  # Safety: no negative shifts
        nshift_up = jnp.minimum(nshift_up, number)  # Can't shift more than we have
        mshift_up = xnew_up * nshift_up

        # Apply upward shift
        Nk_up = Nk_s.at[k].add(-nshift_up)
        Nk_up = Nk_up.at[kk_up].add(nshift_up)

        # Remaining particles in bin k get avg mass = xold, composition = fj
        n_remain_up = number - nshift_up
        Mk_up = Mk_s.at[k].set(xold * n_remain_up * fj)
        # Shifted particles in bin kk get mass = mshift_up * fj
        Mk_up = Mk_up.at[kk_up].add(mshift_up * fj)

        # ===== DOWNWARD SHIFT (avg < xk[k]) =====
        needs_down = (avg < xk_lo) & (k > 0)

        # Find target bin kk such that xk[kk]*1.1 < avg (Fortran goto loop).
        # General formula: xk[j] = xk[0] * p^j where p = xk[1]/xk[0]:
        #   xk[kk]*1.1 < avg => p^kk < avg/(1.1*xk[0])
        #   kk = floor(log_p(avg/(1.1*xk[0])))
        ratio_dn = avg / (1.1 * xk[0] + EPS)
        kk_dn_raw = jnp.floor(jnp.log(jnp.maximum(ratio_dn, 1.0)) / log_p).astype(jnp.int32)
        kk_dn = jnp.clip(kk_dn_raw, 0, jnp.maximum(k - 1, 0))
        xnew_dn = xk[kk_dn] * 1.1

        nshift_dn = (drymass - xold * number) / (xnew_dn - xold + EPS)
        nshift_dn = jnp.maximum(nshift_dn, 0.0)
        nshift_dn = jnp.minimum(nshift_dn, number)
        mshift_dn = xnew_dn * nshift_dn

        # Apply downward shift
        Nk_dn = Nk_s.at[k].add(-nshift_dn)
        Nk_dn = Nk_dn.at[kk_dn].add(nshift_dn)

        n_remain_dn = number - nshift_dn
        Mk_dn = Mk_s.at[k].set(xold * n_remain_dn * fj)
        Mk_dn = Mk_dn.at[kk_dn].add(mshift_dn * fj)

        # ===== SELECT CASE =====
        Nk_out = jnp.where(needs_up, Nk_up,
                 jnp.where(needs_down, Nk_dn,
                 Nk_s))
        Mk_out = jnp.where(needs_up, Mk_up,
                 jnp.where(needs_down, Mk_dn,
                 Mk_s))

        return Nk_out, Mk_out

    Nk, Mk = jax.lax.fori_loop(0, nbins, fix_drift, (Nk, Mk))

    return Nk, Mk
