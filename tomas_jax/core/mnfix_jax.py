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

from .config import ICOMP_NODIAG


def mnfix_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = ICOMP_NODIAG
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
    # Each bin only touches its own row — fully vectorized (identical to the
    # original sequential loop).
    # =========================================================================
    is_empty = Nk < NEPS
    xk_geo = jnp.sqrt(xk[:-1] * xk[1:])
    Nk = jnp.where(is_empty, NEPS, Nk)
    Mk = jnp.where(is_empty[:, None], 0.0, Mk)
    Mk = Mk.at[:, 0].set(jnp.where(is_empty, NEPS * xk_geo, Mk[:, 0]))

    # =========================================================================
    # Phase 2: Fix extreme out-of-range (mnfix.f lines 75-94)
    # avg > xk[ibins+1]: trim mass
    # avg < xk[0]: trim number
    # Per-bin independent — vectorized across bins.
    # =========================================================================
    drymass = jnp.sum(Mk[:, :icomp_nodiag], axis=1)
    avg = drymass / (Nk + EPS)

    # avg > xk[ibins+1] → trim mass
    too_high = avg > xk[nbins]
    mshift = Nk * xk[nbins] / 1.2
    scale = mshift / (drymass + EPS)
    Mk_trimmed = Mk * scale[:, None]
    Mk = jnp.where(too_high[:, None], Mk_trimmed, Mk)

    # Recompute drymass after potential trim
    drymass2 = jnp.where(
        too_high, jnp.sum(Mk_trimmed[:, :icomp_nodiag], axis=1), drymass)

    # avg < xk[0] → trim number
    too_low = drymass2 / (Nk + EPS) < xk[0]
    new_nk = drymass2 / (xk[0] * 1.2)
    Nk = jnp.where(too_low, new_nk, Nk)

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
    # log of bin mass ratio — constant, hoisted out of the loop
    log_p = jnp.log(xk[1] / xk[0])

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
        ratio_up = avg * 1.1 / (xk[0] + EPS)
        kk_up_raw = jnp.ceil(jnp.log(jnp.maximum(ratio_up, 1.0)) / log_p).astype(jnp.int32) - 1
        kk_up = jnp.clip(kk_up_raw, k + 1, nbins - 1)
        xnew_up = xk[jnp.minimum(kk_up + 1, nbins)] / 1.1

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

        # ===== SELECT SHIFT PARAMETERS, APPLY ONCE =====
        # Select (kk, xnew) first, then compute the shift — the arithmetic of
        # whichever branch fires is preserved, but only one scatter pair is
        # materialized instead of both candidate state arrays.
        # (Equivalent to the original two-branch formulation to within
        # compiler-level rounding, ~1e-15 relative.)
        needs_shift = needs_up | needs_down
        kk = jnp.where(needs_up, kk_up, kk_dn)
        xnew = jnp.where(needs_up, xnew_up, xnew_dn)

        nshift = (drymass - xold * number) / (xnew - xold + EPS)
        nshift = jnp.maximum(nshift, 0.0)  # Safety: no negative shifts
        nshift = jnp.minimum(nshift, number)  # Can't shift more than we have
        nshift = jnp.where(needs_shift, nshift, 0.0)
        mshift = xnew * nshift

        Nk_out = Nk_s.at[k].add(-nshift)
        Nk_out = Nk_out.at[kk].add(nshift)

        # Remaining particles in bin k get avg mass = xold, composition = fj;
        # shifted particles in bin kk get mass = mshift * fj
        n_remain = number - nshift
        row_k = jnp.where(needs_shift, xold * n_remain * fj, Mk_s[k])
        Mk_out = Mk_s.at[k].set(row_k)
        Mk_out = Mk_out.at[kk].add(jnp.where(needs_shift, mshift * fj, 0.0))

        return Nk_out, Mk_out

    Nk, Mk = jax.lax.fori_loop(0, nbins, fix_drift, (Nk, Mk))

    return Nk, Mk
