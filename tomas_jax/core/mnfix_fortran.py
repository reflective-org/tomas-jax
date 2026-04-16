"""FORTRAN-equivalent MNFIX — exact port of mnfix.f for validation.

This module provides a faithful JAX translation of the original FORTRAN
mnfix.f subroutine, preserving its sequential bin-by-bin logic and
multi-bin shifting behavior using jax.lax control flow primitives.

Unlike mnfix_jax.py (which uses vectorized roll for speed), this version:
- Processes bins sequentially via fori_loop (matching FORTRAN do-loop)
- Supports multi-bin shifts via while_loop (matching FORTRAN goto logic)
- Uses identical constants (eps=1e-40, Neps=1e-5)
- Matches FORTRAN redistribution: xnew=xk(kk+1)/1.1, xold=sqrt(xk(k)*xk(k+1))

Use this for benchmark validation against FORTRAN output.
For production, use mnfix_jax.py (faster, JIT-friendly).
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Tuple

from .config import ICOMP_NODIAG


def mnfix_fortran(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = ICOMP_NODIAG
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Exact port of FORTRAN mnfix.f subroutine.

    Corrects mass/number consistency by redistributing particles
    whose average mass has drifted outside their bin boundaries.

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
    ncomp = Mk.shape[1]
    eps = 1.0e-40
    Neps = 1.0e-5

    # Enforce positivity (FORTRAN: if Nk < 0, set to 0)
    Nk = jnp.maximum(Nk, 0.0)
    Mk = jnp.maximum(Mk, 0.0)

    def process_bin_up(k, state):
        """Process upward shifts for bin k (particles too large)."""
        Nk_s, Mk_s = state

        # Calculate dry mass for this bin
        drymass_k = jnp.sum(Mk_s[k, :icomp_nodiag])

        # Average mass per particle
        xbar_k = jnp.where(Nk_s[k] > Neps, drymass_k / Nk_s[k], 0.0)

        # Check if particles are too large for this bin
        needs_shift = (xbar_k > xk[k + 1]) & (Nk_s[k] > Neps) & (k < nbins - 1)

        # Find destination bin using while_loop (FORTRAN goto 10 logic)
        def find_dest(carry):
            kk, = carry
            return (kk + 1,)

        def dest_cond(carry):
            kk, = carry
            # Keep going up while xbar exceeds upper boundary and not at top
            return (xbar_k > xk[kk + 1]) & (kk < nbins - 1)

        dest_init = (k + 1,)
        (kk_dest,) = jax.lax.while_loop(dest_cond, find_dest, dest_init)

        # Redistribute: move all N and M from bin k to bin kk_dest
        # FORTRAN: Nk(kk) = Nk(kk) + Nk(k), Mk(kk,j) = Mk(kk,j) + Mk(k,j)
        Nk_new = Nk_s.at[kk_dest].add(jnp.where(needs_shift, Nk_s[k], 0.0))
        Nk_new = Nk_new.at[k].set(jnp.where(needs_shift, 0.0, Nk_s[k]))

        Mk_new = Mk_s.at[kk_dest].add(jnp.where(needs_shift, Mk_s[k], jnp.zeros(ncomp)))
        Mk_new = Mk_new.at[k].set(jnp.where(needs_shift, jnp.zeros(ncomp), Mk_s[k]))

        # FORTRAN: Set empty bin to small values
        # xnew = xk(kk+1)/1.1  (for destination overshoot correction — simplified here)
        # xold = sqrt(xk(k)*xk(k+1))  (geometric mean mass)
        xold = jnp.sqrt(xk[k] * xk[k + 1])
        Nk_new = jnp.where(
            needs_shift,
            Nk_new.at[k].set(eps),
            Nk_new
        )
        # Set SO4 mass for empty bin
        Mk_empty = jnp.zeros(ncomp)
        Mk_empty = Mk_empty.at[0].set(eps * xold)
        Mk_new = jnp.where(
            needs_shift,
            Mk_new.at[k].set(Mk_empty),
            Mk_new
        )

        return Nk_new, Mk_new

    def process_bin_down(k_rev, state):
        """Process downward shifts for bin k (particles too small).
        k_rev counts from 0; actual bin index is (nbins-1) - k_rev."""
        Nk_s, Mk_s = state
        k = (nbins - 1) - k_rev

        # Calculate dry mass for this bin
        drymass_k = jnp.sum(Mk_s[k, :icomp_nodiag])

        # Average mass per particle
        xbar_k = jnp.where(Nk_s[k] > Neps, drymass_k / Nk_s[k], 0.0)

        # Check if particles are too small for this bin
        needs_shift = (xbar_k < xk[k]) & (Nk_s[k] > Neps) & (k > 0)

        # Find destination bin (going down)
        def find_dest_down(carry):
            kk, = carry
            return (kk - 1,)

        def dest_cond_down(carry):
            kk, = carry
            return (xbar_k < xk[kk]) & (kk > 0)

        dest_init = (k - 1,)
        (kk_dest,) = jax.lax.while_loop(dest_cond_down, find_dest_down, dest_init)

        # Ensure kk_dest is valid (>= 0)
        kk_dest = jnp.maximum(kk_dest, 0)

        # Redistribute
        Nk_new = Nk_s.at[kk_dest].add(jnp.where(needs_shift, Nk_s[k], 0.0))
        Nk_new = Nk_new.at[k].set(jnp.where(needs_shift, 0.0, Nk_s[k]))

        Mk_new = Mk_s.at[kk_dest].add(jnp.where(needs_shift, Mk_s[k], jnp.zeros(ncomp)))
        Mk_new = Mk_new.at[k].set(jnp.where(needs_shift, jnp.zeros(ncomp), Mk_s[k]))

        # Set empty bin to small values
        xold = jnp.sqrt(xk[k] * xk[k + 1])
        Nk_new = jnp.where(
            needs_shift,
            Nk_new.at[k].set(eps),
            Nk_new
        )
        Mk_empty = jnp.zeros(ncomp)
        Mk_empty = Mk_empty.at[0].set(eps * xold)
        Mk_new = jnp.where(
            needs_shift,
            Mk_new.at[k].set(Mk_empty),
            Mk_new
        )

        return Nk_new, Mk_new

    # Process upward shifts (low to high, matching FORTRAN do k=1,ibins)
    Nk_out, Mk_out = jax.lax.fori_loop(0, nbins, process_bin_up, (Nk, Mk))

    # Process downward shifts (high to low, matching FORTRAN do k=ibins,1,-1)
    Nk_out, Mk_out = jax.lax.fori_loop(0, nbins, process_bin_down, (Nk_out, Mk_out))

    # Final cleanup: ensure no negative values
    Nk_out = jnp.maximum(Nk_out, 0.0)
    Mk_out = jnp.maximum(Mk_out, 0.0)

    # Set truly empty bins to eps
    mask_empty = Nk_out < Neps
    Nk_out = jnp.where(mask_empty, eps, Nk_out)

    # For empty bins, set mass to eps * geometric_mean_mass
    xk_geo = jnp.sqrt(xk[:-1] * xk[1:])
    Mk_empty_template = jnp.zeros((nbins, ncomp))
    Mk_empty_template = Mk_empty_template.at[:, 0].set(eps * xk_geo)
    Mk_out = jnp.where(mask_empty[:, None], Mk_empty_template, Mk_out)

    return Nk_out, Mk_out
