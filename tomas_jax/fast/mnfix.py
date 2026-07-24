"""Vectorized MNFIX — O(1)-depth mass/number drift correction, batched.

Reimplements core/mnfix_jax.py (port of mnfix.f) without the three
sequential 40-iteration ``fori_loop`` passes, which dominate GPU latency
at large cell counts (~120 dependent kernel launches per call).

Phase semantics are identical to the sequential version:
  Phase 1  — empty-bin reset (per-bin, no cross-bin writes → exact).
  Phase 2  — extreme out-of-range trim (per-bin → exact).
  Phase 3  — partial transfer to a computed target bin. All shifts are
             computed from the pre-sweep state and applied as a one-hot
             scatter; number and mass are conserved by construction
             (identical algebra to the sequential pass). Where the
             sequential ascending-k pass would cascade (a shifting bin
             that also receives a deposit in the same pass), a second
             fixed sweep absorbs the residual. Operation order — not
             conservation — is the only difference from mnfix.f.

All functions accept leading batch axes: Nk (..., B), Mk (..., B, icomp).
"""
import jax.numpy as jnp

# Constants matching Fortran mnfix.f / core/mnfix_jax.py
_EPS = 1.0e-40
_NEPS = 1.0e-5


def mnfix(Nk, Mk, xk, icomp_nodiag=1, n_sweeps=2):
    """Correct mass/number drift (batched, fixed-depth).

    Args:
        Nk: Number [#/cell], shape (..., nbins)
        Mk: Mass [kg/cell], shape (..., nbins, icomp)
        xk: Bin boundaries [kg], shape (nbins+1,)
        icomp_nodiag: Number of prognostic (dry) species.
        n_sweeps: Fixed number of phase-3 sweeps (static).

    Returns:
        (Nk_new, Mk_new)
    """
    nbins = Nk.shape[-1]
    icomp = Mk.shape[-1]

    Nk = jnp.maximum(Nk, 0.0)
    Mk = jnp.maximum(Mk, 0.0)

    # ---- Phase 1: empty-bin reset (exact) ----
    xk_geo = jnp.sqrt(xk[:-1] * xk[1:])                      # (B,)
    Mk_empty = jnp.zeros((nbins, icomp), dtype=Mk.dtype)
    Mk_empty = Mk_empty.at[:, 0].set(_NEPS * xk_geo)         # SO4 column
    is_empty = Nk < _NEPS                                    # (..., B)
    Nk = jnp.where(is_empty, _NEPS, Nk)
    Mk = jnp.where(is_empty[..., None], Mk_empty, Mk)

    # ---- Phase 2: extreme out-of-range trim (exact) ----
    drymass = jnp.sum(Mk[..., :icomp_nodiag], axis=-1)
    avg = drymass / (Nk + _EPS)
    too_high = avg > xk[nbins]
    scale = (Nk * xk[nbins] / 1.2) / (drymass + _EPS)
    Mk = jnp.where(too_high[..., None], Mk * scale[..., None], Mk)

    drymass = jnp.sum(Mk[..., :icomp_nodiag], axis=-1)
    too_low = drymass / (Nk + _EPS) < xk[0]
    Nk = jnp.where(too_low, drymass / (xk[0] * 1.2), Nk)

    # ---- Phase 3: partial transfer, fixed sweeps ----
    for _ in range(n_sweeps):
        Nk, Mk = _drift_sweep(Nk, Mk, xk, icomp_nodiag)

    return Nk, Mk


def _drift_sweep(Nk, Mk, xk, icomp_nodiag):
    """One conservative partial-transfer sweep over all bins in parallel."""
    nbins = Nk.shape[-1]
    k = jnp.arange(nbins)

    drymass = jnp.sum(Mk[..., :icomp_nodiag], axis=-1)       # (..., B)
    number = Nk
    avg = drymass / (number + _EPS)
    # N=0 edge case (Fortran lines 103-110)
    xold = jnp.sqrt(xk[:-1] * xk[1:])                        # (B,)
    avg = jnp.where((number == 0.0) & (drymass <= 0.0), xold, avg)

    fj = Mk / (drymass + _EPS)[..., None]                    # (..., B, icomp)

    log_p = jnp.log(xk[1] / xk[0])

    # Upward shift: avg > xk[k+1]
    needs_up = (avg > xk[1:]) & (k < nbins - 1)
    ratio_up = avg * 1.1 / (xk[0] + _EPS)
    kk_up_raw = jnp.ceil(
        jnp.log(jnp.maximum(ratio_up, 1.0)) / log_p
    ).astype(jnp.int32) - 1
    kk_up = jnp.clip(kk_up_raw, k + 1, nbins - 1)
    xnew_up = xk[jnp.minimum(kk_up + 1, nbins)] / 1.1

    # Downward shift: avg < xk[k]
    needs_down = (avg < xk[:-1]) & (k > 0)
    ratio_dn = avg / (1.1 * xk[0] + _EPS)
    kk_dn_raw = jnp.floor(
        jnp.log(jnp.maximum(ratio_dn, 1.0)) / log_p
    ).astype(jnp.int32)
    kk_dn = jnp.clip(kk_dn_raw, 0, jnp.maximum(k - 1, 0))
    xnew_dn = xk[kk_dn] * 1.1

    shifted = needs_up | needs_down
    kk = jnp.where(needs_up, kk_up, jnp.where(needs_down, kk_dn, k))
    xnew = jnp.where(needs_up, xnew_up, xnew_dn)

    nshift = (drymass - xold * number) / (xnew - xold + _EPS)
    nshift = jnp.clip(nshift, 0.0, number)
    nshift = jnp.where(shifted, nshift, 0.0)
    mshift = xnew * nshift

    # Source update: shifting bins reset to avg mass = xold, composition fj
    n_remain = number - nshift
    Nk_base = jnp.where(shifted, n_remain, number)
    Mk_base = jnp.where(
        shifted[..., None], (xold * n_remain)[..., None] * fj, Mk
    )

    # Deposits: one-hot conservative scatter (B x B contraction, negligible)
    onehot = (
        (kk[..., None] == jnp.arange(nbins)) & shifted[..., None]
    ).astype(Mk.dtype)                                        # (..., B_src, B_dst)
    recv_N = jnp.einsum('...bd,...b->...d', onehot, nshift)
    recv_M = jnp.einsum('...bd,...bi->...di', onehot, mshift[..., None] * fj)

    return Nk_base + recv_N, Mk_base + recv_M
