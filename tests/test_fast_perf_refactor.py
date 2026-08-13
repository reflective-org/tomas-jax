"""Tests for the H100 performance refactor (perf/gpu-fast-h100):

1. calc_coagulation_rates precomputed kij_parts path is bit-identical to
   the default path.
2. The scatter-add mnfix drift sweep conserves number per cell exactly
   and matches the pre-refactor one-hot-einsum deposit on batched states
   with colliding destination bins. (Dry mass is intentionally NOT an
   invariant of the sweep: where nshift clips at the bin's number, the
   sweep renormalizes mass to be consistent with number — Fortran
   mnfix.f semantics, identical before and after the refactor.)
3. The per-segment re-sorted chunked runner routes each cell's state and
   per-cell forcings correctly through sort/chunk/unsort (verified
   against the unchunked runner in a coagulation-quiet regime where the
   shared substep counts are 1 in every chunking, so trajectories are
   batching-invariant).
"""
import numpy as np
import jax.numpy as jnp

from tomas_jax.fast import FastState, run_fast
from tomas_jax.fast.config import NBINS, xk_boundaries
from tomas_jax.fast.mnfix import _drift_sweep
from tomas_jax.physics.coagulation_rates import calc_coagulation_rates


def _xmid(xk):
    return np.sqrt(np.asarray(xk[:-1]) * np.asarray(xk[1:]))


def test_kij_parts_path_matches_default():
    rng = np.random.default_rng(0)
    xk = jnp.asarray(xk_boundaries())
    xmid = _xmid(xk)
    Nk = jnp.asarray(10.0 ** rng.uniform(0.0, 6.0, NBINS))
    Mk = jnp.stack([Nk * xmid, 0.3 * Nk * xmid], axis=1)
    kij = jnp.asarray(rng.uniform(1e-16, 1e-13, (NBINS, NBINS)))
    kij = 0.5 * (kij + kij.T)

    parts = (jnp.tril(kij, k=-1), jnp.triu(kij, k=1), jnp.diag(kij))
    plain = calc_coagulation_rates(Nk, Mk, kij, xk, 1)
    precomp = calc_coagulation_rates(Nk, Mk, None, xk, 1, kij_parts=parts)
    for a, b in zip(plain, precomp):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _drift_sweep_onehot_reference(Nk, Mk, xk, icomp_nodiag):
    """Pre-refactor deposit: identical shift algebra to _drift_sweep but
    with the one-hot einsum scatter it replaced (kept as the equivalence
    reference for the scatter-add)."""
    nbins = Nk.shape[-1]
    k = jnp.arange(nbins)
    eps, xold = 1.0e-40, jnp.sqrt(xk[:-1] * xk[1:])
    drymass = jnp.sum(Mk[..., :icomp_nodiag], axis=-1)
    number = Nk
    avg = drymass / (number + eps)
    avg = jnp.where((number == 0.0) & (drymass <= 0.0), xold, avg)
    fj = Mk / (drymass + eps)[..., None]
    log_p = jnp.log(xk[1] / xk[0])
    needs_up = (avg > xk[1:]) & (k < nbins - 1)
    kk_up = jnp.clip(
        jnp.ceil(jnp.log(jnp.maximum(avg * 1.1 / (xk[0] + eps), 1.0))
                 / log_p).astype(jnp.int32) - 1, k + 1, nbins - 1)
    xnew_up = xk[jnp.minimum(kk_up + 1, nbins)] / 1.1
    needs_down = (avg < xk[:-1]) & (k > 0)
    kk_dn = jnp.clip(
        jnp.floor(jnp.log(jnp.maximum(avg / (1.1 * xk[0] + eps), 1.0))
                  / log_p).astype(jnp.int32), 0, jnp.maximum(k - 1, 0))
    xnew_dn = xk[kk_dn] * 1.1
    shifted = needs_up | needs_down
    kk = jnp.where(needs_up, kk_up, jnp.where(needs_down, kk_dn, k))
    xnew = jnp.where(needs_up, xnew_up, xnew_dn)
    nshift = jnp.clip((drymass - xold * number) / (xnew - xold + eps),
                      0.0, number)
    nshift = jnp.where(shifted, nshift, 0.0)
    mshift = xnew * nshift
    n_remain = number - nshift
    Nk_base = jnp.where(shifted, n_remain, number)
    Mk_base = jnp.where(shifted[..., None],
                        (xold * n_remain)[..., None] * fj, Mk)
    onehot = ((kk[..., None] == jnp.arange(nbins))
              & shifted[..., None]).astype(Mk.dtype)
    recv_N = jnp.einsum('...bd,...b->...d', onehot, nshift)
    recv_M = jnp.einsum('...bd,...bi->...di', onehot, mshift[..., None] * fj)
    return Nk_base + recv_N, Mk_base + recv_M


def test_drift_sweep_scatter_matches_onehot_and_conserves_number():
    rng = np.random.default_rng(1)
    xk = jnp.asarray(xk_boundaries())
    xmid = _xmid(xk)
    C = 64
    Nk = jnp.asarray(10.0 ** rng.uniform(0.0, 6.0, (C, NBINS)))
    # Drift avg masses off-bin so many bins shift, including collisions
    # of several source bins onto the same destination bin.
    drift = 10.0 ** rng.uniform(-1.0, 1.0, (C, NBINS))
    Mk_so4 = Nk * xmid[None, :] * drift
    Mk = jnp.stack([Mk_so4, 0.2 * Mk_so4], axis=-1)

    Nk2, Mk2 = _drift_sweep(Nk, Mk, xk, 1)
    Nr, Mr = _drift_sweep_onehot_reference(Nk, Mk, xk, 1)

    # allclose, not bit-equal: scatter-add accumulation order is not
    # guaranteed on GPU (bit-equal on CPU in practice)
    np.testing.assert_allclose(np.asarray(Nk2), np.asarray(Nr),
                               rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(Mk2), np.asarray(Mr),
                               rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(Nk2, axis=-1)),
        np.asarray(jnp.sum(Nk, axis=-1)), rtol=1e-12,
    )


def test_sorted_chunked_run_routes_cells_and_forcings():
    rng = np.random.default_rng(2)
    xk = jnp.asarray(xk_boundaries())
    xmid = _xmid(xk)
    C, n_steps, dt = 32, 6, 360.0

    # Coagulation-quiet: tiny, cell-distinct number concentrations give
    # distinct sort keys but n_sub == 1 for every possible chunking, so
    # results must be batching-invariant. Tiny gas keeps PPM at 1 substep.
    n_tot = 10.0 ** rng.uniform(-3.0, 2.0, C)
    w = rng.uniform(0.5, 1.5, (C, NBINS))
    w /= w.sum(axis=1, keepdims=True)
    Nk = n_tot[:, None] * w
    Mk = np.zeros((C, NBINS, 2))
    Mk[..., 0] = Nk * xmid[None, :]
    Gc = np.stack(
        [10.0 ** rng.uniform(-20.0, -18.0, C),
         10.0 ** rng.uniform(-12.0, -10.0, C)], axis=1,
    )
    state = FastState.create(
        Nk, Mk, Gc,
        temp=rng.uniform(230.0, 300.0, C),
        pres=rng.uniform(2e4, 1e5, C),
        boxvol=1e6,
        rh=rng.uniform(0.1, 0.9, C),
    )
    # Time-varying, cell-distinct OH exercises the forcing time-slice +
    # permutation routing in the sorted runner.
    oh = 10.0 ** rng.uniform(5.0, 6.5, (n_steps, C))

    ref, dref = run_fast(
        state, n_steps, dt, oh_conc=oh,
        steps_per_segment=4, n_cell_chunks=1,
    )
    out, dout = run_fast(
        state, n_steps, dt, oh_conc=oh,
        steps_per_segment=4, n_cell_chunks=4, sort_by_coag_cost=True,
    )

    assert int(dref["coag_n_sub"].max()) == 1, "regime not coag-quiet"
    assert int(dout["coag_n_sub"].max()) == 1

    # Environment fields are carried unchanged: exact order restoration.
    np.testing.assert_array_equal(np.asarray(out.temp), np.asarray(state.temp))
    np.testing.assert_array_equal(np.asarray(out.rh), np.asarray(state.rh))

    # Per-cell trajectories identical up to fp noise.
    np.testing.assert_allclose(np.asarray(out.Gc), np.asarray(ref.Gc),
                               rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(np.asarray(out.Nk), np.asarray(ref.Nk),
                               rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(np.asarray(out.Mk), np.asarray(ref.Mk),
                               rtol=1e-10, atol=1e-40)

    # Global diagnostics match the unchunked reference.
    np.testing.assert_allclose(dout["N_tot"], dref["N_tot"], rtol=1e-10)
    np.testing.assert_allclose(dout["M_dry"], dref["M_dry"], rtol=1e-10)
