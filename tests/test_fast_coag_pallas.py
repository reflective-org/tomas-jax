"""Equivalence tests: Pallas fused coagulation kernel vs the XLA path.

The Pallas path must reproduce fast/coagulation.coagulation_step to
rtol <= 1e-10 on Nk, Mk, and overflow over a FULL coagulation step
(shared adaptive substep count, positivity clamp, per-substep MNFIX),
for heterogeneous states including stiff (substep-cap-demanding) cells.

Two flavors:
  * interpret mode — the kernel body runs as plain JAX ops; CPU-safe,
    always runs (covers CI, where no GPU is present).
  * compiled Triton — GPU only, skipped cleanly elsewhere.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tomas_jax.fast.coagulation import (
    _kernel_cell,
    _loss_frequency_cell,
    coagulation_step,
)
from tomas_jax.fast.coagulation_pallas import coagulation_step_pallas

from benchmarks.python.bench_fast_1m import make_initial_state

RTOL = 1e-10

_ON_GPU = jax.devices()[0].platform == "gpu"


def _mixed_state(n_cells, pool_factor=32, seed=0):
    """Heterogeneous cells: half median draws, half the stiffest of a
    larger pool (sorted by max loss frequency, like fast/run.py)."""
    pool, _ = make_initial_state(n_cells * pool_factor, seed=seed)
    Nk, Mk, xk = pool.Nk, pool.Mk, pool.xk
    temp, pres, boxvol = pool.temp, pool.pres, pool.boxvol

    kij = jax.vmap(_kernel_cell)(Nk, Mk, temp, pres, boxvol)
    lam = jax.vmap(_loss_frequency_cell, in_axes=(0, 0, 0, None))(
        Nk, Mk, kij, xk
    )
    order = jnp.argsort(jnp.max(lam, axis=-1))
    half = n_cells // 2
    idx = jnp.concatenate([order[:half], order[-(n_cells - half):]])
    return (Nk[idx], Mk[idx], xk, temp[idx], pres[idx], boxvol[idx])


def _assert_equivalent(ref, pal):
    r_nk, r_mk, r_ovf, r_cap, r_nsub = ref
    p_nk, p_mk, p_ovf, p_cap, p_nsub = pal
    assert int(r_nsub) == int(p_nsub)
    assert bool(r_cap) == bool(p_cap)
    for name, r, p in (("Nk", r_nk, p_nk), ("Mk", r_mk, p_mk),
                       ("overflow", r_ovf, p_ovf)):
        r = np.asarray(r)
        p = np.asarray(p)
        assert np.all(np.isfinite(p)), f"{name}: non-finite pallas output"
        denom = np.where(np.abs(r) > 0.0, np.abs(r), 1.0)
        rel = np.abs(p - r) / denom
        assert rel.max() <= RTOL, (
            f"{name}: max rel err {rel.max():.3e} > {RTOL:g} "
            f"(n_sub={int(r_nsub)})"
        )


@pytest.mark.parametrize("dt", [60.0, 360.0])
def test_interpret_equivalence(dt):
    """CPU-safe: kernel semantics vs reference, via interpret mode."""
    args = _mixed_state(6, seed=0)
    ref = jax.jit(coagulation_step)(*args, dt)
    pal = jax.jit(
        lambda *a: coagulation_step_pallas(*a, interpret=True)
    )(*args, dt)
    _assert_equivalent(ref, pal)


def test_interpret_padding_tail():
    """Cell counts that don't divide the persistent grid (padding path)."""
    args = _mixed_state(5, seed=3)
    ref = jax.jit(coagulation_step)(*args, 120.0)
    pal = jax.jit(
        lambda *a: coagulation_step_pallas(*a, n_persistent=2,
                                           interpret=True)
    )(*args, 120.0)
    _assert_equivalent(ref, pal)


@pytest.mark.skipif(not _ON_GPU, reason="requires a CUDA GPU (Triton)")
@pytest.mark.parametrize("seed", [0, 1])
def test_gpu_equivalence(seed):
    """Compiled Triton kernel vs reference on stiff + median cells."""
    args = _mixed_state(256, seed=seed)
    ref = jax.jit(coagulation_step)(*args, 360.0)
    pal = jax.jit(coagulation_step_pallas)(*args, 360.0)
    _assert_equivalent(ref, pal)


@pytest.mark.skipif(not _ON_GPU, reason="requires a CUDA GPU (Triton)")
def test_gpu_equivalence_at_cap():
    """All-stiff chunk that hits the 256-substep cap."""
    pool, _ = make_initial_state(8192, seed=0)
    Nk, Mk, xk = pool.Nk, pool.Mk, pool.xk
    temp, pres, boxvol = pool.temp, pool.pres, pool.boxvol
    kij = jax.vmap(_kernel_cell)(Nk, Mk, temp, pres, boxvol)
    lam = jax.vmap(_loss_frequency_cell, in_axes=(0, 0, 0, None))(
        Nk, Mk, kij, xk
    )
    idx = jnp.argsort(jnp.max(lam, axis=-1))[-512:]
    args = (Nk[idx], Mk[idx], xk, temp[idx], pres[idx], boxvol[idx])
    ref = jax.jit(coagulation_step)(*args, 360.0)
    pal = jax.jit(coagulation_step_pallas)(*args, 360.0)
    assert int(ref[4]) > 100  # genuinely stiff
    _assert_equivalent(ref, pal)
