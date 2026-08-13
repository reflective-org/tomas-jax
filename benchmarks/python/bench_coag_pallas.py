"""Benchmark: XLA vs Pallas coagulation substep loop on the stiff chunk.

Builds the 1M-cell benchmark initial state (seed 0), sorts cells by
coagulation stiffness (max loss frequency, as fast/run.py does), takes
the stiffest chunk, and times one full coagulation_step (which runs the
shared adaptive substep count — the cap, 256 by default, on the stiff
chunk) for both implementations. Reports ms/substep.

Usage:
    python -m benchmarks.python.bench_coag_pallas --pool 1000000 \
        --chunk 125000 --repeats 5 --num-warps 4
"""
import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp

from tomas_jax.fast.coagulation import (
    _kernel_cell,
    _loss_frequency_cell,
    coagulation_step,
)
from tomas_jax.fast.coagulation_pallas import coagulation_step_pallas

from .bench_fast_1m import make_initial_state


def stiff_chunk(pool, chunk, seed=0):
    state, _ = make_initial_state(pool, seed=seed)
    Nk, Mk, xk = state.Nk, state.Mk, state.xk
    temp, pres, boxvol = state.temp, state.pres, state.boxvol

    # Chunked stiffness sort (the (C, B, B) kernel is too big in one go)
    bounds = np.linspace(0, pool, max(pool // 125_000, 1) + 1).astype(int)

    @jax.jit
    def _lam(Nk, Mk, temp, pres, boxvol):
        kij = jax.vmap(_kernel_cell)(Nk, Mk, temp, pres, boxvol)
        lam = jax.vmap(_loss_frequency_cell, in_axes=(0, 0, 0, None))(
            Nk, Mk, kij, xk
        )
        return jnp.max(lam, axis=-1)

    lam = jnp.concatenate([
        _lam(Nk[lo:hi], Mk[lo:hi], temp[lo:hi], pres[lo:hi], boxvol[lo:hi])
        for lo, hi in zip(bounds[:-1], bounds[1:])
    ])
    order = jnp.argsort(lam)[-chunk:]
    return (Nk[order], Mk[order], xk, temp[order], pres[order],
            boxvol[order])


def time_step(fn, args, dt, repeats, **kw):
    stepped = jax.jit(lambda *a: fn(*a, dt, **kw))
    out = stepped(*args)  # compile
    n_sub = int(out[4])
    jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = stepped(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    best = min(times)
    return best, n_sub, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=125_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=360.0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--skip-xla", action="store_true")
    args = ap.parse_args()

    dev = jax.devices()[0]
    print(f"device: {dev.platform} ({dev.device_kind})  x64="
          f"{jax.config.jax_enable_x64}")
    chunk_args = stiff_chunk(args.pool, args.chunk, seed=args.seed)
    print(f"stiff chunk: C={args.chunk:,} of pool {args.pool:,} (seed "
          f"{args.seed})")

    results = {}
    if not args.skip_xla:
        t, n_sub, out_x = time_step(
            coagulation_step, chunk_args, args.dt, args.repeats
        )
        results["xla"] = (t, n_sub)
        print(f"xla:    step {t*1e3:9.2f} ms  n_sub={n_sub}  "
              f"{t/n_sub*1e3:7.3f} ms/substep")

    t, n_sub, out_p = time_step(
        coagulation_step_pallas, chunk_args, args.dt, args.repeats,
        num_warps=args.num_warps,
    )
    results["pallas"] = (t, n_sub)
    print(f"pallas: step {t*1e3:9.2f} ms  n_sub={n_sub}  "
          f"{t/n_sub*1e3:7.3f} ms/substep  (num_warps={args.num_warps})")

    if "xla" in results:
        rel = np.abs(np.asarray(out_p[0]) - np.asarray(out_x[0]))
        denom = np.maximum(np.abs(np.asarray(out_x[0])), 1e-300)
        print(f"speedup: {results['xla'][0] / results['pallas'][0]:.2f}x   "
              f"(Nk max rel diff {np.max(rel / denom):.2e})")


if __name__ == "__main__":
    main()
