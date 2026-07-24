"""GPU benchmark for the fast reduced model: 1M cells x 6h target < 10 s.

Usage:
    # GPU target run
    python -m benchmarks.python.bench_fast_1m --cells 1000000 --hours 6

    # CPU smoke (no GPU) — validates the code path end-to-end
    python -m benchmarks.python.bench_fast_1m --cells 1000 --hours 1 --cpu-smoke

Heterogeneous seeded initial conditions (T, p, RH, size distribution, SO2,
OH sampled over tropospheric + stratospheric ranges) exercise the shared
adaptive substep counts. Compile time is excluded by a warm-up segment.
Fails loudly on non-finite output, sulfur-budget violation, or substep-cap
hits — never silently.
"""
import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp

from tomas_jax.fast import FastState, run_fast
from tomas_jax.fast.config import (
    GH2SO4,
    GSO2,
    MW_H2SO4,
    MW_SO2,
    MW_SO4,
    NBINS,
    SRTSO4,
    xk_boundaries,
)

S_GAS = 32.06 / MW_H2SO4
S_SO2 = 32.06 / MW_SO2
S_AER = 32.06 / MW_SO4


def make_initial_state(n_cells, seed=0, boxvol=1e6):
    """Heterogeneous cells spanning boundary layer to stratosphere."""
    rng = np.random.default_rng(seed)
    xk = np.asarray(xk_boundaries())
    xmid = np.sqrt(xk[:-1] * xk[1:])
    dp = np.cbrt(xmid / 1770.0 * 6 / np.pi)

    gmd = rng.uniform(20e-9, 200e-9, n_cells)
    gsd = rng.uniform(1.5, 2.2, n_cells)
    n_tot = 10 ** rng.uniform(1.0, 4.5, n_cells) * boxvol  # 10 .. 3e4 /cm3

    lnd = np.log(dp)[None, :]
    w = np.exp(-0.5 * ((lnd - np.log(gmd)[:, None]) / np.log(gsd)[:, None]) ** 2)
    w /= w.sum(axis=1, keepdims=True)
    Nk = n_tot[:, None] * w
    Mk = np.zeros((n_cells, NBINS, 2))
    Mk[..., SRTSO4] = Nk * xmid[None, :]

    Gc = np.zeros((n_cells, 2))
    Gc[:, GH2SO4] = 10 ** rng.uniform(-15.0, -12.0, n_cells)  # kg/cell
    Gc[:, GSO2] = 10 ** rng.uniform(-11.0, -8.0, n_cells)

    state = FastState.create(
        Nk, Mk, Gc,
        temp=rng.uniform(200.0, 305.0, n_cells),
        pres=rng.uniform(5e3, 1.02e5, n_cells),
        boxvol=boxvol,
        rh=rng.uniform(0.02, 0.95, n_cells),
    )
    oh_conc = 10 ** rng.uniform(5.5, 6.7, n_cells)
    return state, jnp.asarray(oh_conc)


def total_sulfur(state, ovf_so4=0.0):
    return (
        float(jnp.sum(state.Gc[:, GH2SO4])) * S_GAS
        + float(jnp.sum(state.Gc[:, GSO2])) * S_SO2
        + (float(jnp.sum(state.Mk[..., SRTSO4])) + ovf_so4) * S_AER
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=1_000_000)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=360.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-per-segment", type=int, default=10)
    ap.add_argument("--n-cell-chunks", type=int, default=1,
                    help="Sequential cell chunks; use 4-8 at 1M cells "
                         "(kernel memory) — pair with --sort-by-coag-cost")
    ap.add_argument("--sort-by-coag-cost", action="store_true",
                    help="Order cells by coagulation stiffness before "
                         "chunking so expensive substep counts are "
                         "confined to the chunks that need them")
    ap.add_argument("--target-seconds", type=float, default=10.0)
    ap.add_argument("--cpu-smoke", action="store_true",
                    help="Small CPU validation run; no pass/fail on timing")
    ap.add_argument("--strict", action="store_true",
                    help="Fail (instead of warn) when substep caps are hit")
    args = ap.parse_args()

    n_steps = int(round(args.hours * 3600.0 / args.dt))
    dev = jax.devices()[0]
    print(f"device: {dev.platform} ({dev.device_kind})")
    print(f"cells={args.cells:,}  hours={args.hours}  dt={args.dt}s  "
          f"steps={n_steps}  x64={jax.config.jax_enable_x64}")

    state, oh_conc = make_initial_state(args.cells, seed=args.seed)
    s0 = total_sulfur(state)

    # Warm-up: compile the segment program (same shapes/lengths as the run)
    warm_steps = min(args.steps_per_segment, n_steps)
    t0 = time.perf_counter()
    _ = run_fast(
        state, n_steps=warm_steps, dt=args.dt, oh_conc=oh_conc,
        steps_per_segment=args.steps_per_segment,
        n_cell_chunks=args.n_cell_chunks,
        sort_by_coag_cost=args.sort_by_coag_cost,
    )
    t_compile = time.perf_counter() - t0
    print(f"warm-up ({warm_steps} steps incl. compile): {t_compile:.2f} s")

    # Timed run
    t0 = time.perf_counter()
    out, diags = run_fast(
        state, n_steps=n_steps, dt=args.dt, oh_conc=oh_conc,
        steps_per_segment=args.steps_per_segment,
        n_cell_chunks=args.n_cell_chunks,
        sort_by_coag_cost=args.sort_by_coag_cost,
    )
    jax.block_until_ready(out.Nk)
    wall = time.perf_counter() - t0

    # --- Validation (fail loudly) ---
    if not bool(jnp.all(jnp.isfinite(out.Nk)) & jnp.all(jnp.isfinite(out.Mk))
                & jnp.all(jnp.isfinite(out.Gc))):
        raise RuntimeError("Non-finite state after run — investigate before "
                           "trusting any timing.")
    s1 = total_sulfur(out, ovf_so4=float(np.sum(diags["overflow_so4"])))
    s_err = abs(s1 - s0) / s0
    cond_caps = int(diags["cond_cap_hit"].sum())
    coag_caps = int(diags["coag_cap_hit"].sum())
    print(f"sulfur budget rel err: {s_err:.2e}")
    print(f"substep cap hits: condensation={cond_caps}, coagulation={coag_caps}")
    # Threshold note: the hard floor is MNFIX's Fortran-faithful empty-bin
    # seeding (NEPS*xk_geo SO4 whenever coagulation re-empties a bin, as in
    # the full model / mnfix.f). Very clean cells (~10/cm3) re-seed every
    # MNFIX call, accumulating ~1e-8..1e-7 relative over multi-hour runs.
    # Integration errors proper are caught well below this.
    if s_err > 1e-7:
        raise RuntimeError(f"Sulfur budget violated: {s_err:.2e} > 1e-7")
    if cond_caps or coag_caps:
        msg = (
            f"WARNING: substep caps hit on {cond_caps} (condensation) / "
            f"{coag_caps} (coagulation) of {n_steps} steps — the stiffest "
            "cells ran coarser than the c_max accuracy target on those "
            "steps. Reduce --dt, raise the caps, or accept the degradation."
        )
        if args.strict:
            raise RuntimeError(msg)
        print(msg)

    # --- Report ---
    per_step = wall / n_steps
    print(f"wall: {wall:.2f} s  ({per_step*1e3:.1f} ms/step, "
          f"{args.cells * n_steps / wall:.3g} cell-steps/s)")

    if dev.platform == "gpu":
        stats = dev.memory_stats()
        if stats:
            peak = stats.get("peak_bytes_in_use", 0) / 2**30
            print(f"peak GPU memory: {peak:.1f} GiB")

    if args.cpu_smoke:
        print("CPU smoke run complete — timing target not evaluated.")
        return

    status = "PASS" if wall < args.target_seconds else "FAIL"
    print(f"target: {args.hours}h x {args.cells:,} cells in "
          f"<{args.target_seconds}s -> {status}")
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
