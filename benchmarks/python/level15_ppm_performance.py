"""Level 15: PPM Performance Benchmark.

Compares wall-clock timing of:
- ezcond (TFL) — sequential Python loops
- ezcond_ppm (PPM) — calls JIT-compiled PPM internally
- JIT-compiled ppm_condensation_step (raw PPM, no ezcond overhead)

Also reports CFL substep counts for each case.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time

from .utils import load_csv
from .test_cases import CASES, CASE_KEYS, init_lognormal

from tomas_jax.core.config import (
    SRTSO4, ICOMP, ICOMP_NODIAG, IDIAG, N_GAS_SPECIES,
)
from tomas_jax.physics.condensation_sink import calc_condensation_sink
from tomas_jax.physics.ezcond import ezcond
from tomas_jax.physics.ezcond_ppm import ezcond_ppm
from tomas_jax.physics.condensation_ppm import (
    ppm_condensation_step, compute_substeps, compute_wr,
)


def _compute_tau(Nk, Mk, mcond, sinkfrac, xk):
    """Compute TAU array matching ezcond logic."""
    ibins = len(Nk)
    totsinkfrac = np.sum(sinkfrac)
    tdt = 2.0 / 3.0
    tau = np.zeros(ibins)

    for k in range(ibins):
        mpo = 0.0
        mpw = 0.0
        for j in range(ICOMP - IDIAG):
            mpo += Mk[k, j]
        for j in range(ICOMP):
            mpw += Mk[k, j]
        WR = mpw / mpo if mpo > 0.0 else 1.0

        if Nk[k] > 0.0 and totsinkfrac > 0.0:
            maddp_k = mcond * sinkfrac[k] / totsinkfrac / Nk[k]
            mpw_pp = mpw / Nk[k]
            tau[k] = 1.5 * ((mpw_pp + maddp_k * WR) ** tdt - mpw_pp ** tdt)
    return tau


def run(data_dir, verbose=True):
    """Run Level 15 performance benchmark."""
    results = []

    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    if verbose:
        print("\n=== Level 15: PPM Performance Benchmark ===")
        print(f"  {'Case':<8} {'TFL [ms]':>10} {'PPM [ms]':>10} "
              f"{'PPM-JIT [ms]':>12} {'Speedup':>8} {'CFL subs':>10}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*8} {'-'*10}")

    n_warmup = 1
    n_runs = 3

    for case_key in CASE_KEYS:
        case = CASES[case_key]

        temp = case['temp']
        pres = case['pres']
        boxvol = 1.0e6
        alpha = 1.0

        Nk_init, Mk_init = init_lognormal(case_key, xk)

        Gc = np.zeros(N_GAS_SPECIES)
        Gc[SRTSO4] = case['Gc_so4']

        CS_jax, sf_jax = calc_condensation_sink(
            jnp.array(Nk_init), jnp.array(Mk_init),
            temp, pres, boxvol, accommodation_coeff=alpha
        )
        CS_val = float(CS_jax)
        sinkfrac = np.array(sf_jax)
        dt = 60.0
        mcond = Gc[SRTSO4] * (1.0 - np.exp(-CS_val * dt))

        # --- Time TFL ---
        for _ in range(n_warmup):
            ezcond(Nk_init.copy(), Mk_init.copy(), mcond, SRTSO4,
                   xk, temp, pres, boxvol, alpha)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            ezcond(Nk_init.copy(), Mk_init.copy(), mcond, SRTSO4,
                   xk, temp, pres, boxvol, alpha)
        tfl_time = (time.perf_counter() - t0) / n_runs * 1000

        # --- Time PPM (via ezcond_ppm) ---
        for _ in range(n_warmup):
            ezcond_ppm(Nk_init.copy(), Mk_init.copy(), mcond, SRTSO4,
                       xk, temp, pres, boxvol, alpha)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            ezcond_ppm(Nk_init.copy(), Mk_init.copy(), mcond, SRTSO4,
                       xk, temp, pres, boxvol, alpha)
        ppm_time = (time.perf_counter() - t0) / n_runs * 1000

        # --- Time JIT-compiled PPM step (no ezcond overhead) ---
        tau = _compute_tau(Nk_init, Mk_init, mcond, sinkfrac, xk)
        Nk_jax = jnp.array(Nk_init)
        Mk_jax = jnp.array(Mk_init)
        xk_jax = jnp.array(xk)
        tau_jax = jnp.array(tau)

        ppm_jit = jax.jit(lambda N, M: ppm_condensation_step(
            N, M, xk_jax, tau_jax, SRTSO4, 1.0, ICOMP_NODIAG
        ))

        # Warmup JIT
        _ = ppm_jit(Nk_jax, Mk_jax)
        jax.block_until_ready(_)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            out = ppm_jit(Nk_jax, Mk_jax)
            jax.block_until_ready(out)
        ppm_jit_time = (time.perf_counter() - t0) / n_runs * 1000

        # CFL substeps
        WR = np.array(compute_wr(Mk_jax, ICOMP_NODIAG))
        n_sub = int(compute_substeps(xk_jax, tau_jax, jnp.array(WR), 1.0))

        speedup = tfl_time / ppm_jit_time if ppm_jit_time > 0 else float('inf')

        if verbose:
            print(f"  {case_key:<8} {tfl_time:>10.2f} {ppm_time:>10.2f} "
                  f"{ppm_jit_time:>12.2f} {speedup:>8.1f}x {n_sub:>10d}")

        results.append({
            'name': f"Case {case_key} ({case['name']}) timing",
            'level': '15',
            'passed': True,
            'max_rel_error': 0.0,
            'tfl_ms': tfl_time,
            'ppm_ms': ppm_time,
            'ppm_jit_ms': ppm_jit_time,
            'speedup': speedup,
            'cfl_substeps': n_sub,
        })

    return results
