"""Capture golden outputs for optimization regression testing.

Runs representative simulations (LHC benchmark scenarios) across all major
code paths and saves final state arrays to NPZ for before/after comparison.

Usage::

    python -m benchmarks.python.capture_golden --out /tmp/golden_before.npz
    # ... apply optimization ...
    python -m benchmarks.python.capture_golden --out /tmp/golden_after.npz
    python -m benchmarks.python.capture_golden --compare /tmp/golden_before.npz /tmp/golden_after.npz
"""
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tomas_jax.solvers.diffrax import diffrax_step, coag_euler_step
from tomas_jax.solvers.condensation import (
    run_condensation_scan,
    run_condensation_scan_tfl,
    run_combined_scan_tfl,
    run_combined_scan_ppm,
    run_full_scan,
    make_step,
)
from tomas_jax.core.mnfix_jax import mnfix_jax

from benchmarks.python.scenarios import get_scenarios
from benchmarks.python.time_single_scenario import (
    init_scenario, BOXVOL, DT, NUC_ORG_CONC, NUC_NH3_CONC, NUC_FION,
)

import os
NSTEPS = int(os.environ.get('GOLDEN_NSTEPS', 240))  # 240 = 4h; 1440 = 24h
SCENARIO_IDS = [1, 2, 3]  # diverse: clean/low-N, polluted/high-N, mid


def run_all():
    out = {}
    scenarios = get_scenarios()

    for sid in SCENARIO_IDS:
        sc = scenarios[sid - 1]
        Nk0, Mk0, Gc0, xk = init_scenario(sc)
        temp = jnp.float64(sc['temp'])
        pres = jnp.float64(sc['pres'])
        boxvol = jnp.float64(BOXVOL)
        rh = jnp.float64(sc['RH'])
        alpha = jnp.float64(1.0)
        dt = jnp.float64(DT)
        prod_rate = jnp.float64(sc['h2so4_prod_kg_per_s'])
        tag = f's{sid}'

        # 1. MNFIX direct on perturbed states (stress edge cases)
        rng = np.random.default_rng(sid)
        nbins, icomp = Mk0.shape
        for i, scale in enumerate([1.0, 1e-6, 1e6]):
            Nk_p = Nk0 * jnp.array(rng.uniform(0.0, 2.0, nbins)) * scale
            Mk_p = Mk0 * jnp.array(rng.uniform(0.0, 4.0, (nbins, icomp)))
            Nf, Mf = mnfix_jax(Nk_p, Mk_p, xk)
            out[f'{tag}_mnfix_{i}_N'] = np.array(Nf)
            out[f'{tag}_mnfix_{i}_M'] = np.array(Mf)

        # 2. Euler coagulation, 60 steps
        step = jax.jit(lambda N, M: coag_euler_step(
            N, M, xk, temp, pres, boxvol, dt=dt, n_substeps=3))
        Nk, Mk = Nk0, Mk0
        for _ in range(60):
            Nk, Mk = step(Nk, Mk)
        out[f'{tag}_coag_euler_N'] = np.array(Nk)
        out[f'{tag}_coag_euler_M'] = np.array(Mk)

        # 3. Tsit5 diffrax single 1h call (with overflow)
        Nf, Mf, ovf = jax.jit(lambda N, M: diffrax_step(
            N, M, xk, temp, pres, boxvol, dt=3600.0,
            return_overflow=True))(Nk0, Mk0)
        out[f'{tag}_coag_diffrax_N'] = np.array(Nf)
        out[f'{tag}_coag_diffrax_M'] = np.array(Mf)
        out[f'{tag}_coag_diffrax_ovf'] = np.array(ovf)

        # 4. TFL condensation scan
        Nf, Mf, Gf, hist = run_condensation_scan_tfl(
            Nk0, Mk0, Gc0, xk, temp, pres, boxvol, rh, alpha, dt,
            NSTEPS, prod_rate)
        out[f'{tag}_cond_tfl_N'] = np.array(Nf)
        out[f'{tag}_cond_tfl_M'] = np.array(Mf)
        out[f'{tag}_cond_tfl_G'] = np.array(Gf)
        out[f'{tag}_cond_tfl_hist'] = np.array(hist)

        # 5. PPM condensation scan
        Nf, Mf, Gf, hist = run_condensation_scan(
            Nk0, Mk0, Gc0, xk, temp, pres, boxvol, rh, alpha, dt,
            NSTEPS, prod_rate)
        out[f'{tag}_cond_ppm_N'] = np.array(Nf)
        out[f'{tag}_cond_ppm_M'] = np.array(Mf)
        out[f'{tag}_cond_ppm_G'] = np.array(Gf)
        out[f'{tag}_cond_ppm_hist'] = np.array(hist)

        # 6. Combined coag+cond scans (TFL and PPM)
        Nf, Mf, Gf, _ = run_combined_scan_tfl(
            Nk0, Mk0, Gc0, xk, temp, pres, boxvol, rh, alpha, dt,
            NSTEPS, prod_rate)
        out[f'{tag}_combined_tfl_N'] = np.array(Nf)
        out[f'{tag}_combined_tfl_M'] = np.array(Mf)
        out[f'{tag}_combined_tfl_G'] = np.array(Gf)

        Nf, Mf, Gf, _ = run_combined_scan_ppm(
            Nk0, Mk0, Gc0, xk, temp, pres, boxvol, rh, alpha, dt,
            NSTEPS, prod_rate)
        out[f'{tag}_combined_ppm_N'] = np.array(Nf)
        out[f'{tag}_combined_ppm_M'] = np.array(Mf)
        out[f'{tag}_combined_ppm_G'] = np.array(Gf)

        # 7. Full scan (nucl + coag + cond, TFL)
        Nf, Mf, Gf, _ = run_full_scan(
            Nk0, Mk0, Gc0, xk, temp, pres, boxvol, rh, alpha, dt,
            NSTEPS, prod_rate,
            jnp.float64(NUC_ORG_CONC), jnp.float64(NUC_NH3_CONC),
            jnp.float64(NUC_FION),
            jnp.float64(1.0), jnp.float64(1.0), jnp.float64(1.0),
            1.0)
        out[f'{tag}_full_tfl_N'] = np.array(Nf)
        out[f'{tag}_full_tfl_M'] = np.array(Mf)
        out[f'{tag}_full_tfl_G'] = np.array(Gf)

        # 8. make_step with all processes (ppm_jit) incl. SO2 + dilution
        step = make_step(['so2_chemistry', 'nucleation', 'coagulation',
                          'condensation', 'dilution'], cond_method='ppm_jit')
        Nk, Mk, Gc = Nk0, Mk0, Gc0.at[43].set(1e-12)  # add SO2
        zN, zM, zG = jnp.zeros_like(Nk0), jnp.zeros_like(Mk0), jnp.zeros_like(Gc0)
        for _ in range(20):
            Nk, Mk, Gc = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                              oh_conc=1e6, org_conc=NUC_ORG_CONC,
                              nh3_conc=NUC_NH3_CONC, fion=NUC_FION,
                              kdil=1e-5, Nk_bg=zN, Mk_bg=zM, Gc_bg=zG)
        out[f'{tag}_makestep_N'] = np.array(Nk)
        out[f'{tag}_makestep_M'] = np.array(Mk)
        out[f'{tag}_makestep_G'] = np.array(Gc)

        # 9. zhao2024 nucleation scheme
        step_z = make_step(['nucleation', 'condensation'],
                           cond_method='tfl_jit', nucl_scheme='zhao2024')
        Nk, Mk, Gc = Nk0, Mk0, Gc0
        for _ in range(20):
            Nk, Mk, Gc = step_z(Nk, Mk, Gc, xk, temp, pres, boxvol, rh,
                                alpha, dt,
                                org_conc=NUC_ORG_CONC, nh3_conc=NUC_NH3_CONC,
                                fion=NUC_FION,
                                hno3=1e9, ulvoc=1e6, dma=1e7, hio3=1e6)
        out[f'{tag}_zhao_N'] = np.array(Nk)
        out[f'{tag}_zhao_M'] = np.array(Mk)
        out[f'{tag}_zhao_G'] = np.array(Gc)

        print(f"Scenario S{sid:02d} done")

    # Fail loudly if any path produced NaN — golden data must be clean
    bad = [k for k, v in out.items() if np.any(np.isnan(v))]
    if bad:
        raise RuntimeError(f"NaN detected in golden outputs: {bad}")

    return out


def compare(f1, f2):
    """Compare two golden NPZ files.

    Reports two metrics per differing array:
      - raw max relative diff over all entries
      - max relative diff over *significant* entries (>1e-9 of array max),
        which excludes empty-bin placeholder values (NEPS-scale, ~17 orders
        of magnitude below the distribution peak) where relative differences
        are numerically meaningless.
    """
    a = np.load(f1)
    b = np.load(f2)
    keys = sorted(set(a.files) | set(b.files))
    n_bitwise = 0
    worst = (0.0, None)
    worst_sig = (0.0, None)
    for k in keys:
        if k not in a.files or k not in b.files:
            print(f"  {k}: MISSING in one file")
            continue
        x, y = a[k], b[k]
        if np.array_equal(x, y):
            n_bitwise += 1
            continue
        denom = np.maximum(np.abs(x), 1e-300)
        rel = np.abs(x - y) / denom
        rel_max = np.max(rel)
        sig = np.abs(x) > 1e-9 * np.max(np.abs(x))
        rel_sig = np.max(rel[sig]) if np.any(sig) else 0.0
        print(f"  {k}: max rel diff = {rel_max:.3e} (significant entries: {rel_sig:.3e})")
        if rel_max > worst[0]:
            worst = (rel_max, k)
        if rel_sig > worst_sig[0]:
            worst_sig = (rel_sig, k)
    print(f"\n{n_bitwise}/{len(keys)} arrays bitwise identical")
    if worst[1] is not None:
        print(f"Worst raw rel diff:         {worst[0]:.3e} ({worst[1]})")
        print(f"Worst significant rel diff: {worst_sig[0]:.3e} ({worst_sig[1]})")
        return worst_sig[0]
    return 0.0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', help='Save golden NPZ to this path')
    p.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'))
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
    else:
        out = run_all()
        np.savez(args.out, **out)
        print(f"Saved {len(out)} arrays to {args.out}")
