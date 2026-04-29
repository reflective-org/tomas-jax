"""1000-scenario atmospheric JAX runner (PPM only).

Fork of run_24h_scenarios.py for the tropo→strato benchmark. PPM_JIT only;
writes to benchmarks/results/atmos/ with 4-digit scenario IDs.

Usage::

    # Full 1000 scenarios (4-6 h wall, run under nohup)
    python -m benchmarks.python.run_atmos_scenarios

    # Partial (smoke test)
    python -m benchmarks.python.run_atmos_scenarios --scenarios 1 2 3

    # Specific modes
    python -m benchmarks.python.run_atmos_scenarios --mode coag_only

Skips NPZs that already exist (idempotent); use --force to overwrite.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the existing scenario runner — just swap scenario source and output dir
from benchmarks.python.run_24h_scenarios import (
    run_scenario, NBINS, ICOMP, N_GAS_SPECIES, SRTSO4, SRTH2O, BOXVOL, DT,
    NHOURS, NSTEPS, XK0_LEGACY, init_lognormal_scenario,
)
from benchmarks.python.scenarios_atmos import get_atmos_scenarios

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'atmos'
TIMING_CSV = RESULTS_DIR / 'timing_jax_ppm_atmos.csv'


def run_combined_fast(scenario):
    """PPM_JIT combined mode via run_combined_scan_ppm (scan-fused).

    Bypasses run_24h_scenarios.run_scenario's slow Python-dispatch combined
    path (~900s/scenario) by calling the already-JIT scan directly.
    Gives ~5-10s/scenario after first-call compilation.
    """
    import jax
    import jax.numpy as jnp
    import time as _time
    from tomas_jax.solvers.condensation import run_combined_scan_ppm

    # Build grid identical to run_24h_scenarios.run_scenario
    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = XK0_LEGACY
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk = jnp.array(xk_np)

    Nk_np, Mk_np = init_lognormal_scenario(
        scenario['N_total'], scenario['GMD_um'],
        scenario['GSD'], xk_np,
    )
    Nk = jnp.array(Nk_np)
    Mk = jnp.array(Mk_np)
    Gc = jnp.zeros(N_GAS_SPECIES)
    Gc = Gc.at[SRTSO4].set(scenario['Gc_SO4_kg'])

    temp = jnp.asarray(scenario['temp'])
    pres = jnp.asarray(scenario['pres'])
    rh = jnp.asarray(scenario['RH'])
    alpha = jnp.asarray(1.0)
    prod_rate = jnp.asarray(scenario['h2so4_prod_kg_per_s'])

    steps_per_hour = NSTEPS // NHOURS  # 60

    Nk_hourly = np.zeros((NHOURS, NBINS))
    Mk_hourly = np.zeros((NHOURS, NBINS, ICOMP))
    Gc_hourly = np.zeros((NHOURS, N_GAS_SPECIES))
    N_tot = np.zeros(NHOURS)
    M_tot = np.zeros(NHOURS)
    M_dry = np.zeros(NHOURS)

    t0 = _time.perf_counter()
    Nk_cur, Mk_cur, Gc_cur = Nk, Mk, Gc
    for ihour in range(NHOURS):
        Nk_cur, Mk_cur, Gc_cur, _hist = run_combined_scan_ppm(
            Nk_cur, Mk_cur, Gc_cur, xk,
            temp, pres, jnp.asarray(BOXVOL), rh, alpha,
            jnp.asarray(DT),
            nsteps=steps_per_hour,
            prod_rate=prod_rate,
        )
        Nk_hourly[ihour] = np.asarray(Nk_cur)
        Mk_hourly[ihour] = np.asarray(Mk_cur)
        Gc_hourly[ihour] = np.asarray(Gc_cur)
        N_tot[ihour] = float(jnp.sum(Nk_cur))
        M_tot[ihour] = float(jnp.sum(Mk_cur))
        M_dry[ihour] = float(jnp.sum(Mk_cur[:, :SRTH2O]))
    Nk_cur.block_until_ready()
    wall = _time.perf_counter() - t0

    return {
        'Nk': Nk_hourly,
        'Mk': Mk_hourly,
        'Gc': Gc_hourly,
        'N_tot': N_tot,
        'M_tot': M_tot,
        'M_dry': M_dry,
        'wall_time_s': wall,
        'scenario_params': scenario,
    }


def run_atmos(scenario_ids=None, modes=None, verbose=False, force=False):
    if modes is None:
        modes = ['coag_only', 'cond_only', 'combined']
    # PPM-only for atmos
    method = 'ppm_jit'

    scenarios = get_atmos_scenarios()
    if scenario_ids is not None:
        sid_set = set(scenario_ids)
        scenarios = [s for s in scenarios if s['scenario_id'] in sid_set]
    if not scenarios:
        print("No scenarios selected", file=sys.stderr)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(scenarios) * len(modes)
    run_count = 0
    t0 = time.time()
    timing_records = []

    # Keep timing CSV open for per-run append so crashes don't lose data
    write_header = not TIMING_CSV.exists()
    timing_fh = open(TIMING_CSV, 'a', newline='', buffering=1)
    timing_writer = csv.writer(timing_fh)
    if write_header:
        timing_writer.writerow(['scenario_id', 'mode', 'method',
                                 'regime', 'wall_time_s'])

    try:
        for scenario in scenarios:
            sid = scenario['scenario_id']
            regime = scenario.get('regime', 'unknown')
            for mode in modes:
                run_count += 1
                label = f"s{sid:04d}_{mode}_{method}"
                npz_path = RESULTS_DIR / f"{label}.npz"

                if npz_path.exists() and not force:
                    if run_count % 100 == 0:
                        print(f"[{run_count}/{total_runs}] skip (exists)")
                    continue

                t1 = time.time()
                try:
                    if mode == 'combined':
                        result = run_combined_fast(scenario)
                    else:
                        result = run_scenario(scenario, mode, method, verbose=verbose)
                    np.savez_compressed(
                        npz_path,
                        Nk=result['Nk'],
                        Mk=result['Mk'],
                        Gc=result['Gc'],
                        N_tot=result['N_tot'],
                        M_tot=result['M_tot'],
                        M_dry=result['M_dry'],
                        wall_time_s=np.array(result['wall_time_s']),
                    )
                    wall = result['wall_time_s']
                    timing_writer.writerow([sid, mode, method, regime, f"{wall:.4f}"])
                    timing_records.append({'scenario_id': sid, 'mode': mode,
                                           'regime': regime, 'wall': wall})
                    elapsed = time.time() - t1
                    if run_count <= 5 or run_count % 25 == 0:
                        pct = 100 * run_count / total_runs
                        eta_s = (time.time() - t0) / run_count * (total_runs - run_count)
                        print(f"[{run_count}/{total_runs} {pct:4.1f}%] "
                              f"{label} {elapsed:5.1f}s  "
                              f"ETA {eta_s/60:5.1f} min")
                except Exception as e:
                    print(f"[{run_count}/{total_runs}] ERROR {label}: {e}",
                          file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    # Continue with next scenario
    finally:
        timing_fh.close()

    total_elapsed = time.time() - t0
    print(f"\nDone {len(timing_records)}/{total_runs} runs in "
          f"{total_elapsed:.1f}s ({total_elapsed/3600:.2f} h)")
    print(f"Timing CSV: {TIMING_CSV}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scenarios', nargs='+', type=int, default=None,
                   help='1-based scenario IDs to run. Default: all 1000.')
    p.add_argument('--mode', nargs='+',
                   default=['coag_only', 'cond_only', 'combined'],
                   choices=['coag_only', 'cond_only', 'combined'])
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing NPZs instead of skipping')
    args = p.parse_args()

    run_atmos(
        scenario_ids=args.scenarios,
        modes=args.mode,
        verbose=args.verbose,
        force=args.force,
    )


if __name__ == '__main__':
    main()
