"""JAX 24-hour simulation runner for benchmark scenarios.

Runs all 50 scenarios x 3 modes x 2 methods = 300 simulations,
saving hourly snapshots as NPZ files.

Usage::

    python -m benchmarks.python.run_24h_scenarios
    python -m benchmarks.python.run_24h_scenarios --method ppm
    python -m benchmarks.python.run_24h_scenarios --scenarios 1 5
"""
import os
import sys
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO
)
from tomas_jax.solvers.diffrax import diffrax_step
from tomas_jax.solvers.condensation import condensation_step

from benchmarks.python.scenarios import get_scenarios

# Constants matching Fortran harness exactly
PI_FORTRAN = 3.141592654
DENS_INIT = 1770.0  # kg/m3 (sulfate density)
NEPS = 1.0e-3
BOXVOL = 1.0e6  # cm3

DT = 60.0       # seconds per step
NSTEPS = 1440   # steps in 24 hours
NHOURS = 24


def init_lognormal_scenario(N_total, Dp_gmd_um, sigma_gsd, xk):
    """Initialize lognormal Nk, Mk matching Fortran harness exactly.

    Args:
        N_total: Total number concentration [#/cm3]
        Dp_gmd_um: Geometric mean diameter [um]
        sigma_gsd: Geometric standard deviation
        xk: Bin boundaries [kg], shape (ibins+1,)

    Returns:
        Nk: [#/grid cell], shape (ibins,)
        Mk: [kg/grid cell], shape (ibins, icomp)
    """
    xk_np = np.array(xk)
    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))

    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI_FORTRAN)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI_FORTRAN)) ** 0.3333
        Dk_init = np.sqrt(Dl * Dh)

        np_init = ((N_total * BOXVOL)
                   / (np.sqrt(2.0 * PI_FORTRAN) * Dk_init * np.log(sigma_gsd))
                   * np.exp(-(np.log(Dk_init / Dp_gmd_um) ** 2
                              / (2.0 * np.log(sigma_gsd) ** 2)))
                   * (Dh - Dl))

        Nk[k] = np_init
        Mk[k, SRTSO4] = np_init * np.sqrt(xk_np[k]) * np.sqrt(xk_np[k + 1])

    # Neps preprocessing (match Fortran)
    for k in range(NBINS):
        if Nk[k] < NEPS:
            Nk[k] = NEPS
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = NEPS * 1.4 * xk_np[k]

    return Nk, Mk


def run_scenario(scenario, mode, method='tfl', verbose=False):
    """Run one 24-hour simulation.

    Args:
        scenario: dict with keys from scenarios.py
        mode: 'coag_only' | 'cond_only' | 'combined'
        method: 'tfl' | 'ppm'
        verbose: Print progress

    Returns:
        dict with hourly snapshots:
            'Nk': (24, 36) array
            'Mk': (24, 36, 44) array
            'Gc': (24, 43) array
            'N_tot': (24,) total number
            'M_tot': (24,) total mass
            'scenario_params': dict of scenario parameters
    """
    # Initialize bin boundaries
    xk = jnp.logspace(
        jnp.log10(jnp.array(1e-23)),
        jnp.log10(jnp.array(1e-12)),
        NBINS + 1
    )

    # Use initbounds-style doubling (match Fortran)
    from tomas_jax.physics.properties import calc_particle_properties
    # Actually, Fortran uses initbounds() which does mass doubling from xk(1).
    # Let's compute xk the Fortran way: xk(k+1) = 2*xk(k)
    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = 1.6033e-23  # Match Fortran initbounds xk(1)
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk = jnp.array(xk_np)

    # Initialize lognormal
    Nk_np, Mk_np = init_lognormal_scenario(
        scenario['N_total'], scenario['GMD_um'],
        scenario['GSD'], xk_np
    )

    Nk = jnp.array(Nk_np)
    Mk = jnp.array(Mk_np)

    # Gas phase
    Gc = jnp.zeros(N_GAS_SPECIES)
    Gc = Gc.at[SRTSO4].set(scenario['Gc_SO4_kg'])

    temp = scenario['temp']
    pres = scenario['pres']
    rh = scenario['RH']
    alpha = 1.0
    prod_rate_kg_s = scenario['h2so4_prod_kg_per_s']

    # Storage for hourly snapshots
    Nk_hourly = np.zeros((NHOURS, NBINS))
    Mk_hourly = np.zeros((NHOURS, ICOMP * NBINS))  # Flattened for storage
    Gc_hourly = np.zeros((NHOURS, N_GAS_SPECIES))
    N_tot = np.zeros(NHOURS)
    M_tot = np.zeros(NHOURS)
    M_dry = np.zeros(NHOURS)

    # JIT compile coagulation solver (if needed for this mode)
    if mode in ('coag_only', 'combined'):
        solver_jit = jax.jit(diffrax_step, static_argnames=['icomp_nodiag'])
        # Warmup
        _ = solver_jit(Nk, Mk, xk, temp, pres, BOXVOL, 0.01, ICOMP_NODIAG)

    # Time loop
    t_loop_start = time.perf_counter()
    for istep in range(NSTEPS):
        # 1. Coagulation
        if mode in ('coag_only', 'combined'):
            try:
                Nk, Mk = solver_jit(
                    Nk, Mk, xk, temp, pres, BOXVOL,
                    dt=DT, icomp_nodiag=ICOMP_NODIAG
                )
            except Exception:
                # max_steps exceeded or solver diverged — skip this step
                if verbose and istep < 5:
                    print(f"    WARNING: coag solver failed at step {istep}, skipping")

        # 2. Condensation
        if mode in ('cond_only', 'combined'):
            # Add H2SO4 production
            Gc = Gc.at[SRTSO4].set(Gc[SRTSO4] + prod_rate_kg_s * DT)

            # Run condensation step
            Nk, Mk, Gc = condensation_step(
                Nk, Mk, Gc, xk,
                temp, pres, BOXVOL,
                rh, alpha, DT,
                method=method
            )

        # Hourly snapshot
        if (istep + 1) % 60 == 0:
            ihour = (istep + 1) // 60 - 1  # 0-indexed
            Nk_hourly[ihour] = np.array(Nk)
            Mk_hourly[ihour] = np.array(Mk).flatten()
            Gc_hourly[ihour] = np.array(Gc)
            N_tot[ihour] = float(jnp.sum(Nk))
            M_tot[ihour] = float(jnp.sum(Mk))
            M_dry[ihour] = float(jnp.sum(Mk[:, :SRTH2O]))

            if verbose:
                print(f"    Hour {ihour+1:2d}: N_tot={N_tot[ihour]:.4e}, "
                      f"M_tot={M_tot[ihour]:.4e}, "
                      f"Gc_SO4={float(Gc[SRTSO4]):.4e}")

    wall_time_s = time.perf_counter() - t_loop_start

    return {
        'Nk': Nk_hourly,
        'Mk': Mk_hourly.reshape(NHOURS, NBINS, ICOMP),
        'Gc': Gc_hourly,
        'N_tot': N_tot,
        'M_tot': M_tot,
        'M_dry': M_dry,
        'wall_time_s': wall_time_s,
        'scenario_params': scenario,
    }


def run_all_scenarios(methods=None, scenario_ids=None, modes=None, verbose=False):
    """Run all (or selected) scenarios and save results as NPZ files.

    Args:
        methods: List of methods to run. Default: ['tfl', 'ppm']
        scenario_ids: List of 1-based scenario IDs. Default: all 50.
        modes: List of modes. Default: ['coag_only', 'cond_only', 'combined']
        verbose: Print progress per hour.
    """
    if methods is None:
        methods = ['tfl', 'ppm']
    if modes is None:
        modes = ['coag_only', 'cond_only', 'combined']

    scenarios = get_scenarios(n=50, seed=42)

    if scenario_ids is not None:
        scenarios = [s for s in scenarios if s['scenario_id'] in scenario_ids]

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', '24h')
    os.makedirs(results_dir, exist_ok=True)

    total_runs = len(scenarios) * len(modes) * len(methods)
    run_count = 0
    t0 = time.time()
    timing_records = []

    for scenario in scenarios:
        sid = scenario['scenario_id']
        for mode in modes:
            for method in methods:
                # For coag_only, method doesn't matter (no condensation)
                # Only run once with 'tfl' label
                if mode == 'coag_only' and method == 'ppm':
                    run_count += 1
                    continue

                run_count += 1
                label = f"s{sid:02d}_{mode}_{method}"
                npz_path = os.path.join(results_dir, f"{label}.npz")

                # Skip if already exists
                if os.path.exists(npz_path):
                    print(f"[{run_count}/{total_runs}] {label} — already exists, skipping")
                    continue

                print(f"[{run_count}/{total_runs}] Running {label}...")
                t1 = time.time()

                try:
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
                    elapsed = time.time() - t1
                    print(f"  Done in {elapsed:.1f}s -> {npz_path}")

                    timing_records.append({
                        'scenario_id': sid,
                        'mode': mode,
                        'method': method,
                        'wall_time_s': result['wall_time_s'],
                    })

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Write timing CSV
    if timing_records:
        import csv
        timing_path = os.path.join(results_dir, 'timing_jax.csv')
        with open(timing_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scenario_id', 'mode', 'method', 'wall_time_s'])
            for rec in timing_records:
                writer.writerow([rec['scenario_id'], rec['mode'],
                                 rec['method'], f"{rec['wall_time_s']:.4f}"])
        print(f"Timing CSV: {timing_path}")

    total_elapsed = time.time() - t0
    print(f"\nAll runs complete in {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 24h JAX benchmark scenarios")
    parser.add_argument('--method', nargs='+', default=['tfl', 'ppm'],
                        choices=['tfl', 'ppm'],
                        help='Condensation methods to run')
    parser.add_argument('--mode', nargs='+',
                        default=['coag_only', 'cond_only', 'combined'],
                        choices=['coag_only', 'cond_only', 'combined'],
                        help='Simulation modes to run')
    parser.add_argument('--scenarios', nargs='+', type=int, default=None,
                        help='Scenario IDs to run (1-based). Default: all')
    parser.add_argument('--verbose', action='store_true',
                        help='Print hourly progress')
    args = parser.parse_args()

    run_all_scenarios(
        methods=args.method,
        scenario_ids=args.scenarios,
        modes=args.mode,
        verbose=args.verbose,
    )
