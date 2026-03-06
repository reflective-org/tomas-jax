"""Clean apples-to-apples timing benchmark for a single scenario.

Times every JAX solver combination and compares against Fortran S01 timing.
Each test gets a separate JIT warmup run, then 3 timed runs (median reported).

Usage::

    python -m benchmarks.python.time_single_scenario
    python -m benchmarks.python.time_single_scenario --scenario 5
    python -m benchmarks.python.time_single_scenario --runs 5
"""
import os
import csv
import time
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO,
)
from tomas_jax.solvers.diffrax import diffrax_step, coag_euler_step
from tomas_jax.solvers.condensation import (
    run_condensation_scan,
    run_condensation_scan_tfl,
    run_combined_scan_ppm,
    run_combined_scan_tfl,
    run_nucleation_condensation_scan,
    run_full_scan,
)

from benchmarks.python.scenarios import get_scenarios

# Constants matching Fortran harness
PI_FORTRAN = 3.141592654
DENS_INIT = 1770.0
NEPS = 1.0e-3
BOXVOL = 1.0e6
DT = 60.0
NSTEPS = 1440

# Nucleation parameters
NUC_ORG_CONC = 1e7
NUC_NH3_CONC = 1e9
NUC_FION = 3.0


def init_scenario(scenario):
    """Initialize a scenario's state arrays."""
    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = 1.6033e-23
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk = jnp.array(xk_np)

    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))
    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI_FORTRAN)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI_FORTRAN)) ** 0.3333
        Dk_init = np.sqrt(Dl * Dh)
        np_init = ((scenario['N_total'] * BOXVOL)
                   / (np.sqrt(2.0 * PI_FORTRAN) * Dk_init * np.log(scenario['GSD']))
                   * np.exp(-(np.log(Dk_init / scenario['GMD_um']) ** 2
                              / (2.0 * np.log(scenario['GSD']) ** 2)))
                   * (Dh - Dl))
        Nk[k] = np_init
        Mk[k, SRTSO4] = np_init * np.sqrt(xk_np[k]) * np.sqrt(xk_np[k + 1])
    for k in range(NBINS):
        if Nk[k] < NEPS:
            Nk[k] = NEPS
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = NEPS * 1.4 * xk_np[k]

    Nk = jnp.array(Nk)
    Mk = jnp.array(Mk)
    Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(scenario['Gc_SO4_kg'])

    return Nk, Mk, Gc, xk


def time_fn(fn, n_runs=3):
    """Time a function n_runs times, return median wall time in seconds."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        # Force JAX to finish
        if isinstance(result, tuple):
            jax.block_until_ready(result[0])
        else:
            jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    return np.median(times)


def load_fortran_timing(scenario_id):
    """Load Fortran timing for a single scenario from CSV."""
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'tomas_fortran', 'output', '24h', 'timing_fortran.csv'
    )
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['scenario_id'].strip())
            if sid == scenario_id:
                return {
                    k.strip(): float(v.strip())
                    for k, v in row.items() if k.strip() != 'scenario_id'
                }
    return None


def main():
    parser = argparse.ArgumentParser(description='Single-scenario timing benchmark')
    parser.add_argument('--scenario', type=int, default=1, help='Scenario ID (default: 1)')
    parser.add_argument('--runs', type=int, default=3, help='Timed runs per test (default: 3)')
    args = parser.parse_args()

    scenarios = get_scenarios()
    scenario = scenarios[args.scenario - 1]
    sid = scenario['scenario_id']
    n_runs = args.runs

    print(f"=== Timing Benchmark: Scenario S{sid:02d} ===")
    print(f"  N_total={scenario['N_total']:.1e} #/cm3, "
          f"GMD={scenario['GMD_um']:.3f} um, GSD={scenario['GSD']:.2f}")
    print(f"  T={scenario['temp']:.0f} K, P={scenario['pres']:.0f} Pa, "
          f"RH={scenario['RH']:.2f}")
    print(f"  Gc_SO4={scenario['Gc_SO4_kg']:.1e} kg/cell, "
          f"prod={scenario['h2so4_prod']:.1e} molec/cm3/s")
    print(f"  Runs per test: {n_runs} (reporting median)")
    print()

    Nk, Mk, Gc, xk = init_scenario(scenario)
    temp = jnp.float64(scenario['temp'])
    pres = jnp.float64(scenario['pres'])
    boxvol = jnp.float64(BOXVOL)
    rh = jnp.float64(scenario['RH'])
    alpha = jnp.float64(1.0)
    dt = jnp.float64(DT)
    prod_rate = jnp.float64(scenario['h2so4_prod_kg_per_s'])

    # Nucleation params
    org_conc = jnp.float64(NUC_ORG_CONC)
    nh3_conc = jnp.float64(NUC_NH3_CONC)
    fion = jnp.float64(NUC_FION)
    enable_org = jnp.float64(1.0)
    enable_inorg = jnp.float64(1.0)
    fn_scale = jnp.float64(1.0)
    use_tfl = 1.0

    results = {}

    # ------------------------------------------------------------------
    # 1. Tsit5 coag-only (Python loop, diffrax adaptive)
    # ------------------------------------------------------------------
    print("Compiling Tsit5 coag-only...", end=" ", flush=True)
    solver_jit = jax.jit(diffrax_step, static_argnames=['icomp_nodiag'])
    # Warmup
    _ = solver_jit(Nk, Mk, xk, temp, pres, boxvol, jnp.float64(0.01), ICOMP_NODIAG)
    jax.block_until_ready(_[0])

    def run_tsit5_coag():
        Nk_c, Mk_c = Nk, Mk
        for _ in range(NSTEPS):
            Nk_c, Mk_c = solver_jit(
                Nk_c, Mk_c, xk, temp, pres, boxvol, dt, ICOMP_NODIAG
            )
        return Nk_c, Mk_c

    t = time_fn(run_tsit5_coag, n_runs)
    results['tsit5_coag'] = t
    print(f"{t:.3f}s")

    # ------------------------------------------------------------------
    # 2. Euler coag-only (scan-fused via lax.scan inside coag_euler_step)
    # ------------------------------------------------------------------
    print("Compiling Euler coag-only...", end=" ", flush=True)

    @jax.jit
    def run_euler_coag_scan(Nk, Mk):
        def step(carry, _):
            Nk_c, Mk_c = carry
            Nk_c, Mk_c = coag_euler_step(
                Nk_c, Mk_c, xk, temp, pres, boxvol,
                dt=dt, icomp_nodiag=ICOMP_NODIAG, n_substeps=3,
            )
            return (Nk_c, Mk_c), None
        (Nk_f, Mk_f), _ = jax.lax.scan(step, (Nk, Mk), None, length=NSTEPS)
        return Nk_f, Mk_f

    # Warmup
    _ = run_euler_coag_scan(Nk, Mk)
    jax.block_until_ready(_[0])

    t = time_fn(lambda: run_euler_coag_scan(Nk, Mk), n_runs)
    results['euler_coag'] = t
    print(f"{t:.3f}s")

    # ------------------------------------------------------------------
    # 3. PPM cond-only (scan-fused)
    # ------------------------------------------------------------------
    print("Compiling PPM cond-only...", end=" ", flush=True)
    # Warmup
    _ = run_condensation_scan(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS, prod_rate
    )
    jax.block_until_ready(_[0])

    t = time_fn(
        lambda: run_condensation_scan(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS, prod_rate
        ),
        n_runs,
    )
    results['ppm_cond'] = t
    print(f"{t:.3f}s")

    # ------------------------------------------------------------------
    # 4. TFL cond-only (scan-fused)
    # ------------------------------------------------------------------
    print("Compiling TFL cond-only...", end=" ", flush=True)
    _ = run_condensation_scan_tfl(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS, prod_rate
    )
    jax.block_until_ready(_[0])

    t = time_fn(
        lambda: run_condensation_scan_tfl(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS, prod_rate
        ),
        n_runs,
    )
    results['tfl_cond'] = t
    print(f"{t:.3f}s")

    # ------------------------------------------------------------------
    # 5. Euler + PPM combined (scan-fused)
    # ------------------------------------------------------------------
    print("Compiling Euler+PPM combined...", end=" ", flush=True)
    _ = run_combined_scan_ppm(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS, prod_rate
    )
    jax.block_until_ready(_[0])

    t = time_fn(
        lambda: run_combined_scan_ppm(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS, prod_rate
        ),
        n_runs,
    )
    results['euler_ppm_combined'] = t
    print(f"{t:.3f}s")

    # ------------------------------------------------------------------
    # 6. Euler + TFL combined (scan-fused)
    # ------------------------------------------------------------------
    print("Compiling Euler+TFL combined...", end=" ", flush=True)
    _ = run_combined_scan_tfl(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS, prod_rate
    )
    jax.block_until_ready(_[0])

    t = time_fn(
        lambda: run_combined_scan_tfl(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS, prod_rate
        ),
        n_runs,
    )
    results['euler_tfl_combined'] = t
    print(f"{t:.3f}s")

    # ------------------------------------------------------------------
    # 7. Nucl + TFL cond (scan-fused)
    # ------------------------------------------------------------------
    print("Compiling Nucl+TFL cond...", end=" ", flush=True)
    _ = run_nucleation_condensation_scan(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS,
        prod_rate, org_conc, nh3_conc, fion, enable_org, enable_inorg,
        fn_scale, use_tfl,
    )
    jax.block_until_ready(_[0])

    t = time_fn(
        lambda: run_nucleation_condensation_scan(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS,
            prod_rate, org_conc, nh3_conc, fion, enable_org, enable_inorg,
            fn_scale, use_tfl,
        ),
        n_runs,
    )
    results['nucl_cond'] = t
    print(f"{t:.3f}s")

    # ------------------------------------------------------------------
    # 8. Full: Nucl + Euler coag + TFL cond (scan-fused)
    # ------------------------------------------------------------------
    print("Compiling Full (nucl+coag+cond)...", end=" ", flush=True)
    _ = run_full_scan(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS,
        prod_rate, org_conc, nh3_conc, fion, enable_org, enable_inorg,
        fn_scale, use_tfl,
    )
    jax.block_until_ready(_[0])

    t = time_fn(
        lambda: run_full_scan(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, NSTEPS,
            prod_rate, org_conc, nh3_conc, fion, enable_org, enable_inorg,
            fn_scale, use_tfl,
        ),
        n_runs,
    )
    results['full'] = t
    print(f"{t:.3f}s")

    # ------------------------------------------------------------------
    # Load Fortran timing
    # ------------------------------------------------------------------
    fortran = load_fortran_timing(sid)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print()
    print(f"{'='*72}")
    print(f"  TIMING RESULTS — Scenario S{sid:02d} (24h, 1440 steps, dt=60s)")
    print(f"{'='*72}")
    print(f"  {'Test':<30s} {'JAX (s)':>8s} {'Fortran (s)':>12s} {'Ratio':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*8}")

    rows = [
        ('Tsit5 coag-only',       'tsit5_coag',        'coag_s'),
        ('Euler coag-only',       'euler_coag',         'coag_s'),
        ('PPM cond-only',         'ppm_cond',           'cond_s'),
        ('TFL cond-only',         'tfl_cond',           'cond_s'),
        ('Euler+PPM combined',    'euler_ppm_combined',  'combined_s'),
        ('Euler+TFL combined',    'euler_tfl_combined',  'combined_s'),
        ('Nucl+TFL cond',         'nucl_cond',          'nucl_cond_s'),
        ('Full (nucl+coag+cond)', 'full',               'full_s'),
    ]

    for label, jax_key, fort_key in rows:
        jax_t = results.get(jax_key)
        fort_t = fortran.get(fort_key) if fortran else None
        jax_str = f"{jax_t:.3f}" if jax_t is not None else "—"
        if fort_t is not None and fort_t > 0:
            fort_str = f"{fort_t:.4f}"
            ratio_str = f"{jax_t / fort_t:.2f}x" if jax_t else "—"
        else:
            fort_str = "—"
            ratio_str = "—"
        print(f"  {label:<30s} {jax_str:>8s} {fort_str:>12s} {ratio_str:>8s}")

    print(f"{'='*72}")

    if fortran:
        print(f"\n  Fortran source: tomas_fortran/output/24h/timing_fortran.csv")
    else:
        print(f"\n  (Fortran timing CSV not found)")

    print(f"  Ratio = JAX / Fortran (lower is better, <1 means JAX wins)")
    print()


if __name__ == '__main__':
    main()
