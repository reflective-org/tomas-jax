"""Side-by-side comparison of all condensation methods on a single scenario.

Runs S01 cond-only with Fortran (loaded), TFL, TFL_JIT, PPM, PPM_JIT.
Produces:
  1. Size distribution comparison at hours 0,1,2,3,4,5,6
  2. Timing table
  3. Conservation table (Nk, M_dry, M_dry+Gc)

Usage:
    source .venv/bin/activate
    python -m benchmarks.python.compare_methods
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS_LEGACY as NBINS, XK0_LEGACY,
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO
)
from tomas_jax.solvers.condensation import (
    condensation_step,
    condensation_step_jit,
    condensation_step_tfl_jit,
    run_condensation_scan,
    run_condensation_scan_tfl,
)
from tomas_jax.solvers.diffrax import diffrax_step
from benchmarks.python.scenarios import get_scenarios

# =========================================================================
# Constants
# =========================================================================
BOXVOL = 1.0e6
DENS_INIT = 1770.0
PI_F = 3.141592654
DT = 60.0
NSTEPS = 1440
SNAPSHOT_HOURS = [0, 1, 2, 3, 4, 5, 6]

FORTRAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'fortran', 'output', '24h')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', '24h')


def init_lognormal(N_total, GMD_um, GSD, xk_np):
    """Initialize lognormal matching Fortran harness."""
    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))
    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI_F)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI_F)) ** 0.3333
        Dk = np.sqrt(Dl * Dh)
        np_init = ((N_total * BOXVOL)
                   / (np.sqrt(2.0 * PI_F) * Dk * np.log(GSD))
                   * np.exp(-(np.log(Dk / GMD_um) ** 2
                              / (2.0 * np.log(GSD) ** 2)))
                   * (Dh - Dl))
        Nk[k] = np_init
        Mk[k, SRTSO4] = np_init * np.sqrt(xk_np[k]) * np.sqrt(xk_np[k + 1])
    for k in range(NBINS):
        if Nk[k] < 1e-3:
            Nk[k] = 1e-3
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = 1e-3 * 1.4 * xk_np[k]
    return Nk, Mk


def load_fortran(scenario_id, mode='cond'):
    """Load Fortran hourly snapshots."""
    hourly = {}
    for h in range(1, 25):
        prefix = f"s{scenario_id:02d}_{mode}_hour{h:02d}"
        nk_path = os.path.join(FORTRAN_DIR, f"{prefix}_Nk.csv")
        mk_path = os.path.join(FORTRAN_DIR, f"{prefix}_Mk.csv")
        gc_path = os.path.join(FORTRAN_DIR, f"{prefix}_Gc.csv")
        if not os.path.exists(nk_path):
            continue
        Nk = np.loadtxt(nk_path)
        Mk = np.loadtxt(mk_path, delimiter=',')
        Gc = np.loadtxt(gc_path)
        hourly[h] = {'Nk': Nk, 'Mk': Mk, 'Gc': Gc}
    return hourly


def run_jax_method(method, Nk0, Mk0, Gc0, xk, scenario, snapshot_hours):
    """Run a JAX method and return hourly snapshots + timing."""
    Nk = jnp.array(Nk0)
    Mk = jnp.array(Mk0)
    Gc = jnp.array(Gc0)
    xk_j = jnp.array(xk)

    temp = scenario['temp']
    pres = scenario['pres']
    rh = scenario['RH']
    alpha = 1.0
    prod_rate = scenario['h2so4_prod_kg_per_s']

    # JIT warmup
    if method == 'ppm_jit':
        _ = condensation_step_jit(
            Nk, Mk, Gc, xk_j,
            jnp.array(temp), jnp.array(pres),
            jnp.array(BOXVOL), jnp.array(rh),
            jnp.array(alpha), jnp.array(DT)
        )
    elif method == 'tfl_jit':
        _ = condensation_step_tfl_jit(
            Nk, Mk, Gc, xk_j,
            jnp.array(temp), jnp.array(pres),
            jnp.array(BOXVOL), jnp.array(rh),
            jnp.array(alpha), jnp.array(DT)
        )

    snapshots = {}
    # Hour 0
    snapshots[0] = {
        'Nk': np.array(Nk), 'Mk': np.array(Mk), 'Gc': np.array(Gc)
    }

    t_start = time.perf_counter()

    for istep in range(NSTEPS):
        # Add production
        Gc = Gc.at[SRTSO4].add(prod_rate * DT)

        # Condensation
        if method == 'ppm_jit':
            Nk, Mk, Gc = condensation_step_jit(
                Nk, Mk, Gc, xk_j,
                jnp.array(temp), jnp.array(pres),
                jnp.array(BOXVOL), jnp.array(rh),
                jnp.array(alpha), jnp.array(DT)
            )
        elif method == 'tfl_jit':
            Nk, Mk, Gc = condensation_step_tfl_jit(
                Nk, Mk, Gc, xk_j,
                jnp.array(temp), jnp.array(pres),
                jnp.array(BOXVOL), jnp.array(rh),
                jnp.array(alpha), jnp.array(DT)
            )
        else:
            Nk, Mk, Gc = condensation_step(
                Nk, Mk, Gc, xk_j,
                temp, pres, BOXVOL, rh, alpha, DT,
                method=method
            )

        hour = (istep + 1) // 60
        if (istep + 1) % 60 == 0 and hour in snapshot_hours:
            # Force sync for timing
            Nk.block_until_ready()
            snapshots[hour] = {
                'Nk': np.array(Nk), 'Mk': np.array(Mk), 'Gc': np.array(Gc)
            }

    Nk.block_until_ready()
    wall_time = time.perf_counter() - t_start

    return snapshots, wall_time


def compute_conservation(snapshots, hour, Nk0, Mk0, Gc0):
    """Compute conservation metrics at a given hour."""
    snap = snapshots.get(hour)
    if snap is None:
        return None
    Nk = snap['Nk']
    Mk = snap['Mk']
    Gc = snap['Gc']

    N_tot = np.sum(Nk)
    N_tot0 = np.sum(Nk0)
    M_dry = np.sum(Mk[:, :SRTH2O])
    M_dry0 = np.sum(Mk0[:, :SRTH2O])
    G_tot = np.sum(Gc)
    G_tot0 = np.sum(Gc0)

    return {
        'N_ratio': N_tot / max(N_tot0, 1e-30),
        'M_dry': M_dry,
        'M_dry_plus_G': M_dry + G_tot,
        'M_dry_plus_G_init': M_dry0 + G_tot0,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get scenario 1
    scenarios = get_scenarios(n=50, seed=42)
    scenario = scenarios[0]
    sid = scenario['scenario_id']
    print(f"Scenario {sid}: N={scenario['N_total']:.1f}/cm3, "
          f"GMD={scenario['GMD_um']:.4f}um, GSD={scenario['GSD']:.2f}")

    # Init bin boundaries
    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = XK0_LEGACY
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]

    # Init state
    Nk0, Mk0 = init_lognormal(
        scenario['N_total'], scenario['GMD_um'], scenario['GSD'], xk_np
    )
    Gc0 = np.zeros(N_GAS_SPECIES)
    Gc0[SRTSO4] = scenario['Gc_SO4_kg']

    # =====================================================================
    # Load Fortran
    # =====================================================================
    print("\nLoading Fortran results...")
    fortran_hourly = load_fortran(sid, 'cond')
    fortran_timing_path = os.path.join(FORTRAN_DIR, 'timing_fortran.csv')
    fortran_time = None
    if os.path.exists(fortran_timing_path):
        import csv
        with open(fortran_timing_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['scenario_id'].strip()) == sid:
                    fortran_time = float(row['cond_s'].strip())
                    break
    print(f"  Fortran: {len(fortran_hourly)} hours loaded, "
          f"timing={'%.3fs' % fortran_time if fortran_time else 'N/A'}")

    # Fortran snapshots (0-indexed hours)
    fortran_snaps = {0: {'Nk': Nk0.copy(), 'Mk': Mk0.copy(), 'Gc': Gc0.copy()}}
    for h, data in fortran_hourly.items():
        fortran_snaps[h] = data

    # =====================================================================
    # Run JAX methods
    # =====================================================================
    methods = ['tfl', 'tfl_jit', 'ppm', 'ppm_jit']
    results = {}

    for method in methods:
        print(f"\nRunning {method.upper()}...")
        snaps, wt = run_jax_method(
            method, Nk0, Mk0, Gc0, xk_np, scenario, SNAPSHOT_HOURS
        )
        results[method] = {'snapshots': snaps, 'wall_time': wt}
        print(f"  Wall time: {wt:.3f}s")

    # =====================================================================
    # Timing Table
    # =====================================================================
    print("\n" + "=" * 60)
    print("TIMING COMPARISON (S01 cond-only, 24h)")
    print("=" * 60)
    print(f"{'Method':<15} {'Wall Time [s]':>15} {'Speedup vs PPM':>18}")
    print("-" * 50)

    ppm_time = results['ppm']['wall_time']

    if fortran_time:
        print(f"{'Fortran':<15} {fortran_time:>15.3f} {ppm_time/fortran_time:>17.1f}x")
    for method in methods:
        wt = results[method]['wall_time']
        speedup = ppm_time / wt if wt > 0 else float('inf')
        print(f"{method.upper():<15} {wt:>15.3f} {speedup:>17.1f}x")

    # =====================================================================
    # Conservation Table
    # =====================================================================
    # Compute total production added over time
    prod_rate = scenario['h2so4_prod_kg_per_s']

    print("\n" + "=" * 60)
    print("CONSERVATION (hours 0-6)")
    print("=" * 60)

    all_sources = {'Fortran': fortran_snaps}
    for m in methods:
        all_sources[m.upper()] = results[m]['snapshots']

    # Header
    print(f"\n{'Method':<12} {'Hour':>4} {'N_ratio':>10} {'M_dry [kg]':>14} "
          f"{'M_dry+Gc [kg]':>14} {'Cons. Err':>12}")
    print("-" * 70)

    for name, snaps in all_sources.items():
        for h in SNAPSHOT_HOURS:
            if h not in snaps:
                continue
            cons = compute_conservation(snaps, h, Nk0, Mk0, Gc0)
            if cons is None:
                continue
            # Expected total = init + produced
            total_produced = prod_rate * DT * h * 60  # h hours of production
            expected = cons['M_dry_plus_G_init'] + total_produced
            actual = cons['M_dry_plus_G']
            err = (actual - expected) / max(abs(expected), 1e-30) if h > 0 else 0.0

            print(f"{name:<12} {h:>4d} {cons['N_ratio']:>10.6f} "
                  f"{cons['M_dry']:>14.6e} {cons['M_dry_plus_G']:>14.6e} "
                  f"{err:>12.2e}")

    # =====================================================================
    # Size Distribution Plots
    # =====================================================================
    print("\nGenerating size distribution plots...")

    # Compute diameters for x-axis
    Dp_mid = np.zeros(NBINS)
    for k in range(NBINS):
        m_mid = np.sqrt(xk_np[k] * xk_np[k + 1])
        Dp_mid[k] = 1e9 * ((6.0 * m_mid) / (DENS_INIT * np.pi)) ** (1.0 / 3.0)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'S01 Cond-Only: Size Distribution Comparison\n'
                 f'N={scenario["N_total"]:.0f}/cm3, GMD={scenario["GMD_um"]:.3f}um, '
                 f'GSD={scenario["GSD"]:.2f}', fontsize=14)

    colors = {'Fortran': 'black', 'TFL': 'blue', 'TFL_JIT': 'green', 'PPM': 'orange', 'PPM_JIT': 'red'}
    styles = {'Fortran': '-', 'TFL': '--', 'TFL_JIT': '-.', 'PPM': '-.', 'PPM_JIT': ':'}

    for i, hour in enumerate(SNAPSHOT_HOURS):
        row, col = divmod(i, 4)
        ax = axes[row, col]

        for name, snaps in all_sources.items():
            label = name
            if hour not in snaps:
                continue
            Nk_h = snaps[hour]['Nk']
            # dN/dlogDp
            dlogDp = np.log10(xk_np[1:] / xk_np[:-1]) / 3.0
            dNdlogDp = Nk_h / BOXVOL / dlogDp

            ax.plot(Dp_mid, dNdlogDp,
                    color=colors.get(label, 'gray'),
                    linestyle=styles.get(label, '-'),
                    linewidth=2 if label == 'Fortran' else 1.5,
                    label=label)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-2, 1e8)
        ax.set_xlabel('Dp [nm]')
        ax.set_ylabel('dN/dlogDp [#/cm3]')
        ax.set_title(f'Hour {hour}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    # Last subplot: Mass evolution
    ax = axes[1, 3]
    for name, snaps in all_sources.items():
        hours_avail = sorted([h for h in snaps if h <= 6])
        if not hours_avail:
            continue
        M_dry_series = []
        for h in hours_avail:
            M_dry_series.append(np.sum(snaps[h]['Mk'][:, :SRTH2O]))
        ax.plot(hours_avail, M_dry_series,
                color=colors.get(name, 'gray'),
                linestyle=styles.get(name, '-'),
                linewidth=2 if name == 'Fortran' else 1.5,
                label=name, marker='o', markersize=3)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Total Dry Mass [kg]')
    ax.set_title('Dry Mass Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'method_comparison_s01.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()

    # =====================================================================
    # Conservation plot
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('S01 Cond-Only: Conservation Metrics', fontsize=14)

    for name, snaps in all_sources.items():
        hours_avail = sorted([h for h in snaps if h <= 24])
        N_ratios = []
        M_dry_vals = []
        total_mass_err = []
        for h in hours_avail:
            cons = compute_conservation(snaps, h, Nk0, Mk0, Gc0)
            if cons:
                N_ratios.append(cons['N_ratio'])
                M_dry_vals.append(cons['M_dry'])
                total_produced = prod_rate * DT * h * 60
                expected = cons['M_dry_plus_G_init'] + total_produced
                err = (cons['M_dry_plus_G'] - expected) / max(abs(expected), 1e-30) if h > 0 else 0.0
                total_mass_err.append(err)

        c = colors.get(name, 'gray')
        ls = styles.get(name, '-')
        lw = 2 if name == 'Fortran' else 1.5

        axes[0].plot(hours_avail, N_ratios, color=c, linestyle=ls, linewidth=lw,
                     label=name, marker='o', markersize=3)
        axes[1].plot(hours_avail, M_dry_vals, color=c, linestyle=ls, linewidth=lw,
                     label=name, marker='o', markersize=3)
        axes[2].plot(hours_avail, total_mass_err, color=c, linestyle=ls, linewidth=lw,
                     label=name, marker='o', markersize=3)

    axes[0].set_ylabel('N(t)/N(0)')
    axes[0].set_title('Number Conservation')
    axes[0].axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylabel('M_dry [kg]')
    axes[1].set_title('Dry Mass')
    axes[2].set_ylabel('Relative Error')
    axes[2].set_title('Total Mass Conservation')
    axes[2].axhline(0.0, color='gray', linestyle=':', alpha=0.5)

    for ax in axes:
        ax.set_xlabel('Hour')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    cons_path = os.path.join(OUTPUT_DIR, 'method_conservation_s01.png')
    plt.savefig(cons_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {cons_path}")
    plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
