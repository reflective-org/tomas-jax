"""Plot: Fortran vs JAX combined (coag+cond) size distributions at multiple hours.

Runs coagulation (Forward Euler JIT) + condensation (TFL_JIT) in operator-split
mode, compares with Fortran combined-mode output.
"""
import os
import sys
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, xk_boundaries
)
from tomas_jax.solvers.condensation import condensation_step_tfl_jit
from tomas_jax.solvers.coagulation_jax import coagulation_step_jit
from benchmarks.python.scenarios import get_scenarios

BOXVOL = 1.0e6
DENS_INIT = 1770.0
PI_F = 3.141592654
DT = 60.0
FORTRAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'fortran', 'output', '24h')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def init_lognormal(N_total, GMD_um, GSD, xk_np):
    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))
    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI_F)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI_F)) ** 0.3333
        Dk = np.sqrt(Dl * Dh)
        n = ((N_total * BOXVOL) / (np.sqrt(2.0 * PI_F) * Dk * np.log(GSD))
             * np.exp(-(np.log(Dk / GMD_um) ** 2 / (2.0 * np.log(GSD) ** 2))) * (Dh - Dl))
        Nk[k] = n
        Mk[k, SRTSO4] = n * np.sqrt(xk_np[k]) * np.sqrt(xk_np[k + 1])
    for k in range(NBINS):
        if Nk[k] < 1e-3:
            Nk[k] = 1e-3
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = 1e-3 * 1.4 * xk_np[k]
    return Nk, Mk


def run_jax_combined_24h(sc, xk_j, xk_np):
    """Run 24h JAX combined (coag+cond) simulation, returning hourly snapshots."""
    Nk0, Mk0 = init_lognormal(sc['N_total'], sc['GMD_um'], sc['GSD'], xk_np)
    Gc0 = np.zeros(N_GAS_SPECIES)
    Gc0[SRTSO4] = sc['Gc_SO4_kg']

    Nk = jnp.array(Nk0)
    Mk = jnp.array(Mk0)
    Gc = jnp.array(Gc0)

    temp_j = jnp.array(sc['temp'])
    pres_j = jnp.array(sc['pres'])
    boxvol_j = jnp.array(BOXVOL)
    rh_j = jnp.array(sc['RH'])
    alpha_j = jnp.array(1.0)
    dt_j = jnp.array(DT)

    # JIT warmup — coagulation
    _ = coagulation_step_jit(Nk, Mk, xk_j, temp_j, pres_j, boxvol_j, dt_j)
    # JIT warmup — condensation
    _ = condensation_step_tfl_jit(Nk, Mk, Gc, xk_j, temp_j, pres_j, boxvol_j, rh_j, alpha_j, dt_j)

    # Fresh start for timed run
    Nk = jnp.array(Nk0)
    Mk = jnp.array(Mk0)
    Gc = jnp.array(Gc0)

    hourly = {}
    t0 = time.perf_counter()
    for step in range(1440):
        # 1. Coagulation
        Nk, Mk = coagulation_step_jit(Nk, Mk, xk_j, temp_j, pres_j, boxvol_j, dt_j)

        # 2. Condensation (add H2SO4 production, then condense)
        Gc = Gc.at[SRTSO4].add(sc['h2so4_prod_kg_per_s'] * DT)
        Nk, Mk, Gc = condensation_step_tfl_jit(
            Nk, Mk, Gc, xk_j, temp_j, pres_j, boxvol_j, rh_j, alpha_j, dt_j)

        if (step + 1) % 60 == 0:
            h = (step + 1) // 60
            Nk_np = np.array(Nk)
            Mk_np = np.array(Mk)
            hourly[h] = {'Nk': Nk_np, 'Mk': Mk_np}

    jax.block_until_ready(Nk)
    wall_time = time.perf_counter() - t0
    return hourly, wall_time, Nk0, Mk0


def load_fortran_hourly(sid, mode='combined'):
    hourly = {}
    for h in range(1, 25):
        nk_path = os.path.join(FORTRAN_DIR, f"s{sid:02d}_{mode}_hour{h:02d}_Nk.csv")
        mk_path = os.path.join(FORTRAN_DIR, f"s{sid:02d}_{mode}_hour{h:02d}_Mk.csv")
        if os.path.exists(nk_path) and os.path.exists(mk_path):
            hourly[h] = {
                'Nk': np.loadtxt(nk_path),
                'Mk': np.loadtxt(mk_path, delimiter=',')
            }
    return hourly


def bin_diameters(xk_np):
    dp = np.zeros(NBINS)
    for k in range(NBINS):
        mass_geo = np.sqrt(xk_np[k] * xk_np[k + 1])
        dp[k] = 1e6 * ((6.0 * mass_geo) / (DENS_INIT * PI_F)) ** (1.0/3.0)
    return dp


def main():
    scenarios = get_scenarios(n=50, seed=42)
    xk_np = np.array(xk_boundaries())
    xk_j = jnp.array(xk_np)
    dp = bin_diameters(xk_np)

    test_cases = [
        (20, 'S20: N=73923, GMD=0.060um, GSD=2.29'),
        (10, 'S10: N=1354, GMD=0.118um, GSD=1.46'),
        (39, 'S39: N=37273, GMD=0.032um, GSD=2.15'),
    ]

    plot_hours = [1, 4, 8, 12, 18, 24]

    # Load Fortran timing
    timing_path = os.path.join(FORTRAN_DIR, 'timing_fortran.csv')
    fortran_times = {}
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row['scenario_id'])
                fortran_times[sid] = float(row['combined_s'])

    # =====================================================================
    # Figure 1: Multi-hour size distribution comparison (combined mode)
    # =====================================================================
    fig, axes = plt.subplots(len(test_cases), len(plot_hours),
                             figsize=(24, 4 * len(test_cases)),
                             sharex=True)

    jax_times = {}

    for row, (sid, label) in enumerate(test_cases):
        sc = scenarios[sid - 1]
        print(f"Running combined {label}...")

        jax_hourly, wall_time, Nk0, Mk0 = run_jax_combined_24h(sc, xk_j, xk_np)
        jax_times[sid] = wall_time

        fort_hourly = load_fortran_hourly(sid, mode='combined')

        for col, h in enumerate(plot_hours):
            ax = axes[row, col]

            # Initial
            ax.semilogy(dp, Nk0 / BOXVOL, 'k--', alpha=0.3, linewidth=0.8,
                       label='Initial' if col == 0 else '')

            # Fortran
            if h in fort_hourly:
                Nk_f = fort_hourly[h]['Nk']
                ax.semilogy(dp, Nk_f / BOXVOL, 'k-', linewidth=2.0,
                           label='Fortran' if col == 0 else '')

            # JAX combined
            if h in jax_hourly:
                Nk_j = jax_hourly[h]['Nk']
                ax.semilogy(dp, Nk_j / BOXVOL, 'b--', linewidth=1.5,
                           label='JAX (coag+cond)' if col == 0 else '')

            ax.set_xlim(0.005, 5.0)
            ax.set_ylim(1e-2, 1e8)
            ax.set_xscale('log')
            if col == 0:
                ax.set_ylabel(f'{label.split(":")[0]}\ndN/dlogDp [#/cm$^3$]')
            if row == 0:
                ax.set_title(f'Hour {h}', fontsize=12, fontweight='bold')
            if row == len(test_cases) - 1:
                ax.set_xlabel('Dp [um]')
            ax.grid(True, alpha=0.3)

        axes[row, 0].legend(fontsize=8, loc='upper right')

    # Timing annotation
    timing_text = "Timing (24h combined coag+cond):\n"
    for sid, label in test_cases:
        f_time = fortran_times.get(sid, 0)
        j_time = jax_times.get(sid, 0)
        speedup = f_time / j_time if j_time > 0 else 0
        timing_text += f"  {label.split(':')[0]}: Fortran={f_time:.3f}s, JAX={j_time:.3f}s ({speedup:.2f}x)\n"

    fig.text(0.02, 0.01, timing_text, fontsize=9, fontfamily='monospace',
             verticalalignment='bottom')

    fig.suptitle('Fortran vs JAX Combined (Coag+Cond TFL_JIT) Size Distributions',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    out_path = os.path.join(RESULTS_DIR, 'combined_sizedist_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    # =====================================================================
    # Figure 2: Error evolution over 24 hours
    # =====================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (sid, label) in enumerate(test_cases):
        sc = scenarios[sid - 1]
        fort_hourly = load_fortran_hourly(sid, mode='combined')

        # Re-run JAX to get fresh hourly data
        jax_hourly, _, Nk0, _ = run_jax_combined_24h(sc, xk_j, xk_np)

        hours = []
        n_err = []
        peak_err = []

        for h in range(1, 25):
            if h not in fort_hourly or h not in jax_hourly:
                continue
            Nk_f = fort_hourly[h]['Nk']
            Nk_j = jax_hourly[h]['Nk']

            N_f = np.sum(Nk_f)
            N_j = np.sum(Nk_j)
            n_err.append(abs(N_j - N_f) / max(N_f, 1e-30))

            peak = np.argmax(Nk_f)
            lo = max(0, peak - 3)
            hi = min(NBINS, peak + 4)
            max_rel = np.max(np.abs(Nk_j[lo:hi] - Nk_f[lo:hi])) / max(np.max(Nk_f[lo:hi]), 1e-30)
            peak_err.append(max_rel)

            hours.append(h)

        ax = axes2[idx]
        ax.semilogy(hours, n_err, 'b-o', markersize=4, label='N_tot rel error')
        ax.semilogy(hours, peak_err, 'r-s', markersize=4, label='Peak-region max rel error')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Relative Error')
        ax.set_title(label.split(':')[0] + ' Error Evolution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-7, 10)
        ax.set_xlim(0, 25)

    fig2.suptitle('Error Evolution: JAX Combined vs Fortran',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path2 = os.path.join(RESULTS_DIR, 'combined_error_evolution.png')
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path2}")
    plt.close(fig2)

    # =====================================================================
    # Figure 3: Timing bar chart
    # =====================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    sids = [sid for sid, _ in test_cases]
    labels = [label.split(':')[0] for _, label in test_cases]
    fort_t = [fortran_times.get(sid, 0) for sid in sids]
    jax_t = [jax_times.get(sid, 0) for sid in sids]

    x = np.arange(len(sids))
    w = 0.35
    ax3.bar(x - w/2, fort_t, w, label='Fortran', color='#333333')
    ax3.bar(x + w/2, jax_t, w, label='JAX (coag+cond)', color='#4488cc')

    for i in range(len(sids)):
        if jax_t[i] > 0 and fort_t[i] > 0:
            ratio = fort_t[i] / jax_t[i]
            max_h = max(fort_t[i], jax_t[i])
            ax3.text(x[i], max_h * 1.05, f'{ratio:.2f}x', ha='center', fontsize=10, fontweight='bold')

    ax3.set_ylabel('Wall-clock time [s]')
    ax3.set_title('24h Combined (Coag+Cond): Fortran vs JAX', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out_path3 = os.path.join(RESULTS_DIR, 'combined_timing.png')
    fig3.savefig(out_path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path3}")
    plt.close(fig3)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Combined Mode (Coag + Cond TFL_JIT)")
    print("=" * 60)
    for sid, label in test_cases:
        f_time = fortran_times.get(sid, 0)
        j_time = jax_times.get(sid, 0)
        speedup = f_time / j_time if j_time > 0 else 0
        print(f"  {label}")
        print(f"    Fortran: {f_time:.3f}s, JAX: {j_time:.3f}s")
        print(f"    Ratio: {speedup:.2f}x {'(JAX faster)' if speedup > 1 else '(Fortran faster)'}")


if __name__ == '__main__':
    main()
