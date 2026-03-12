"""Comprehensive Fortran vs JAX (Euler+TFL_JIT) comparison across all scenarios.

Runs 49 scenarios x 2 modes (cond_only, combined) and generates:
  1. Size distribution comparison (representative scenarios, hour 24)
  2. N conservation scatter (Fortran vs JAX, all scenarios)
  3. Mass conservation error (all scenarios)
  4. Error evolution over time
  5. Timing comparison
  6. Per-bin error heatmaps

Usage:
    source .venv/bin/activate
    python -m benchmarks.python.compare_fortran_jax
"""
import os
import sys
import time
import csv
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
from tomas_jax.solvers.condensation import condensation_step_tfl_jit
from tomas_jax.solvers.coagulation_jax import coagulation_step_jit
from benchmarks.python.scenarios import get_scenarios

# Constants
BOXVOL = 1.0e6
DENS_INIT = 1770.0
PI_F = 3.141592654
DT = 60.0
NSTEPS = 1440
NHOURS = 24

FORTRAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'fortran', 'output', '24h')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', '24h')


def init_lognormal(N_total, GMD_um, GSD, xk_np):
    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))
    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI_F)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI_F)) ** 0.3333
        Dk = np.sqrt(Dl * Dh)
        n = ((N_total * BOXVOL)
             / (np.sqrt(2.0 * PI_F) * Dk * np.log(GSD))
             * np.exp(-(np.log(Dk / GMD_um) ** 2 / (2.0 * np.log(GSD) ** 2)))
             * (Dh - Dl))
        Nk[k] = n
        Mk[k, SRTSO4] = n * np.sqrt(xk_np[k]) * np.sqrt(xk_np[k + 1])
    for k in range(NBINS):
        if Nk[k] < 1e-3:
            Nk[k] = 1e-3
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = 1e-3 * 1.4 * xk_np[k]
    return Nk, Mk


def load_fortran_hourly(scenario_id, mode='cond'):
    hourly = {}
    for h in range(1, 25):
        prefix = f"s{scenario_id:02d}_{mode}_hour{h:02d}"
        nk_path = os.path.join(FORTRAN_DIR, f"{prefix}_Nk.csv")
        mk_path = os.path.join(FORTRAN_DIR, f"{prefix}_Mk.csv")
        gc_path = os.path.join(FORTRAN_DIR, f"{prefix}_Gc.csv")
        if not os.path.exists(nk_path):
            continue
        hourly[h] = {
            'Nk': np.loadtxt(nk_path),
            'Mk': np.loadtxt(mk_path, delimiter=','),
            'Gc': np.loadtxt(gc_path),
        }
    return hourly


def load_fortran_timing():
    timing = {}
    path = os.path.join(FORTRAN_DIR, 'timing_fortran.csv')
    if not os.path.exists(path):
        return timing
    with open(path) as f:
        for row in csv.DictReader(f):
            sid = int(row['scenario_id'].strip())
            timing[sid] = {
                'coag': float(row['coag_s'].strip()),
                'cond': float(row['cond_s'].strip()),
                'combined': float(row['combined_s'].strip()),
            }
    return timing


def run_jax_scenario(mode, Nk0, Mk0, Gc0, xk_j, sc):
    """Run one scenario with JAX Euler+TFL_JIT, return hourly snapshots."""
    Nk = jnp.array(Nk0)
    Mk = jnp.array(Mk0)
    Gc = jnp.array(Gc0)
    temp = sc['temp']
    pres = sc['pres']
    rh = sc['RH']
    prod = sc['h2so4_prod_kg_per_s']

    hourly = {}

    t0 = time.perf_counter()
    for i in range(NSTEPS):
        # Coagulation (combined mode only)
        if mode == 'combined':
            Nk, Mk = coagulation_step_jit(
                Nk, Mk, xk_j,
                jnp.array(temp), jnp.array(pres),
                jnp.array(BOXVOL), jnp.array(DT))

        # Production + Condensation
        Gc = Gc.at[SRTSO4].add(prod * DT)
        Nk, Mk, Gc = condensation_step_tfl_jit(
            Nk, Mk, Gc, xk_j,
            jnp.array(temp), jnp.array(pres),
            jnp.array(BOXVOL), jnp.array(rh),
            jnp.array(1.0), jnp.array(DT))

        if (i + 1) % 60 == 0:
            h = (i + 1) // 60
            Nk.block_until_ready()
            hourly[h] = {
                'Nk': np.array(Nk), 'Mk': np.array(Mk), 'Gc': np.array(Gc)
            }

    Nk.block_until_ready()
    wall_time = time.perf_counter() - t0
    return hourly, wall_time


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenarios = get_scenarios(n=50, seed=42)
    # Use scenarios 1-49 (Fortran has these)
    scenarios = [s for s in scenarios if s['scenario_id'] <= 49]
    n_sc = len(scenarios)
    print(f"Running {n_sc} scenarios x 2 modes (cond_only, combined)")

    # Bin boundaries
    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = XK0_LEGACY
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk_j = jnp.array(xk_np)

    # Diameters for plotting
    Dp = np.array([1e9 * ((6 * np.sqrt(xk_np[k] * xk_np[k + 1])) / (DENS_INIT * np.pi)) ** (1.0 / 3.0)
                    for k in range(NBINS)])
    dlogDp = np.log10(xk_np[1:] / xk_np[:-1]) / 3.0

    # Load Fortran timing
    ft_timing = load_fortran_timing()

    # JIT warmup
    print("JIT warmup...")
    sc0 = scenarios[0]
    Nk0, Mk0 = init_lognormal(sc0['N_total'], sc0['GMD_um'], sc0['GSD'], xk_np)
    Gc0 = np.zeros(N_GAS_SPECIES)
    Gc0[SRTSO4] = sc0['Gc_SO4_kg']
    _ = coagulation_step_jit(jnp.array(Nk0), jnp.array(Mk0), xk_j,
                              jnp.array(sc0['temp']), jnp.array(sc0['pres']),
                              jnp.array(BOXVOL), jnp.array(DT))
    _ = condensation_step_tfl_jit(jnp.array(Nk0), jnp.array(Mk0), jnp.array(Gc0), xk_j,
                                   jnp.array(sc0['temp']), jnp.array(sc0['pres']),
                                   jnp.array(BOXVOL), jnp.array(sc0['RH']),
                                   jnp.array(1.0), jnp.array(DT))
    print("Warmup done.\n")

    # =========================================================================
    # Run all scenarios
    # =========================================================================
    modes = ['cond', 'combined']
    # Storage: results[mode][sid] = {fortran_hourly, jax_hourly, jax_time, ...}
    results = {m: {} for m in modes}

    for sc in scenarios:
        sid = sc['scenario_id']
        Nk0, Mk0 = init_lognormal(sc['N_total'], sc['GMD_um'], sc['GSD'], xk_np)
        Gc0 = np.zeros(N_GAS_SPECIES)
        Gc0[SRTSO4] = sc['Gc_SO4_kg']

        N0 = np.sum(Nk0)
        Mdry0 = np.sum(Mk0[:, :SRTH2O])
        G0 = np.sum(Gc0)
        total0 = Mdry0 + G0

        for mode in modes:
            # Load Fortran
            f_mode = mode
            f_hourly = load_fortran_hourly(sid, f_mode)
            if not f_hourly:
                continue

            # Run JAX
            jax_mode = 'combined' if mode == 'combined' else 'cond_only'
            j_hourly, j_time = run_jax_scenario(
                'combined' if mode == 'combined' else 'cond_only',
                Nk0, Mk0, Gc0, xk_j, sc)

            # Fortran timing
            ft = ft_timing.get(sid, {})
            ft_time = ft.get(mode, ft.get('cond', None))

            results[mode][sid] = {
                'fortran': f_hourly,
                'jax': j_hourly,
                'jax_time': j_time,
                'fortran_time': ft_time,
                'Nk0': Nk0, 'Mk0': Mk0, 'Gc0': Gc0,
                'N0': N0, 'Mdry0': Mdry0, 'total0': total0,
                'prod': sc['h2so4_prod_kg_per_s'],
                'scenario': sc,
            }

        print(f"  S{sid:02d} done (N={sc['N_total']:.0f}, GMD={sc['GMD_um']:.4f})")

    print(f"\nAll scenarios complete.")

    # =========================================================================
    # FIGURE 1: Size distributions at hour 24 (6 representative scenarios)
    # =========================================================================
    for mode in modes:
        mode_label = 'Cond-Only' if mode == 'cond' else 'Combined'
        sids_avail = sorted(results[mode].keys())
        # Pick 6 evenly spaced
        pick_idx = np.linspace(0, len(sids_avail) - 1, 6, dtype=int)
        pick_sids = [sids_avail[i] for i in pick_idx]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{mode_label}: Size Distribution at Hour 24 — Fortran vs JAX',
                     fontsize=14)

        for idx, sid in enumerate(pick_sids):
            ax = axes[idx // 3, idx % 3]
            r = results[mode][sid]
            sc = r['scenario']

            if 24 in r['fortran']:
                dN_f = r['fortran'][24]['Nk'] / BOXVOL / dlogDp
                ax.plot(Dp, dN_f, 'k-', lw=2.5, label='Fortran')
            if 24 in r['jax']:
                dN_j = r['jax'][24]['Nk'] / BOXVOL / dlogDp
                ax.plot(Dp, dN_j, 'g--', lw=2, label='JAX')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1, 5000)
            ax.set_ylim(1e-2, 1e8)
            ax.set_xlabel('Dp [nm]')
            ax.set_ylabel('dN/dlogDp [#/cm3]')
            ax.set_title(f'S{sid:02d}: N={sc["N_total"]:.0f}, '
                         f'GMD={sc["GMD_um"]:.3f}um', fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=9)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f'fig1_sizedist_h24_{mode}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # =========================================================================
    # FIGURE 2: N conservation scatter — N(24h)/N(0) Fortran vs JAX
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('N(24h)/N(0): Fortran vs JAX (Euler+TFL_JIT)', fontsize=14)

    for mi, mode in enumerate(modes):
        ax = axes[mi]
        mode_label = 'Cond-Only' if mode == 'cond' else 'Combined'
        f_ratios = []
        j_ratios = []

        for sid, r in sorted(results[mode].items()):
            if 24 not in r['fortran'] or 24 not in r['jax']:
                continue
            Nr_f = np.sum(r['fortran'][24]['Nk']) / r['N0']
            Nr_j = np.sum(r['jax'][24]['Nk']) / r['N0']
            f_ratios.append(Nr_f)
            j_ratios.append(Nr_j)

        f_ratios = np.array(f_ratios)
        j_ratios = np.array(j_ratios)

        ax.scatter(f_ratios, j_ratios, c='green', s=30, alpha=0.7, edgecolors='k', linewidths=0.5)
        lo = min(f_ratios.min(), j_ratios.min()) * 0.99
        hi = max(f_ratios.max(), j_ratios.max()) * 1.01
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('Fortran N(24h)/N(0)')
        ax.set_ylabel('JAX N(24h)/N(0)')
        ax.set_title(mode_label)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_N_conservation_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # =========================================================================
    # FIGURE 3: Mass conservation error — bar chart per scenario
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle('Total Mass Conservation Error at Hour 24 (|M_dry+Gc - Expected| / Expected)',
                 fontsize=14)

    for mi, mode in enumerate(modes):
        ax = axes[mi]
        mode_label = 'Cond-Only' if mode == 'cond' else 'Combined'

        sids_sorted = sorted(results[mode].keys())
        f_errs = []
        j_errs = []
        labels = []

        for sid in sids_sorted:
            r = results[mode][sid]
            if 24 not in r['fortran'] or 24 not in r['jax']:
                continue

            expected = r['total0'] + r['prod'] * DT * 24 * 60

            f_Md = np.sum(r['fortran'][24]['Mk'][:, :SRTH2O])
            f_Gc = np.sum(r['fortran'][24]['Gc'])
            f_err = abs((f_Md + f_Gc) - expected) / abs(expected)

            j_Md = np.sum(r['jax'][24]['Mk'][:, :SRTH2O])
            j_Gc = np.sum(r['jax'][24]['Gc'])
            j_err = abs((j_Md + j_Gc) - expected) / abs(expected)

            f_errs.append(f_err)
            j_errs.append(j_err)
            labels.append(f'S{sid:02d}')

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, f_errs, w, label='Fortran', color='black', alpha=0.7)
        ax.bar(x + w / 2, j_errs, w, label='JAX', color='green', alpha=0.7)
        ax.set_ylabel('Relative Error')
        ax.set_title(mode_label)
        ax.set_yscale('log')
        ax.set_xticks(x[::2])
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], rotation=45, fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_mass_conservation_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # =========================================================================
    # FIGURE 4: Error evolution over time (median across scenarios)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Error Evolution: Fortran vs JAX (median and IQR across scenarios)', fontsize=14)

    for mi, mode in enumerate(modes):
        mode_label = 'Cond-Only' if mode == 'cond' else 'Combined'

        # Collect per-hour metrics across all scenarios
        hours = list(range(1, 25))
        N_rel_err = np.full((len(results[mode]), 24), np.nan)
        M_rel_err = np.full((len(results[mode]), 24), np.nan)
        mass_cons_err = np.full((len(results[mode]), 24), np.nan)

        for si, (sid, r) in enumerate(sorted(results[mode].items())):
            for h in hours:
                if h not in r['fortran'] or h not in r['jax']:
                    continue
                N_f = np.sum(r['fortran'][h]['Nk'])
                N_j = np.sum(r['jax'][h]['Nk'])
                N_rel_err[si, h - 1] = abs(N_f - N_j) / max(abs(N_f), 1e-30)

                Md_f = np.sum(r['fortran'][h]['Mk'][:, :SRTH2O])
                Md_j = np.sum(r['jax'][h]['Mk'][:, :SRTH2O])
                M_rel_err[si, h - 1] = abs(Md_f - Md_j) / max(abs(Md_f), 1e-30)

                expected = r['total0'] + r['prod'] * DT * h * 60
                j_total = np.sum(r['jax'][h]['Mk'][:, :SRTH2O]) + np.sum(r['jax'][h]['Gc'])
                mass_cons_err[si, h - 1] = abs(j_total - expected) / abs(expected)

        for ci, (data, ylabel, title) in enumerate([
            (N_rel_err, 'Rel. Error', f'{mode_label}: N_tot |Fortran-JAX|/Fortran'),
            (M_rel_err, 'Rel. Error', f'{mode_label}: M_dry |Fortran-JAX|/Fortran'),
            (mass_cons_err, 'Rel. Error', f'{mode_label}: JAX Mass Conservation'),
        ]):
            ax = axes[mi, ci]
            median = np.nanmedian(data, axis=0)
            q25 = np.nanpercentile(data, 25, axis=0)
            q75 = np.nanpercentile(data, 75, axis=0)

            ax.plot(hours, median, 'g-', lw=2, label='Median')
            ax.fill_between(hours, q25, q75, alpha=0.2, color='green', label='IQR')
            ax.set_xlabel('Hour')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_error_evolution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # =========================================================================
    # FIGURE 5: Timing comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Wall-Clock Timing: Fortran vs JAX (Euler+TFL_JIT)', fontsize=14)

    for mi, mode in enumerate(modes):
        ax = axes[mi]
        mode_label = 'Cond-Only' if mode == 'cond' else 'Combined'

        f_times = []
        j_times = []
        for sid in sorted(results[mode].keys()):
            r = results[mode][sid]
            ft = r['fortran_time']
            if ft is None:
                continue
            f_times.append(ft)
            j_times.append(r['jax_time'])

        f_times = np.array(f_times)
        j_times = np.array(j_times)

        ax.scatter(f_times, j_times, c='green', s=30, alpha=0.7, edgecolors='k', linewidths=0.5)
        # 1:1 line
        hi = max(f_times.max(), j_times.max()) * 1.1
        ax.plot([0, hi], [0, hi], 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlabel('Fortran [s]')
        ax.set_ylabel('JAX [s]')
        ax.set_title(f'{mode_label}\nMedian ratio: {np.median(j_times/f_times):.1f}x')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig5_timing_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # =========================================================================
    # FIGURE 6: Per-bin error heatmap at hour 24
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Per-Bin Relative Error at Hour 24: |Fortran - JAX| / max(Fortran)', fontsize=14)

    for mi, mode in enumerate(modes):
        mode_label = 'Cond-Only' if mode == 'cond' else 'Combined'
        sids_sorted = sorted(results[mode].keys())

        Nk_err = np.full((len(sids_sorted), NBINS), np.nan)
        Mk_err = np.full((len(sids_sorted), NBINS), np.nan)

        for si, sid in enumerate(sids_sorted):
            r = results[mode][sid]
            if 24 not in r['fortran'] or 24 not in r['jax']:
                continue
            Nk_f = r['fortran'][24]['Nk']
            Nk_j = r['jax'][24]['Nk']
            Nk_max = np.max(Nk_f)
            if Nk_max > 0:
                Nk_err[si] = np.abs(Nk_f - Nk_j) / Nk_max

            Mk_f = np.sum(r['fortran'][24]['Mk'][:, :SRTH2O], axis=1)
            Mk_j = np.sum(r['jax'][24]['Mk'][:, :SRTH2O], axis=1)
            Mk_max = np.max(Mk_f)
            if Mk_max > 0:
                Mk_err[si] = np.abs(Mk_f - Mk_j) / Mk_max

        for ci, (data, title) in enumerate([
            (Nk_err, f'{mode_label}: Nk Error'),
            (Mk_err, f'{mode_label}: Mk_dry Error'),
        ]):
            ax = axes[mi, ci]
            im = ax.imshow(data, aspect='auto', cmap='hot_r',
                           vmin=0, vmax=np.nanpercentile(data, 95),
                           extent=[0.5, NBINS + 0.5, len(sids_sorted) + 0.5, 0.5])
            ax.set_xlabel('Bin')
            ax.set_ylabel('Scenario')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='Relative Error')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig6_perbin_error_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # =========================================================================
    # FIGURE 7: All 49 scenarios hour-24 size distributions (7x7 grid)
    # =========================================================================
    for mode in modes:
        mode_label = 'Cond-Only' if mode == 'cond' else 'Combined'
        sids_sorted = sorted(results[mode].keys())

        ncols = 7
        nrows = (len(sids_sorted) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
        fig.suptitle(f'{mode_label}: All Scenarios Hour 24 — Fortran (black) vs JAX (green)',
                     fontsize=14)

        for idx, sid in enumerate(sids_sorted):
            row, col = divmod(idx, ncols)
            ax = axes[row, col] if nrows > 1 else axes[col]
            r = results[mode][sid]

            if 24 in r['fortran']:
                ax.plot(Dp, r['fortran'][24]['Nk'] / BOXVOL / dlogDp, 'k-', lw=1.5)
            if 24 in r['jax']:
                ax.plot(Dp, r['jax'][24]['Nk'] / BOXVOL / dlogDp, 'g--', lw=1)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(1, 5000)
            ax.set_ylim(1e-2, 1e8)
            ax.set_title(f'S{sid:02d}', fontsize=8)
            ax.tick_params(labelsize=6)
            if row == nrows - 1:
                ax.set_xlabel('Dp [nm]', fontsize=7)
            if col == 0:
                ax.set_ylabel('dN/dlogDp', fontsize=7)

        # Hide empty subplots
        for idx in range(len(sids_sorted), nrows * ncols):
            row, col = divmod(idx, ncols)
            ax = axes[row, col] if nrows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f'fig7_all_scenarios_h24_{mode}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # =========================================================================
    # Summary stats
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for mode in modes:
        mode_label = 'Cond-Only' if mode == 'cond' else 'Combined'
        print(f"\n--- {mode_label} ({len(results[mode])} scenarios) ---")

        N_errs_24 = []
        M_errs_24 = []
        cons_errs_24 = []
        timing_ratios = []

        for sid, r in sorted(results[mode].items()):
            if 24 not in r['fortran'] or 24 not in r['jax']:
                continue

            N_f = np.sum(r['fortran'][24]['Nk'])
            N_j = np.sum(r['jax'][24]['Nk'])
            N_errs_24.append(abs(N_f - N_j) / max(abs(N_f), 1e-30))

            Md_f = np.sum(r['fortran'][24]['Mk'][:, :SRTH2O])
            Md_j = np.sum(r['jax'][24]['Mk'][:, :SRTH2O])
            M_errs_24.append(abs(Md_f - Md_j) / max(abs(Md_f), 1e-30))

            expected = r['total0'] + r['prod'] * DT * 24 * 60
            j_total = np.sum(r['jax'][24]['Mk'][:, :SRTH2O]) + np.sum(r['jax'][24]['Gc'])
            cons_errs_24.append(abs(j_total - expected) / abs(expected))

            if r['fortran_time']:
                timing_ratios.append(r['jax_time'] / r['fortran_time'])

        N_errs_24 = np.array(N_errs_24)
        M_errs_24 = np.array(M_errs_24)
        cons_errs_24 = np.array(cons_errs_24)
        timing_ratios = np.array(timing_ratios)

        print(f"  N_tot |F-J|/F at h24:  median={np.median(N_errs_24):.2e}, "
              f"max={np.max(N_errs_24):.2e}")
        print(f"  M_dry |F-J|/F at h24:  median={np.median(M_errs_24):.2e}, "
              f"max={np.max(M_errs_24):.2e}")
        print(f"  JAX mass cons at h24:  median={np.median(cons_errs_24):.2e}, "
              f"max={np.max(cons_errs_24):.2e}")
        print(f"  Timing JAX/Fortran:    median={np.median(timing_ratios):.1f}x, "
              f"range=[{np.min(timing_ratios):.1f}x, {np.max(timing_ratios):.1f}x]")

    print("\nDone!")


if __name__ == '__main__':
    main()
