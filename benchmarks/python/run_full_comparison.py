"""Full Fortran vs JAX comparison: 49 scenarios x 3 modes.

Runs coag_only, cond_only, combined using Forward Euler coagulation
and TFL_JIT condensation, then generates comprehensive plots.

Output goes to benchmarks/results/YYYY-MM-DD/

Usage:
    PYTHONPATH=. python benchmarks/python/run_full_comparison.py
"""
import os
import sys
import time
import csv
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
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
TODAY = datetime.date.today().strftime('%Y-%m-%d')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', TODAY)

# Mode config: (fortran_prefix, has_coag, has_cond, display_name)
MODES = {
    'coag':     ('coag',     True,  False, 'Coag-Only'),
    'cond':     ('cond',     False, True,  'Cond-Only'),
    'combined': ('combined', True,  True,  'Combined'),
}


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


def load_fortran_hourly(scenario_id, mode_prefix):
    hourly = {}
    for h in range(1, 25):
        prefix = f"s{scenario_id:02d}_{mode_prefix}_hour{h:02d}"
        nk_path = os.path.join(FORTRAN_DIR, f"{prefix}_Nk.csv")
        mk_path = os.path.join(FORTRAN_DIR, f"{prefix}_Mk.csv")
        gc_path = os.path.join(FORTRAN_DIR, f"{prefix}_Gc.csv")
        if not os.path.exists(nk_path):
            continue
        entry = {
            'Nk': np.loadtxt(nk_path),
            'Mk': np.loadtxt(mk_path, delimiter=','),
        }
        if os.path.exists(gc_path):
            entry['Gc'] = np.loadtxt(gc_path)
        hourly[h] = entry
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


def run_jax_scenario(mode_key, Nk0, Mk0, Gc0, xk_j, sc):
    """Run one scenario, return hourly snapshots and wall time."""
    has_coag = MODES[mode_key][1]
    has_cond = MODES[mode_key][2]

    Nk = jnp.array(Nk0)
    Mk = jnp.array(Mk0)
    Gc = jnp.array(Gc0)
    temp_j = jnp.array(sc['temp'])
    pres_j = jnp.array(sc['pres'])
    boxvol_j = jnp.array(BOXVOL)
    rh_j = jnp.array(sc['RH'])
    alpha_j = jnp.array(1.0)
    dt_j = jnp.array(DT)

    hourly = {}
    t0 = time.perf_counter()
    for i in range(NSTEPS):
        if has_coag:
            Nk, Mk = coagulation_step_jit(Nk, Mk, xk_j, temp_j, pres_j, boxvol_j, dt_j)

        if has_cond:
            Gc = Gc.at[SRTSO4].add(sc['h2so4_prod_kg_per_s'] * DT)
            Nk, Mk, Gc = condensation_step_tfl_jit(
                Nk, Mk, Gc, xk_j, temp_j, pres_j, boxvol_j, rh_j, alpha_j, dt_j)

        if (i + 1) % 60 == 0:
            h = (i + 1) // 60
            Nk.block_until_ready()
            hourly[h] = {'Nk': np.array(Nk), 'Mk': np.array(Mk), 'Gc': np.array(Gc)}

    Nk.block_until_ready()
    wall_time = time.perf_counter() - t0
    return hourly, wall_time


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenarios = get_scenarios(n=50, seed=42)
    scenarios = [s for s in scenarios if s['scenario_id'] <= 49]
    n_sc = len(scenarios)

    # Bin boundaries
    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = 1.6033e-23
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk_j = jnp.array(xk_np)

    # Diameters [nm] for plotting
    Dp = np.array([1e9 * ((6 * np.sqrt(xk_np[k] * xk_np[k + 1]))
                          / (DENS_INIT * np.pi)) ** (1.0 / 3.0) for k in range(NBINS)])
    dlogDp = np.log10(xk_np[1:] / xk_np[:-1]) / 3.0

    ft_timing = load_fortran_timing()

    # JIT warmup
    print("JIT warmup...")
    sc0 = scenarios[0]
    Nk0, Mk0 = init_lognormal(sc0['N_total'], sc0['GMD_um'], sc0['GSD'], xk_np)
    Gc0 = np.zeros(N_GAS_SPECIES); Gc0[SRTSO4] = sc0['Gc_SO4_kg']
    _ = coagulation_step_jit(jnp.array(Nk0), jnp.array(Mk0), xk_j,
                              jnp.array(sc0['temp']), jnp.array(sc0['pres']),
                              jnp.array(BOXVOL), jnp.array(DT))
    _ = condensation_step_tfl_jit(jnp.array(Nk0), jnp.array(Mk0), jnp.array(Gc0), xk_j,
                                   jnp.array(sc0['temp']), jnp.array(sc0['pres']),
                                   jnp.array(BOXVOL), jnp.array(sc0['RH']),
                                   jnp.array(1.0), jnp.array(DT))
    print("Warmup done.\n")

    # =========================================================================
    # Run all scenarios x all modes
    # =========================================================================
    mode_keys = list(MODES.keys())
    results = {m: {} for m in mode_keys}

    total_runs = n_sc * len(mode_keys)
    run_count = 0

    for sc in scenarios:
        sid = sc['scenario_id']
        Nk0, Mk0 = init_lognormal(sc['N_total'], sc['GMD_um'], sc['GSD'], xk_np)
        Gc0 = np.zeros(N_GAS_SPECIES); Gc0[SRTSO4] = sc['Gc_SO4_kg']

        N0 = np.sum(Nk0)
        Mdry0 = np.sum(Mk0[:, :SRTH2O])
        G0 = np.sum(Gc0)

        for mode_key in mode_keys:
            fort_prefix = MODES[mode_key][0]
            f_hourly = load_fortran_hourly(sid, fort_prefix)
            if not f_hourly:
                run_count += 1
                continue

            j_hourly, j_time = run_jax_scenario(mode_key, Nk0, Mk0, Gc0, xk_j, sc)

            ft = ft_timing.get(sid, {})
            ft_time = ft.get(mode_key, ft.get(fort_prefix, None))

            results[mode_key][sid] = {
                'fortran': f_hourly,
                'jax': j_hourly,
                'jax_time': j_time,
                'fortran_time': ft_time,
                'Nk0': Nk0, 'Mk0': Mk0, 'Gc0': Gc0,
                'N0': N0, 'Mdry0': Mdry0, 'total0': Mdry0 + G0,
                'prod': sc['h2so4_prod_kg_per_s'],
                'scenario': sc,
            }
            run_count += 1

        print(f"  [{run_count}/{total_runs}] S{sid:02d} done "
              f"(N={sc['N_total']:.0f}, GMD={sc['GMD_um']:.4f})")

    print(f"\nAll {run_count} runs complete. Generating plots...\n")

    # =========================================================================
    # FIGURE 1: Size distributions at hour 24 (6 representative scenarios per mode)
    # =========================================================================
    for mode_key in mode_keys:
        mode_label = MODES[mode_key][3]
        sids_avail = sorted(results[mode_key].keys())
        if len(sids_avail) < 6:
            continue
        pick_idx = np.linspace(0, len(sids_avail) - 1, 6, dtype=int)
        pick_sids = [sids_avail[i] for i in pick_idx]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{mode_label}: Size Distribution at Hour 24 — Fortran vs JAX',
                     fontsize=14, fontweight='bold')

        for idx, sid in enumerate(pick_sids):
            ax = axes[idx // 3, idx % 3]
            r = results[mode_key][sid]
            sc = r['scenario']

            if 24 in r['fortran']:
                dN_f = r['fortran'][24]['Nk'] / BOXVOL / dlogDp
                ax.plot(Dp, dN_f, 'k-', lw=2.5, label='Fortran')
            if 24 in r['jax']:
                dN_j = r['jax'][24]['Nk'] / BOXVOL / dlogDp
                ax.plot(Dp, dN_j, 'b--', lw=2, label='JAX')

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(1, 5000); ax.set_ylim(1e-2, 1e8)
            ax.set_xlabel('Dp [nm]'); ax.set_ylabel('dN/dlogDp [#/cm3]')
            ax.set_title(f'S{sid:02d}: N={sc["N_total"]:.0f}, '
                         f'GMD={sc["GMD_um"]:.3f}um', fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=9)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f'fig1_sizedist_h24_{mode_key}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 2: N(24h)/N(0) conservation scatter
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('N(24h)/N(0): Fortran vs JAX', fontsize=14, fontweight='bold')

    for mi, mode_key in enumerate(mode_keys):
        ax = axes[mi]
        mode_label = MODES[mode_key][3]
        f_ratios, j_ratios = [], []

        for sid, r in sorted(results[mode_key].items()):
            if 24 not in r['fortran'] or 24 not in r['jax']:
                continue
            Nr_f = np.sum(r['fortran'][24]['Nk']) / r['N0']
            Nr_j = np.sum(r['jax'][24]['Nk']) / r['N0']
            f_ratios.append(Nr_f)
            j_ratios.append(Nr_j)

        f_ratios = np.array(f_ratios)
        j_ratios = np.array(j_ratios)

        ax.scatter(f_ratios, j_ratios, c='#4488cc', s=30, alpha=0.7, edgecolors='k', linewidths=0.5)
        lo = min(f_ratios.min(), j_ratios.min()) * 0.95
        hi = max(f_ratios.max(), j_ratios.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlabel('Fortran N(24h)/N(0)')
        ax.set_ylabel('JAX N(24h)/N(0)')
        ax.set_title(mode_label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_N_conservation_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 3: Mass conservation error bar chart
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle('Total Mass Conservation Error at Hour 24', fontsize=14, fontweight='bold')

    for mi, mode_key in enumerate(mode_keys):
        ax = axes[mi]
        mode_label = MODES[mode_key][3]
        has_cond = MODES[mode_key][2]
        sids_sorted = sorted(results[mode_key].keys())

        f_errs, j_errs, labels = [], [], []
        for sid in sids_sorted:
            r = results[mode_key][sid]
            if 24 not in r['fortran'] or 24 not in r['jax']:
                continue

            if has_cond:
                expected = r['total0'] + r['prod'] * DT * 24 * 60
            else:
                expected = r['Mdry0']  # coag-only: dry mass is conserved

            f_Md = np.sum(r['fortran'][24]['Mk'][:, :SRTH2O])
            f_Gc = np.sum(r['fortran'][24].get('Gc', np.zeros(1)))
            if has_cond:
                f_total = f_Md + f_Gc
            else:
                f_total = f_Md
            f_err = abs(f_total - expected) / abs(expected) if expected != 0 else 0

            j_Md = np.sum(r['jax'][24]['Mk'][:, :SRTH2O])
            j_Gc = np.sum(r['jax'][24].get('Gc', np.zeros(1)))
            if has_cond:
                j_total = j_Md + j_Gc
            else:
                j_total = j_Md
            j_err = abs(j_total - expected) / abs(expected) if expected != 0 else 0

            f_errs.append(max(f_err, 1e-16))
            j_errs.append(max(j_err, 1e-16))
            labels.append(f'S{sid:02d}')

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, f_errs, w, label='Fortran', color='black', alpha=0.7)
        ax.bar(x + w/2, j_errs, w, label='JAX', color='#4488cc', alpha=0.7)
        ax.set_ylabel('Relative Error')
        ax.set_title(mode_label)
        ax.set_yscale('log')
        ax.set_xticks(x[::2])
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], rotation=45, fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_mass_conservation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 4: Error evolution over time (N_tot, M_dry, mass conservation)
    # =========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Error Evolution: Fortran vs JAX (median and IQR across scenarios)',
                 fontsize=14, fontweight='bold')

    for mi, mode_key in enumerate(mode_keys):
        mode_label = MODES[mode_key][3]
        has_cond = MODES[mode_key][2]
        hours = list(range(1, 25))
        n_scenarios = len(results[mode_key])

        N_rel_err = np.full((n_scenarios, 24), np.nan)
        M_rel_err = np.full((n_scenarios, 24), np.nan)
        mass_cons_err = np.full((n_scenarios, 24), np.nan)

        for si, (sid, r) in enumerate(sorted(results[mode_key].items())):
            for h in hours:
                if h not in r['fortran'] or h not in r['jax']:
                    continue
                N_f = np.sum(r['fortran'][h]['Nk'])
                N_j = np.sum(r['jax'][h]['Nk'])
                N_rel_err[si, h-1] = abs(N_f - N_j) / max(abs(N_f), 1e-30)

                Md_f = np.sum(r['fortran'][h]['Mk'][:, :SRTH2O])
                Md_j = np.sum(r['jax'][h]['Mk'][:, :SRTH2O])
                M_rel_err[si, h-1] = abs(Md_f - Md_j) / max(abs(Md_f), 1e-30)

                if has_cond:
                    expected = r['total0'] + r['prod'] * DT * h * 60
                    j_total = np.sum(r['jax'][h]['Mk'][:, :SRTH2O]) + np.sum(r['jax'][h]['Gc'])
                else:
                    expected = r['Mdry0']
                    j_total = np.sum(r['jax'][h]['Mk'][:, :SRTH2O])
                mass_cons_err[si, h-1] = abs(j_total - expected) / max(abs(expected), 1e-30)

        for ci, (data, ylabel, title) in enumerate([
            (N_rel_err, 'Rel. Error', f'{mode_label}: N_tot |F-J|/F'),
            (M_rel_err, 'Rel. Error', f'{mode_label}: M_dry |F-J|/F'),
            (mass_cons_err, 'Rel. Error', f'{mode_label}: JAX Mass Conservation'),
        ]):
            ax = axes[mi, ci]
            median = np.nanmedian(data, axis=0)
            q25 = np.nanpercentile(data, 25, axis=0)
            q75 = np.nanpercentile(data, 75, axis=0)

            ax.plot(hours, median, 'b-', lw=2, label='Median')
            ax.fill_between(hours, q25, q75, alpha=0.2, color='#4488cc', label='IQR')
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
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 5: Timing comparison scatter
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Wall-Clock Timing: Fortran vs JAX (Euler+TFL_JIT)', fontsize=14, fontweight='bold')

    for mi, mode_key in enumerate(mode_keys):
        ax = axes[mi]
        mode_label = MODES[mode_key][3]

        f_times, j_times = [], []
        for sid in sorted(results[mode_key].keys()):
            r = results[mode_key][sid]
            ft = r['fortran_time']
            if ft is None:
                continue
            f_times.append(ft)
            j_times.append(r['jax_time'])

        f_times = np.array(f_times)
        j_times = np.array(j_times)

        ax.scatter(f_times, j_times, c='#4488cc', s=30, alpha=0.7, edgecolors='k', linewidths=0.5)
        hi = max(f_times.max(), j_times.max()) * 1.1
        ax.plot([0, hi], [0, hi], 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlabel('Fortran [s]')
        ax.set_ylabel('JAX [s]')
        median_ratio = np.median(j_times / f_times)
        ax.set_title(f'{mode_label}\nMedian JAX/Fortran: {median_ratio:.2f}x')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig5_timing_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 6: Per-bin error heatmap at hour 24
    # =========================================================================
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('Per-Bin Relative Error at Hour 24: |Fortran - JAX| / max(Fortran)',
                 fontsize=14, fontweight='bold')

    for mi, mode_key in enumerate(mode_keys):
        mode_label = MODES[mode_key][3]
        sids_sorted = sorted(results[mode_key].keys())

        Nk_err = np.full((len(sids_sorted), NBINS), np.nan)
        Mk_err = np.full((len(sids_sorted), NBINS), np.nan)

        for si, sid in enumerate(sids_sorted):
            r = results[mode_key][sid]
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
            vmax = np.nanpercentile(data, 95)
            if vmax == 0 or np.isnan(vmax):
                vmax = 1.0
            im = ax.imshow(data, aspect='auto', cmap='hot_r',
                           vmin=0, vmax=vmax,
                           extent=[0.5, NBINS+0.5, len(sids_sorted)+0.5, 0.5])
            ax.set_xlabel('Bin')
            ax.set_ylabel('Scenario')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='Relative Error')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig6_perbin_error_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 7: All scenarios hour-24 size distributions (7x7 grid, per mode)
    # =========================================================================
    for mode_key in mode_keys:
        mode_label = MODES[mode_key][3]
        sids_sorted = sorted(results[mode_key].keys())
        if not sids_sorted:
            continue

        ncols = 7
        nrows = (len(sids_sorted) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2.5*nrows))
        fig.suptitle(f'{mode_label}: All Scenarios Hour 24 — Fortran (black) vs JAX (blue)',
                     fontsize=14, fontweight='bold')

        for idx, sid in enumerate(sids_sorted):
            row, col = divmod(idx, ncols)
            ax = axes[row, col] if nrows > 1 else axes[col]
            r = results[mode_key][sid]

            if 24 in r['fortran']:
                ax.plot(Dp, r['fortran'][24]['Nk'] / BOXVOL / dlogDp, 'k-', lw=1.5)
            if 24 in r['jax']:
                ax.plot(Dp, r['jax'][24]['Nk'] / BOXVOL / dlogDp, 'b--', lw=1)

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(1, 5000); ax.set_ylim(1e-2, 1e8)
            ax.set_title(f'S{sid:02d}', fontsize=8)
            ax.tick_params(labelsize=6)
            if row == nrows - 1:
                ax.set_xlabel('Dp [nm]', fontsize=7)
            if col == 0:
                ax.set_ylabel('dN/dlogDp', fontsize=7)

        for idx in range(len(sids_sorted), nrows * ncols):
            row, col = divmod(idx, ncols)
            ax = axes[row, col] if nrows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f'fig7_all_scenarios_h24_{mode_key}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 8: N_tot and M_dry timeseries (all scenarios, normalized)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Total N and Dry Mass Timeseries (normalized by hour-0)',
                 fontsize=14, fontweight='bold')

    for mi, mode_key in enumerate(mode_keys):
        mode_label = MODES[mode_key][3]

        for si, (sid, r) in enumerate(sorted(results[mode_key].items())):
            hours_f, N_f_series = [], []
            hours_j, N_j_series = [], []
            hours_f_m, M_f_series = [], []
            hours_j_m, M_j_series = [], []

            for h in range(1, 25):
                if h in r['fortran']:
                    N_f_series.append(np.sum(r['fortran'][h]['Nk']) / r['N0'])
                    M_f_series.append(np.sum(r['fortran'][h]['Mk'][:, :SRTH2O]) / max(r['Mdry0'], 1e-30))
                    hours_f.append(h)
                    hours_f_m.append(h)
                if h in r['jax']:
                    N_j_series.append(np.sum(r['jax'][h]['Nk']) / r['N0'])
                    M_j_series.append(np.sum(r['jax'][h]['Mk'][:, :SRTH2O]) / max(r['Mdry0'], 1e-30))
                    hours_j.append(h)
                    hours_j_m.append(h)

            # N_tot
            ax = axes[0, mi]
            ax.plot(hours_f, N_f_series, 'k-', alpha=0.15, lw=0.5)
            ax.plot(hours_j, N_j_series, 'b--', alpha=0.15, lw=0.5)

            # M_dry
            ax = axes[1, mi]
            ax.plot(hours_f_m, M_f_series, 'k-', alpha=0.15, lw=0.5)
            ax.plot(hours_j_m, M_j_series, 'b--', alpha=0.15, lw=0.5)

        axes[0, mi].set_title(f'{mode_label}: N(t)/N(0)')
        axes[0, mi].set_ylabel('N(t)/N(0)')
        axes[0, mi].set_xlabel('Hour')
        axes[0, mi].set_yscale('log')
        axes[0, mi].grid(True, alpha=0.3)
        # Dummy lines for legend
        axes[0, mi].plot([], [], 'k-', lw=1.5, label='Fortran')
        axes[0, mi].plot([], [], 'b--', lw=1.5, label='JAX')
        axes[0, mi].legend(fontsize=8)

        axes[1, mi].set_title(f'{mode_label}: M_dry(t)/M_dry(0)')
        axes[1, mi].set_ylabel('M_dry(t)/M_dry(0)')
        axes[1, mi].set_xlabel('Hour')
        axes[1, mi].grid(True, alpha=0.3)
        axes[1, mi].plot([], [], 'k-', lw=1.5, label='Fortran')
        axes[1, mi].plot([], [], 'b--', lw=1.5, label='JAX')
        axes[1, mi].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig8_N_M_timeseries.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # Summary stats
    # =========================================================================
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append(f"SUMMARY STATISTICS — {TODAY}")
    summary_lines.append("=" * 70)

    for mode_key in mode_keys:
        mode_label = MODES[mode_key][3]
        has_cond = MODES[mode_key][2]
        n = len(results[mode_key])
        summary_lines.append(f"\n--- {mode_label} ({n} scenarios) ---")

        N_errs, M_errs, cons_errs = [], [], []
        timing_ratios = []

        for sid, r in sorted(results[mode_key].items()):
            if 24 not in r['fortran'] or 24 not in r['jax']:
                continue

            N_f = np.sum(r['fortran'][24]['Nk'])
            N_j = np.sum(r['jax'][24]['Nk'])
            N_errs.append(abs(N_f - N_j) / max(abs(N_f), 1e-30))

            Md_f = np.sum(r['fortran'][24]['Mk'][:, :SRTH2O])
            Md_j = np.sum(r['jax'][24]['Mk'][:, :SRTH2O])
            M_errs.append(abs(Md_f - Md_j) / max(abs(Md_f), 1e-30))

            if has_cond:
                expected = r['total0'] + r['prod'] * DT * 24 * 60
                j_total = np.sum(r['jax'][24]['Mk'][:, :SRTH2O]) + np.sum(r['jax'][24]['Gc'])
            else:
                expected = r['Mdry0']
                j_total = np.sum(r['jax'][24]['Mk'][:, :SRTH2O])
            cons_errs.append(abs(j_total - expected) / max(abs(expected), 1e-30))

            if r['fortran_time']:
                timing_ratios.append(r['jax_time'] / r['fortran_time'])

        N_errs = np.array(N_errs)
        M_errs = np.array(M_errs)
        cons_errs = np.array(cons_errs)
        timing_ratios = np.array(timing_ratios)

        summary_lines.append(f"  N_tot |F-J|/F at h24:  median={np.median(N_errs):.2e}, "
                             f"max={np.max(N_errs):.2e}")
        summary_lines.append(f"  M_dry |F-J|/F at h24:  median={np.median(M_errs):.2e}, "
                             f"max={np.max(M_errs):.2e}")
        summary_lines.append(f"  JAX mass cons at h24:  median={np.median(cons_errs):.2e}, "
                             f"max={np.max(cons_errs):.2e}")
        if len(timing_ratios) > 0:
            summary_lines.append(f"  Timing JAX/Fortran:    median={np.median(timing_ratios):.2f}x, "
                                 f"range=[{np.min(timing_ratios):.2f}x, {np.max(timing_ratios):.2f}x]")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save summary to file
    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text + "\n")
    print(f"\n  Saved: {summary_path}")
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
