"""PPM Analytical Mass-Weighted Flux Benchmark: 49 scenarios x 4 modes.

Compares Fortran with JAX using both TFL_JIT and PPM_JIT condensation,
plus coag-only, generating comprehensive plots.

Output: benchmarks/results/2026-03-02-ppm-analytical/

Usage:
    .venv/bin/python benchmarks/python/run_ppm_analytical_benchmark.py
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
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO
)
from tomas_jax.solvers.condensation import (
    condensation_step_tfl_jit, condensation_step_jit,
    run_condensation_scan, run_condensation_scan_tfl,
)
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', '2026-03-02-ppm-analytical')

# Mode config: (fortran_prefix, has_coag, has_cond, cond_method, display_name, color)
MODES = {
    'coag':         ('coag',     True,  False, None,       'Coag-Only',       '#2ca02c'),
    'cond_tfl':     ('cond',     False, True,  'tfl_jit',  'Cond TFL_JIT',    '#1f77b4'),
    'cond_ppm':     ('cond',     False, True,  'ppm_jit',  'Cond PPM_JIT',    '#ff7f0e'),
    'combined_tfl': ('combined', True,  True,  'tfl_jit',  'Combined TFL_JIT','#1f77b4'),
    'combined_ppm': ('combined', True,  True,  'ppm_jit',  'Combined PPM_JIT','#ff7f0e'),
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
    _, has_coag, has_cond, cond_method, _, _ = MODES[mode_key]

    Nk = jnp.array(Nk0)
    Mk = jnp.array(Mk0)
    Gc = jnp.array(Gc0)
    temp_j = jnp.array(sc['temp'])
    pres_j = jnp.array(sc['pres'])
    boxvol_j = jnp.array(BOXVOL)
    rh_j = jnp.array(sc['RH'])
    alpha_j = jnp.array(1.0)
    dt_j = jnp.array(DT)

    cond_fn = condensation_step_jit if cond_method == 'ppm_jit' else condensation_step_tfl_jit

    hourly = {}
    t0 = time.perf_counter()
    for i in range(NSTEPS):
        if has_coag:
            Nk, Mk = coagulation_step_jit(Nk, Mk, xk_j, temp_j, pres_j, boxvol_j, dt_j)

        if has_cond:
            Gc = Gc.at[SRTSO4].add(sc['h2so4_prod_kg_per_s'] * DT)
            Nk, Mk, Gc = cond_fn(Nk, Mk, Gc, xk_j, temp_j, pres_j, boxvol_j, rh_j, alpha_j, dt_j)

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
    _ = condensation_step_jit(jnp.array(Nk0), jnp.array(Mk0), jnp.array(Gc0), xk_j,
                               jnp.array(sc0['temp']), jnp.array(sc0['pres']),
                               jnp.array(BOXVOL), jnp.array(sc0['RH']),
                               jnp.array(1.0), jnp.array(DT))
    jax.block_until_ready(_[0])
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
            # Map mode to Fortran timing column
            fort_timing_key = {'coag': 'coag', 'cond_tfl': 'cond', 'cond_ppm': 'cond',
                               'combined_tfl': 'combined', 'combined_ppm': 'combined'}
            ft_time = ft.get(fort_timing_key.get(mode_key, ''), None)

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
    # FIGURE 1: Size distributions at hour 24 — TFL vs PPM vs Fortran
    # (6 representative scenarios, cond-only comparison)
    # =========================================================================
    sids_avail = sorted(set(results['cond_tfl'].keys()) & set(results['cond_ppm'].keys()))
    if len(sids_avail) >= 6:
        pick_idx = np.linspace(0, len(sids_avail) - 1, 6, dtype=int)
        pick_sids = [sids_avail[i] for i in pick_idx]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Cond-Only Hour 24: Fortran vs TFL_JIT vs PPM_JIT',
                     fontsize=14, fontweight='bold')

        for idx, sid in enumerate(pick_sids):
            ax = axes[idx // 3, idx % 3]
            r_tfl = results['cond_tfl'][sid]
            r_ppm = results['cond_ppm'][sid]
            sc = r_tfl['scenario']

            if 24 in r_tfl['fortran']:
                dN_f = r_tfl['fortran'][24]['Nk'] / BOXVOL / dlogDp
                ax.plot(Dp, dN_f, 'k-', lw=2.5, label='Fortran')
            if 24 in r_tfl['jax']:
                dN_tfl = r_tfl['jax'][24]['Nk'] / BOXVOL / dlogDp
                ax.plot(Dp, dN_tfl, 'b--', lw=2, label='TFL_JIT')
            if 24 in r_ppm['jax']:
                dN_ppm = r_ppm['jax'][24]['Nk'] / BOXVOL / dlogDp
                ax.plot(Dp, dN_ppm, 'r:', lw=2, label='PPM_JIT')

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(1, 5000); ax.set_ylim(1e-2, 1e8)
            ax.set_xlabel('Dp [nm]'); ax.set_ylabel('dN/dlogDp [#/cm3]')
            ax.set_title(f'S{sid:02d}: N={sc["N_total"]:.0f}, '
                         f'GMD={sc["GMD_um"]:.3f}um', fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=9)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'fig1_sizedist_h24_3way.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 2: N(24h)/N(0) scatter for all modes
    # =========================================================================
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle('N(24h)/N(0): Fortran vs JAX', fontsize=14, fontweight='bold')

    for mi, mode_key in enumerate(mode_keys):
        ax = axes[mi]
        mode_label = MODES[mode_key][4]
        color = MODES[mode_key][5]
        f_ratios, j_ratios = [], []

        for sid, r in sorted(results[mode_key].items()):
            if 24 not in r['fortran'] or 24 not in r['jax']:
                continue
            Nr_f = np.sum(r['fortran'][24]['Nk']) / r['N0']
            Nr_j = np.sum(r['jax'][24]['Nk']) / r['N0']
            f_ratios.append(Nr_f)
            j_ratios.append(Nr_j)

        if not f_ratios:
            continue
        f_ratios = np.array(f_ratios)
        j_ratios = np.array(j_ratios)

        ax.scatter(f_ratios, j_ratios, c=color, s=30, alpha=0.7, edgecolors='k', linewidths=0.5)
        lo = min(f_ratios.min(), j_ratios.min()) * 0.95
        hi = max(f_ratios.max(), j_ratios.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlabel('Fortran N(24h)/N(0)')
        ax.set_ylabel('JAX N(24h)/N(0)')
        ax.set_title(mode_label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_N_conservation_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 3: Error evolution (median+IQR) — TFL vs PPM side by side
    # =========================================================================
    cond_modes = ['cond_tfl', 'cond_ppm']
    combined_modes = ['combined_tfl', 'combined_ppm']

    for group_name, group_modes in [('Cond-Only', cond_modes), ('Combined', combined_modes)]:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f'{group_name}: Error Evolution (median + IQR across 49 scenarios)',
                     fontsize=14, fontweight='bold')

        for mi, mode_key in enumerate(group_modes):
            mode_label = MODES[mode_key][4]
            color = MODES[mode_key][5]
            hours = list(range(1, 25))

            sids = sorted(results[mode_key].keys())
            n_scenarios = len(sids)
            N_rel_err = np.full((n_scenarios, 24), np.nan)
            M_rel_err = np.full((n_scenarios, 24), np.nan)
            mass_cons_err = np.full((n_scenarios, 24), np.nan)

            for si, sid in enumerate(sids):
                r = results[mode_key][sid]
                for h in hours:
                    if h not in r['fortran'] or h not in r['jax']:
                        continue
                    N_f = np.sum(r['fortran'][h]['Nk'])
                    N_j = np.sum(r['jax'][h]['Nk'])
                    N_rel_err[si, h-1] = abs(N_f - N_j) / max(abs(N_f), 1e-30)

                    Md_f = np.sum(r['fortran'][h]['Mk'][:, :SRTH2O])
                    Md_j = np.sum(r['jax'][h]['Mk'][:, :SRTH2O])
                    M_rel_err[si, h-1] = abs(Md_f - Md_j) / max(abs(Md_f), 1e-30)

                    expected = r['total0'] + r['prod'] * DT * h * 60
                    j_total = np.sum(r['jax'][h]['Mk'][:, :SRTH2O]) + np.sum(r['jax'][h]['Gc'])
                    mass_cons_err[si, h-1] = abs(j_total - expected) / max(abs(expected), 1e-30)

            for ci, (data, ylabel, col_title) in enumerate([
                (N_rel_err, 'Rel. Error', 'N_tot |F-J|/F'),
                (M_rel_err, 'Rel. Error', 'M_dry |F-J|/F'),
                (mass_cons_err, 'Rel. Error', 'JAX Mass Conservation'),
            ]):
                ax = axes[mi, ci]
                median = np.nanmedian(data, axis=0)
                q25 = np.nanpercentile(data, 25, axis=0)
                q75 = np.nanpercentile(data, 75, axis=0)
                ax.plot(hours, median, '-', lw=2, color=color, label='Median')
                ax.fill_between(hours, q25, q75, alpha=0.2, color=color, label='IQR')
                ax.set_xlabel('Hour')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{mode_label}: {col_title}', fontsize=10)
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

        plt.tight_layout()
        slug = group_name.lower().replace('-', '_')
        path = os.path.join(OUTPUT_DIR, f'fig3_error_evolution_{slug}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 4: Timing comparison — TFL vs PPM vs Fortran
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Wall-Clock Timing: Fortran vs TFL_JIT vs PPM_JIT', fontsize=14, fontweight='bold')

    for mi, (group_label, tfl_key, ppm_key) in enumerate([
        ('Coag-Only', 'coag', 'coag'),
        ('Cond-Only', 'cond_tfl', 'cond_ppm'),
        ('Combined', 'combined_tfl', 'combined_ppm'),
    ]):
        ax = axes[mi]
        sids_common = sorted(set(results[tfl_key].keys()) & set(results[ppm_key].keys()))

        f_t, tfl_t, ppm_t = [], [], []
        for sid in sids_common:
            r_tfl = results[tfl_key][sid]
            r_ppm = results[ppm_key][sid]
            ft = r_tfl['fortran_time']
            if ft is None:
                continue
            f_t.append(ft)
            tfl_t.append(r_tfl['jax_time'])
            ppm_t.append(r_ppm['jax_time'])

        if not f_t:
            continue
        f_t = np.array(f_t)
        tfl_t = np.array(tfl_t)
        ppm_t = np.array(ppm_t)

        if tfl_key != ppm_key:
            ax.scatter(f_t, tfl_t, c='#1f77b4', s=30, alpha=0.7, edgecolors='k',
                      linewidths=0.5, label=f'TFL_JIT (med {np.median(tfl_t/f_t):.2f}x)')
            ax.scatter(f_t, ppm_t, c='#ff7f0e', s=30, alpha=0.7, edgecolors='k',
                      linewidths=0.5, label=f'PPM_JIT (med {np.median(ppm_t/f_t):.2f}x)')
        else:
            ax.scatter(f_t, tfl_t, c='#2ca02c', s=30, alpha=0.7, edgecolors='k',
                      linewidths=0.5, label=f'JAX (med {np.median(tfl_t/f_t):.2f}x)')

        hi = max(f_t.max(), tfl_t.max(), ppm_t.max()) * 1.1
        ax.plot([0, hi], [0, hi], 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlabel('Fortran [s]')
        ax.set_ylabel('JAX [s]')
        ax.set_title(group_label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_timing_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 5: Per-bin error heatmap at hour 24 (cond_tfl vs cond_ppm)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Per-Bin Nk Relative Error at Hour 24: |Fortran - JAX| / max(Fortran)',
                 fontsize=14, fontweight='bold')

    for mi, mode_key in enumerate(['cond_tfl', 'cond_ppm']):
        mode_label = MODES[mode_key][4]
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
    path = os.path.join(OUTPUT_DIR, 'fig5_perbin_error_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 6: All scenarios hour-24 size distributions (7x7 grid, PPM cond)
    # =========================================================================
    for mode_key in ['cond_ppm', 'combined_ppm']:
        mode_label = MODES[mode_key][4]
        sids_sorted = sorted(results[mode_key].keys())
        if not sids_sorted:
            continue

        ncols = 7
        nrows = (len(sids_sorted) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2.5*nrows))
        fig.suptitle(f'{mode_label}: All Scenarios Hour 24 — Fortran (black) vs JAX (orange)',
                     fontsize=14, fontweight='bold')

        for idx, sid in enumerate(sids_sorted):
            row, col = divmod(idx, ncols)
            ax = axes[row, col] if nrows > 1 else axes[col]
            r = results[mode_key][sid]

            if 24 in r['fortran']:
                ax.plot(Dp, r['fortran'][24]['Nk'] / BOXVOL / dlogDp, 'k-', lw=1.5)
            if 24 in r['jax']:
                ax.plot(Dp, r['jax'][24]['Nk'] / BOXVOL / dlogDp, color='#ff7f0e',
                       linestyle='--', lw=1)

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
        slug = mode_key
        path = os.path.join(OUTPUT_DIR, f'fig6_all_scenarios_h24_{slug}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 7: N_tot timeseries (cond-only: TFL vs PPM overlay)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('N(t)/N(0) Timeseries: TFL_JIT vs PPM_JIT (all scenarios)',
                 fontsize=14, fontweight='bold')

    for mi, mode_key in enumerate(['cond_tfl', 'cond_ppm']):
        ax = axes[mi]
        mode_label = MODES[mode_key][4]
        color = MODES[mode_key][5]

        for si, (sid, r) in enumerate(sorted(results[mode_key].items())):
            hours_f, N_f_s = [], []
            hours_j, N_j_s = [], []

            for h in range(1, 25):
                if h in r['fortran']:
                    N_f_s.append(np.sum(r['fortran'][h]['Nk']) / r['N0'])
                    hours_f.append(h)
                if h in r['jax']:
                    N_j_s.append(np.sum(r['jax'][h]['Nk']) / r['N0'])
                    hours_j.append(h)

            ax.plot(hours_f, N_f_s, 'k-', alpha=0.12, lw=0.5)
            ax.plot(hours_j, N_j_s, '--', color=color, alpha=0.12, lw=0.5)

        ax.set_title(mode_label)
        ax.set_ylabel('N(t)/N(0)')
        ax.set_xlabel('Hour')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.plot([], [], 'k-', lw=1.5, label='Fortran')
        ax.plot([], [], '--', color=color, lw=1.5, label=mode_label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig7_N_timeseries_cond.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # Summary stats
    # =========================================================================
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("SUMMARY — PPM Analytical Mass-Weighted Flux Benchmark")
    summary_lines.append("Date: 2026-03-02")
    summary_lines.append("=" * 70)

    for mode_key in mode_keys:
        mode_label = MODES[mode_key][4]
        has_cond = MODES[mode_key][2]
        n = len(results[mode_key])
        summary_lines.append(f"\n--- {mode_label} ({n} scenarios) ---")

        N_errs, M_errs, cons_errs = [], [], []
        timing_ratios = []
        jax_times_list = []

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

            jax_times_list.append(r['jax_time'])
            if r['fortran_time']:
                timing_ratios.append(r['jax_time'] / r['fortran_time'])

        N_errs = np.array(N_errs)
        M_errs = np.array(M_errs)
        cons_errs = np.array(cons_errs)
        timing_ratios = np.array(timing_ratios)
        jax_times_list = np.array(jax_times_list)

        summary_lines.append(f"  N_tot |F-J|/F at h24:  median={np.median(N_errs):.2e}, "
                             f"max={np.max(N_errs):.2e}")
        summary_lines.append(f"  M_dry |F-J|/F at h24:  median={np.median(M_errs):.2e}, "
                             f"max={np.max(M_errs):.2e}")
        summary_lines.append(f"  JAX mass cons at h24:  median={np.median(cons_errs):.2e}, "
                             f"max={np.max(cons_errs):.2e}")
        summary_lines.append(f"  JAX wall time:         median={np.median(jax_times_list):.3f}s, "
                             f"range=[{np.min(jax_times_list):.3f}s, {np.max(jax_times_list):.3f}s]")
        if len(timing_ratios) > 0:
            summary_lines.append(f"  Timing JAX/Fortran:    median={np.median(timing_ratios):.2f}x, "
                                 f"range=[{np.min(timing_ratios):.2f}x, {np.max(timing_ratios):.2f}x]")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text + "\n")
    print(f"\n  Saved: {summary_path}")
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
