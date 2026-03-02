"""Compare Fortran TFL vs Fortran PPM condensation — 50 scenarios, 24h.

Reads output from:
  tomas_fortran/output/24h/s{01-50}_{coag|cond|combined}_hour{01-24}_{Nk|Mk|Gc}.csv   (TFL)
  tomas_fortran/output/24h/ppm_s{01-50}_{coag|cond|combined}_hour{01-24}_{Nk|Mk|Gc}.csv (PPM)

Generates comparison plots in:
  benchmarks/results/fortran_tfl_vs_ppm/

Usage:
  python benchmarks/python/compare_fortran_tfl_ppm.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Constants ───────────────────────────────────────────────────────
NBINS = 36
ICOMP = 44
SRTH2O = 44  # 1-based Fortran index for water
NHOURS = 24
NSCENARIOS = 50
DENSITY = 1770.0  # kg/m3 for diameter calculation
PI = 3.141592654

# ── Paths ───────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FORTRAN_DIR = os.path.join(REPO, 'tomas_fortran', 'output', '24h')
OUTPUT_DIR = os.path.join(REPO, 'benchmarks', 'results', 'fortran_tfl_vs_ppm')

MODE_MAP = {'cond_only': 'cond', 'combined': 'combined', 'coag_only': 'coag'}


# ── Bin geometry ────────────────────────────────────────────────────
def compute_bin_geometry():
    """Compute bin-edge masses, bin-center diameters, and dlogDp."""
    xk = np.zeros(NBINS + 1)
    xk[0] = 1.6033e-23  # kg, smallest bin lower bound
    for k in range(NBINS):
        xk[k + 1] = 2.0 * xk[k]
    # Bin-center diameter [um]
    Dp_center = np.zeros(NBINS)
    for k in range(NBINS):
        mp = np.sqrt(xk[k] * xk[k + 1])
        Dp_center[k] = 1e6 * (6.0 * mp / (DENSITY * PI)) ** (1.0 / 3.0)
    # dlogDp for each bin
    Dp_lo = 1e6 * (6.0 * xk[:-1] / (DENSITY * PI)) ** (1.0 / 3.0)
    Dp_hi = 1e6 * (6.0 * xk[1:] / (DENSITY * PI)) ** (1.0 / 3.0)
    dlogDp = np.log10(Dp_hi) - np.log10(Dp_lo)
    return xk, Dp_center, dlogDp


# ── I/O ─────────────────────────────────────────────────────────────
def load_fortran_hourly(scenario_id, mode_short, hour, prefix=''):
    """Load one hourly snapshot from Fortran CSV output.

    prefix='ppm_' for PPM files, '' for TFL.
    Returns (Nk, Mk, Gc) or None if files missing.
    """
    base = os.path.join(FORTRAN_DIR,
                        f'{prefix}s{scenario_id:02d}_{mode_short}_hour{hour:02d}')
    nk_file = base + '_Nk.csv'
    mk_file = base + '_Mk.csv'
    gc_file = base + '_Gc.csv'
    if not os.path.exists(nk_file):
        return None
    Nk = np.loadtxt(nk_file, dtype=np.float64)
    Mk = np.loadtxt(mk_file, delimiter=',', dtype=np.float64).reshape(NBINS, ICOMP)
    Gc = np.loadtxt(gc_file, dtype=np.float64)
    return Nk, Mk, Gc


def load_all_hours(scenario_id, mode_short, prefix=''):
    """Load all 24 hours for one scenario+mode. Returns dict of arrays or None."""
    Nk_all = np.zeros((NHOURS, NBINS))
    Mk_all = np.zeros((NHOURS, NBINS, ICOMP))
    Gc_all = np.zeros((NHOURS, ICOMP - 1))
    for h in range(1, NHOURS + 1):
        data = load_fortran_hourly(scenario_id, mode_short, h, prefix)
        if data is None:
            return None
        Nk_all[h - 1] = data[0]
        Mk_all[h - 1] = data[1]
        Gc_all[h - 1] = data[2]
    # Derived totals
    N_tot = Nk_all.sum(axis=1)  # (24,)
    M_dry = Mk_all[:, :, :ICOMP - 1].sum(axis=(1, 2))  # exclude water (last col, 1-based idx 44 = 0-based 43)
    return dict(Nk=Nk_all, Mk=Mk_all, Gc=Gc_all, N_tot=N_tot, M_dry=M_dry)


# ── Metrics ─────────────────────────────────────────────────────────
def rel_error(a, b, atol=1e-30):
    """Element-wise relative error |a-b|/max(|a|,|b|,atol)."""
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), atol)
    return np.abs(a - b) / denom


def scalar_rel_error(a, b, atol=1e-30):
    return abs(a - b) / max(abs(a), abs(b), atol)


# ── Data Loading ────────────────────────────────────────────────────
def load_comparison_data(modes=('cond_only', 'combined')):
    """Load TFL and PPM data for all scenarios and specified modes."""
    results = {}
    for mode in modes:
        ms = MODE_MAP[mode]
        tfl_list, ppm_list = [], []
        for sid in range(1, NSCENARIOS + 1):
            tfl = load_all_hours(sid, ms, prefix='')
            ppm = load_all_hours(sid, ms, prefix='ppm_')
            tfl_list.append(tfl)
            ppm_list.append(ppm)
        results[mode] = (tfl_list, ppm_list)
    return results


# ── Plot 1: Total N timeseries ──────────────────────────────────────
def plot_total_N_timeseries(results, output_dir):
    """Total number concentration over 24h — TFL vs PPM for each mode."""
    modes = list(results.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    hours = np.arange(1, NHOURS + 1)

    for ax, mode in zip(axes, modes):
        tfl_list, ppm_list = results[mode]
        for sid in range(NSCENARIOS):
            tfl, ppm = tfl_list[sid], ppm_list[sid]
            if tfl is None or ppm is None:
                continue
            ax.plot(hours, tfl['N_tot'], color='steelblue', alpha=0.15, lw=0.8)
            ax.plot(hours, ppm['N_tot'], color='orangered', alpha=0.15, lw=0.8)
        # Legend proxies
        ax.plot([], [], color='steelblue', lw=2, label='TFL')
        ax.plot([], [], color='orangered', lw=2, label='PPM')
        ax.set_title(mode.replace('_', ' ').title())
        ax.set_xlabel('Hour')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Total N [#/cell]')
    fig.suptitle('Total Number Concentration: Fortran TFL vs PPM', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '01_total_N_timeseries.png'), dpi=150)
    plt.close(fig)
    print('  [1/8] Total N timeseries')


# ── Plot 2: Total dry mass timeseries ───────────────────────────────
def plot_total_M_timeseries(results, output_dir):
    modes = list(results.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    hours = np.arange(1, NHOURS + 1)

    for ax, mode in zip(axes, modes):
        tfl_list, ppm_list = results[mode]
        for sid in range(NSCENARIOS):
            tfl, ppm = tfl_list[sid], ppm_list[sid]
            if tfl is None or ppm is None:
                continue
            ax.plot(hours, tfl['M_dry'], color='steelblue', alpha=0.15, lw=0.8)
            ax.plot(hours, ppm['M_dry'], color='orangered', alpha=0.15, lw=0.8)
        ax.plot([], [], color='steelblue', lw=2, label='TFL')
        ax.plot([], [], color='orangered', lw=2, label='PPM')
        ax.set_title(mode.replace('_', ' ').title())
        ax.set_xlabel('Hour')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Total dry mass [kg/cell]')
    fig.suptitle('Total Dry Mass: Fortran TFL vs PPM', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '02_total_M_timeseries.png'), dpi=150)
    plt.close(fig)
    print('  [2/8] Total M timeseries')


# ── Plot 3: Relative error evolution ────────────────────────────────
def plot_error_evolution(results, output_dir):
    """N_tot and M_dry relative error vs time for all scenarios."""
    modes = list(results.keys())
    fig, axes = plt.subplots(2, len(modes), figsize=(6 * len(modes), 8), sharex=True)
    if len(modes) == 1:
        axes = axes.reshape(-1, 1)
    hours = np.arange(1, NHOURS + 1)

    for col, mode in enumerate(modes):
        tfl_list, ppm_list = results[mode]
        for sid in range(NSCENARIOS):
            tfl, ppm = tfl_list[sid], ppm_list[sid]
            if tfl is None or ppm is None:
                continue
            n_err = np.array([scalar_rel_error(tfl['N_tot'][h], ppm['N_tot'][h])
                              for h in range(NHOURS)])
            m_err = np.array([scalar_rel_error(tfl['M_dry'][h], ppm['M_dry'][h])
                              for h in range(NHOURS)])
            axes[0, col].plot(hours, np.maximum(n_err, 1e-16), color='royalblue', alpha=0.15, lw=0.7)
            axes[1, col].plot(hours, np.maximum(m_err, 1e-16), color='forestgreen', alpha=0.15, lw=0.7)

        # Median lines
        n_errs_all = []
        m_errs_all = []
        for sid in range(NSCENARIOS):
            tfl, ppm = tfl_list[sid], ppm_list[sid]
            if tfl is None or ppm is None:
                continue
            n_errs_all.append([scalar_rel_error(tfl['N_tot'][h], ppm['N_tot'][h])
                               for h in range(NHOURS)])
            m_errs_all.append([scalar_rel_error(tfl['M_dry'][h], ppm['M_dry'][h])
                               for h in range(NHOURS)])
        if n_errs_all:
            med_n = np.median(n_errs_all, axis=0)
            med_m = np.median(m_errs_all, axis=0)
            axes[0, col].plot(hours, np.maximum(med_n, 1e-16), 'r-', lw=2, label='median')
            axes[1, col].plot(hours, np.maximum(med_m, 1e-16), 'r-', lw=2, label='median')

        axes[0, col].set_title(f'{mode.replace("_", " ").title()}')
        axes[1, col].set_xlabel('Hour')
        for row in range(2):
            axes[row, col].set_yscale('log')
            axes[row, col].set_ylim(1e-16, 1e1)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=8)

    axes[0, 0].set_ylabel('|N_tot relative error|')
    axes[1, 0].set_ylabel('|M_dry relative error|')
    fig.suptitle('PPM vs TFL Relative Error Evolution (Fortran)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '03_error_evolution.png'), dpi=150)
    plt.close(fig)
    print('  [3/8] Error evolution')


# ── Plot 4: Scatter TFL vs PPM at hour 24 ──────────────────────────
def plot_scatter_24h(results, output_dir):
    modes = list(results.keys())
    fig, axes = plt.subplots(2, len(modes), figsize=(6 * len(modes), 10))
    if len(modes) == 1:
        axes = axes.reshape(-1, 1)

    for col, mode in enumerate(modes):
        tfl_list, ppm_list = results[mode]
        tfl_N, ppm_N = [], []
        tfl_M, ppm_M = [], []
        for sid in range(NSCENARIOS):
            tfl, ppm = tfl_list[sid], ppm_list[sid]
            if tfl is None or ppm is None:
                continue
            tfl_N.append(tfl['N_tot'][-1])
            ppm_N.append(ppm['N_tot'][-1])
            tfl_M.append(tfl['M_dry'][-1])
            ppm_M.append(ppm['M_dry'][-1])

        for row, (tv, pv, label) in enumerate([
            (tfl_N, ppm_N, 'N_tot at 24h'),
            (tfl_M, ppm_M, 'M_dry at 24h'),
        ]):
            ax = axes[row, col]
            tv, pv = np.array(tv), np.array(pv)
            ax.scatter(tv, pv, s=20, alpha=0.7, c='royalblue', edgecolors='k', lw=0.3)
            lims = [min(tv.min(), pv.min()) * 0.5, max(tv.max(), pv.max()) * 2]
            ax.plot(lims, lims, 'k-', lw=1, label='1:1')
            ax.plot(lims, [l * 1.2 for l in lims], 'k--', lw=0.5, alpha=0.5)
            ax.plot(lims, [l * 0.8 for l in lims], 'k--', lw=0.5, alpha=0.5, label='+/- 20%')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(f'TFL {label}')
            ax.set_ylabel(f'PPM {label}')
            ax.set_title(mode.replace('_', ' ').title())
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

    fig.suptitle('TFL vs PPM Scatter at Hour 24 (Fortran)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '04_scatter_24h.png'), dpi=150)
    plt.close(fig)
    print('  [4/8] Scatter at hour 24')


# ── Plot 5: Size distribution comparison (selected scenarios) ──────
def plot_size_distributions(results, output_dir):
    """Size distribution at hours 1,6,12,24 for 3 representative scenarios."""
    xk, Dp, dlogDp = compute_bin_geometry()
    mode = 'cond_only' if 'cond_only' in results else list(results.keys())[0]
    tfl_list, ppm_list = results[mode]

    # Pick 3 representative: small, medium, large GMD by hour-1 peak bin
    peak_bins = []
    for sid in range(NSCENARIOS):
        tfl = tfl_list[sid]
        if tfl is None:
            peak_bins.append(-1)
            continue
        peak_bins.append(np.argmax(tfl['Nk'][0]))
    peak_bins = np.array(peak_bins)
    valid = peak_bins >= 0
    sorted_idx = np.argsort(peak_bins[valid])
    valid_ids = np.where(valid)[0]
    picks = [valid_ids[sorted_idx[0]], valid_ids[sorted_idx[len(sorted_idx) // 2]],
             valid_ids[sorted_idx[-1]]]

    hours_plot = [0, 5, 11, 23]  # 0-indexed: hour 1,6,12,24
    hour_labels = ['Hour 1', 'Hour 6', 'Hour 12', 'Hour 24']

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    for col_idx, sid in enumerate(picks):
        tfl = tfl_list[sid]
        ppm = ppm_list[sid]
        if tfl is None or ppm is None:
            continue
        for row_idx, (hi, hl) in enumerate(zip(hours_plot, hour_labels)):
            ax = axes[row_idx, col_idx]
            dNdlogDp_tfl = tfl['Nk'][hi] / dlogDp
            dNdlogDp_ppm = ppm['Nk'][hi] / dlogDp
            ax.plot(Dp, dNdlogDp_tfl, 'b-', lw=1.5, label='TFL')
            ax.plot(Dp, dNdlogDp_ppm, 'r--', lw=1.5, label='PPM')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-3)
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(f'Scenario {sid + 1}', fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f'{hl}\ndN/dlogDp')
            if row_idx == 3:
                ax.set_xlabel('Dp [um]')
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f'Size Distribution Evolution: TFL vs PPM ({mode.replace("_"," ").title()})',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '05_size_distribution_evolution.png'), dpi=150)
    plt.close(fig)
    print('  [5/8] Size distribution evolution')


# ── Plot 6: All 50 scenarios size distribution at hour 24 ──────────
def plot_all_size_dist_24h(results, output_dir):
    xk, Dp, dlogDp = compute_bin_geometry()
    mode = 'combined' if 'combined' in results else list(results.keys())[0]
    tfl_list, ppm_list = results[mode]

    nrows, ncols = 10, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 24))
    for sid in range(NSCENARIOS):
        row, col = divmod(sid, ncols)
        ax = axes[row, col]
        tfl = tfl_list[sid]
        ppm = ppm_list[sid]
        if tfl is None or ppm is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'S{sid + 1:02d}', fontsize=8)
            continue
        dNdlogDp_tfl = tfl['Nk'][-1] / dlogDp
        dNdlogDp_ppm = ppm['Nk'][-1] / dlogDp
        ax.plot(Dp, dNdlogDp_tfl, 'b-', lw=1, label='TFL')
        ax.plot(Dp, dNdlogDp_ppm, 'r--', lw=1, label='PPM')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3)
        ax.set_title(f'S{sid + 1:02d}', fontsize=8)
        ax.tick_params(labelsize=6)
        if row == 0 and col == 0:
            ax.legend(fontsize=6)

    fig.suptitle(f'All 50 Scenarios — Size Dist at Hour 24 ({mode.replace("_"," ").title()})',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '06_all_size_dist_24h.png'), dpi=120)
    plt.close(fig)
    print('  [6/8] All 50 size distributions at hour 24')


# ── Plot 7: Per-bin relative error heatmap at hour 24 ──────────────
def plot_perbin_error_heatmap(results, output_dir):
    xk, Dp, dlogDp = compute_bin_geometry()

    modes_to_plot = [m for m in ('cond_only', 'combined') if m in results]
    fig, axes = plt.subplots(len(modes_to_plot), 2, figsize=(16, 5 * len(modes_to_plot)))
    if len(modes_to_plot) == 1:
        axes = axes.reshape(1, -1)

    for row, mode in enumerate(modes_to_plot):
        tfl_list, ppm_list = results[mode]

        nk_err_matrix = np.full((NSCENARIOS, NBINS), np.nan)
        mk_err_matrix = np.full((NSCENARIOS, NBINS), np.nan)

        for sid in range(NSCENARIOS):
            tfl, ppm = tfl_list[sid], ppm_list[sid]
            if tfl is None or ppm is None:
                continue
            # Per-bin Nk relative error at hour 24
            nk_err = rel_error(tfl['Nk'][-1], ppm['Nk'][-1])
            nk_err_matrix[sid] = nk_err
            # Per-bin dry mass relative error at hour 24
            mk_tfl = tfl['Mk'][-1, :, :ICOMP - 1].sum(axis=1)
            mk_ppm = ppm['Mk'][-1, :, :ICOMP - 1].sum(axis=1)
            mk_err = rel_error(mk_tfl, mk_ppm)
            mk_err_matrix[sid] = mk_err

        for col, (matrix, vlabel) in enumerate([
            (nk_err_matrix, 'Nk per-bin'),
            (mk_err_matrix, 'Mk_dry per-bin'),
        ]):
            ax = axes[row, col]
            safe = np.where(matrix > 0, np.log10(matrix), -16)
            im = ax.pcolormesh(np.arange(NBINS + 1), np.arange(NSCENARIOS + 1),
                               safe, cmap='RdYlGn_r', vmin=-6, vmax=0)
            ax.set_xlabel('Bin index')
            ax.set_ylabel('Scenario ID')
            ax.set_title(f'{vlabel} — {mode.replace("_"," ").title()}')
            plt.colorbar(im, ax=ax, label='log10(relative error)')

    fig.suptitle('Per-Bin Relative Error at Hour 24: PPM vs TFL (Fortran)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '07_perbin_error_heatmap.png'), dpi=150)
    plt.close(fig)
    print('  [7/8] Per-bin error heatmap')


# ── Plot 8: Timing comparison ──────────────────────────────────────
def plot_timing(output_dir):
    """Bar chart comparing TFL vs PPM wall-clock time from timing CSVs."""
    tfl_csv = os.path.join(FORTRAN_DIR, 'timing_fortran.csv')
    ppm_csv = os.path.join(FORTRAN_DIR, 'timing_fortran_ppm.csv')

    if not os.path.exists(tfl_csv) or not os.path.exists(ppm_csv):
        print('  [8/8] Timing: CSV not found, skipping')
        return

    tfl_data = np.genfromtxt(tfl_csv, delimiter=',', skip_header=1)
    ppm_data = np.genfromtxt(ppm_csv, delimiter=',', skip_header=1)

    n_tfl = len(tfl_data)
    n_ppm = len(ppm_data)
    n = min(n_tfl, n_ppm)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mode_labels = ['Coag-only', 'Cond-only', 'Combined']
    for col in range(3):
        ax = axes[col]
        t_tfl = tfl_data[:n, col + 1]
        t_ppm = ppm_data[:n, col + 1]
        x = np.arange(n)
        w = 0.35
        ax.bar(x - w / 2, t_tfl, w, label='TFL', color='steelblue', alpha=0.8)
        ax.bar(x + w / 2, t_ppm, w, label='PPM', color='orangered', alpha=0.8)
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Wall time [s]')
        ax.set_title(mode_labels[col])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Summary stats
        med_tfl = np.median(t_tfl)
        med_ppm = np.median(t_ppm)
        ratio = med_ppm / med_tfl if med_tfl > 0 else float('nan')
        ax.text(0.98, 0.95, f'TFL med: {med_tfl:.3f}s\nPPM med: {med_ppm:.3f}s\nratio: {ratio:.2f}x',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('Wall-Clock Timing: Fortran TFL vs PPM (24h, 50 scenarios)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '08_timing_comparison.png'), dpi=150)
    plt.close(fig)
    print('  [8/8] Timing comparison')


# ── Summary text ────────────────────────────────────────────────────
def write_summary(results, output_dir):
    """Write a text summary of the comparison metrics."""
    lines = ['Fortran TFL vs PPM Comparison Summary', '=' * 50, '']

    for mode in results:
        tfl_list, ppm_list = results[mode]
        n_err_24 = []
        m_err_24 = []
        max_nk_err_24 = []
        for sid in range(NSCENARIOS):
            tfl, ppm = tfl_list[sid], ppm_list[sid]
            if tfl is None or ppm is None:
                continue
            n_err_24.append(scalar_rel_error(tfl['N_tot'][-1], ppm['N_tot'][-1]))
            m_err_24.append(scalar_rel_error(tfl['M_dry'][-1], ppm['M_dry'][-1]))
            max_nk_err_24.append(rel_error(tfl['Nk'][-1], ppm['Nk'][-1]).max())

        n_err_24 = np.array(n_err_24)
        m_err_24 = np.array(m_err_24)
        max_nk_err_24 = np.array(max_nk_err_24)

        lines.append(f'Mode: {mode}')
        lines.append(f'  Scenarios compared: {len(n_err_24)}')
        lines.append(f'  N_tot relative error at 24h:')
        lines.append(f'    median: {np.median(n_err_24):.4e}')
        lines.append(f'    max:    {np.max(n_err_24):.4e}')
        lines.append(f'    mean:   {np.mean(n_err_24):.4e}')
        lines.append(f'  M_dry relative error at 24h:')
        lines.append(f'    median: {np.median(m_err_24):.4e}')
        lines.append(f'    max:    {np.max(m_err_24):.4e}')
        lines.append(f'    mean:   {np.mean(m_err_24):.4e}')
        lines.append(f'  Max per-bin Nk error at 24h:')
        lines.append(f'    median: {np.median(max_nk_err_24):.4e}')
        lines.append(f'    max:    {np.max(max_nk_err_24):.4e}')
        lines.append('')

    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nSummary written to {summary_path}')
    for line in lines:
        print(f'  {line}')


# ── Main ────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check that output exists
    test_file = os.path.join(FORTRAN_DIR, 's01_cond_hour01_Nk.csv')
    ppm_test = os.path.join(FORTRAN_DIR, 'ppm_s01_cond_hour01_Nk.csv')
    if not os.path.exists(test_file):
        print(f'ERROR: TFL output not found at {test_file}')
        print('Run: cd tomas_fortran && make run_24h')
        sys.exit(1)
    if not os.path.exists(ppm_test):
        print(f'ERROR: PPM output not found at {ppm_test}')
        print('Run: cd tomas_fortran && make run_24h_ppm')
        sys.exit(1)

    print('Loading Fortran TFL and PPM output...')
    results = load_comparison_data(modes=('cond_only', 'combined'))

    # Count successful loads
    for mode in results:
        tfl_list, ppm_list = results[mode]
        n_ok = sum(1 for t, p in zip(tfl_list, ppm_list) if t is not None and p is not None)
        print(f'  {mode}: {n_ok}/{NSCENARIOS} scenarios loaded')

    print('\nGenerating plots...')
    plot_total_N_timeseries(results, OUTPUT_DIR)
    plot_total_M_timeseries(results, OUTPUT_DIR)
    plot_error_evolution(results, OUTPUT_DIR)
    plot_scatter_24h(results, OUTPUT_DIR)
    plot_size_distributions(results, OUTPUT_DIR)
    plot_all_size_dist_24h(results, OUTPUT_DIR)
    plot_perbin_error_heatmap(results, OUTPUT_DIR)
    plot_timing(OUTPUT_DIR)

    write_summary(results, OUTPUT_DIR)

    print(f'\nAll plots saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
