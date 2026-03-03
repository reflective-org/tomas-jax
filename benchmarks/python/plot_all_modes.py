"""Comprehensive benchmark plots for all 5 modes (coag, cond, combined, nucl_cond, full).

Generates 8 figures comparing JAX TFL vs Fortran across all modes,
including the new nucleation modes.

Usage::

    python -m benchmarks.python.plot_all_modes
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NBINS = 36
NHOURS = 24
HOURS = np.arange(1, NHOURS + 1)
N_SCENARIOS = 50

ALL_MODES = ['coag_only', 'cond_only', 'combined', 'nucl_cond', 'full']
MODE_LABELS = {
    'coag_only': 'Coag Only',
    'cond_only': 'Cond Only',
    'combined': 'Coag+Cond',
    'nucl_cond': 'Nucl+Cond',
    'full': 'Full (N+Co+Cd)',
}

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', '24h')
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'nucleation_benchmark')


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def _load_comparison(mode):
    """Load comparison NPZ for a mode."""
    path = os.path.join(_RESULTS_DIR, f'comparison_{mode}.npz')
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def _load_all():
    """Load all modes' comparison data."""
    data = {}
    for mode in ALL_MODES:
        d = _load_comparison(mode)
        if d is not None:
            data[mode] = d
    return data


# =========================================================================
# Figure 1: N_tot timeseries — 5 panels
# =========================================================================

def plot_N_timeseries(data, output_dir):
    """Total number evolution: Fortran vs JAX TFL for all 5 modes."""
    _ensure_dir(output_dir)
    n_modes = len(data)
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 5), squeeze=False)
    fig.suptitle('Total Number Concentration (N_tot) — 24h Evolution', fontsize=14)

    for idx, mode in enumerate(ALL_MODES):
        if mode not in data:
            continue
        ax = axes[0, idx]
        d = data[mode]
        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_xlabel('Hour')
        if idx == 0:
            ax.set_ylabel('N_tot [#/cell]')

        # Plot each scenario
        for s in range(N_SCENARIOS):
            n_fort = d['N_tot_fortran'][s]
            n_tfl = d['N_tot_tfl'][s]

            if not np.all(np.isnan(n_fort)):
                ax.plot(HOURS, n_fort, color='0.6', alpha=0.3, linewidth=0.6)
            if not np.all(np.isnan(n_tfl)):
                ax.plot(HOURS, n_tfl, color='#1f77b4', alpha=0.3, linewidth=0.6)

        # Legend entries
        ax.plot([], [], color='0.6', linewidth=2, label='Fortran')
        ax.plot([], [], color='#1f77b4', linewidth=2, label='JAX TFL')
        ax.legend(fontsize=8)
        ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_N_tot_timeseries.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Figure 2: M_tot timeseries — 5 panels
# =========================================================================

def plot_M_timeseries(data, output_dir):
    """Total dry mass evolution: Fortran vs JAX TFL for all 5 modes."""
    _ensure_dir(output_dir)
    n_modes = len(data)
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 5), squeeze=False)
    fig.suptitle('Total Dry Mass (M_dry) — 24h Evolution', fontsize=14)

    for idx, mode in enumerate(ALL_MODES):
        if mode not in data:
            continue
        ax = axes[0, idx]
        d = data[mode]
        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_xlabel('Hour')
        if idx == 0:
            ax.set_ylabel('M_dry [kg/cell]')

        for s in range(N_SCENARIOS):
            m_fort = d['M_dry_fortran'][s]
            m_tfl = d['M_dry_tfl'][s]

            if not np.all(np.isnan(m_fort)):
                ax.plot(HOURS, m_fort, color='0.6', alpha=0.3, linewidth=0.6)
            if not np.all(np.isnan(m_tfl)):
                ax.plot(HOURS, m_tfl, color='#1f77b4', alpha=0.3, linewidth=0.6)

        ax.plot([], [], color='0.6', linewidth=2, label='Fortran')
        ax.plot([], [], color='#1f77b4', linewidth=2, label='JAX TFL')
        ax.legend(fontsize=8)
        ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_M_dry_timeseries.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Figure 3: TFL vs Fortran N_tot relative error — 5 panels
# =========================================================================

def plot_N_error_evolution(data, output_dir):
    """N_tot relative error over time: each scenario as a thin line, median thick."""
    _ensure_dir(output_dir)
    n_modes = len(data)
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 5), squeeze=False)
    fig.suptitle('TFL vs Fortran: N_tot Relative Error Evolution', fontsize=14)

    for idx, mode in enumerate(ALL_MODES):
        if mode not in data:
            continue
        ax = axes[0, idx]
        d = data[mode]
        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_xlabel('Hour')
        if idx == 0:
            ax.set_ylabel('|N_tfl - N_fort| / N_fort')

        errs = d['tfl_vs_fortran_N_tot_relerr']  # (50, 24)
        for s in range(N_SCENARIOS):
            e = errs[s]
            if not np.all(np.isnan(e)):
                ax.plot(HOURS, e, alpha=0.15, linewidth=0.4, color='tab:blue')

        # Median
        medians = np.nanmedian(errs, axis=0)
        ax.plot(HOURS, medians, color='tab:red', linewidth=2, label='Median')

        # P90
        p90 = np.nanpercentile(errs, 90, axis=0)
        ax.plot(HOURS, p90, color='tab:orange', linewidth=1.5, linestyle='--', label='P90')

        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-10, top=10)
        ax.axhline(0.01, color='k', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(0.05, color='k', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_N_error_evolution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Figure 4: TFL vs Fortran M_tot relative error — 5 panels
# =========================================================================

def plot_M_error_evolution(data, output_dir):
    """M_tot relative error over time."""
    _ensure_dir(output_dir)
    n_modes = len(data)
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 5), squeeze=False)
    fig.suptitle('TFL vs Fortran: M_dry Relative Error Evolution', fontsize=14)

    for idx, mode in enumerate(ALL_MODES):
        if mode not in data:
            continue
        ax = axes[0, idx]
        d = data[mode]
        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_xlabel('Hour')
        if idx == 0:
            ax.set_ylabel('|M_tfl - M_fort| / M_fort')

        errs = d['tfl_vs_fortran_M_tot_relerr']
        for s in range(N_SCENARIOS):
            e = errs[s]
            if not np.all(np.isnan(e)):
                ax.plot(HOURS, e, alpha=0.15, linewidth=0.4, color='tab:green')

        medians = np.nanmedian(errs, axis=0)
        ax.plot(HOURS, medians, color='tab:red', linewidth=2, label='Median')
        p90 = np.nanpercentile(errs, 90, axis=0)
        ax.plot(HOURS, p90, color='tab:orange', linewidth=1.5, linestyle='--', label='P90')

        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-10, top=10)
        ax.axhline(0.01, color='k', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_M_error_evolution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Figure 5: Hour-24 scatter (TFL vs Fortran)
# =========================================================================

def plot_scatter_hour24(data, output_dir):
    """Scatter plots: JAX TFL vs Fortran at hour 24 for N and M."""
    _ensure_dir(output_dir)
    n_modes = len(data)
    fig, axes = plt.subplots(2, n_modes, figsize=(4 * n_modes, 9), squeeze=False)
    fig.suptitle('JAX TFL vs Fortran at Hour 24', fontsize=14)

    quantities = [
        ('N_tot', 'N_tot [#/cell]'),
        ('M_dry', 'M_dry [kg/cell]'),
    ]

    for row, (qname, qlabel) in enumerate(quantities):
        for col, mode in enumerate(ALL_MODES):
            if mode not in data:
                continue
            ax = axes[row, col]
            d = data[mode]

            fort = d[f'{qname}_fortran'][:, -1]
            tfl = d[f'{qname}_tfl'][:, -1]

            valid = ~np.isnan(fort) & ~np.isnan(tfl) & (fort > 0) & (tfl > 0)
            if not np.any(valid):
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center')
                continue

            f_v = fort[valid]
            t_v = tfl[valid]

            ax.scatter(f_v, t_v, s=15, alpha=0.6, c='tab:blue', edgecolors='none')

            vmin = min(f_v.min(), t_v.min()) * 0.5
            vmax = max(f_v.max(), t_v.max()) * 2.0
            ax.plot([vmin, vmax], [vmin, vmax], 'k-', linewidth=1, label='1:1')
            ax.plot([vmin, vmax], [vmin * 0.9, vmax * 0.9], 'k:', linewidth=0.5)
            ax.plot([vmin, vmax], [vmin * 1.1, vmax * 1.1], 'k:', linewidth=0.5,
                    label='10% band')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_aspect('equal', adjustable='datalim')

            if row == 0:
                ax.set_title(MODE_LABELS[mode], fontsize=11)
            if col == 0:
                ax.set_ylabel(f'JAX TFL {qlabel}')
            if row == 1:
                ax.set_xlabel(f'Fortran {qlabel}')
            ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_scatter_hour24.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Figure 6: Per-scenario final error bar chart
# =========================================================================

def plot_per_scenario_error(data, output_dir):
    """Bar chart of hour-24 N_tot and M_dry relative error per scenario."""
    _ensure_dir(output_dir)

    # Focus on nucl_cond and full modes (new results)
    modes_to_plot = [m for m in ['nucl_cond', 'full'] if m in data]
    if not modes_to_plot:
        return

    fig, axes = plt.subplots(len(modes_to_plot), 2, figsize=(18, 5 * len(modes_to_plot)),
                             squeeze=False)
    fig.suptitle('TFL vs Fortran: Per-Scenario Error at Hour 24', fontsize=14)

    sids = np.arange(1, N_SCENARIOS + 1)

    for row, mode in enumerate(modes_to_plot):
        d = data[mode]

        for col, (key, label) in enumerate([
            ('tfl_vs_fortran_N_tot_relerr', 'N_tot Rel Error'),
            ('tfl_vs_fortran_M_tot_relerr', 'M_dry Rel Error'),
        ]):
            ax = axes[row, col]
            errs = d[key][:, -1]  # Hour 24

            colors = np.where(errs < 0.01, 'tab:green',
                     np.where(errs < 0.05, 'tab:orange', 'tab:red'))
            ax.bar(sids, errs, color=colors, width=0.8)

            ax.set_xlabel('Scenario ID')
            ax.set_ylabel(label)
            ax.set_title(f'{MODE_LABELS[mode]} — {label}')
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-6, top=10)
            ax.axhline(0.01, color='k', linestyle='--', linewidth=0.5, label='1%')
            ax.axhline(0.05, color='k', linestyle=':', linewidth=0.5, label='5%')
            ax.legend(fontsize=7)
            ax.set_xticks(sids[::5])

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_per_scenario_error.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Figure 7: Error heatmap (scenario x hour)
# =========================================================================

def plot_error_heatmap(data, output_dir):
    """Heatmap of N_tot relative error: scenario (y) x hour (x)."""
    _ensure_dir(output_dir)

    fig, axes = plt.subplots(1, len(ALL_MODES), figsize=(4 * len(ALL_MODES), 12),
                             squeeze=False)
    fig.suptitle('TFL vs Fortran: N_tot Relative Error Heatmap', fontsize=14)

    for idx, mode in enumerate(ALL_MODES):
        if mode not in data:
            continue
        ax = axes[0, idx]
        d = data[mode]
        errs = d['tfl_vs_fortran_N_tot_relerr']  # (50, 24)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_err = np.log10(errs)
        log_err = np.where(np.isfinite(log_err), log_err, np.nan)

        im = ax.pcolormesh(
            np.arange(NHOURS + 1) + 0.5,
            np.arange(N_SCENARIOS + 1) + 0.5,
            log_err,
            cmap='RdYlGn_r', vmin=-6, vmax=0, shading='flat'
        )
        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Scenario ID')
        ax.set_yticks(np.arange(0, 50, 5) + 1)

    fig.colorbar(im, ax=axes[0, -1], label='log10(rel error)', shrink=0.6)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig7_error_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Figure 8: Summary statistics table
# =========================================================================

def plot_summary_table(data, output_dir):
    """Summary statistics as a text table figure."""
    _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    rows = []
    headers = ['Mode', 'N Scenarios', 'N<1%', 'N<5%', 'N<10%',
               'N Med Err', 'N Max Err', 'M Med Err', 'M Max Err']

    for mode in ALL_MODES:
        if mode not in data:
            continue
        d = data[mode]
        n_errs = d['tfl_vs_fortran_N_tot_relerr'][:, -1]
        m_errs = d['tfl_vs_fortran_M_tot_relerr'][:, -1]

        valid_n = n_errs[~np.isnan(n_errs)]
        valid_m = m_errs[~np.isnan(m_errs)]

        n_valid = len(valid_n)
        n_lt1 = np.sum(valid_n < 0.01)
        n_lt5 = np.sum(valid_n < 0.05)
        n_lt10 = np.sum(valid_n < 0.10)
        n_med = np.median(valid_n) if len(valid_n) > 0 else np.nan
        n_max = np.max(valid_n) if len(valid_n) > 0 else np.nan
        m_med = np.median(valid_m) if len(valid_m) > 0 else np.nan
        m_max = np.max(valid_m) if len(valid_m) > 0 else np.nan

        rows.append([
            MODE_LABELS[mode],
            f'{n_valid}',
            f'{n_lt1}',
            f'{n_lt5}',
            f'{n_lt10}',
            f'{n_med:.2e}',
            f'{n_max:.2e}',
            f'{m_med:.2e}',
            f'{m_max:.2e}',
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color cells based on error quality
    for i, row in enumerate(rows):
        # N Max Err column (index 7 in cellText, but +1 for header row)
        for j, val in enumerate(row):
            cell = table[i + 1, j]
            if j >= 5:  # Error columns
                try:
                    v = float(val)
                    if v < 0.01:
                        cell.set_facecolor('#c8e6c9')
                    elif v < 0.05:
                        cell.set_facecolor('#fff9c4')
                    elif v < 0.10:
                        cell.set_facecolor('#ffe0b2')
                    else:
                        cell.set_facecolor('#ffcdd2')
                except ValueError:
                    pass

    ax.set_title('TFL vs Fortran: Hour-24 Error Summary', fontsize=14, pad=20)

    path = os.path.join(output_dir, 'fig8_summary_table.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Main
# =========================================================================

def generate_all_plots():
    """Generate all 8 summary figures."""
    print("Loading comparison data for all modes...")
    data = _load_all()
    print(f"  Loaded modes: {list(data.keys())}")

    output_dir = _OUTPUT_DIR

    print("\nFigure 1: N_tot timeseries...")
    plot_N_timeseries(data, output_dir)

    print("Figure 2: M_dry timeseries...")
    plot_M_timeseries(data, output_dir)

    print("Figure 3: N_tot error evolution...")
    plot_N_error_evolution(data, output_dir)

    print("Figure 4: M_dry error evolution...")
    plot_M_error_evolution(data, output_dir)

    print("Figure 5: Scatter plots (hour 24)...")
    plot_scatter_hour24(data, output_dir)

    print("Figure 6: Per-scenario error bars...")
    plot_per_scenario_error(data, output_dir)

    print("Figure 7: Error heatmap...")
    plot_error_heatmap(data, output_dir)

    print("Figure 8: Summary table...")
    plot_summary_table(data, output_dir)

    # Print text summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY: TFL vs Fortran at Hour 24")
    print("=" * 70)
    for mode in ALL_MODES:
        if mode not in data:
            continue
        d = data[mode]
        n_errs = d['tfl_vs_fortran_N_tot_relerr'][:, -1]
        m_errs = d['tfl_vs_fortran_M_tot_relerr'][:, -1]
        valid_n = n_errs[~np.isnan(n_errs)]
        valid_m = m_errs[~np.isnan(m_errs)]

        print(f"\n{MODE_LABELS[mode]:>20s}: {len(valid_n)} scenarios with Fortran data")
        if len(valid_n) > 0:
            print(f"  N_tot: median={np.median(valid_n):.2e}  max={np.max(valid_n):.2e}  "
                  f"<1%={np.sum(valid_n<0.01)}/50  <5%={np.sum(valid_n<0.05)}/50")
        if len(valid_m) > 0:
            print(f"  M_dry: median={np.median(valid_m):.2e}  max={np.max(valid_m):.2e}  "
                  f"<1%={np.sum(valid_m<0.01)}/50  <5%={np.sum(valid_m<0.05)}/50")
    print()


if __name__ == '__main__':
    generate_all_plots()
