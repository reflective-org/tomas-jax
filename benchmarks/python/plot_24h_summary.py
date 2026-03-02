"""Summary plots for 24-hour benchmark results.

Generates 6 figures comparing PPM vs TFL vs Fortran across 50 scenarios.

Usage::

    python -m benchmarks.python.plot_24h_summary
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks.python.compare_24h import (
    compare_scenario, load_jax_results, load_fortran_hourly,
    FORTRAN_DIR, RESULTS_DIR, MODE_NAMES, NHOURS, SRTH2O
)
from benchmarks.python.scenarios import get_scenarios

NBINS = 36
ICOMP = 44
HOURS = np.arange(1, NHOURS + 1)
SCENARIO_IDS = list(range(1, 51))
MODES = ['coag_only', 'cond_only', 'combined']
MODE_LABELS = {'coag_only': 'Coag Only', 'cond_only': 'Cond Only', 'combined': 'Combined'}

_RESULTS_OUT = os.path.join(os.path.dirname(__file__), '..', 'results')


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def _load_all_metrics():
    """Load comparison metrics for all scenarios and modes."""
    all_metrics = {}
    for mode in MODES:
        all_metrics[mode] = []
        for sid in SCENARIO_IDS:
            try:
                m = compare_scenario(sid, mode)
                all_metrics[mode].append(m)
            except Exception:
                pass
    return all_metrics


# =========================================================================
# Figure 1: Total N time series
# =========================================================================

def plot_total_N_timeseries(all_metrics, output_dir=None):
    """Total number evolution: 3 panels (coag, cond, combined)."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    _ensure_dir(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle('Total Number Concentration Evolution (24h)', fontsize=14)

    for idx, mode in enumerate(MODES):
        ax = axes[idx]
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel('Hour')
        ax.set_ylabel('N_tot / N_tot(t=1h)')

        metrics_list = all_metrics[mode]
        style = [
            ('N_tot_fortran', '0.3',       '-',  'Fortran'),
            ('N_tot_tfl',     '#1f77b4',   '--', 'TFL'),
            ('N_tot_ppm',     '#d62728',   ':',  'PPM'),
        ]
        for m in metrics_list:
            for key, color, ls, label in style:
                vals = m[key]
                if np.all(np.isnan(vals)):
                    continue
                ref = vals[0] if vals[0] > 0 else 1.0
                ax.plot(HOURS, vals / ref, color=color, alpha=0.25,
                        linewidth=0.8, linestyle=ls)

        # Dummy lines for legend
        for key, color, ls, label in style:
            ax.plot([], [], color=color, linewidth=2, linestyle=ls, label=label)
        ax.legend(fontsize=9)
        ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_total_N_timeseries.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Figure 2: Total M time series
# =========================================================================

def plot_total_M_timeseries(all_metrics, output_dir=None):
    """Total mass evolution: 3 panels."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    _ensure_dir(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle('Total Dry Mass Evolution (24h)', fontsize=14)

    for idx, mode in enumerate(MODES):
        ax = axes[idx]
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel('Hour')
        ax.set_ylabel('M_tot / M_tot(t=1h)')

        style = [
            ('M_dry_fortran', '0.3',       '-',  'Fortran'),
            ('M_dry_tfl',     '#1f77b4',   '--', 'TFL'),
            ('M_dry_ppm',     '#d62728',   ':',  'PPM'),
        ]
        for m in all_metrics[mode]:
            for key, color, ls, label in style:
                vals = m[key]
                if np.all(np.isnan(vals)):
                    continue
                ref = vals[0] if vals[0] > 0 else 1.0
                ax.plot(HOURS, vals / ref, color=color, alpha=0.25,
                        linewidth=0.8, linestyle=ls)

        for key, color, ls, label in style:
            ax.plot([], [], color=color, linewidth=2, linestyle=ls, label=label)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_total_M_timeseries.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Figure 3: Error evolution
# =========================================================================

def plot_error_evolution(all_metrics, output_dir=None):
    """Relative errors over time: 2x3 grid (TFL-Fort, PPM-Fort) x modes."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    _ensure_dir(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Relative Error Evolution (N_tot)', fontsize=14)

    comparisons = [
        ('tfl_vs_fortran_N_tot_relerr', 'TFL vs Fortran'),
        ('ppm_vs_fortran_N_tot_relerr', 'PPM vs Fortran'),
    ]

    for row, (err_key, row_label) in enumerate(comparisons):
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(MODE_LABELS[mode])
            ax.set_ylabel(row_label)
            ax.set_xlabel('Hour')

            # Collect error data per hour for box plots
            hourly_errors = [[] for _ in range(NHOURS)]
            for m in all_metrics[mode]:
                errs = m[err_key]
                for h in range(NHOURS):
                    if not np.isnan(errs[h]):
                        hourly_errors[h].append(errs[h])

            # Plot as box-whisker every 4 hours, individual lines otherwise
            for m in all_metrics[mode]:
                errs = m[err_key]
                if not np.all(np.isnan(errs)):
                    ax.plot(HOURS, errs, alpha=0.1, linewidth=0.3, color='tab:blue')

            # Median line
            medians = [np.median(he) if he else np.nan for he in hourly_errors]
            ax.plot(HOURS, medians, color='tab:red', linewidth=2, label='Median')

            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-10)
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_error_evolution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Figure 4: PPM vs TFL scatter
# =========================================================================

def plot_ppm_vs_tfl_scatter(all_metrics, output_dir=None):
    """PPM vs TFL final-hour agreement scatter plot."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    _ensure_dir(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('PPM vs TFL: Hour 24 Total Number', fontsize=14)

    for idx, mode in enumerate(MODES):
        ax = axes[idx]
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel('TFL N_tot(24h)')
        ax.set_ylabel('PPM N_tot(24h)')

        tfl_vals = []
        ppm_vals = []
        gc_vals = []

        for m in all_metrics[mode]:
            t = m['N_tot_tfl'][23]
            p = m['N_tot_ppm'][23]
            if not np.isnan(t) and not np.isnan(p) and t > 0 and p > 0:
                tfl_vals.append(t)
                ppm_vals.append(p)

        if tfl_vals:
            tfl_vals = np.array(tfl_vals)
            ppm_vals = np.array(ppm_vals)

            ax.scatter(tfl_vals, ppm_vals, alpha=0.6, s=20)

            # 1:1 line
            vmin = min(tfl_vals.min(), ppm_vals.min()) * 0.5
            vmax = max(tfl_vals.max(), ppm_vals.max()) * 2.0
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, label='1:1')
            ax.plot([vmin, vmax], [vmin * 0.8, vmax * 0.8], 'k:', linewidth=0.5)
            ax.plot([vmin, vmax], [vmin * 1.2, vmax * 1.2], 'k:', linewidth=0.5,
                    label='20% bands')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_ppm_vs_tfl_scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Figure 5: Mass conservation
# =========================================================================

def plot_mass_conservation(output_dir=None):
    """Mass balance error at hour 24 for all scenarios."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    _ensure_dir(output_dir)

    scenarios = get_scenarios()
    modes = ['cond_only', 'combined']
    methods = ['tfl', 'ppm']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Mass Conservation Error at Hour 24', fontsize=14)

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel('Scenario ID')
        ax.set_ylabel('Relative Mass Error')

        for method, color, offset in [('tfl', 'tab:blue', -0.15), ('ppm', 'tab:red', 0.15)]:
            errors = []
            sids = []
            for scen in scenarios:
                sid = scen['scenario_id']
                data = load_jax_results(sid, mode, method)
                if data is None:
                    continue

                # Dry aerosol mass at hour 24 + gas
                M_aero = np.sum(data['Mk'][23, :, :SRTH2O])
                M_gas = np.sum(data['Gc'][23])
                total_24 = M_aero + M_gas

                # Dry aerosol mass at hour 1 + gas
                M_aero_1 = np.sum(data['Mk'][0, :, :SRTH2O])
                M_gas_1 = np.sum(data['Gc'][0])
                total_1 = M_aero_1 + M_gas_1

                # Expected production between hours 1-24
                prod = scen['h2so4_prod_kg_per_s'] * 23 * 3600.0
                expected = total_1 + prod

                if expected > 1e-30:
                    err = (total_24 - expected) / expected
                    errors.append(err)
                    sids.append(sid)

            if errors:
                ax.bar(np.array(sids) + offset, np.abs(errors), width=0.3,
                       color=color, alpha=0.7, label=method.upper())

        ax.set_yscale('log')
        ax.axhline(0.01, color='k', linestyle='--', linewidth=0.5, label='1% target')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_mass_conservation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Figure 6: Size distribution evolution (3 representative scenarios)
# =========================================================================

def plot_size_distribution_evolution(output_dir=None):
    """Size distribution evolution for 3 representative scenarios."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    _ensure_dir(output_dir)

    # Pick 3 representative scenarios
    scenarios = get_scenarios()
    # Small particles (smallest GMD), large particles (largest GMD), middle
    gmds = [(s['GMD_um'], s['scenario_id']) for s in scenarios]
    gmds.sort()
    rep_sids = [gmds[0][1], gmds[len(gmds)//2][1], gmds[-1][1]]
    rep_labels = ['Small GMD', 'Medium GMD', 'Large GMD']

    # Bin boundaries (Fortran-style doubling)
    xk = np.zeros(NBINS + 1)
    xk[0] = 1.6033e-23
    for k in range(NBINS):
        xk[k + 1] = 2.0 * xk[k]

    # Geometric mean diameters [um]
    pi = 3.141592654
    rho = 1770.0
    Dp_mid = np.zeros(NBINS)
    for k in range(NBINS):
        m_mid = np.sqrt(xk[k] * xk[k + 1])
        Dp_mid[k] = 1e6 * (6.0 * m_mid / (rho * pi)) ** (1.0/3.0)

    hours_to_plot = [1, 6, 12, 24]

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle('Size Distribution Evolution (dN/dlogDp)', fontsize=14, y=0.98)

    for col, (sid, slabel) in enumerate(zip(rep_sids, rep_labels)):
        for row, hour in enumerate(hours_to_plot):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(f'{slabel} (S{sid:02d})', fontsize=10)
            ax.set_ylabel(f'Hour {hour}')

            # dlogDp
            dlogDp = np.log10(Dp_mid[1:]) - np.log10(Dp_mid[:-1])
            dlogDp = np.append(dlogDp, dlogDp[-1])

            # Load TFL and PPM
            for method, color, label in [('tfl', 'tab:blue', 'TFL'),
                                          ('ppm', 'tab:red', 'PPM')]:
                data = load_jax_results(sid, 'combined', method)
                if data is None:
                    continue
                Nk = data['Nk'][hour - 1]
                dNdlogDp = Nk / dlogDp
                ax.plot(Dp_mid, dNdlogDp, color=color, linewidth=1.0, label=label)

            # Try Fortran
            try:
                Nk_f, _, _ = load_fortran_hourly(sid, 'combined', hour)
                dNdlogDp_f = Nk_f / dlogDp
                ax.plot(Dp_mid, dNdlogDp_f, color='0.5', linewidth=1.0,
                        linestyle='--', label='Fortran')
            except Exception:
                pass

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(bottom=1)
            if row == 3:
                ax.set_xlabel('Dp [um]')
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_size_distribution_evolution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Figure 7: Size distribution at hour 24 — all scenarios
# =========================================================================

def _get_bin_diameters():
    """Return bin-center diameters [um] and dlogDp for TOMAS bins."""
    xk = np.zeros(NBINS + 1)
    xk[0] = 1.6033e-23
    for k in range(NBINS):
        xk[k + 1] = 2.0 * xk[k]
    pi = 3.141592654
    rho = 1770.0
    Dp_mid = np.zeros(NBINS)
    for k in range(NBINS):
        m_mid = np.sqrt(xk[k] * xk[k + 1])
        Dp_mid[k] = 1e6 * (6.0 * m_mid / (rho * pi)) ** (1.0 / 3.0)
    dlogDp = np.diff(np.log10(Dp_mid))
    dlogDp = np.append(dlogDp, dlogDp[-1])
    return Dp_mid, dlogDp


def plot_size_distribution_24h_all(output_dir=None):
    """Figure 7: 10x5 grid of size distributions at hour 24, combined mode."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    _ensure_dir(output_dir)

    Dp_mid, dlogDp = _get_bin_diameters()
    nrows, ncols = 10, 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 28), sharex=True)
    fig.suptitle('Size Distribution at Hour 24 — Combined Mode (all 50 scenarios)',
                 fontsize=14, y=0.995)

    for idx, sid in enumerate(SCENARIO_IDS):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Fortran
        try:
            Nk_f, _, _ = load_fortran_hourly(sid, 'combined', 24)
            ax.plot(Dp_mid, Nk_f / dlogDp, color='0.5', linewidth=1.0,
                    linestyle='--', label='Fortran')
        except Exception:
            pass

        # TFL & PPM
        for method, color, label in [('tfl', 'tab:blue', 'TFL'),
                                      ('ppm', 'tab:red', 'PPM')]:
            data = load_jax_results(sid, 'combined', method)
            if data is None:
                continue
            Nk = data['Nk'][23]
            ax.plot(Dp_mid, Nk / dlogDp, color=color, linewidth=0.8, label=label)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)
        ax.set_title(f'S{sid:02d}', fontsize=8, pad=2)
        ax.tick_params(labelsize=6)

        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')
        if col == 0:
            ax.set_ylabel('dN/dlogDp', fontsize=7)
        if row == nrows - 1:
            ax.set_xlabel('Dp [um]', fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_size_distribution_24h_all.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Figure 8: Per-bin error heatmap (PPM vs Fortran at hour 24)
# =========================================================================

def plot_size_distribution_error_heatmap(output_dir=None):
    """Figure 8: Heatmap of per-bin PPM vs Fortran relative error at hour 24."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    _ensure_dir(output_dir)

    Dp_mid, _ = _get_bin_diameters()
    n_scen = len(SCENARIO_IDS)
    error_matrix = np.full((n_scen, NBINS), np.nan)

    for idx, sid in enumerate(SCENARIO_IDS):
        # PPM data
        ppm_data = load_jax_results(sid, 'combined', 'ppm')
        if ppm_data is None:
            continue
        Nk_ppm = ppm_data['Nk'][23]

        # Fortran data
        try:
            Nk_f, _, _ = load_fortran_hourly(sid, 'combined', 24)
        except Exception:
            continue

        denom = np.maximum(np.abs(Nk_f), 1e-20)
        rel_err = np.abs(Nk_ppm - Nk_f) / denom
        error_matrix[idx, :] = rel_err

    fig, ax = plt.subplots(figsize=(14, 10))
    # Use log10 of relative error for colormap
    with np.errstate(divide='ignore', invalid='ignore'):
        log_err = np.log10(error_matrix)
    log_err = np.where(np.isfinite(log_err), log_err, np.nan)

    im = ax.pcolormesh(
        np.arange(NBINS + 1), np.arange(n_scen + 1),
        log_err, cmap='RdYlGn_r', vmin=-6, vmax=0,
        shading='flat'
    )
    cbar = fig.colorbar(im, ax=ax, label='log10(relative error)')

    # Label x-axis with Dp values
    tick_positions = np.arange(0, NBINS, 6) + 0.5
    tick_labels = [f'{Dp_mid[i]:.3f}' for i in range(0, NBINS, 6)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax.set_xlabel('Dp [um]')
    ax.set_ylabel('Scenario ID')
    ax.set_yticks(np.arange(n_scen) + 0.5)
    ax.set_yticklabels([f'S{sid:02d}' for sid in SCENARIO_IDS], fontsize=6)
    ax.set_title('PPM vs Fortran: Per-Bin Relative Error at Hour 24 (Combined)',
                 fontsize=12)

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_size_distribution_error_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Main
# =========================================================================

def generate_all_plots():
    """Generate all 6 summary figures."""
    print("Loading comparison metrics...")
    all_metrics = _load_all_metrics()

    print("\nGenerating Figure 1: Total N timeseries...")
    plot_total_N_timeseries(all_metrics)

    print("Generating Figure 2: Total M timeseries...")
    plot_total_M_timeseries(all_metrics)

    print("Generating Figure 3: Error evolution...")
    plot_error_evolution(all_metrics)

    print("Generating Figure 4: PPM vs TFL scatter...")
    plot_ppm_vs_tfl_scatter(all_metrics)

    print("Generating Figure 5: Mass conservation...")
    plot_mass_conservation()

    print("Generating Figure 6: Size distribution evolution...")
    plot_size_distribution_evolution()

    print("Generating Figure 7: Size distribution at hour 24 (all scenarios)...")
    plot_size_distribution_24h_all()

    print("Generating Figure 8: Per-bin error heatmap...")
    plot_size_distribution_error_heatmap()

    print("\nAll plots saved to benchmarks/results/")


if __name__ == '__main__':
    generate_all_plots()
