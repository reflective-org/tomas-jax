"""Statistical summary plots for Fortran-vs-JAX comparison.

Reads ``per_scenario_metrics.csv`` + ``per_scenario_metrics_summary.csv``
(produced by ``export_metrics_csv.py``) plus the existing timing CSVs,
and emits seven presentation-ready figures into
``benchmarks/results/24h/stats/``.

Figures:
    A. Taylor diagram (sigma_test/sigma_ref, correlation) per mode
    B. Q-Q plot of log10(N_tot_test / N_tot_ref)
    C. CDF of |N_tot_relerr| and |M_dry_relerr|
    D. Per-bin error heatmap (scenarios × bins)
    E. N_tot_relerr vs each LHC parameter
    F. Wall-clock comparison (Fortran vs JAX methods)
    G. JAX-only convergence (36/40/80 bins)

Usage::

    python -m benchmarks.python.plot_statistical_suite
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

from benchmarks.python import compare_24h as cmp24
from benchmarks.python.scenarios import get_scenarios, PARAM_NAMES

# --- Paths ------------------------------------------------------------------

_BASE = os.path.join(os.path.dirname(__file__), '..')
STATS_DIR = os.path.abspath(os.path.join(_BASE, 'results', '24h', 'stats'))
PER_SCENARIO_CSV = os.path.join(STATS_DIR, 'per_scenario_metrics.csv')
SUMMARY_CSV = os.path.join(STATS_DIR, 'per_scenario_metrics_summary.csv')
TIMING_JAX_CSV = os.path.abspath(os.path.join(_BASE, 'results', '24h', 'timing_jax.csv'))
TIMING_FORT_CSV = os.path.abspath(os.path.join(
    _BASE, '..', 'tomas_fortran', 'tomas_fortran', 'output', '24h', 'timing_fortran.csv'))

# --- Config -----------------------------------------------------------------

MODES = ('coag_only', 'cond_only', 'combined', 'nucl_cond', 'full')
PAIRS_FORTRAN = ('tfl_vs_fortran', 'ppm_vs_fortran')  # skip ppm_vs_tfl in comparisons
REPORT_HOUR = '24'

mpl.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 180,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# --- Data loaders -----------------------------------------------------------

def load_per_scenario() -> pd.DataFrame:
    df = pd.read_csv(PER_SCENARIO_CSV)
    # Normalise hour column (may come as int or str)
    df['hour'] = df['hour'].astype(str)
    return df


# --- Figure A: Taylor diagram -----------------------------------------------

def _plot_taylor_polar(ax, points: Iterable[tuple[float, float, str]],
                        ref_std: float = 1.0, max_std: float = 2.0,
                        title: str = ''):
    """Taylor diagram on a polar axis (upper-half only).

    `points`: iterable of (correlation r, sigma_test/sigma_ref, label).
    `ref_std`: reference standard deviation (plotted at (0, ref_std)).

    Correlation maps to angle via theta = arccos(r).  Radial = std ratio.
    RMSE-skill arcs are centered on the reference point.
    """
    # Correlation grid lines (meridians)
    corr_ticks = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    for c in corr_ticks:
        theta = np.arccos(c)
        ax.plot([theta, theta], [0, max_std], color='0.85', linewidth=0.7, zorder=1)
        ax.text(theta, max_std * 1.02, f'{c:g}', ha='center', va='bottom',
                fontsize=7, color='0.2')

    # Standard-deviation circles
    stds = np.linspace(0, max_std, 6)[1:]
    theta_arc = np.linspace(0, np.pi / 2, 90)
    for s in stds:
        ax.plot(theta_arc, np.full_like(theta_arc, s),
                color='0.85', linewidth=0.7, zorder=1)
    ax.set_rmax(max_std)

    # Reference point
    ax.plot([0], [ref_std], marker='*', color='black', ms=12, zorder=4,
            label='Reference')

    # Plot data points
    styles = {
        'tfl_vs_fortran': dict(marker='o', color='#1f77b4', label='TFL vs Fortran'),
        'ppm_vs_fortran': dict(marker='s', color='#d62728', label='PPM vs Fortran'),
        'ppm_vs_tfl':     dict(marker='^', color='#2ca02c', label='PPM vs TFL'),
    }
    seen_labels = set()
    for r, sigma_ratio, pair in points:
        if not np.isfinite(r) or not np.isfinite(sigma_ratio):
            continue
        r_clip = np.clip(r, -0.99, 0.99)  # arccos domain safety
        theta = np.arccos(r_clip)
        s = styles.get(pair, dict(marker='x', color='black'))
        lbl = s.get('label') if s.get('label') not in seen_labels else None
        if lbl:
            seen_labels.add(lbl)
        ax.scatter(theta, min(sigma_ratio, max_std), marker=s['marker'],
                   c=s['color'], s=32, alpha=0.7, edgecolors='white',
                   linewidths=0.5, zorder=3, label=lbl)

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_xlabel('')
    ax.set_rlabel_position(135)
    ax.set_title(title, pad=16, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=8)


def figure_a_taylor(df: pd.DataFrame, out_path: str):
    """Taylor diagram per mode; sigma ratio = std(test_Nk) / std(ref_Nk).

    Because we logged `Nk_pearson_r` + don't have per-row sigma directly,
    we approximate sigma_ratio from the identity
        r^2 * sigma_ratio^2 + (sigma_ratio - r)^2 = 1 - R2
    i.e. sigma_ratio solves 1 - R2 = 1 - 2 r sigma_ratio + sigma_ratio^2,
    which rearranges to sigma_ratio = r +/- sqrt(R2 + 2 r sigma_ratio - 1)…
    Simpler: emit sigma_ratio via Nk_rmse and Nk_std? We lack those columns.

    Workaround: use `sigma_ratio = 1 + Nk_bias / max(Nk_rmse, atol)` is also
    wrong. Since std data isn't in the CSV, we recompute from source NPZs/CSVs
    on the fly via a small helper that takes advantage of compare_24h loaders.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                              subplot_kw=dict(projection='polar'))
    axes = axes.ravel()

    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        sub = df[(df['mode'] == mode) & (df['hour'] == REPORT_HOUR)]
        points = []
        for _, row in sub.iterrows():
            r = float(row['Nk_pearson_r'])
            # sigma_ratio derived from r^2 and R^2 identity:
            # R2 = 1 - (1 - 2 r alpha + alpha^2) when means match → alpha = r (optimal)
            # Better: compute sigma_ratio from raw source data on demand.
            sigma_ratio = _sigma_ratio_from_source(
                int(row['scenario_id']), mode, row['method_pair'])
            points.append((r, sigma_ratio, row['method_pair']))
        _plot_taylor_polar(ax, points, title=f'{mode}  (h24)')

    # Hide unused axes
    for i in range(len(MODES), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Taylor diagrams — Nk at hour 24 (one point per scenario)',
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig_A → {out_path}")


_SIGMA_CACHE: dict = {}

def _sigma_ratio_from_source(scenario_id: int, mode: str, pair: str) -> float:
    """Load Nk at hour 24 from NPZ/CSV, compute std_test/std_ref."""
    key = (scenario_id, mode, pair)
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]

    fort_mode = cmp24.MODE_NAMES[mode]
    try:
        Nk_f, _, _ = cmp24.load_fortran_hourly(scenario_id, fort_mode, 24)
    except Exception:
        Nk_f = None

    tfl = cmp24.load_jax_results(scenario_id, mode, 'tfl_jit') \
        or cmp24.load_jax_results(scenario_id, mode, 'tfl')
    ppm = cmp24.load_jax_results(scenario_id, mode, 'ppm_jit') \
        or cmp24.load_jax_results(scenario_id, mode, 'ppm')
    if mode == 'coag_only':
        ppm = tfl

    def std_arr(data):
        if data is None:
            return float('nan')
        return float(np.std(data['Nk'][23]))

    std_f = float('nan') if Nk_f is None else float(np.std(Nk_f))
    std_t = std_arr(tfl)
    std_p = std_arr(ppm)

    if pair == 'tfl_vs_fortran':
        ratio = std_t / std_f if std_f else float('nan')
    elif pair == 'ppm_vs_fortran':
        ratio = std_p / std_f if std_f else float('nan')
    elif pair == 'ppm_vs_tfl':
        ratio = std_p / std_t if std_t else float('nan')
    else:
        ratio = float('nan')
    _SIGMA_CACHE[key] = ratio
    return ratio


# --- Figure B: Q-Q plot of log10 ratios -------------------------------------

def figure_b_qq(df: pd.DataFrame, out_path: str):
    """Q-Q of log10(N_tot_test / N_tot_ref) at h24, one subplot per mode."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.ravel()

    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        sub = df[(df['mode'] == mode) & (df['hour'] == REPORT_HOUR)]
        any_plotted = False
        for pair in PAIRS_FORTRAN:
            s = sub[sub['method_pair'] == pair]
            ref = s['N_tot_ref'].values.astype(float)
            test = s['N_tot_test'].values.astype(float)
            mask = np.isfinite(ref) & np.isfinite(test) & (ref > 0) & (test > 0)
            if mask.sum() < 3:
                continue
            log_ratio = np.log10(test[mask] / ref[mask])
            log_ratio = np.sort(log_ratio)
            n = len(log_ratio)
            # Theoretical normal quantiles
            quantiles = (np.arange(1, n + 1) - 0.5) / n
            theoretical = _norm_ppf(quantiles)
            ax.plot(theoretical, log_ratio, 'o', ms=4, alpha=0.7,
                    label=pair.replace('_', ' '))
            any_plotted = True

        if any_plotted:
            ax.axhline(0, color='black', lw=0.5, alpha=0.5)
            ax.set_xlabel('Theoretical normal quantile')
            ax.set_ylabel(r'log$_{10}$(N$_{test}$/N$_{ref}$)')
            ax.legend(fontsize=8, loc='best')
        ax.set_title(mode)
    for i in range(len(MODES), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(r'Q-Q plot — log$_{10}$(N$_{tot,\,test}$/N$_{tot,\,ref}$) at h24',
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig_B → {out_path}")


def _norm_ppf(p):
    """Inverse normal CDF via rational approximation (no scipy dep)."""
    from math import sqrt, log
    p = np.asarray(p, dtype=np.float64)
    # Beasley-Springer-Moro approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    out = np.empty_like(p)
    plow = 0.02425
    phigh = 1 - plow
    low = p < plow
    high = p > phigh
    mid = ~(low | high)

    q = np.sqrt(-2 * np.log(p[low]))
    out[low] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    q = np.sqrt(-2 * np.log(1 - p[high]))
    out[high] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    return out


# --- Figure C: CDF of relative errors ---------------------------------------

def figure_c_cdf(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        sub = df[(df['mode'] == mode) & (df['hour'] == REPORT_HOUR)]
        for pair in PAIRS_FORTRAN:
            s = sub[sub['method_pair'] == pair]
            for col, style in (('N_tot_relerr', '-'), ('M_dry_relerr', '--')):
                vals = np.abs(s[col].values.astype(float))
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                sorted_v = np.sort(vals)
                ecdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
                label = f"{pair.split('_vs_')[0].upper()} {col.split('_')[0]}"
                ax.plot(sorted_v, ecdf, style, lw=1.3, label=label, alpha=0.85)
        ax.set_xscale('symlog', linthresh=1e-10)
        ax.set_xlim(left=1e-10)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('|relative error|')
        ax.set_ylabel('Empirical CDF')
        ax.set_title(mode)
        ax.legend(fontsize=7, loc='lower right')

    for i in range(len(MODES), len(axes)):
        axes[i].set_visible(False)
    fig.suptitle('Empirical CDF of relative errors at h24', fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig_C → {out_path}")


# --- Figure D: per-bin error heatmap ----------------------------------------

def figure_d_heatmap(df: pd.DataFrame, out_path: str):
    """Heatmap of log_bias(Nk) across scenarios × bins at h24.

    Builds the (n_scen × 36) matrix by reloading the Nk arrays for the
    TFL_JIT vs Fortran comparison (most straightforward representative).
    """
    n_bins = 36
    scens = sorted(df['scenario_id'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for ax, mode in zip(axes, ('coag_only', 'cond_only', 'combined')):
        heat = np.full((len(scens), n_bins), np.nan)
        gmd_vals = np.full(len(scens), np.nan)

        fort_mode = cmp24.MODE_NAMES[mode]
        scenarios = {s['scenario_id']: s for s in get_scenarios()}
        for i, sid in enumerate(scens):
            try:
                Nk_f, _, _ = cmp24.load_fortran_hourly(sid, fort_mode, 24)
            except Exception:
                continue
            tfl = cmp24.load_jax_results(sid, mode, 'tfl_jit') \
                or cmp24.load_jax_results(sid, mode, 'tfl')
            if tfl is None:
                continue
            Nk_t = tfl['Nk'][23]
            # log_bias per bin
            atol = 1e-5
            with np.errstate(invalid='ignore', divide='ignore'):
                heat[i, :] = np.log10(np.maximum(Nk_t, atol)) - \
                             np.log10(np.maximum(Nk_f, atol))
            gmd_vals[i] = scenarios[sid].get('GMD_um', np.nan)

        # Sort by GMD
        order = np.argsort(gmd_vals)
        heat_sorted = heat[order, :]

        # Symmetric colormap around zero
        vmax = np.nanpercentile(np.abs(heat), 99)
        im = ax.imshow(heat_sorted, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, n_bins, len(scens), 0])
        ax.set_xlabel('Bin index')
        ax.set_ylabel('Scenario (sorted by GMD)')
        ax.set_title(f'{mode}  TFL vs Fortran')
        plt.colorbar(im, ax=ax, label=r'log$_{10}$(N$_{JAX}$/N$_{Fortran}$)',
                     shrink=0.85)

    fig.suptitle('Per-bin log-bias at hour 24', fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig_D → {out_path}")


# --- Figure E: error vs LHC parameters --------------------------------------

def figure_e_error_vs_params(df: pd.DataFrame, out_path: str):
    sub = df[(df['mode'] == 'combined') &
             (df['hour'] == REPORT_HOUR) &
             (df['method_pair'] == 'tfl_vs_fortran')].copy()
    if sub.empty:
        print('  [warn] no data for fig_E')
        return
    sub['abs_N_relerr'] = np.abs(sub['N_tot_relerr'].astype(float))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for ax, pname in zip(axes, PARAM_NAMES):
        x = sub[pname].astype(float).values
        y = sub['abs_N_relerr'].values
        ax.scatter(x, y, alpha=0.7, s=28, edgecolors='white', linewidths=0.5)
        if pname in ('N_total', 'GMD_um', 'Gc_SO4_kg', 'h2so4_prod'):
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(pname)
        ax.set_ylabel(r'|N$_{tot}$ relerr| (h24)')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Combined-mode N_tot error vs LHC parameter (TFL vs Fortran, h24)',
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig_E → {out_path}")


# --- Figure F: wall-clock comparison ----------------------------------------

def _load_timing_fortran():
    if not os.path.exists(TIMING_FORT_CSV):
        print(f'  [warn] missing Fortran timing: {TIMING_FORT_CSV}')
        return None
    df = pd.read_csv(TIMING_FORT_CSV)
    # strip whitespace from columns
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_timing_jax():
    if not os.path.exists(TIMING_JAX_CSV):
        return None
    return pd.read_csv(TIMING_JAX_CSV)


def figure_f_walltime(out_path: str):
    fort = _load_timing_fortran()
    jax_df = _load_timing_jax()

    if fort is None and jax_df is None:
        print('  [warn] no timing data; skipping fig_F')
        return

    # Build a long-form dataframe: (solver_label, mode, wall_time_s)
    records = []
    if fort is not None:
        for _, row in fort.iterrows():
            for mode_col, mode_name in (('coag_s', 'coag_only'),
                                         ('cond_s', 'cond_only'),
                                         ('combined_s', 'combined')):
                v = float(row.get(mode_col, -1))
                if v > 0:
                    records.append(dict(solver='Fortran', mode=mode_name, t=v))
    if jax_df is not None:
        for _, row in jax_df.iterrows():
            t = float(row['wall_time_s'])
            if t <= 0:
                continue
            records.append(dict(solver=row['method'].upper(),
                                mode=row['mode'], t=t))

    if not records:
        print('  [warn] no timing records')
        return
    tdf = pd.DataFrame(records)

    modes = ['coag_only', 'cond_only', 'combined']
    solvers = sorted(tdf['solver'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.8 / len(solvers)
    colors = plt.cm.tab10(np.linspace(0, 1, len(solvers)))

    for i, solver in enumerate(solvers):
        meds = []
        p05 = []
        p95 = []
        for mode in modes:
            v = tdf[(tdf['solver'] == solver) & (tdf['mode'] == mode)]['t'].values
            if v.size:
                meds.append(np.median(v))
                p05.append(np.percentile(v, 5))
                p95.append(np.percentile(v, 95))
            else:
                meds.append(np.nan)
                p05.append(np.nan)
                p95.append(np.nan)
        xs = np.arange(len(modes)) + (i - len(solvers)/2 + 0.5) * width
        meds = np.array(meds)
        yerr = np.vstack([meds - np.array(p05), np.array(p95) - meds])
        ax.bar(xs, meds, width=width * 0.9, color=colors[i],
               yerr=yerr, capsize=3, label=solver, alpha=0.88)

    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_ylabel('Wall time per 24h scenario [s]  (log scale)')
    ax.set_title('Fortran vs JAX wall-clock (median, 5/95 pctile bars)')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig_F → {out_path}")


# --- Figure G: JAX-only convergence (36/40/80 bins) -------------------------

def figure_g_convergence(out_path: str):
    """Run short JAX-only convergence test for 2 scenarios × 3 resolutions.

    Uses tomas_jax directly (not the benchmark harness) since Fortran can't
    match arbitrary bin counts.  Demonstrates JAX scales cleanly to 80 bins.
    """
    try:
        import jax.numpy as jnp
        from tomas_jax.core.config import (
            make_grid, XK0_LEGACY, make_grid_80bin,
        )
        from tomas_jax.core.state import TomasState
        from tomas_jax.solvers.diffrax import coag_euler_step
        from benchmarks.python.convergence_test import init_lognormal
    except Exception as e:
        print(f'  [warn] skipping fig_G (imports failed: {e})')
        return

    import time

    scens = get_scenarios()
    lo = min(scens, key=lambda s: s['N_total'])
    hi = max(scens, key=lambda s: s['N_total'])

    # Grid configurations: (nbins, label)
    resolutions = [
        (36, make_grid(36, XK0_LEGACY, 2.0), 'make_grid(36, 2x)'),
        (40, make_grid(40, XK0_LEGACY, 2.0), 'make_grid(40, 2x)'),
        (80, make_grid_80bin(),               '80-bin (√2×)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scen, title in ((axes[0], lo, f"Low-N  (N={lo['N_total']:.1e}#/cc)"),
                             (axes[1], hi, f"High-N (N={hi['N_total']:.1e}#/cc)")):
        for nb, xk, lbl in resolutions:
            try:
                Nk0_np, Mk0_np = init_lognormal(
                    scen['N_total'], scen['GMD_um'], scen['GSD'], xk)
                Nk_j = jnp.array(Nk0_np)
                Mk_j = jnp.array(Mk0_np)
                xk_j = jnp.array(xk)
                dt = 3600.0
                nsteps = 24
                t0 = time.time()
                for _ in range(nsteps):
                    Nk_j, Mk_j = coag_euler_step(
                        Nk_j, Mk_j, xk_j,
                        float(scen['temp']), float(scen['pres']), 1.0e6,
                        dt=dt, n_substeps=60,
                    )
                # Block on device so wall time is meaningful
                Nk_j.block_until_ready()
                wall = time.time() - t0
                Nk_final = np.asarray(Nk_j)
                Dpk = (6.0 * np.asarray(xk)[:-1] / (np.pi * 1770.0)) ** (1/3) * 1e6
                ax.plot(Dpk, np.maximum(Nk_final, 1.0),
                        marker='o', ms=3, lw=1.0,
                        label=f'{nb} bins  (wall {wall:.2f}s)')
            except Exception as e:
                print(f'  [warn] fig_G {nb}-bin failed: {e}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'D$_p$ [$\mu$m]')
        ax.set_ylabel(r'N$_k$ at t=24h')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle('JAX coagulation convergence across bin resolutions '
                 '(no Fortran — Fortran harness hardcodes 36 bins)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig_G → {out_path}")


# --- Entrypoint -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--skip', nargs='+', default=(),
                   help='Figure letters to skip (A,B,C,D,E,F,G)')
    args = p.parse_args()

    skip = set(s.upper() for s in args.skip)
    os.makedirs(STATS_DIR, exist_ok=True)
    df = load_per_scenario()

    print("Generating statistical suite...")
    if 'A' not in skip:
        figure_a_taylor(df, os.path.join(STATS_DIR, 'fig_A_taylor.png'))
    if 'B' not in skip:
        figure_b_qq(df, os.path.join(STATS_DIR, 'fig_B_qq.png'))
    if 'C' not in skip:
        figure_c_cdf(df, os.path.join(STATS_DIR, 'fig_C_cdf.png'))
    if 'D' not in skip:
        figure_d_heatmap(df, os.path.join(STATS_DIR, 'fig_D_perbin_heatmap.png'))
    if 'E' not in skip:
        figure_e_error_vs_params(df, os.path.join(STATS_DIR, 'fig_E_error_vs_params.png'))
    if 'F' not in skip:
        figure_f_walltime(os.path.join(STATS_DIR, 'fig_F_walltime.png'))
    if 'G' not in skip:
        figure_g_convergence(os.path.join(STATS_DIR, 'fig_G_convergence.png'))

    print("Done.")


if __name__ == '__main__':
    main()
