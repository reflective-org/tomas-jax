"""Comprehensive plot suite for the 1000-scenario atmospheric benchmark.

Twelve figures summarising JAX PPM_JIT vs Fortran PPM across troposphere,
UTLS, and stratosphere.  Reads the per-scenario metrics CSV produced by
export_atmos_metrics.py plus original NPZ/CSV data for representative
scenario overlays.

Usage::

    python -m benchmarks.python.plot_atmos_suite
    python -m benchmarks.python.plot_atmos_suite --skip 02 11   # skip figs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from benchmarks.python import export_atmos_metrics as exp_mod
from benchmarks.python.export_atmos_metrics import (
    JAX_DIR, FORTRAN_DIR, STATS_DIR, NBINS, SRTH2O, REPORT_HOURS,
    load_jax_npz, load_fortran_hourly, classify_regime,
)
from benchmarks.python.scenarios_atmos import (
    get_atmos_scenarios, REGIMES, REGIME_RANGES,
)
from benchmarks.python.metrics import log_bias

# --- Styling ----------------------------------------------------------------

mpl.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 180,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

REGIME_COLORS = {'tropo': '#1f77b4', 'utls': '#ff7f0e', 'strato': '#d62728'}
MODE_MARKERS = {'coag_only': 'o', 'cond_only': 's', 'combined': '^'}
REPORT_HOUR = 24

PER_SCENARIO_CSV = STATS_DIR / 'per_scenario_metrics.csv'
SUMMARY_CSV = STATS_DIR / 'per_scenario_metrics_summary.csv'

# US Standard Atmosphere nominal profile (approx, for annotation)
_US_STD_T_K = np.array([288, 280, 270, 250, 230, 217, 217, 220, 235, 260, 270, 250, 220])
_US_STD_P_PA = np.array([101325, 70000, 50000, 30000, 18000, 10000, 5500, 3000, 1000, 200, 50, 10, 2])


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(PER_SCENARIO_CSV)
    df['hour'] = df['hour'].astype(int)
    df['regime'] = df['regime'].astype(str)
    return df


# --- Figure 01: atmospheric coverage (T, P) scatter -------------------------

def fig_01_coverage_TP(df: pd.DataFrame, out_path: Path):
    sub = df[df['hour'] == REPORT_HOUR].drop_duplicates('scenario_id')
    fig, ax = plt.subplots(figsize=(10, 7))
    for regime in REGIMES:
        m = sub['regime'] == regime
        ax.scatter(sub.loc[m, 'pres'] / 100.0, sub.loc[m, 'temp'],
                   s=15, alpha=0.7, label=regime,
                   color=REGIME_COLORS[regime], edgecolors='white',
                   linewidths=0.3)
    # Overlay US Standard Atmosphere
    ax.plot(_US_STD_P_PA / 100.0, _US_STD_T_K,
            'k-', lw=1.3, alpha=0.7, label='US Std Atm (nominal)')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('Pressure [hPa] (log, surface → top)')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Atmospheric coverage of the 1000-scenario LHC '
                 f'({len(sub)} scenarios)')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 02: 8-parameter pairplot (compact) ------------------------------

def fig_02_pairplot(df: pd.DataFrame, out_path: Path):
    cols = ['temp', 'pres', 'RH', 'N_total', 'GMD_um', 'GSD', 'Gc_SO4_kg',
            'h2so4_prod']
    sub = df[df['hour'] == REPORT_HOUR].drop_duplicates('scenario_id')
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(16, 16),
                              sharex=False, sharey=False)

    log_cols = {'pres', 'RH', 'N_total', 'GMD_um', 'Gc_SO4_kg', 'h2so4_prod'}

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram per regime
                for regime in REGIMES:
                    m = sub['regime'] == regime
                    vals = sub.loc[m, ci].values
                    if ci in log_cols:
                        vals = np.log10(np.maximum(vals, 1e-30))
                    ax.hist(vals, bins=15, alpha=0.5,
                             color=REGIME_COLORS[regime], label=regime,
                             histtype='stepfilled')
                if ci in log_cols:
                    ax.set_xlabel('log10 ' + ci, fontsize=7)
                else:
                    ax.set_xlabel(ci, fontsize=7)
                ax.tick_params(labelsize=6)
                if i == 0 and j == 0:
                    ax.legend(fontsize=6, loc='best')
            else:
                for regime in REGIMES:
                    m = sub['regime'] == regime
                    x = sub.loc[m, cj].values
                    y = sub.loc[m, ci].values
                    ax.scatter(x, y, s=3, alpha=0.4,
                               color=REGIME_COLORS[regime])
                if cj in log_cols: ax.set_xscale('log')
                if ci in log_cols: ax.set_yscale('log')
                ax.tick_params(labelsize=6)
                if i == n - 1:
                    ax.set_xlabel(cj, fontsize=7)
                if j == 0:
                    ax.set_ylabel(ci, fontsize=7)
    fig.suptitle('8-parameter pairplot, regime-colored', y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 03: Taylor diagram per regime (with σ zoom) ---------------------

_SIGMA_CACHE: dict = {}


def _sigma_ratio(sid: int, mode: str) -> float:
    """Compute std(test Nk) / std(ref Nk) at h24 from raw NPZ/CSV."""
    key = (sid, mode)
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]
    Nk_f, _ = load_fortran_hourly(sid, mode, 24)
    jax_data = load_jax_npz(sid, mode)
    if Nk_f is None or jax_data is None:
        _SIGMA_CACHE[key] = float('nan')
        return float('nan')
    Nk_j = jax_data['Nk'][23]
    std_f = float(np.std(Nk_f))
    std_j = float(np.std(Nk_j))
    ratio = std_j / std_f if std_f > 0 else float('nan')
    _SIGMA_CACHE[key] = ratio
    return ratio


def _taylor_polar_to_xy(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def _taylor_axes_cartesian(ax, min_corr=0.8, min_std=0.5, max_std=1.3):
    """Zoomed Taylor diagram on Cartesian axes (tight fit, no polar waste).

    The wedge spans theta in [0, arccos(min_corr)] with radius in
    [min_std, max_std]. Axes are plain x/y — we draw the polar structure
    (arcs, meridians) as line artists so aspect='equal' works and the
    bounding box matches the wedge exactly.
    """
    theta_max = np.arccos(min_corr)
    corr_ticks = [v for v in (0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)
                  if v >= min_corr]
    theta_arc = np.linspace(0, theta_max, 120)

    # sigma-ratio arcs
    std_ticks = np.linspace(min_std, max_std, 5)
    for s in std_ticks:
        style = dict(color='0.85', lw=0.6)
        if abs(s - 1.0) < 1e-6:
            style = dict(color='0.55', lw=1.1)
        x, y = _taylor_polar_to_xy(s, theta_arc)
        ax.plot(x, y, zorder=1, **style)

    # correlation meridians + labels at the max_std arc
    for c in corr_ticks:
        theta = np.arccos(c)
        x0, y0 = _taylor_polar_to_xy(min_std, theta)
        x1, y1 = _taylor_polar_to_xy(max_std, theta)
        ax.plot([x0, x1], [y0, y1], color='0.85', lw=0.6, zorder=1)
        lx, ly = _taylor_polar_to_xy(max_std * 1.03, theta)
        ax.text(lx, ly, f'{c:g}', ha='center', va='bottom',
                fontsize=9, color='0.25')

    # Radial (sigma) tick labels along theta=0 line
    for s in std_ticks:
        ax.text(s, -0.02, f'{s:.2g}', ha='center', va='top',
                fontsize=9, color='0.35')

    # Bounding arcs (outer, inner, top-meridian, bottom edge)
    ax.plot(*_taylor_polar_to_xy(max_std, theta_arc),
            color='0.25', lw=1.0, zorder=2)
    ax.plot(*_taylor_polar_to_xy(min_std, theta_arc),
            color='0.25', lw=1.0, zorder=2)
    x0, y0 = _taylor_polar_to_xy(min_std, 0.0)
    x1, y1 = _taylor_polar_to_xy(max_std, 0.0)
    ax.plot([x0, x1], [y0, y1], color='0.25', lw=1.0, zorder=2)
    x0, y0 = _taylor_polar_to_xy(min_std, theta_max)
    x1, y1 = _taylor_polar_to_xy(max_std, theta_max)
    ax.plot([x0, x1], [y0, y1], color='0.25', lw=1.0, zorder=2)

    # Reference point (sigma=1, r=1)
    ax.plot([1.0], [0.0], marker='*', color='black', ms=18, zorder=5,
            markeredgecolor='white', markeredgewidth=1.0)

    # Tight bbox around the wedge
    x_min = min_std * np.cos(theta_max)
    x_max = max_std * 1.06
    y_min = -0.05
    y_max = max_std * np.sin(theta_max) * 1.10
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def fig_03_taylor_regime(df: pd.DataFrame, out_path: Path):
    sub = df[df['hour'] == REPORT_HOUR].copy()

    corrs = sub['Nk_pearson_r'].astype(float).values
    corrs = corrs[np.isfinite(corrs)]
    auto_min = float(np.nanpercentile(corrs, 1)) if corrs.size else 0.8
    min_corr = float(np.clip(auto_min - 0.02, 0.80, 0.95))

    mode_colors = {'coag_only': '#1f77b4',
                   'cond_only': '#ff7f0e',
                   'combined':  '#2ca02c'}

    min_std, max_std = 0.5, 1.3
    theta_max = np.arccos(min_corr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    legend_handles: dict = {}
    for ax, regime in zip(axes, REGIMES):
        _taylor_axes_cartesian(ax, min_corr=min_corr,
                                min_std=min_std, max_std=max_std)
        bucket = sub[sub['regime'] == regime]
        for mode in ('coag_only', 'cond_only', 'combined'):
            m = bucket[bucket['mode'] == mode]
            xs, ys = [], []
            for _, r in m.iterrows():
                corr = float(r['Nk_pearson_r'])
                if not np.isfinite(corr) or corr < min_corr:
                    continue
                sigma = _sigma_ratio(int(r['scenario_id']), mode)
                if not np.isfinite(sigma):
                    continue
                theta = np.arccos(np.clip(corr, min_corr, 0.9999))
                sigma = float(np.clip(sigma, min_std, max_std))
                x, y = _taylor_polar_to_xy(sigma, theta)
                xs.append(x); ys.append(y)
            if xs:
                h = ax.scatter(xs, ys, marker=MODE_MARKERS[mode],
                                s=36, alpha=0.55,
                                edgecolors='white', linewidths=0.4,
                                color=mode_colors[mode], label=mode)
                legend_handles[mode] = h

        n_uniq = len(bucket["scenario_id"].unique())
        ax.set_title(f'{regime}  (n={n_uniq})',
                      fontsize=13, fontweight='bold',
                      color=REGIME_COLORS[regime], pad=8)

    handles = list(legend_handles.values())
    labels = list(legend_handles.keys())
    ref_h = plt.Line2D([0], [0], marker='*', color='w',
                        markerfacecolor='black', markeredgecolor='white',
                        markersize=12, linestyle='None')
    handles.append(ref_h)
    labels.append('reference  (sigma=1, r=1)')
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle('Taylor diagrams: Nk at h24, JAX PPM_JIT vs Fortran PPM\n'
                 'angle = correlation (r), radius = sigma_JAX / sigma_Fortran',
                 fontsize=13, fontweight='bold', y=0.99)

    fig.tight_layout(rect=[0, 0.07, 1, 0.91])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 04: violin of skill metrics -------------------------------------

def fig_04_violin_skill(df: pd.DataFrame, out_path: Path):
    sub = df[df['hour'] == REPORT_HOUR].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    mode_order = ('coag_only', 'cond_only', 'combined')

    for ax, metric, title in zip(axes,
                                  ('Nk_r2', 'Nk_kge'),
                                  (r'N$_k$ R$^2$', r'N$_k$ KGE')):
        data = []
        labels = []
        colors = []
        pos = []
        x = 0
        for regime in REGIMES:
            for mode in mode_order:
                vals = sub[(sub['regime'] == regime)
                            & (sub['mode'] == mode)][metric].values
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                data.append(vals)
                labels.append(f'{regime}\n{mode}')
                colors.append(REGIME_COLORS[regime])
                pos.append(x)
                x += 1
            x += 0.5  # gap between regimes

        vp = ax.violinplot(data, positions=pos, widths=0.8,
                            showmedians=True, showextrema=False)
        for b, c in zip(vp['bodies'], colors):
            b.set_facecolor(c)
            b.set_alpha(0.6)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(1.0, color='k', lw=0.5, alpha=0.5, linestyle='--')
        ax.set_ylabel(title)
        ax.set_title(title + ' distribution at h24')
        ax.set_ylim(-0.2, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 05: CDF of |N_tot| and |M_dry| errors ---------------------------

def fig_05_cdf_regime(df: pd.DataFrame, out_path: Path):
    sub = df[df['hour'] == REPORT_HOUR].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, col, title in zip(axes,
                                ('N_tot_relerr', 'M_dry_relerr'),
                                (r'$|\Delta N_{tot}|$ relerr',
                                 r'$|\Delta M_{dry}|$ relerr')):
        for regime in REGIMES:
            vals = np.abs(sub[sub['regime'] == regime][col].values)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            vals.sort()
            ecdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, ecdf, lw=1.5, color=REGIME_COLORS[regime],
                    label=f'{regime} (n={len(vals)})')
        ax.set_xscale('symlog', linthresh=1e-10)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(title)
        ax.set_ylabel('Empirical CDF')
        ax.set_title(title + ' across regimes (h24)')
        ax.legend(loc='lower right')
        ax.axhline(0.95, color='k', lw=0.5, alpha=0.4, ls=':')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 06: error vs pressure ------------------------------------------

def fig_06_error_vs_pressure(df: pd.DataFrame, out_path: Path):
    sub = df[(df['hour'] == REPORT_HOUR) & (df['mode'] == 'combined')].copy()
    sub['abs_N_relerr'] = np.abs(sub['N_tot_relerr'].astype(float))
    fig, ax = plt.subplots(figsize=(10, 6))
    for regime in REGIMES:
        m = sub['regime'] == regime
        ax.scatter(sub.loc[m, 'pres'] / 100.0,
                   sub.loc[m, 'abs_N_relerr'],
                   s=14, alpha=0.65, color=REGIME_COLORS[regime],
                   label=regime, edgecolors='white', linewidths=0.3)
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_yscale('log')
    ax.set_xlabel('Pressure [hPa]')
    ax.set_ylabel(r'$|N_{tot}$ relerr$|$ at h24, combined mode')
    ax.set_title('Combined-mode error vs pressure across the 1000-scenario LHC')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 07: error hex-bin in (T, P) plane ------------------------------

def fig_07_error_contour_TP(df: pd.DataFrame, out_path: Path):
    sub = df[(df['hour'] == REPORT_HOUR) & (df['mode'] == 'combined')].copy()
    x = np.log10(sub['pres'].astype(float).values / 100.0)
    y = sub['temp'].astype(float).values
    c = np.log10(np.abs(sub['N_tot_relerr'].astype(float).values) + 1e-12)
    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(x, y, C=c, reduce_C_function=np.median,
                   gridsize=25, cmap='viridis')
    cbar = fig.colorbar(hb, ax=ax,
                        label=r'log$_{10}$(median $|\Delta N|$ relerr)')
    ax.set_xlabel(r'log$_{10}$(Pressure [hPa])')
    ax.set_ylabel('Temperature [K]')
    ax.invert_xaxis()
    ax.set_title('Median combined-mode N error in (log P, T) plane (h24)')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 08: per-bin heatmap stratified by regime -----------------------

def fig_08_perbin_heatmap(df: pd.DataFrame, out_path: Path):
    scens = {s['scenario_id']: s for s in get_atmos_scenarios()}
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)

    for ax, regime in zip(axes, REGIMES):
        sids = sorted(
            [s['scenario_id'] for s in scens.values() if s['regime'] == regime]
        )
        gmds = np.array([scens[sid]['GMD_um'] for sid in sids])
        order = np.argsort(gmds)
        sids = [sids[i] for i in order]

        heat = np.full((len(sids), NBINS), np.nan)
        for i, sid in enumerate(sids):
            Nk_f, _ = load_fortran_hourly(sid, 'combined', 24)
            jax_data = load_jax_npz(sid, 'combined')
            if Nk_f is None or jax_data is None:
                continue
            Nk_j = jax_data['Nk'][23]
            atol = 1.0
            with np.errstate(invalid='ignore', divide='ignore'):
                heat[i, :] = (np.log10(np.maximum(Nk_j, atol))
                              - np.log10(np.maximum(Nk_f, atol)))

        vmax = np.nanpercentile(np.abs(heat), 99) if np.any(np.isfinite(heat)) else 1
        im = ax.imshow(heat, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[0, NBINS, len(sids), 0])
        ax.set_xlabel('Bin index')
        ax.set_ylabel(f'{regime} scenarios (sorted by GMD)')
        ax.set_title(f'{regime} (n={len(sids)})')
        plt.colorbar(im, ax=ax,
                     label=r'log$_{10}$(N$_{JAX}$/N$_{Fortran}$)',
                     shrink=0.7)
    fig.suptitle('Per-bin log-bias heatmap, combined mode h24, '
                 'scenarios sorted by GMD within regime',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 09: error evolution vs time -------------------------------------

def fig_09_error_evolution(out_path: Path):
    """Reload JAX NPZ + Fortran CSV for each hour ∈ {1,6,12,24} and compute
    median ± IQR of |N_tot_relerr(t)| per (regime, mode)."""
    scens = get_atmos_scenarios()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, mode in zip(axes, ('coag_only', 'cond_only', 'combined')):
        for regime in REGIMES:
            sids = [s['scenario_id'] for s in scens if s['regime'] == regime]
            hours = list(range(1, 25))  # every hour
            med = np.full(len(hours), np.nan)
            lo = np.full(len(hours), np.nan)
            hi = np.full(len(hours), np.nan)
            for hi_idx, hour in enumerate(hours):
                vals = []
                for sid in sids:
                    Nk_f, _ = load_fortran_hourly(sid, mode, hour)
                    jax_data = load_jax_npz(sid, mode)
                    if Nk_f is None or jax_data is None:
                        continue
                    Ntot_f = float(np.sum(Nk_f))
                    Ntot_j = float(jax_data['N_tot'][hour - 1])
                    if abs(Ntot_f) > 1e-30:
                        vals.append(abs(Ntot_j - Ntot_f) / abs(Ntot_f))
                if vals:
                    vals = np.array(vals)
                    med[hi_idx] = np.median(vals)
                    lo[hi_idx] = np.percentile(vals, 25)
                    hi[hi_idx] = np.percentile(vals, 75)
            ax.plot(hours, med, lw=1.4, color=REGIME_COLORS[regime],
                    label=regime)
            ax.fill_between(hours, lo, hi, alpha=0.2,
                             color=REGIME_COLORS[regime])
        ax.set_yscale('log')
        ax.set_xlabel('Hour')
        ax.set_title(mode)
        ax.set_ylim(bottom=1e-10)
        if mode == 'coag_only':
            ax.set_ylabel(r'|$\Delta N_{tot}$| relerr (median, IQR)')
            ax.legend(fontsize=8)

    fig.suptitle('Error evolution over 24 h (median ± IQR across regime)',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 10: mass conservation -------------------------------------------

def fig_10_mass_conservation(out_path: Path):
    scens = get_atmos_scenarios()
    fig, ax = plt.subplots(figsize=(10, 6))
    for regime in REGIMES:
        ratios = []
        for s in scens:
            if s['regime'] != regime:
                continue
            for mode in ('coag_only', 'cond_only', 'combined'):
                jax_data = load_jax_npz(s['scenario_id'], mode)
                if jax_data is None:
                    continue
                m_dry = jax_data['M_dry'] if 'M_dry' in jax_data.files \
                    else np.sum(jax_data['Mk'][:, :, :SRTH2O], axis=(1, 2))
                if len(m_dry) > 0 and abs(m_dry[0]) > 1e-30:
                    ratios.append(m_dry[-1] / m_dry[0])
        if ratios:
            ax.hist(ratios, bins=80, alpha=0.5, label=f'{regime} (n={len(ratios)})',
                    color=REGIME_COLORS[regime])
    ax.axvline(1.0, color='k', lw=1, ls='--')
    ax.set_xlabel(r'$M_{dry}$(h24) / $M_{dry}$(h0)   JAX PPM_JIT')
    ax.set_ylabel('Count')
    ax.set_title('JAX PPM mass conservation across 3000 runs')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 11: representative scenario overlays ----------------------------

def fig_11_representative_sizedist(df: pd.DataFrame, out_path: Path):
    """Pick 2 scenarios per regime: worst and best N_tot error."""
    scens = {s['scenario_id']: s for s in get_atmos_scenarios()}
    sub = df[(df['hour'] == REPORT_HOUR) & (df['mode'] == 'combined')].copy()
    sub['abs_err'] = np.abs(sub['N_tot_relerr'].astype(float))

    picks = []
    for regime in REGIMES:
        b = sub[sub['regime'] == regime].sort_values('abs_err')
        b = b[np.isfinite(b['abs_err'])]
        if len(b) >= 2:
            picks.append((regime, 'best',  int(b.iloc[0]['scenario_id'])))
            picks.append((regime, 'worst', int(b.iloc[-1]['scenario_id'])))

    if not picks:
        print('  [warn] fig_11: no data')
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    for ax_idx, (regime, flavor, sid) in enumerate(picks):
        r, c = ax_idx // 3, ax_idx % 3
        ax = axes[r, c]
        Nk_f, _ = load_fortran_hourly(sid, 'combined', 24)
        jax_data = load_jax_npz(sid, 'combined')
        if Nk_f is None or jax_data is None:
            ax.set_visible(False)
            continue
        Nk_j = jax_data['Nk'][23]
        xk = np.logspace(-9, -5, NBINS + 1)   # dummy x — we plot by bin idx
        bins = np.arange(NBINS)
        ax.step(bins, Nk_f + 1, where='mid', color='k', lw=1.3, label='Fortran PPM')
        ax.step(bins, np.asarray(Nk_j) + 1, where='mid',
                color=REGIME_COLORS[regime], lw=1.3,
                linestyle='--', label='JAX PPM_JIT')
        ax.set_yscale('log')
        ax.set_xlabel('Bin index')
        ax.set_ylabel(r'N$_k$ + 1')
        s = scens[sid]
        ax.set_title(f'{regime} — {flavor}\n'
                     f'sid={sid}, T={s["temp"]:.0f}K, P={s["pres"]/100:.1f}hPa',
                     fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle('Representative scenarios — Fortran vs JAX PPM size dists at h24',
                 y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Figure 12: wall-clock per regime ---------------------------------------

def fig_12_walltime_regime(out_path: Path):
    tcsv = JAX_DIR / 'timing_jax_ppm_atmos.csv'
    if not tcsv.exists():
        print('  [warn] fig_12: missing timing CSV')
        return
    df = pd.read_csv(tcsv)
    # regime column might exist or not
    if 'regime' not in df.columns:
        scens = {s['scenario_id']: s['regime'] for s in get_atmos_scenarios()}
        df['regime'] = df['scenario_id'].map(scens).astype(str)
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    positions = []
    labels = []
    x = 0
    for regime in REGIMES:
        for mode in ('coag_only', 'cond_only', 'combined'):
            vals = df[(df['regime'] == regime)
                        & (df['mode'] == mode)]['wall_time_s'].values
            if vals.size == 0:
                continue
            data.append(vals)
            positions.append(x)
            labels.append(f'{regime}\n{mode}')
            x += 1
        x += 0.5

    bp = ax.boxplot(data, positions=positions, widths=0.65,
                     patch_artist=True, showfliers=False)
    for patch, lbl in zip(bp['boxes'], labels):
        regime = lbl.split('\n')[0]
        patch.set_facecolor(REGIME_COLORS[regime])
        patch.set_alpha(0.65)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('JAX PPM_JIT wall time per scenario [s]')
    ax.set_title('JAX wall-clock distribution by regime and mode')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_path.name}")


# --- Entrypoint -------------------------------------------------------------

FIGURES = {
    '01': ('coverage_TP',        fig_01_coverage_TP,        True),
    '02': ('pairplot',           fig_02_pairplot,           True),
    '03': ('taylor_regime',      fig_03_taylor_regime,      True),
    '04': ('violin_skill',       fig_04_violin_skill,       True),
    '05': ('cdf_regime',         fig_05_cdf_regime,         True),
    '06': ('error_vs_pressure',  fig_06_error_vs_pressure,  True),
    '07': ('error_contour_TP',   fig_07_error_contour_TP,   True),
    '08': ('perbin_heatmap',     fig_08_perbin_heatmap,     True),
    '09': ('error_evolution',    fig_09_error_evolution,    False),
    '10': ('mass_conservation',  fig_10_mass_conservation,  False),
    '11': ('representative',     fig_11_representative_sizedist, True),
    '12': ('walltime_regime',    fig_12_walltime_regime,    False),
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--skip', nargs='+', default=[])
    p.add_argument('--only', nargs='+', default=None,
                   help='Render only the specified figure numbers (e.g. 01 03 07)')
    args = p.parse_args()

    skip = {s.zfill(2) for s in args.skip}
    only = {s.zfill(2) for s in args.only} if args.only else None

    STATS_DIR.mkdir(parents=True, exist_ok=True)

    print("Rendering atmos suite...")
    df = None
    for key, (name, fn, needs_df) in FIGURES.items():
        if key in skip:
            continue
        if only is not None and key not in only:
            continue
        if needs_df and df is None:
            df = load_metrics()
        out_path = STATS_DIR / f'atmos_fig_{key}_{name}.png'
        try:
            if needs_df:
                fn(df, out_path)
            else:
                fn(out_path)
        except Exception as e:
            print(f"  [error] {key} failed: {e}")
            import traceback
            traceback.print_exc()
    print("Done.")


if __name__ == '__main__':
    main()
