"""Presentation-quality plots for TOMAS-JAX benchmark talk.

Generates 7 polished figures for an academic presentation:
  1. Parameter space coverage (LHC scatter)
  2. Coagulation: Fortran vs JAX size distributions
  3. TFL vs PPM vs Fortran condensation comparison
  4. Error evolution over 24h (median + IQR)
  5. Timing bar chart (Fortran vs JAX)
  6. Mass/number conservation scatter
  7. All scenarios at hour 24 (small multiples grid)

Usage::

    python benchmarks/python/plot_presentation.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmarks.python.scenarios import get_scenarios
from benchmarks.python.utils import load_csv

# ── Constants ──────────────────────────────────────────────────────────────
NBINS = 36
ICOMP = 44
SRTH2O = 43
NHOURS = 24
BOXVOL = 1.0e6
DENS_INIT = 1770.0
SRTSO4 = 0

# ── Directories ────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), '..', '..')
FORTRAN_DIR = os.path.join(_BASE, 'tomas_fortran', 'output', '24h')
JAX_DIR = os.path.join(_BASE, 'benchmarks', 'results', '24h')
OUTPUT_DIR = os.path.join(_BASE, 'benchmarks', 'results', '2026-03-03-presentation-coag-cond')

# ── Bin boundaries & diameters ─────────────────────────────────────────────
xk = np.zeros(NBINS + 1)
xk[0] = 1.6033e-23
for k in range(NBINS):
    xk[k + 1] = 2.0 * xk[k]

Dp_nm = np.array([
    1e9 * ((6 * np.sqrt(xk[k] * xk[k + 1])) / (DENS_INIT * np.pi)) ** (1.0 / 3.0)
    for k in range(NBINS)
])
dlogDp = np.log10(xk[1:] / xk[:-1]) / 3.0

# ── Color palette ──────────────────────────────────────────────────────────
C_FORTRAN = '#bbbbbb'   # light gray (thick background reference)
C_TFL     = '#1f77b4'   # blue
C_PPM     = '#d62728'   # red (stronger than #e74c3c)
C_COAG    = '#2ca02c'   # green
C_ACCENT  = '#ff7f0e'   # orange

# ── Line style constants for size distribution plots ──────────────────────
# Fortran: thick light gray background; TFL/PPM: bold colored on top
LS_FORTRAN = dict(color=C_FORTRAN, lw=4.0, ls='-',  zorder=1, alpha=0.9)
LS_TFL     = dict(color=C_TFL,     lw=2.5, ls='--', zorder=3,
                  dashes=(6, 3))     # long dashes
LS_PPM     = dict(color=C_PPM,     lw=2.5, ls='-',  zorder=2,
                  dashes=(2, 2))     # short dashes (dotted-like)
LS_INITIAL = dict(color='#cccccc',  lw=1.5, ls='-',  zorder=0)
# Smaller versions for the 10x5 grid
LS_FORTRAN_SM = dict(color=C_FORTRAN, lw=2.5, ls='-',  zorder=1, alpha=0.9)
LS_TFL_SM     = dict(color=C_TFL,     lw=1.8, ls='--', zorder=3,
                     dashes=(5, 2.5))
LS_PPM_SM     = dict(color=C_PPM,     lw=1.8, ls='-',  zorder=2,
                     dashes=(2, 2))
LS_INITIAL_SM = dict(color='#dddddd', lw=0.8, ls='-',  zorder=0)

# ── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'font.family': 'sans-serif',
})


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_fortran_hourly(sid, mode_prefix, hour):
    prefix = f"s{sid:02d}_{mode_prefix}_hour{hour:02d}"
    nk_p = os.path.join(FORTRAN_DIR, f"{prefix}_Nk.csv")
    mk_p = os.path.join(FORTRAN_DIR, f"{prefix}_Mk.csv")
    gc_p = os.path.join(FORTRAN_DIR, f"{prefix}_Gc.csv")
    if not os.path.exists(nk_p):
        return None
    Nk = load_csv(nk_p).flatten()
    Mk = load_csv(mk_p).reshape(NBINS, ICOMP)
    Gc = load_csv(gc_p).flatten() if os.path.exists(gc_p) else np.zeros(43)
    return {'Nk': Nk, 'Mk': Mk, 'Gc': Gc}


def load_fortran_all_hours(sid, mode_prefix):
    hourly = {}
    for h in range(1, 25):
        d = load_fortran_hourly(sid, mode_prefix, h)
        if d is not None:
            hourly[h] = d
    return hourly


def load_jax_npz(sid, mode, method):
    path = os.path.join(JAX_DIR, f"s{sid:02d}_{mode}_{method}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    result = {
        'Nk': data['Nk'],          # (24, 36)
        'Mk': data['Mk'],          # (24, 36, 44)
        'N_tot': data['N_tot'],     # (24,)
        'M_tot': data['M_tot'],     # (24,)
        'wall_time_s': float(data.get('wall_time_s', np.nan)),
    }
    if 'M_dry' in data:
        result['M_dry'] = data['M_dry']
    else:
        result['M_dry'] = np.sum(data['Mk'][:, :, :SRTH2O], axis=(1, 2))
    if 'Gc' in data:
        result['Gc'] = data['Gc']
    return result


def load_fortran_timing():
    timing = {}
    path = os.path.join(FORTRAN_DIR, 'timing_fortran.csv')
    if not os.path.exists(path):
        return timing
    with open(path) as f:
        for row in csv.DictReader(f):
            sid = int(row['scenario_id'].strip())
            timing[sid] = {}
            for col in ['coag_s', 'cond_s', 'combined_s']:
                timing[sid][col.replace('_s', '')] = float(row[col].strip())
    return timing


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Parameter space coverage
# ═══════════════════════════════════════════════════════════════════════════

def fig1_parameter_space(scenarios):
    params = {
        'N [#/cm$^3$]':   ([s['N_total'] for s in scenarios], 'log'),
        'GMD [$\\mu$m]':   ([s['GMD_um'] for s in scenarios], 'log'),
        'GSD':             ([s['GSD'] for s in scenarios], 'linear'),
        'T [K]':           ([s['temp'] for s in scenarios], 'linear'),
        'P [Pa]':          ([s['pres'] for s in scenarios], 'linear'),
        'RH':              ([s['RH'] for s in scenarios], 'linear'),
        'Gc$_{SO4}$ [kg]': ([s['Gc_SO4_kg'] for s in scenarios], 'log'),
        'H$_2$SO$_4$ prod': ([s['h2so4_prod'] for s in scenarios], 'log'),
    }

    keys = list(params.keys())
    n = len(keys)

    # Pick 4 most important pairs for a cleaner layout
    pairs = [
        ('N [#/cm$^3$]', 'GMD [$\\mu$m]'),
        ('T [K]', 'RH'),
        ('N [#/cm$^3$]', 'Gc$_{SO4}$ [kg]'),
        ('GMD [$\\mu$m]', 'GSD'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    fig.suptitle('Latin Hypercube Sampling: 50 Scenarios', fontsize=16, y=1.02)

    for idx, (xkey, ykey) in enumerate(pairs):
        ax = axes[idx]
        xvals, xscale = params[xkey]
        yvals, yscale = params[ykey]
        ax.scatter(xvals, yvals, c=C_TFL, s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        if xscale == 'log':
            ax.set_xscale('log')
        if yscale == 'log':
            ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'slide04_parameter_space.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Coagulation — Fortran vs JAX size distributions
# ═══════════════════════════════════════════════════════════════════════════

def fig2_coagulation_sizedist(scenarios):
    # Pick 3 representative scenarios: low N, medium N, high N
    sorted_sc = sorted(scenarios, key=lambda s: s['N_total'])
    picks = [sorted_sc[5], sorted_sc[24], sorted_sc[44]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Coagulation: Fortran vs JAX (Hour 24)', fontsize=16, y=1.02)

    for idx, sc in enumerate(picks):
        ax = axes[idx]
        sid = sc['scenario_id']

        # Initial distribution (hour 1 from Fortran or JAX)
        fort_init = load_fortran_hourly(sid, 'coag', 1)
        if fort_init is not None:
            dN_init = fort_init['Nk'] / BOXVOL / dlogDp
            mask = dN_init > 1e-3
            ax.plot(Dp_nm[mask], dN_init[mask], **LS_INITIAL,
                    label='Initial (t=0)')

        fort = load_fortran_hourly(sid, 'coag', 24)
        jax_d = load_jax_npz(sid, 'coag_only', 'tfl')

        if fort is not None:
            dN_f = fort['Nk'] / BOXVOL / dlogDp
            mask = dN_f > 1e-3
            ax.plot(Dp_nm[mask], dN_f[mask], **LS_FORTRAN,
                    label='Fortran (24h)')

        if jax_d is not None:
            dN_j = jax_d['Nk'][23] / BOXVOL / dlogDp
            mask = dN_j > 1e-3
            ax.plot(Dp_nm[mask], dN_j[mask], color=C_COAG, lw=2.5,
                    ls='--', dashes=(6, 3), zorder=3,
                    label='JAX (24h)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 5000)
        ax.set_xlabel('$D_p$ [nm]')
        if idx == 0:
            ax.set_ylabel('$dN/d\\log D_p$ [#/cm$^3$]')
        ax.set_title(f'S{sid:02d}: N={sc["N_total"]:.0e}, GMD={sc["GMD_um"]:.2f} $\\mu$m',
                     fontsize=13)
        if idx == 0:
            ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'slide05_coagulation_sizedist.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Condensation — TFL vs PPM vs Fortran
# ═══════════════════════════════════════════════════════════════════════════

def fig3_condensation_tfl_vs_ppm(scenarios):
    MW_H2SO4 = 0.098  # kg/mol
    AVOGADRO = 6.022e23

    # Find scenarios that have all 3 data sources AND visible condensation growth
    valid = []
    for sc in scenarios:
        sid = sc['scenario_id']
        has_fort = load_fortran_hourly(sid, 'cond', 24) is not None
        has_tfl = load_jax_npz(sid, 'cond_only', 'tfl_jit') is not None
        has_ppm = load_jax_npz(sid, 'cond_only', 'ppm_jit') is not None
        if has_fort and has_tfl and has_ppm:
            # Check if peak shifted (visible growth)
            fort_init = load_fortran_hourly(sid, 'cond', 1)
            fort_final = load_fortran_hourly(sid, 'cond', 24)
            if fort_init is not None and fort_final is not None:
                peak_init = np.argmax(fort_init['Nk'])
                peak_final = np.argmax(fort_final['Nk'])
                sc['_peak_shift'] = peak_final - peak_init
                valid.append(sc)

    # Only keep scenarios with visible growth (peak shifted >= 1 bin)
    valid_growth = [s for s in valid if s.get('_peak_shift', 0) >= 1]
    if len(valid_growth) < 3:
        valid_growth = valid  # fallback

    # Pick 3: small, medium, medium-large GMD (all with visible growth)
    valid_sorted = sorted(valid_growth, key=lambda s: s['GMD_um'])
    n = len(valid_sorted)
    picks = [valid_sorted[n // 6], valid_sorted[n // 2], valid_sorted[5 * n // 6]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Condensation: Fortran vs TFL$_{JIT}$ vs PPM$_{JIT}$ (Hour 24)',
                 fontsize=16, y=1.02)

    for idx, sc in enumerate(picks):
        ax = axes[idx]
        sid = sc['scenario_id']

        # Compute SO4 budget for annotation
        prod_molec = sc.get('h2so4_prod', 0)
        prod_24h_kg = sc['h2so4_prod_kg_per_s'] * 86400
        gc_init_molec = sc['Gc_SO4_kg'] / (MW_H2SO4 / AVOGADRO) / BOXVOL

        # Initial distribution
        fort_init = load_fortran_hourly(sid, 'cond', 1)
        if fort_init is not None:
            dN_init = fort_init['Nk'] / BOXVOL / dlogDp
            mask = dN_init > 1e-3
            ax.plot(Dp_nm[mask], dN_init[mask], **LS_INITIAL,
                    label='Initial (t=0)')

        fort = load_fortran_hourly(sid, 'cond', 24)
        tfl_d = load_jax_npz(sid, 'cond_only', 'tfl_jit')
        ppm_d = load_jax_npz(sid, 'cond_only', 'ppm_jit')

        if fort is not None:
            dN_f = fort['Nk'] / BOXVOL / dlogDp
            mask = dN_f > 1e-3
            ax.plot(Dp_nm[mask], dN_f[mask], **LS_FORTRAN,
                    label='Fortran (24h)')

        if tfl_d is not None:
            dN_t = tfl_d['Nk'][23] / BOXVOL / dlogDp
            mask = dN_t > 1e-3
            ax.plot(Dp_nm[mask], dN_t[mask], **LS_TFL,
                    label='TFL$_{JIT}$ (24h)')

        if ppm_d is not None:
            dN_p = ppm_d['Nk'][23] / BOXVOL / dlogDp
            mask = dN_p > 1e-3
            ax.plot(Dp_nm[mask], dN_p[mask], **LS_PPM,
                    label='PPM$_{JIT}$ (24h)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 5000)
        ax.set_xlabel('$D_p$ [nm]')
        if idx == 0:
            ax.set_ylabel('$dN/d\\log D_p$ [#/cm$^3$]')

        # Title with scenario info
        ax.set_title(f'S{sid:02d}: GMD={sc["GMD_um"]:.3f} $\\mu$m, '
                     f'N={sc["N_total"]:.1e} cm$^{{-3}}$',
                     fontsize=12)

        # SO4 annotation box
        so4_text = (f'H$_2$SO$_4$ prod: {prod_molec:.1e} cm$^{{-3}}$s$^{{-1}}$\n'
                    f'24h total: {prod_24h_kg:.1e} kg')
        ax.text(0.97, 0.03, so4_text, transform=ax.transAxes,
                fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                          alpha=0.8, edgecolor='#cccccc'))

        if idx == 0:
            ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'slide06_condensation_tfl_vs_ppm.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Error evolution over 24 hours (median + IQR)
# ═══════════════════════════════════════════════════════════════════════════

def _collect_errors(scenarios, jax_mode, fort_mode, jax_method):
    """Collect hourly N and M relative errors vs Fortran for all scenarios."""
    sids = [s['scenario_id'] for s in scenarios]
    N_errors = np.full((len(sids), 24), np.nan)
    M_errors = np.full((len(sids), 24), np.nan)

    for i, sid in enumerate(sids):
        jax_d = load_jax_npz(sid, jax_mode, jax_method)
        if jax_d is None:
            continue

        for h in range(24):
            fort = load_fortran_hourly(sid, fort_mode, h + 1)
            if fort is None:
                continue

            N_f = np.sum(fort['Nk'])
            N_j = jax_d['N_tot'][h]
            if N_f > 1e-30:
                N_errors[i, h] = abs(N_f - N_j) / N_f

            M_f = np.sum(fort['Mk'][:, :SRTH2O])
            M_j = jax_d['M_dry'][h]
            if M_f > 1e-30:
                M_errors[i, h] = abs(M_f - M_j) / M_f

    return N_errors, M_errors


def _plot_error_band(ax, hours, errors, color, label, ls='-'):
    """Plot median + IQR shading for an error array."""
    med = np.nanmedian(errors, axis=0)
    q25 = np.nanpercentile(errors, 25, axis=0)
    q75 = np.nanpercentile(errors, 75, axis=0)
    ax.fill_between(hours, q25, q75, alpha=0.15, color=color)
    ax.plot(hours, med, color=color, lw=2.5, ls=ls, label=label)


def fig4_error_evolution(scenarios):
    hours = np.arange(1, 25)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Relative Error vs Fortran Over 24 Hours\n(median + IQR over 49 scenarios)',
                 fontsize=16, y=1.02)

    # ── Top-left: Coagulation (Euler) ──
    ax = axes[0, 0]
    N_err, M_err = _collect_errors(scenarios, 'coag_only', 'coag', 'euler')
    # Fall back to tfl (diffrax) if euler not available
    if np.all(np.isnan(N_err)):
        N_err, M_err = _collect_errors(scenarios, 'coag_only', 'coag', 'tfl')
    _plot_error_band(ax, hours, N_err, C_TFL, '$N_{tot}$')
    _plot_error_band(ax, hours, M_err, C_PPM, '$M_{dry}$')
    ax.set_yscale('log')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Relative Error')
    ax.set_title('Coagulation (Euler)', fontsize=14)
    ax.set_xlim(1, 24)
    ax.legend(loc='best', framealpha=0.9)

    # ── Top-right: Condensation TFL_JIT ──
    ax = axes[0, 1]
    N_err, M_err = _collect_errors(scenarios, 'cond_only', 'cond', 'tfl_jit')
    if np.all(np.isnan(N_err)):
        N_err, M_err = _collect_errors(scenarios, 'cond_only', 'cond', 'tfl')
    _plot_error_band(ax, hours, N_err, C_TFL, '$N_{tot}$')
    _plot_error_band(ax, hours, M_err, C_PPM, '$M_{dry}$')
    ax.set_yscale('log')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Relative Error')
    ax.set_title('Condensation (TFL$_{JIT}$)', fontsize=14)
    ax.set_xlim(1, 24)
    ax.legend(loc='best', framealpha=0.9)

    # ── Bottom-left: Condensation PPM_JIT ──
    ax = axes[1, 0]
    N_err, M_err = _collect_errors(scenarios, 'cond_only', 'cond', 'ppm_jit')
    _plot_error_band(ax, hours, N_err, C_TFL, '$N_{tot}$')
    _plot_error_band(ax, hours, M_err, C_PPM, '$M_{dry}$')
    ax.set_yscale('log')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Relative Error')
    ax.set_title('Condensation (PPM$_{JIT}$)', fontsize=14)
    ax.set_xlim(1, 24)
    ax.legend(loc='best', framealpha=0.9)

    # ── Bottom-right: Combined (Coag+Cond, TFL_JIT) ──
    ax = axes[1, 1]
    N_err, M_err = _collect_errors(scenarios, 'combined', 'combined', 'tfl_jit')
    if np.all(np.isnan(N_err)):
        N_err, M_err = _collect_errors(scenarios, 'combined', 'combined', 'tfl')
    _plot_error_band(ax, hours, N_err, C_TFL, '$N_{tot}$')
    _plot_error_band(ax, hours, M_err, C_PPM, '$M_{dry}$')
    ax.set_yscale('log')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Relative Error')
    ax.set_title('Combined — Coag + Cond (TFL$_{JIT}$)', fontsize=14)
    ax.set_xlim(1, 24)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'slide07_error_evolution.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Timing comparison (bar chart)
# ═══════════════════════════════════════════════════════════════════════════

def fig5_timing_bars(scenarios):
    """Bar chart computed from actual NPZ wall_time_s data."""
    ft = load_fortran_timing()

    # Collect per-scenario timing from NPZ files
    configs = [
        ('Coag Only',  'coag_only', 'euler',   'coag'),
        ('Cond Only',  'cond_only', 'ppm_jit', 'cond'),
        ('Combined',   'combined',  'ppm_jit', 'combined'),
    ]

    fortran_medians = []
    jax_medians = []
    ratios = []

    for label, jax_mode, jax_method, fort_key in configs:
        jt_list = []
        ft_list = []
        for sc in scenarios:
            sid = sc['scenario_id']
            d = load_jax_npz(sid, jax_mode, jax_method)
            if d is not None and not np.isnan(d['wall_time_s']):
                jt_list.append(d['wall_time_s'])
            if sid in ft:
                ft_list.append(ft[sid].get(fort_key, np.nan))

        jax_med = np.median(jt_list) if jt_list else np.nan
        fort_med = np.median(ft_list) if ft_list else np.nan
        fortran_medians.append(fort_med)
        jax_medians.append(jax_med)
        ratios.append(jax_med / fort_med if fort_med > 0 else np.nan)

    modes = [c[0] for c in configs]
    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_f = ax.bar(x - width / 2, fortran_medians, width,
                    color=C_FORTRAN, alpha=0.85, label='Fortran', edgecolor='white')
    bars_j = ax.bar(x + width / 2, jax_medians, width,
                    color=C_TFL, alpha=0.85, label='JAX (best)', edgecolor='white')

    for i, (ratio, bj) in enumerate(zip(ratios, bars_j)):
        y = bj.get_height()
        if np.isnan(ratio):
            continue
        if ratio < 1:
            txt = f'{ratio:.2f}x\n(faster!)'
            color = '#2ca02c'
        else:
            txt = f'{ratio:.1f}x'
            color = C_PPM if ratio > 2 else C_ACCENT
        ax.annotate(txt, xy=(bj.get_x() + bj.get_width() / 2, y),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=12, fontweight='bold', color=color)

    ax.set_ylabel('Wall-Clock Time [s]')
    ax.set_title('Performance: Fortran vs JAX (24h, 1440 steps, median over 49 scenarios)',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=13)
    ax.legend(fontsize=13)
    ax.set_ylim(0, max(jax_medians) * 1.4)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'slide08_timing_bars.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5b: Timing scatter (Fortran vs JAX per scenario)
# ═══════════════════════════════════════════════════════════════════════════

def fig5b_timing_scatter(scenarios):
    """Per-scenario timing scatter: 2x2 grid for coag, cond TFL, cond PPM, combined."""
    ft = load_fortran_timing()
    if not ft:
        print("  Skipping timing scatter — no Fortran timing data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Wall-Clock Timing: Fortran vs JAX (per scenario)',
                 fontsize=16, y=1.02)

    mode_configs = [
        ('coag_only', 'coag',     'euler',   'Coagulation',        C_COAG),
        ('cond_only', 'cond',     'tfl_jit', 'Condensation TFL',   C_TFL),
        ('cond_only', 'cond',     'ppm_jit', 'Condensation PPM',   C_PPM),
        ('combined',  'combined', 'ppm_jit', 'Combined (Coag+PPM)', C_ACCENT),
    ]

    for idx, (jax_mode, fort_key, method, title, color) in enumerate(mode_configs):
        ax = axes[idx // 2, idx % 2]

        fort_times = []
        jax_times = []

        for sc in scenarios:
            sid = sc['scenario_id']
            jax_d = load_jax_npz(sid, jax_mode, method)
            if jax_d is None or sid not in ft:
                continue

            ft_time = ft[sid].get(fort_key, None)
            jt = jax_d['wall_time_s']
            if ft_time is not None and not np.isnan(jt):
                fort_times.append(ft_time)
                jax_times.append(jt)

        if not fort_times:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        fort_times = np.array(fort_times)
        jax_times = np.array(jax_times)

        ax.scatter(fort_times, jax_times, c=color, s=50, alpha=0.7,
                   edgecolors='white', linewidth=0.5)

        lim = max(np.max(fort_times), np.max(jax_times)) * 1.15
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel('Fortran [s]')
        ax.set_ylabel('JAX [s]')

        med_ratio = np.median(jax_times / fort_times)
        if med_ratio < 1:
            ratio_str = f'{med_ratio:.2f}x (faster!)'
        else:
            ratio_str = f'{med_ratio:.1f}x'
        ax.set_title(f'{title}\nMedian ratio: {ratio_str}', fontsize=13)
        ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'slide08b_timing_scatter.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Conservation scatter (mass and number)
# ═══════════════════════════════════════════════════════════════════════════

def fig6_conservation(scenarios):
    sids = [s['scenario_id'] for s in scenarios]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Conservation Properties Across 49 Scenarios', fontsize=16, y=1.02)

    # --- Panel 1: Coag mass conservation ---
    ax = axes[0]
    coag_cons = []
    coag_sids = []
    for sc in scenarios:
        sid = sc['scenario_id']
        d = load_jax_npz(sid, 'coag_only', 'tfl')
        if d is None:
            continue
        M0 = d['M_dry'][0]
        M24 = d['M_dry'][23]
        if M0 > 1e-30:
            coag_cons.append(abs(M24 - M0) / M0)
            coag_sids.append(sid)

    if coag_cons:
        ax.scatter(coag_sids, coag_cons, c=C_COAG, s=50, alpha=0.7,
                   edgecolors='white', linewidth=0.5)
        ax.axhline(1e-6, color='gray', ls=':', lw=1, label='$10^{-6}$')
        ax.set_yscale('log')
        ax.set_xlabel('Scenario ID')
        ax.set_ylabel('|$M_{24h}$ - $M_0$| / $M_0$')
        ax.set_title('Coagulation\nMass Conservation', fontsize=13)
        ax.legend(fontsize=10)

    # --- Panel 2: Cond N conservation (TFL_JIT) ---
    ax = axes[1]
    cond_N_cons = []
    cond_sids = []
    for sc in scenarios:
        sid = sc['scenario_id']
        d = load_jax_npz(sid, 'cond_only', 'tfl_jit')
        if d is None:
            d = load_jax_npz(sid, 'cond_only', 'tfl')
        if d is None:
            continue
        N0 = d['N_tot'][0]
        N24 = d['N_tot'][23]
        if N0 > 1e-30:
            cond_N_cons.append(abs(N24 - N0) / N0)
            cond_sids.append(sid)

    if cond_N_cons:
        ax.scatter(cond_sids, cond_N_cons, c=C_TFL, s=50, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label='TFL')
        ax.set_yscale('log')
        ax.set_xlabel('Scenario ID')
        ax.set_ylabel('|$N_{24h}$ - $N_0$| / $N_0$')
        ax.set_title('Condensation\nNumber Conservation', fontsize=13)
        ax.legend(fontsize=10)

    # --- Panel 3: Cond mass conservation budget ---
    # Budget: delta(M_aero) + delta(Gc) should equal production over same interval
    # NPZ stores hourly: index 0 = hour 1, index 23 = hour 24 → 23 hours between them
    ax = axes[2]
    cond_M_cons_tfl = []
    cond_M_cons_ppm = []
    cond_M_sids = []
    for sc in scenarios:
        sid = sc['scenario_id']
        for method, store, color, label in [
            ('tfl_jit', cond_M_cons_tfl, C_TFL, 'TFL'),
            ('ppm_jit', cond_M_cons_ppm, C_PPM, 'PPM'),
        ]:
            d = load_jax_npz(sid, 'cond_only', method)
            if d is None:
                d = load_jax_npz(sid, 'cond_only', method.replace('_jit', ''))
            if d is None:
                store.append(np.nan)
                continue
            M0 = d['M_dry'][0]
            M24 = d['M_dry'][23]
            Gc0 = np.sum(d['Gc'][0]) if 'Gc' in d else 0
            Gc24 = np.sum(d['Gc'][23]) if 'Gc' in d else 0
            # 23 hours between hour-1 snapshot and hour-24 snapshot
            prod_23h = sc['h2so4_prod_kg_per_s'] * 23 * 3600
            net_change = (M24 - M0) + (Gc24 - Gc0)
            if prod_23h > 1e-30:
                store.append(abs(net_change - prod_23h) / prod_23h)
            else:
                store.append(np.nan)
        cond_M_sids.append(sid)

    cond_M_sids = np.array(cond_M_sids)
    cond_M_cons_tfl = np.array(cond_M_cons_tfl)
    cond_M_cons_ppm = np.array(cond_M_cons_ppm)

    valid_tfl = ~np.isnan(cond_M_cons_tfl)
    valid_ppm = ~np.isnan(cond_M_cons_ppm)

    if np.any(valid_tfl):
        ax.scatter(cond_M_sids[valid_tfl], cond_M_cons_tfl[valid_tfl],
                   c=C_TFL, s=50, alpha=0.7, edgecolors='white', linewidth=0.5,
                   label='TFL', marker='o')
    if np.any(valid_ppm):
        ax.scatter(cond_M_sids[valid_ppm] + 0.3, cond_M_cons_ppm[valid_ppm],
                   c=C_PPM, s=50, alpha=0.7, edgecolors='white', linewidth=0.5,
                   label='PPM', marker='s')

    ax.set_yscale('log')
    ax.set_xlabel('Scenario ID')
    ax.set_ylabel('Mass Budget Error')
    ax.set_title('Condensation\nMass Conservation', fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'slide09_conservation.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: All scenarios at Hour 24 — small multiples (10x5 grid)
# ═══════════════════════════════════════════════════════════════════════════

def fig7_all_scenarios_grid(scenarios, mode='combined', fort_mode='combined'):
    n_sc = min(len(scenarios), 50)
    ncols = 10
    nrows = (n_sc + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, nrows * 2.4),
                             sharex=True, sharey=True)
    mode_label = mode.replace('_', ' ').title()
    fig.suptitle(f'{mode_label}: Size Distribution at Hour 24 — All Scenarios',
                 fontsize=18, y=1.01)

    for idx in range(nrows * ncols):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]

        if idx >= n_sc:
            ax.set_visible(False)
            continue

        sc = scenarios[idx]
        sid = sc['scenario_id']

        # Initial distribution (thin gray)
        fort_init = load_fortran_hourly(sid, fort_mode, 1)
        if fort_init is not None:
            dN_init = fort_init['Nk'] / BOXVOL / dlogDp
            mask_init = dN_init > 1e-3
            ax.plot(Dp_nm[mask_init], dN_init[mask_init], **LS_INITIAL_SM)

        fort = load_fortran_hourly(sid, fort_mode, 24)
        tfl_d = load_jax_npz(sid, mode, 'tfl_jit')
        if tfl_d is None:
            tfl_d = load_jax_npz(sid, mode, 'tfl')
        ppm_d = load_jax_npz(sid, mode, 'ppm_jit')

        if fort is not None:
            dN_f = fort['Nk'] / BOXVOL / dlogDp
            mask = dN_f > 1e-3
            ax.plot(Dp_nm[mask], dN_f[mask], **LS_FORTRAN_SM)

        if tfl_d is not None:
            dN_t = tfl_d['Nk'][23] / BOXVOL / dlogDp
            mask = dN_t > 1e-3
            ax.plot(Dp_nm[mask], dN_t[mask], **LS_TFL_SM)

        if ppm_d is not None:
            dN_p = ppm_d['Nk'][23] / BOXVOL / dlogDp
            mask = dN_p > 1e-3
            ax.plot(Dp_nm[mask], dN_p[mask], **LS_PPM_SM)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 5000)
        ax.set_ylim(1e-2, 1e8)
        ax.set_title(f'S{sid:02d}', fontsize=9, pad=2)

        # Clean up tick labels for inner plots
        if row < nrows - 1:
            ax.set_xticklabels([])
        if col > 0:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

        # Minimal grid
        ax.grid(True, alpha=0.15, linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared axis labels
    fig.text(0.5, -0.01, '$D_p$ [nm]', ha='center', fontsize=14)
    fig.text(-0.01, 0.5, '$dN/d\\log D_p$ [#/cm$^3$]', va='center',
             rotation='vertical', fontsize=14)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], **LS_INITIAL_SM, label='Initial (t=0)'),
        Line2D([0], [0], color=C_FORTRAN, lw=3, ls='-', alpha=0.9,
               label='Fortran (24h)'),
        Line2D([0], [0], color=C_TFL, lw=2.5, ls='--', dashes=(5, 2.5),
               label='JAX TFL (24h)'),
        Line2D([0], [0], color=C_PPM, lw=2.5, ls='-', dashes=(2, 2),
               label='JAX PPM (24h)'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=12,
               bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'slide10_all_scenarios_{mode}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Operator splitting schematic
# ═══════════════════════════════════════════════════════════════════════════

def fig8_operator_splitting():
    """Architecture diagram showing operator splitting within one timestep."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_aspect('equal')

    # ── Colors ──
    C_BOX_COAG = '#2ca02c'    # green
    C_BOX_COND = '#1f77b4'    # blue
    C_BOX_NUCL = '#9467bd'    # purple
    C_BOX_PROD = '#ff7f0e'    # orange
    C_BOX_STATE = '#555555'   # dark gray
    C_BOX_MNFIX = '#e377c2'   # pink

    def draw_box(x, y, w, h, label, sublabel, color, fontsize=13):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor='white',
                             linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2 + 0.15, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white')
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25, sublabel,
                    ha='center', va='center', fontsize=9,
                    color='white', alpha=0.9, style='italic')

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#444444',
                                    lw=2.5, connectionstyle='arc3,rad=0'))

    # ── Title ──
    ax.text(7, 6.6, 'Operator Splitting: One Timestep ($\\Delta t$ = 60 s)',
            ha='center', va='center', fontsize=18, fontweight='bold')

    # ── Time loop box (outer frame) ──
    loop_box = FancyBboxPatch((0.3, 0.3), 13.4, 5.8,
                               boxstyle="round,pad=0.2",
                               facecolor='none', edgecolor='#aaaaaa',
                               linewidth=1.5, linestyle='--')
    ax.add_patch(loop_box)
    ax.text(7, 0.65, 'Repeat for $t = 1, 2, \\ldots, 1440$ steps  (24 hours)',
            ha='center', va='center', fontsize=11, color='#888888', style='italic')

    # ── Row 1: Main pipeline (y=4.2) ──
    y_main = 4.0
    bw, bh = 2.4, 1.2

    # State input
    draw_box(0.5, y_main, 1.8, bh, '$N_k, M_k, G_c$', 'State at $t$', C_BOX_STATE, 11)

    # Step 1: H2SO4 Production
    x1 = 3.0
    draw_box(x1, y_main, bw, bh, 'H$_2$SO$_4$ Prod', '$G_c += P \\cdot \\Delta t$', C_BOX_PROD)
    draw_arrow(2.3, y_main + bh / 2, x1, y_main + bh / 2)

    # Step 2: Coagulation
    x2 = 6.1
    draw_box(x2, y_main, bw, bh, 'Coagulation', 'Forward Euler (JIT)', C_BOX_COAG)
    draw_arrow(x1 + bw, y_main + bh / 2, x2, y_main + bh / 2)

    # Step 3: Condensation
    x3 = 9.2
    draw_box(x3, y_main, bw, bh, 'Condensation', 'TFL or PPM (JIT)', C_BOX_COND)
    draw_arrow(x2 + bw, y_main + bh / 2, x3, y_main + bh / 2)

    # State output
    x_out = 12.2
    draw_box(x_out, y_main, 1.4, bh, '$N_k, M_k, G_c$', 'at $t+\\Delta t$', C_BOX_STATE, 11)
    draw_arrow(x3 + bw, y_main + bh / 2, x_out, y_main + bh / 2)

    # ── Row 2: Detail boxes (y=1.5) ──
    y_det = 1.5
    det_w, det_h = 3.4, 1.8

    # Coagulation detail
    xd1 = 1.0
    det_box1 = FancyBboxPatch((xd1, y_det), det_w, det_h,
                               boxstyle="round,pad=0.15",
                               facecolor='#f0f8f0', edgecolor=C_BOX_COAG,
                               linewidth=2)
    ax.add_patch(det_box1)
    ax.text(xd1 + det_w / 2, y_det + det_h - 0.25, 'Coagulation',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=C_BOX_COAG)
    detail_coag = (
        '36-bin sectional kernel\n'
        'Brownian + van der Waals\n'
        'Forward Euler + MNFIX\n'
        'JIT-compiled (JAX)'
    )
    ax.text(xd1 + det_w / 2, y_det + det_h / 2 - 0.2, detail_coag,
            ha='center', va='center', fontsize=9, color='#333333',
            linespacing=1.4)

    # Condensation TFL detail
    xd2 = 5.3
    det_box2 = FancyBboxPatch((xd2, y_det), det_w, det_h,
                               boxstyle="round,pad=0.15",
                               facecolor='#f0f0f8', edgecolor=C_BOX_COND,
                               linewidth=2)
    ax.add_patch(det_box2)
    ax.text(xd2 + det_w / 2, y_det + det_h - 0.25, 'Condensation — TFL',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=C_BOX_COND)
    detail_tfl = (
        'Top-hat / Flat (Lagrangian)\n'
        'Analytical growth: dmdt_int\n'
        'Partial-transfer remapping\n'
        'Fortran-faithful (JIT)'
    )
    ax.text(xd2 + det_w / 2, y_det + det_h / 2 - 0.2, detail_tfl,
            ha='center', va='center', fontsize=9, color='#333333',
            linespacing=1.4)

    # Condensation PPM detail
    xd3 = 9.6
    det_box3 = FancyBboxPatch((xd3, y_det), det_w, det_h,
                               boxstyle="round,pad=0.15",
                               facecolor='#fff0f0', edgecolor=C_PPM,
                               linewidth=2)
    ax.add_patch(det_box3)
    ax.text(xd3 + det_w / 2, y_det + det_h - 0.25, 'Condensation — PPM',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=C_PPM)
    detail_ppm = (
        'Piecewise Parabolic (Eulerian)\n'
        'Analytical mass-weighted flux\n'
        'CFL-substepped advection\n'
        '3rd-order accuracy (JIT)'
    )
    ax.text(xd3 + det_w / 2, y_det + det_h / 2 - 0.2, detail_ppm,
            ha='center', va='center', fontsize=9, color='#333333',
            linespacing=1.4)

    # Dashed arrows from main boxes to detail boxes
    ax.annotate('', xy=(xd1 + det_w / 2, y_det + det_h),
                xytext=(x2 + bw / 2, y_main),
                arrowprops=dict(arrowstyle='->', color=C_BOX_COAG,
                                lw=1.5, linestyle='--', alpha=0.5))
    ax.annotate('', xy=(xd2 + det_w / 2, y_det + det_h),
                xytext=(x3 + bw / 2 - 0.3, y_main),
                arrowprops=dict(arrowstyle='->', color=C_BOX_COND,
                                lw=1.5, linestyle='--', alpha=0.5))
    ax.annotate('', xy=(xd3 + det_w / 2, y_det + det_h),
                xytext=(x3 + bw / 2 + 0.3, y_main),
                arrowprops=dict(arrowstyle='->', color=C_PPM,
                                lw=1.5, linestyle='--', alpha=0.5))

    # "OR" label between TFL and PPM detail boxes
    ax.text((xd2 + det_w + xd3) / 2, y_det + det_h / 2, 'OR',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#888888',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#cccccc'))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'slide03_operator_splitting.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scenarios = get_scenarios(n=50, seed=42)
    # Exclude scenario 50 (crashes Fortran)
    scenarios_49 = [s for s in scenarios if s['scenario_id'] <= 49]

    print("Generating presentation figures...")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}/\n")

    print("[0/8] Operator splitting diagram...")
    fig8_operator_splitting()

    print("[1/8] Parameter space...")
    fig1_parameter_space(scenarios)

    print("[2/7] Coagulation size distributions...")
    fig2_coagulation_sizedist(scenarios_49)

    print("[3/7] TFL vs PPM condensation...")
    fig3_condensation_tfl_vs_ppm(scenarios_49)

    print("[4/7] Error evolution...")
    fig4_error_evolution(scenarios_49)

    print("[5/7] Timing bars...")
    fig5_timing_bars(scenarios_49)
    fig5b_timing_scatter(scenarios_49)

    print("[6/7] Conservation...")
    fig6_conservation(scenarios_49)

    print("[7/7] All scenarios grid (combined)...")
    fig7_all_scenarios_grid(scenarios_49, mode='combined', fort_mode='combined')

    print(f"\nDone! All figures saved to:\n  {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == '__main__':
    main()
