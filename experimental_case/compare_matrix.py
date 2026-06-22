"""3×6 comparison figures across the run matrix.

Rows = baselines (B1/B2/B3), columns = snapshot times (12/24/48/72/168/240 h);
each panel overlays the 5 dilution regimes (D1–D5) for that baseline & time.

Four figures:
  compare_dN.png       — dN/dlogDp   [#/cm³]
  compare_dA.png       — dA/dlogDp   [m²/cm³]
  compare_dV.png       — dV/dlogDp   [µm³/cm³]
  compare_dN_perS.png  — dN/dlogDp normalized to total sulfur [SO2 + gas H2SO4 +
                         particulate SO4]  → corrected for dilution, so residual
                         differences between D1–D5 are the microphysical
                         (coagulation-rate) effect, not the trivial dilution scale.

Usage::
    python -m experimental_case.compare_matrix
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tomas_jax.core.config import AVOGADRO, MW_H2SO4, NBINS
from .run_marianna_dilution import (BASELINES, DILUTIONS, make_matrix,
                                     make_background, _RESULTS_ROOT)
from .plot_marianna_dilution import _grid_geometry, _load, BOXVOL, save_fig

SNAPSHOT_HOURS = [12, 24, 48, 96, 168, 240]
MW_S = 32.06

# D1..D5 styles — user-specified colors (final naming).
DIL_STYLE = {
    'D1': ('#0047FF', '-', 'Low Kz'),
    'D2': ('#FF7F0E', '-', 'Med Kz'),
    'D3': ('#008B8B', '-', 'High Kz'),
    'D4': ('#C000CC', '-', 'Burst'),
    'D5': ('#E41A1C', '-', 'Low Lx'),
}

_QTY = {
    'dN':      ('dN/dlogDp [#/cm³]',        'compare_dN.png'),
    'dA':      ('dA/dlogDp [m²/cm³]',       'compare_dA.png'),
    'dV':      ('dV/dlogDp [µm³/cm³]',      'compare_dV.png'),
    'dN_perS': ('dN/dlogDp ÷ total S [per molec-S]', 'compare_dN_perS.png'),
    'dA_perS': ('dA/dlogDp ÷ total S [m²·cm³/cm³ per molec-S]', 'compare_dA_perS.png'),
    'dV_perS': ('dV/dlogDp ÷ total S [µm³ per molec-S]', 'compare_dV_perS.png'),
    # Injected-only (SO2-derived), background excluded by subtraction. Numerator
    # is (total per-bin distribution − constant entrained background); denominator
    # is total INJECTED S = (SO2+H2SO4)_init × inert_tracer.
    'dN_injPerS': ('injected dN/dlogDp ÷ injected S [per molec-S]', 'compare_dN_injPerS.png'),
    'dA_injPerS': ('injected dA/dlogDp ÷ injected S [m²·cm³/cm³ per molec-S]', 'compare_dA_injPerS.png'),
    'dV_injPerS': ('injected dV/dlogDp ÷ injected S [µm³ per molec-S]', 'compare_dV_injPerS.png'),
}


# Compact y-labels for the B1 poster layout (the _QTY labels above are too long
# and overlap when rotated on a 2-row figure).
_B1_YLABEL = {
    'dN': 'dN/dlogDp [cm⁻³]',
    'dA': 'dA/dlogDp [m²/cm³]',
    'dV': 'dV/dlogDp [µm³/cm³]',
    'dN_perS': 'dN/dlogDp ÷ total S',
    'dA_perS': 'dA/dlogDp ÷ total S',
    'dV_perS': 'dV/dlogDp ÷ total S',
    'dN_injPerS': 'dN/dlogDp ÷ injected S',
    'dA_injPerS': 'dA/dlogDp ÷ injected S',
    'dV_injPerS': 'dV/dlogDp ÷ injected S',
}


def _base_dist(base, Nk_cm3, dp_m, dp_um, dlogDp):
    if base == 'dN':
        return Nk_cm3 / dlogDp
    if base == 'dA':
        return (np.pi * dp_m ** 2 * Nk_cm3) / dlogDp        # m²/cm³
    if base == 'dV':
        return (np.pi / 6.0 * dp_um ** 3 * Nk_cm3) / dlogDp  # µm³/cm³
    raise ValueError(base)


def _total_sulfur(d, i):
    """Total S concentration [molec/cm³] = SO2 + gas H2SO4 + particulate SO4."""
    SO2 = float(d['SO2_molec_cm3'][i])
    H2SO4g = float(d['SO4_molec_cm3'][i])
    aer = float(d['M_dry_every'][i]) * 1e3 / MW_H2SO4 * AVOGADRO / 1e6
    return max(SO2 + H2SO4g + aer, 1e-300)


def _injected_sulfur(d, i):
    """Total INJECTED (SO2-derived) S [molec/cm³] = (SO2+H2SO4)_init × tracer.

    The injected sulfur is conserved by chemistry/cond/nucl/coag and removed only
    by dilution toward background=0, so it tracks the inert dilution tracer
    exactly (validated to ~3e-9). Background SO2/aerosol are excluded."""
    inj0 = float(d['so2_init_molec_cm3']) + float(d['h2so4_init'])
    return max(inj0 * float(d['tracer_every'][i]), 1e-300)


def _snap_distribution(d, qty, dp_m, dp_um, dlogDp, t_target_s, bg_Nk_cm3=None):
    """Per-bin distribution of `qty` at the snapshot nearest t_target_s.

    `qty` is dN/dA/dV, optionally suffixed:
      '_perS'    → divide by total S (mixed plume+background, dilution-corrected)
      '_injPerS' → INJECTED only: subtract the constant entrained background
                   distribution (bg_Nk_cm3) from the per-bin number, then divide
                   by the injected sulfur budget."""
    t_s = d['t_seconds']
    if t_target_s > t_s[-1] + d['dts'][-1] + 1.0:
        return None                                   # past this run's cutoff → prune
    i = int(np.argmin(np.abs(t_s - t_target_s)))
    Nk_cm3 = d['Nk_every'][i] / BOXVOL

    if qty.endswith('_injPerS'):
        base = qty[:-8]
        Nk_inj = np.maximum(Nk_cm3 - (bg_Nk_cm3 if bg_Nk_cm3 is not None else 0.0), 0.0)
        y = _base_dist(base, Nk_inj, dp_m, dp_um, dlogDp)
        return y / _injected_sulfur(d, i)

    per_s = qty.endswith('_perS')
    base = qty[:-5] if per_s else qty
    y = _base_dist(base, Nk_cm3, dp_m, dp_um, dlogDp)
    if per_s:
        y = y / _total_sulfur(d, i)
    return y


_BG_NK_CACHE = {}


def _bg_Nk_cm3(bid, nbins):
    """Constant entrained-background number distribution [#/cm³ per bin] for a
    baseline (same red-circles ambient aerosol for all its dilution regimes)."""
    key = (bid, nbins)
    if key not in _BG_NK_CACHE:
        cfg = make_matrix(nbins)[f'{bid}-D1']
        Nk_bg, _, _, _ = make_background(cfg, nbins)
        _BG_NK_CACHE[key] = np.asarray(Nk_bg) / BOXVOL
    return _BG_NK_CACHE[key]


def _load_runs(nbins=NBINS):
    """Load the 15 NPZs into {run_id: dict}; skip any missing (with a warning)."""
    runs = {}
    for rid, cfg in make_matrix(nbins).items():
        if os.path.exists(cfg.npz):
            runs[rid] = _load(cfg.npz)
        else:
            print(f'  WARNING: missing {cfg.npz} — {rid} will be blank.')
    return runs


def plot_matrix_comparison(qty, nbins=NBINS, yscale='log', runs=None, outdir=None):
    """One 3×6 comparison figure. yscale='log' (loglog) or 'linear' (log-x,
    linear-y). x-axis shared across the grid; y shared per row."""
    ylabel, fname = _QTY[qty]
    if yscale == 'linear':
        fname = fname.replace('.png', '_linear.png')
    runs = runs or _load_runs(nbins)
    baselines = list(BASELINES)
    dp_nm, dlogDp, dp_m, dp_um = _grid_geometry(nbins)

    fig, axes = plt.subplots(len(baselines), len(SNAPSHOT_HOURS),
                             figsize=(15, 8), squeeze=False,
                             sharex=True, sharey='row', layout='constrained')
    fig.suptitle(
        f'{ylabel}\n'
        f'dilution-regime comparison · rows: baselines · columns: time · '
        f'{nbins}-bin · {yscale}-y',
        fontsize=12, fontweight='bold')

    bg_needed = qty.endswith('_injPerS')
    for r, bid in enumerate(baselines):
        bg_Nk = _bg_Nk_cm3(bid, nbins) if bg_needed else None
        row_max = 0.0
        for c, h in enumerate(SNAPSHOT_HOURS):
            ax = axes[r][c]
            ax.set_xscale('log')
            if yscale == 'log':
                ax.set_yscale('log')
            for did, (color, ls, lbl) in DIL_STYLE.items():
                d = runs.get(f'{bid}-{did}')
                if d is None:
                    continue
                y = _snap_distribution(d, qty, dp_m, dp_um, dlogDp, h * 3600.0,
                                       bg_Nk_cm3=bg_Nk)
                if y is None:
                    continue
                # Mask non-positive bins (e.g. background-subtracted zeros) as NaN
                # so the line breaks cleanly instead of drawing a floor artifact.
                yp = np.where(np.asarray(y) > 0, y, np.nan)
                if not np.any(np.isfinite(yp)):
                    continue
                ax.plot(dp_nm, yp, color=color, ls=ls,
                        label=lbl if (r == 0 and c == 0) else None)
                row_max = max(row_max, float(np.nanmax(yp)))
            if r == 0:
                ax.set_title(f'{h} h')
            if c == 0:
                ax.set_ylabel(f'{bid}\n{BASELINES[bid]["name"]}',
                              fontsize=9, fontweight='medium')
            ax.set_xlim(1, 2e4)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25, which='both')
        if row_max > 0:
            if yscale == 'log':
                axes[r][0].set_ylim(row_max * 3 / 1e8, row_max * 3)
            else:
                axes[r][0].set_ylim(0, row_max * 1.05)
    for c in range(len(SNAPSHOT_HOURS)):
        axes[-1][c].set_xlabel('Dp [nm]')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5,
               handlelength=2.4, columnspacing=2.0)

    if outdir is None:
        sub = 'comparison' if nbins == NBINS else f'comparison_{nbins}bin'
        outdir = os.path.join(_RESULTS_ROOT, sub)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, fname)
    save_fig(fig, out, vector=True)   # 300-dpi PNG + vector PDF (best for poster)
    return out


def plot_all_comparisons(nbins=NBINS, yscale='both', outdir=None):
    runs = _load_runs(nbins)
    scales = ['log', 'linear'] if yscale == 'both' else [yscale]
    outs = [plot_matrix_comparison(q, nbins=nbins, yscale=s, runs=runs, outdir=outdir)
            for s in scales for q in _QTY]
    print(f'Saved {nbins}-bin comparison figures:')
    for o in outs:
        print(f'  {o}')
    return outs


# =========================================================================
# Baseline-1-only comparison: the 6 snapshot times reflowed into a 2x3 grid
# (poster layout). Top row = first 3 times, bottom row = last 3.
# =========================================================================
def plot_b1_comparison(qty, nbins=NBINS, yscale='linear', runs=None, outdir=None,
                       bid='B1'):
    """B1-only comparison in a 2x3 time grid (poster layout). Shared x and y;
    poster-scale fonts."""
    _, fname = _QTY[qty]
    ylabel = _B1_YLABEL[qty]   # compact label (the matrix labels are too long here)
    fname = fname.replace('compare_', f'compare_{bid}_')
    if yscale == 'linear':
        fname = fname.replace('.png', '_linear.png')
    runs = runs or _load_runs(nbins)
    dp_nm, dlogDp, dp_m, dp_um = _grid_geometry(nbins)
    bg_Nk = _bg_Nk_cm3(bid, nbins) if qty.endswith('_injPerS') else None

    LBL, TICK, TITLE, LEG = 18, 16, 18, 15
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False,
                             sharex=True, sharey=True, layout='constrained')
    flat_ax = axes.ravel()
    gmax = 0.0
    for k, h in enumerate(SNAPSHOT_HOURS):
        ax = flat_ax[k]
        ax.set_xscale('log')
        if yscale == 'log':
            ax.set_yscale('log')
        for did, (color, ls, lbl) in DIL_STYLE.items():
            d = runs.get(f'{bid}-{did}')
            if d is None:
                continue
            y = _snap_distribution(d, qty, dp_m, dp_um, dlogDp, h * 3600.0,
                                   bg_Nk_cm3=bg_Nk)
            if y is None:
                continue
            yp = np.where(np.asarray(y) > 0, y, np.nan)
            if not np.any(np.isfinite(yp)):
                continue
            ax.plot(dp_nm, yp, color=color, ls=ls, lw=2.2,
                    label=lbl if k == 0 else None)
            gmax = max(gmax, float(np.nanmax(yp)))
        ax.set_title(f'{h} h', fontsize=TITLE, fontweight='medium')
        ax.set_xlim(1, 2e4)
        ax.tick_params(labelsize=TICK)
        ax.yaxis.get_offset_text().set_fontsize(TICK)   # the '1e-21' exponent
        ax.grid(True, alpha=0.25, which='both')
    if gmax > 0:
        flat_ax[0].set_ylim((gmax * 3 / 1e8, gmax * 3) if yscale == 'log'
                            else (0, gmax * 1.05))
    for ax in axes[-1, :]:
        ax.set_xlabel('Dp [nm]', fontsize=LBL)
    fig.supylabel(ylabel, fontsize=LBL)   # single centered label (no overlap)

    handles, labels = flat_ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5,
               fontsize=LEG, handlelength=2.4, columnspacing=2.0)

    if outdir is None:
        sub = 'comparison_b1' if nbins == NBINS else f'comparison_b1_{nbins}bin'
        outdir = os.path.join(_RESULTS_ROOT, sub)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, fname)
    save_fig(fig, out, vector=True)
    return out


def plot_all_b1_comparisons(nbins=NBINS, yscale='both', outdir=None):
    runs = _load_runs(nbins)
    scales = ['log', 'linear'] if yscale == 'both' else [yscale]
    outs = [plot_b1_comparison(q, nbins=nbins, yscale=s, runs=runs, outdir=outdir)
            for s in scales for q in _QTY]
    print(f'Saved {nbins}-bin B1 comparison figures:')
    for o in outs:
        print(f'  {o}')
    return outs


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nbins', type=int, default=NBINS)
    ap.add_argument('--yscale', default='both', choices=['log', 'linear', 'both'])
    ap.add_argument('--b1', action='store_true',
                    help='B1-only 2x3 layout instead of the 3x6 matrix')
    args = ap.parse_args()
    if args.b1:
        plot_all_b1_comparisons(args.nbins, yscale=args.yscale)
    else:
        plot_all_comparisons(args.nbins, yscale=args.yscale)
