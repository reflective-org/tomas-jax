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
from .run_marianna_dilution import BASELINES, DILUTIONS, make_matrix, _RESULTS_ROOT
from .plot_marianna_dilution import (_grid_geometry, _load, BOXVOL, save_fig,
                                     TOL_BLUE, TOL_CYAN, TOL_GREEN, TOL_RED, TOL_PURPLE)

SNAPSHOT_HOURS = [12, 24, 48, 72, 168, 240]
MW_S = 32.06

# D1..D5 styles — Paul Tol bright (colorblind-safe). Cool→warm = low→high Kz;
# D4 (burst) is the odd one out: purple + dashed.
DIL_STYLE = {
    'D1': (TOL_BLUE,   '-',  'D1 Low Kz'),
    'D2': (TOL_CYAN,   '-',  'D2 Med Kz'),
    'D3': (TOL_GREEN,  '-',  'D3 High Kz'),
    'D4': (TOL_PURPLE, '--', 'D4 Burst'),
    'D5': (TOL_RED,    '-',  'D5 Very High'),
}

_QTY = {
    'dN':      ('dN/dlogDp [#/cm³]',        'compare_dN.png'),
    'dA':      ('dA/dlogDp [m²/cm³]',       'compare_dA.png'),
    'dV':      ('dV/dlogDp [µm³/cm³]',      'compare_dV.png'),
    'dN_perS': ('dN/dlogDp ÷ total S [per molec-S]', 'compare_dN_perS.png'),
    'dA_perS': ('dA/dlogDp ÷ total S [m²·cm³/cm³ per molec-S]', 'compare_dA_perS.png'),
    'dV_perS': ('dV/dlogDp ÷ total S [µm³ per molec-S]', 'compare_dV_perS.png'),
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


def _snap_distribution(d, qty, dp_m, dp_um, dlogDp, t_target_s):
    """Per-bin distribution of `qty` at the snapshot nearest t_target_s.

    `qty` is dN/dA/dV, optionally suffixed '_perS' to normalize by total
    sulfur concentration (dilution-corrected)."""
    t_s = d['t_seconds']
    if t_target_s > t_s[-1] + d['dts'][-1] + 1.0:
        return None
    i = int(np.argmin(np.abs(t_s - t_target_s)))
    Nk_cm3 = d['Nk_every'][i] / BOXVOL
    per_s = qty.endswith('_perS')
    base = qty[:-5] if per_s else qty
    y = _base_dist(base, Nk_cm3, dp_m, dp_um, dlogDp)
    if per_s:
        y = y / _total_sulfur(d, i)
    return y


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

    for r, bid in enumerate(baselines):
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
                y = _snap_distribution(d, qty, dp_m, dp_um, dlogDp, h * 3600.0)
                if y is None:
                    continue
                ax.plot(dp_nm, np.maximum(y, 1e-300), color=color, ls=ls,
                        label=lbl if (r == 0 and c == 0) else None)
                row_max = max(row_max, float(np.max(y)))
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


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nbins', type=int, default=NBINS)
    ap.add_argument('--yscale', default='both', choices=['log', 'linear', 'both'])
    args = ap.parse_args()
    plot_all_comparisons(args.nbins, yscale=args.yscale)
