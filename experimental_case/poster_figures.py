"""Poster figures for the Marianna dilution study (baseline 1).

Outputs go to ``results/marianna/figs/poster/`` (PNG @ poster DPI + vector PDF).

Currently:
  * banana_medkz_burst — B1 Med Kz (D2) and Burst (D4) banana plots stacked
    vertically, shared x-axis (time) and a single shared colorbar.

Usage::
    python -m experimental_case.poster_figures            # 80-bin (default)
    python -m experimental_case.poster_figures --nbins 40
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tomas_jax.core.config import NBINS
from .run_marianna_dilution import make_matrix, _RESULTS_ROOT
from .plot_marianna_dilution import _load, _banana_field, _despine, save_fig

POSTER_DIR = os.path.join(_RESULTS_ROOT, 'figs', 'poster')

# (run_id, panel label) — Med Kz on top, Burst below.
_STACK = [('B1-D2', 'Med Kz'), ('B1-D4', 'Burst')]


def banana_medkz_burst(nbins=NBINS, qty='dN', outdir=None):
    """Stacked B1 banana plots (Med Kz over Burst), shared x + shared colorbar."""
    matrix = make_matrix(nbins)
    runs = []
    for rid, label in _STACK:
        npz = matrix[rid].npz
        if not os.path.exists(npz):
            raise FileNotFoundError(f'Missing {npz}. Run {rid} (nbins={nbins}) first.')
        runs.append((label, _load(npz)))

    # Common color scale across both panels (shared colorbar).
    fields = []
    for _, d in runs:
        t_h, dp_nm, field = _banana_field(d, qty)
        fields.append((t_h / 24.0, dp_nm, field))   # x in days
    vmax = max(float(np.nanmax(f)) for _, _, f in fields)
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError('Banana: no positive data to plot.')
    norm = mcolors.LogNorm(vmin=vmax / 1e8, vmax=vmax)   # fixed 8-decade range

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                             layout='constrained')
    # Poster-scale fonts (larger than the default rcParams for print legibility).
    LBL, TICK, PANEL, CBAR = 19, 15, 20, 18
    pcm = None
    for ax, (label, _), (t_d, dp_nm, field) in zip(axes, runs, fields):
        pcm = ax.pcolormesh(t_d, dp_nm, field.T, norm=norm,
                            cmap='inferno', shading='nearest', rasterized=True)
        ax.set_yscale('log')
        ax.set_ylim(1, 2e4)
        ax.set_ylabel('Dp [nm]', fontsize=LBL)
        ax.tick_params(labelsize=TICK)
        ax.text(0.015, 0.93, label, transform=ax.transAxes,
                fontsize=PANEL, fontweight='bold', color='white', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.45, ec='none'))
        _despine(ax)
    axes[-1].set_xlabel('Time [days]', fontsize=LBL)

    cbar_label = {'dN': 'dN/dlogDp [#/cm³]',
                  'dA': 'dA/dlogDp [m²/cm³]',
                  'dV': 'dV/dlogDp [µm³/cm³]'}[qty]
    cb = fig.colorbar(pcm, ax=axes, label=cbar_label, pad=0.015, aspect=40)
    cb.set_label(cbar_label, fontsize=LBL)
    cb.ax.tick_params(labelsize=TICK)

    outdir = outdir or POSTER_DIR
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'banana_medkz_burst_{qty}.png')
    save_fig(fig, out, vector=True)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nbins', type=int, default=NBINS)
    ap.add_argument('--qty', default='dN', choices=['dN', 'dA', 'dV'])
    args = ap.parse_args()
    out = banana_medkz_burst(nbins=args.nbins, qty=args.qty)
    print(f'Saved: {out}')
    print(f'       {os.path.splitext(out)[0]}.pdf')
