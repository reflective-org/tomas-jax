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

from tomas_jax.core.config import AVOGADRO, MW_H2SO4
from .run_marianna_dilution import BASELINES, DILUTIONS, SCENARIOS, _RESULTS_ROOT
from .plot_marianna_dilution import _grid_geometry, _load, BOXVOL

SNAPSHOT_HOURS = [12, 24, 48, 72, 168, 240]
MW_S = 32.06

# D1..D5 styles (low→high Kz: blue→red; D4 burst dashed purple)
DIL_STYLE = {
    'D1': ('#1565C0', '-',  'D1 Low Kz'),
    'D2': ('#2E7D32', '-',  'D2 Med Kz'),
    'D3': ('#EF6C00', '-',  'D3 High Kz'),
    'D4': ('#8E24AA', '--', 'D4 Burst'),
    'D5': ('#C62828', '-',  'D5 Very High'),
}

_QTY = {
    'dN':      ('dN/dlogDp [#/cm³]',        'compare_dN.png'),
    'dA':      ('dA/dlogDp [m²/cm³]',       'compare_dA.png'),
    'dV':      ('dV/dlogDp [µm³/cm³]',      'compare_dV.png'),
    'dN_perS': ('dN/dlogDp ÷ total S [per molec-S]', 'compare_dN_perS.png'),
}


def _snap_distribution(d, qty, dp_m, dp_um, dlogDp, t_target_s):
    """Per-bin distribution of `qty` at the snapshot nearest t_target_s."""
    t_s = d['t_seconds']
    if t_target_s > t_s[-1] + d['dts'][-1] + 1.0:
        return None
    i = int(np.argmin(np.abs(t_s - t_target_s)))
    Nk_cm3 = d['Nk_every'][i] / BOXVOL
    if qty == 'dN':
        y = Nk_cm3 / dlogDp
    elif qty == 'dA':
        y = (np.pi * dp_m ** 2 * Nk_cm3) / dlogDp
    elif qty == 'dV':
        y = (np.pi / 6.0 * dp_um ** 3 * Nk_cm3) / dlogDp
    elif qty == 'dN_perS':
        SO2 = float(d['SO2_molec_cm3'][i])
        H2SO4g = float(d['SO4_molec_cm3'][i])
        aer = float(d['M_dry_every'][i]) * 1e3 / MW_H2SO4 * AVOGADRO / 1e6
        S_tot = max(SO2 + H2SO4g + aer, 1e-300)
        y = (Nk_cm3 / dlogDp) / S_tot
    return y


def _load_runs():
    """Load the 15 NPZs into {run_id: dict}; skip any missing (with a warning)."""
    runs = {}
    for rid, cfg in SCENARIOS.items():
        if os.path.exists(cfg.npz):
            runs[rid] = _load(cfg.npz)
        else:
            print(f'  WARNING: missing {cfg.npz} — {rid} will be blank.')
    return runs


def plot_matrix_comparison(qty, runs=None, outdir=None):
    ylabel, fname = _QTY[qty]
    runs = runs or _load_runs()
    baselines = list(BASELINES)
    dp_nm, dlogDp, dp_m, dp_um = _grid_geometry(40)

    fig, axes = plt.subplots(len(baselines), len(SNAPSHOT_HOURS),
                             figsize=(22, 11), squeeze=False)
    fig.suptitle(f'Dilution-regime comparison — {ylabel}\n'
                 f'rows = baselines, columns = time; curves = D1–D5',
                 fontsize=14, fontweight='bold')

    for r, bid in enumerate(baselines):
        row_vals = []
        for c, h in enumerate(SNAPSHOT_HOURS):
            ax = axes[r][c]
            for did, (color, ls, lbl) in DIL_STYLE.items():
                rid = f'{bid}-{did}'
                d = runs.get(rid)
                if d is None:
                    continue
                y = _snap_distribution(d, qty, dp_m, dp_um, dlogDp, h * 3600.0)
                if y is None:
                    continue
                ax.loglog(dp_nm, np.maximum(y, 1e-300), color=color, ls=ls,
                          lw=1.6, label=lbl if (r == 0 and c == 0) else None)
                row_vals.append(np.max(y))
            if r == 0:
                ax.set_title(f'{h} h')
            if c == 0:
                ax.set_ylabel(f'{bid} {BASELINES[bid]["name"]}\n{ylabel}', fontsize=8)
            ax.set_xlim(1, 2e4)
            ax.grid(True, alpha=0.2, which='both')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        # shared y-limits per row
        if row_vals:
            ymax = max(row_vals) * 3
            for c in range(len(SNAPSHOT_HOURS)):
                axes[r][c].set_ylim(ymax / 1e8, ymax)
    for c in range(len(SNAPSHOT_HOURS)):
        axes[-1][c].set_xlabel('Dp [nm]')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if outdir is None:
        outdir = os.path.join(_RESULTS_ROOT, 'comparison')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    return out


def plot_all_comparisons(outdir=None):
    runs = _load_runs()
    outs = [plot_matrix_comparison(q, runs=runs, outdir=outdir) for q in _QTY]
    print('Saved comparison figures:')
    for o in outs:
        print(f'  {o}')
    return outs


if __name__ == '__main__':
    plot_all_comparisons()
