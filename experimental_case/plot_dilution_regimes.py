"""Overlay the 5 dilution regimes (D1–D5) — V(t)/V0 and the inert tracer V0/V.

A pre-flight sanity figure: visualize every dilution curve in the matrix before
launching the (expensive) 30-run re-run. Also prints V/V0 at the snapshot hours
and a breakpoint-continuity check for each regime.

Usage::
    python -m experimental_case.plot_dilution_regimes
"""
import os
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .run_marianna_dilution import (DILUTIONS, V_ratio, SNAPSHOT_HOURS,
                                     _RESULTS_ROOT, dilution_str)
from .compare_matrix import DIL_STYLE
from .plot_marianna_dilution import save_fig

MAX_HOURS = 336.0


def _cfg(did):
    """Lightweight shim exposing only what V_ratio needs."""
    return SimpleNamespace(dilution=DILUTIONS[did]['segments'])


def _breakpoints(did):
    """Segment-boundary times [s] (interior breaks only)."""
    segs = DILUTIONS[did]['segments']
    return [t_end for t_end, _ in segs[:-1]]


def print_table():
    snap_s = np.array(SNAPSHOT_HOURS, dtype=float) * 3600.0
    print('\nV(t)/V0 at snapshot hours:')
    hdr = '  regime        ' + ''.join(f'{h:>11d}h' for h in SNAPSHOT_HOURS)
    print(hdr)
    for did in DILUTIONS:
        v = V_ratio(snap_s, _cfg(did))
        row = ''.join(f'{x:>12.3e}' for x in v)
        print(f'  {did} {DILUTIONS[did]["name"]:<10s}{row}')

    print('\nBreakpoint continuity (|left/right - 1|, should be ~0):')
    for did in DILUTIONS:
        brks = _breakpoints(did)
        if not brks:
            print(f'  {did}: (single segment — no interior breaks)')
            continue
        msgs = []
        for tb in brks:
            lo = float(V_ratio(np.array([tb - 1e-3]), _cfg(did))[0])
            hi = float(V_ratio(np.array([tb + 1e-3]), _cfg(did))[0])
            msgs.append(f't={tb:g}s: {abs(lo / hi - 1.0):.2e}')
        print(f'  {did}: ' + ' | '.join(msgs))

    print('\nRegime definitions:')
    for did in DILUTIONS:
        print(f'  {did} {DILUTIONS[did]["name"]}: {dilution_str(DILUTIONS[did]["segments"])}')


def plot(outdir=None):
    t = np.logspace(np.log10(0.09 * 86400.0), np.log10(10.0 * 86400.0), 2000)  # 0.09–10 d
    t_d = t / 86400.0   # days

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout='constrained')
    for did, (color, ls, lbl) in DIL_STYLE.items():
        v = V_ratio(t, _cfg(did))
        axes[0].plot(t_d, v, color=color, ls=ls, lw=2.0, label=lbl)
        axes[1].plot(t_d, 1.0 / v, color=color, ls=ls, lw=2.0, label=lbl)

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('time [days]')
        ax.set_xlim(0.09, 10.0)
        for h in SNAPSHOT_HOURS:
            ax.axvline(h / 24.0, color='0.8', lw=0.8, zorder=0)
        ax.grid(True, which='both', alpha=0.25)
    axes[0].set_ylabel('V(t) / V₀   (plume volume expansion)')
    axes[0].set_title('Dilution: volume ratio V(t)/V₀')
    axes[1].set_ylabel('V₀ / V(t)   (inert tracer / concentration factor)')
    axes[1].set_title('Inert tracer V₀/V(t)')
    axes[0].legend(title='dilution regime', frameon=False, fontsize=9)
    fig.suptitle('Marianna dilution regimes (D1–D5) · grey lines = snapshots '
                 '(0.5/1/2/3/7/10 d)', fontsize=12, fontweight='bold')

    outdir = outdir or _RESULTS_ROOT
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 'dilution_regimes.png')
    save_fig(fig, out, vector=True)
    print(f'\nSaved: {out}')
    return out


if __name__ == '__main__':
    print_table()
    plot()
