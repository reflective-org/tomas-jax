"""Plots for the Marianna SAI dilution case (reads the NPZ written by
run_marianna_dilution.run()).

Figures (saved under results/marianna/figs/):
  1. dilution_trend.png  — inert tracer & V(t)/V0 vs time (abs + ratio)
  2. gas_timeseries.png  — SO2 & H2SO4 (molec/cm³) vs time
  3. banana.png          — dN/dlogDp(Dp, t)
  4. sizedist_dN.png     — dN/dlogDp at snapshot times (log-y + linear-y)
  5. sizedist_dA.png     — dA/dlogDp (surface area) at snapshot times
  6. sizedist_dV.png     — dV/dlogDp (volume) at snapshot times
  7. ntotal.png          — N_total vs time

Per project rules, plot functions raise if the NPZ is missing rather than
producing empty figures.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tomas_jax.core.config import make_grid, XK0, PI

DENS_INIT = 1770.0   # kg/m³
BOXVOL    = 1.0e6    # cm³
SNAPSHOT_HOURS = [12, 24, 48, 72, 168, 240]
SNAP_COLORS = ['#333333', '#1E88E5', '#43A047', '#E53935', '#FB8C00', '#8E24AA']


def _grid_geometry(nbins):
    """Return (dp_nm, dlogDp, dp_m, dp_um) for the 40-bin (mass-doubling) grid."""
    xk = np.array(make_grid(nbins, XK0, 2.0))
    m_mid = np.sqrt(xk[:-1] * xk[1:])
    dp_m  = (6.0 * m_mid / (DENS_INIT * PI)) ** (1.0 / 3.0)
    dp_edges = (6.0 * xk / (DENS_INIT * PI)) ** (1.0 / 3.0)
    dlogDp   = np.log10(dp_edges[1:] / dp_edges[:-1])
    return dp_m * 1e9, dlogDp, dp_m, dp_m * 1e6


def _load(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f'Missing NPZ: {npz_path}. Run the simulation first '
            f'(python -m experimental_case.run_marianna_dilution).')
    return np.load(npz_path, allow_pickle=True)


def _snap_index(t_seconds, snap_t):
    return int(np.argmin(np.abs(t_seconds - snap_t)))


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# =========================================================================
# 1. Dilution trend
# =========================================================================

def plot_dilution_trend(d, figdir):
    t_h = d['t_seconds'] / 3600.0
    tracer = d['tracer_every']
    Vr = d['V_ratio_every']
    inv_V = 1.0 / np.maximum(Vr, 1e-300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dilution trend — inert tracer vs V(t)/V0',
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(t_h, tracer, color='#1565C0', lw=2.0, label='inert tracer (Cbg=0)')
    ax.plot(t_h, inv_V, color='#E53935', lw=1.4, ls='--', label='V0/V(t)')
    ax.set_yscale('log'); ax.set_xlabel('Time [h]')
    ax.set_ylabel('tracer  /  V0/V(t)')
    ax.set_title('Dilution factor (log-y)')
    ax.legend(frameon=False); _despine(ax); ax.grid(True, alpha=0.25, which='both')

    ax = axes[1]
    ax.plot(t_h, Vr, color='#2E7D32', lw=2.0)
    ax.set_yscale('log'); ax.set_xlabel('Time [h]')
    ax.set_ylabel('V(t)/V0'); ax.set_title('Volume expansion V(t)/V0')
    _despine(ax); ax.grid(True, alpha=0.25, which='both')

    fig.tight_layout()
    out = os.path.join(figdir, 'dilution_trend.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    return out


# =========================================================================
# 2. Gas timeseries
# =========================================================================

def plot_gas_timeseries(d, figdir):
    t_h = d['t_seconds'] / 3600.0
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_h, d['SO2_molec_cm3'], color='#E53935', lw=2.0, label='SO2')
    ax.plot(t_h, d['SO4_molec_cm3'], color='#1565C0', lw=2.0, label='H2SO4 (gas)')
    ax.set_yscale('log'); ax.set_xlabel('Time [h]')
    ax.set_ylabel('Gas concentration [molec/cm³]')
    ax.set_title('Gas-phase SO2 and H2SO4 vs time')
    ax.legend(frameon=False); _despine(ax); ax.grid(True, alpha=0.25, which='both')
    fig.tight_layout()
    out = os.path.join(figdir, 'gas_timeseries.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    return out


# =========================================================================
# 3. Banana plot
# =========================================================================

def plot_banana(d, figdir):
    nbins = int(d['nbins'])
    dp_nm, dlogDp, _, _ = _grid_geometry(nbins)
    t_h = d['t_seconds'] / 3600.0
    dNdlogDp = (d['Nk_every'] / BOXVOL) / dlogDp[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = max(1e1, float(np.nanmax(dNdlogDp)))
    pcm = ax.pcolormesh(
        t_h, dp_nm, dNdlogDp.T,
        norm=mcolors.LogNorm(vmin=1.0, vmax=vmax),
        cmap='inferno', shading='nearest')
    ax.set_yscale('log'); ax.set_ylim(1, 2e4)
    ax.set_xlabel('Time [h]'); ax.set_ylabel('Dp [nm]')
    ax.set_title(f'Banana — dN/dlogDp  (T={float(d["temp"]):.0f}K, '
                 f'P={float(d["pres"])/100:.0f}hPa)')
    _despine(ax)
    fig.colorbar(pcm, ax=ax, label='dN/dlogDp [#/cm³]', pad=0.01)
    fig.tight_layout()
    out = os.path.join(figdir, 'banana.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    return out


# =========================================================================
# 4-6. Size / area / volume distributions at snapshot times
# =========================================================================

def _dist_from_Nk(Nk_row, dlogDp, dp_m, dp_um, label, t_h):
    """Build the dN/dA/dV size distributions for a single Nk record."""
    Nk_cm3 = Nk_row / BOXVOL
    return {
        'label': label,
        't_h': t_h,
        'dN': Nk_cm3 / dlogDp,
        'dA': (np.pi * dp_m ** 2 * Nk_cm3) / dlogDp,        # m²/cm³
        'dV': (np.pi / 6.0 * dp_um ** 3 * Nk_cm3) / dlogDp,  # µm³/cm³
    }


def _snapshots(d):
    """Return (dp_nm, initial_snap, time_snaps). initial_snap is the t=0 state."""
    nbins = int(d['nbins'])
    dp_nm, dlogDp, dp_m, dp_um = _grid_geometry(nbins)
    t_s = d['t_seconds']
    Nk_every = d['Nk_every']
    t_end = float(t_s[-1] + d['dts'][-1])   # total simulated time [s]

    initial = _dist_from_Nk(Nk_every[0], dlogDp, dp_m, dp_um,
                            label='t=0 (initial)', t_h=t_s[0] / 3600.0)
    snaps = []
    for h in SNAPSHOT_HOURS:
        if h * 3600.0 > t_end + 1.0:
            continue
        idx = _snap_index(t_s, h * 3600.0)
        snaps.append(_dist_from_Nk(Nk_every[idx], dlogDp, dp_m, dp_um,
                                   label=f'{h}h', t_h=t_s[idx] / 3600.0))
    return dp_nm, initial, snaps


def _plot_sizedist(d, figdir, qty, ylabel, fname):
    dp_nm, initial, snaps = _snapshots(d)
    if not snaps:
        raise ValueError(f'No snapshots available for {qty} — run longer.')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{qty} size distribution — Marianna  '
                 f'(T={float(d["temp"]):.0f}K, P={float(d["pres"])/100:.0f}hPa)',
                 fontsize=11, fontweight='bold')
    # Initial distribution as a black dashed reference curve
    axes[0].loglog(dp_nm, np.maximum(initial[qty], 1e-30), color='k', lw=2.4,
                   ls='--', label=initial['label'], zorder=6)
    axes[1].semilogx(dp_nm, initial[qty], color='k', lw=2.4, ls='--',
                     label=initial['label'], zorder=6)
    for s, c in zip(snaps, SNAP_COLORS):
        axes[0].loglog(dp_nm, np.maximum(s[qty], 1e-30), color=c, lw=1.8,
                       label=s['label'])
        axes[1].semilogx(dp_nm, s[qty], color=c, lw=1.8, label=s['label'])
    for ax, title in zip(axes, ['log-y', 'linear-y']):
        _despine(ax)
        ax.set_xlabel('Dp [nm]'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xlim(1, 2e4); ax.grid(True, alpha=0.25, which='both')
        ax.legend(fontsize=9, frameon=False, title='snapshot')
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    out = os.path.join(figdir, fname)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    return out


# =========================================================================
# 7. N_total
# =========================================================================

def plot_ntotal(d, figdir):
    t_h = d['t_seconds'] / 3600.0
    N = d['N_tot_every'] / BOXVOL
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_h, N, color='#6A1B9A', lw=2.0)
    ax.set_yscale('log'); ax.set_xlabel('Time [h]')
    ax.set_ylabel('N_total [#/cm³]'); ax.set_title('Total number concentration vs time')
    _despine(ax); ax.grid(True, alpha=0.25, which='both')
    fig.tight_layout()
    out = os.path.join(figdir, 'ntotal.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    return out


# =========================================================================
# Driver
# =========================================================================

def plot_all(npz_path):
    d = _load(npz_path)
    figdir = os.path.join(os.path.dirname(npz_path), 'figs')
    os.makedirs(figdir, exist_ok=True)
    outs = []
    outs.append(plot_dilution_trend(d, figdir))
    outs.append(plot_gas_timeseries(d, figdir))
    outs.append(plot_banana(d, figdir))
    outs.append(_plot_sizedist(d, figdir, 'dN', 'dN/dlogDp [#/cm³]', 'sizedist_dN.png'))
    outs.append(_plot_sizedist(d, figdir, 'dA', 'dA/dlogDp [m²/cm³]', 'sizedist_dA.png'))
    outs.append(_plot_sizedist(d, figdir, 'dV', 'dV/dlogDp [µm³/cm³]', 'sizedist_dV.png'))
    outs.append(plot_ntotal(d, figdir))
    print('Saved figures:')
    for o in outs:
        print(f'  {o}')
    return outs


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=os.path.join(
        os.path.dirname(__file__), 'results', 'marianna', 'marianna_dilution.npz'))
    args = ap.parse_args()
    plot_all(args.npz)
