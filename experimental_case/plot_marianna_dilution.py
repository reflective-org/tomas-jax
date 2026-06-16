"""Plots for the Marianna SAI dilution case (reads the NPZ written by
run_marianna_dilution.run()).

Figures (saved under results/marianna/<scenario_slug>/figs/):
  0. parameters.png      — table of all input parameters for the scenario
  1. dilution_trend.png  — inert tracer & V(t)/V0 vs time (abs + ratio)
  2. gas_timeseries.png  — SO2 & H2SO4 (molec/cm³) vs time
  3. banana_dN/dA/dV.png — dN/dA/dV per dlogDp vs (Dp, t)
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

from tomas_jax.core.config import PI
from .run_marianna_dilution import make_grid_for

# ---- Global figure style (clean, consistent, colorblind-safe) ----------------
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 11, 'axes.titleweight': 'medium',
    'axes.labelsize': 10.5,
    'legend.fontsize': 9.5, 'legend.frameon': False,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'lines.linewidth': 1.8,
    'figure.dpi': 110, 'savefig.dpi': 130, 'savefig.bbox': 'tight',
    'mathtext.default': 'regular',
})

DENS_INIT = 1770.0   # kg/m³
BOXVOL    = 1.0e6    # cm³
SNAPSHOT_HOURS = [12, 24, 48, 72, 168, 240]

# Paul Tol "bright" qualitative palette — colorblind-safe (all 7 mutually distinct)
TOL_BRIGHT = ['#4477AA', '#66CCEE', '#228833', '#CCBB44', '#EE6677', '#AA3377', '#BBBBBB']
TOL_BLUE, TOL_CYAN, TOL_GREEN, TOL_YELLOW, TOL_RED, TOL_PURPLE, TOL_GREY = TOL_BRIGHT
SNAP_COLORS = TOL_BRIGHT[:6]   # 6 snapshot times (initial dist drawn separately, black)


def _grid_geometry(nbins):
    """Return (dp_nm, dlogDp, dp_m, dp_um) for the given resolution's grid."""
    xk = np.array(make_grid_for(nbins))
    m_mid = np.sqrt(xk[:-1] * xk[1:])
    dp_m  = (6.0 * m_mid / (DENS_INIT * PI)) ** (1.0 / 3.0)
    dp_edges = (6.0 * xk / (DENS_INIT * PI)) ** (1.0 / 3.0)
    dlogDp   = np.log10(dp_edges[1:] / dp_edges[:-1])
    return dp_m * 1e9, dlogDp, dp_m, dp_m * 1e6


def _get(d, key, default=None):
    """Read a (possibly 0-d) value from an NPZ, with a default for old files."""
    if key not in d.files:
        return default
    v = d[key]
    return v.item() if getattr(v, 'shape', None) == () else v


def _sname(d):
    """Scenario tag for plot titles, e.g. 'Scenario 1: <name>'."""
    sid = _get(d, 'scenario_id')
    name = _get(d, 'scenario_name')
    if sid is None and name is None:
        return f'(T={float(d["temp"]):.0f}K, P={float(d["pres"])/100:.0f}hPa)'
    return f'Scenario {sid}: {name}'


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
# 3. Banana plots (number / surface area / volume)
# =========================================================================

# qty -> (title, colorbar label, dyn-range floor relative to vmax)
_BANANA_QTY = {
    'dN': ('dN/dlogDp', 'dN/dlogDp [#/cm³]'),
    'dA': ('dA/dlogDp', 'dA/dlogDp [m²/cm³]'),
    'dV': ('dV/dlogDp', 'dV/dlogDp [µm³/cm³]'),
}


def _banana_field(d, qty):
    """Return (t_h, dp_nm, field[nt, nbins]) for the chosen quantity."""
    nbins = int(d['nbins'])
    dp_nm, dlogDp, dp_m, dp_um = _grid_geometry(nbins)
    t_h = d['t_seconds'] / 3600.0
    Nk_cm3 = d['Nk_every'] / BOXVOL                      # [nt, nbins]
    if qty == 'dN':
        per = np.ones(nbins)
    elif qty == 'dA':
        per = np.pi * dp_m ** 2                          # m²/particle
    elif qty == 'dV':
        per = np.pi / 6.0 * dp_um ** 3                   # µm³/particle
    else:
        raise ValueError(f'Unknown banana qty {qty!r}')
    field = (Nk_cm3 * per[np.newaxis, :]) / dlogDp[np.newaxis, :]
    return t_h, dp_nm, field


def plot_banana(d, figdir, qty='dN', fname=None):
    title, cbar_label = _BANANA_QTY[qty]
    t_h, dp_nm, field = _banana_field(d, qty)

    vmax = float(np.nanmax(field))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError(f'Banana {qty}: no positive data to plot.')
    vmin = vmax / 1e8   # fixed 8-decade dynamic range

    fig, ax = plt.subplots(figsize=(12, 5))
    pcm = ax.pcolormesh(
        t_h, dp_nm, field.T,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        cmap='inferno', shading='nearest')
    ax.set_yscale('log'); ax.set_ylim(1, 2e4)
    ax.set_xlabel('Time [h]'); ax.set_ylabel('Dp [nm]')
    ax.set_title(f'Banana — {title}  •  {_sname(d)}')
    _despine(ax)
    fig.colorbar(pcm, ax=ax, label=cbar_label, pad=0.01)
    fig.tight_layout()
    out = os.path.join(figdir, fname or f'banana_{qty}.png')
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
    fig.suptitle(f'{qty} size distribution  •  {_sname(d)}',
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
# 8. Input-parameter summary table
# =========================================================================

def plot_parameter_summary(d, figdir):
    """Render all input parameters of the simulation as a table PNG."""
    g = lambda k, dv=None: _get(d, k, dv)
    sid, name = g('scenario_id', '?'), g('scenario_name', 'Marianna dilution')

    sections = [
        ('Ambient conditions', [
            ('Temperature', f"{float(d['temp']):.0f} K"),
            ('Pressure', f"{float(d['pres'])/100:.0f} hPa ({float(d['pres']):.0f} Pa)"),
            ('H2O', f"{g('h2o_ppm','?')} ppm  (rh = {float(d['rh']):.4f})"),
            ('Air number density', f"{g('n_air_cm3',0):.3e} molec/cm³"),
        ]),
        ('Chemistry / nucleation', [
            ('OH (constant)', f"{g('oh_conc',0):.1e} molec/cm³"),
            ('Ion-pair production (fion)', f"{g('fion',0):.0f} pairs/cm³/s"),
            ('Organic nucleation', f"{g('nuc_org',0):.0f}"),
            ('NH3 nucleation', f"{g('nuc_nh3',0):.0f}"),
            ('Nucleation scheme', 'ricco_dunne (binary neutral + binary ion)'),
        ]),
        ('Initial plume', [
            ('SO2', f"{g('so2_init_ppt',0):.2e} ppt = {float(d['so2_init_molec_cm3']):.3e} molec/cm³"),
            ('H2SO4', f"{g('h2so4_init',0):.1e} molec/cm³"),
            ('Aerosol distribution', f"{g('init_dist','?')}"
             f"{' (STP→ambient)' if g('init_to_ambient', False) else ''}"),
            ('Initial N_total', f"{g('N_init',0):.2f} /cm³"),
        ]),
        ('Background (entrained air)', [
            ('Aerosol distribution', f"{g('bg_dist','?')}"
             f"{' (STP→ambient)' if g('bg_to_ambient', False) else ''}"),
            ('Background N_total', f"{g('N_bg',0):.2f} /cm³"),
            ('SO2', f"{g('bg_so2_ppb',0)} ppb ({g('bg_so2_cm3',0):.2e} molec/cm³)"),
            ('H2SO4', f"{g('bg_h2so4',0):.0e} molec/cm³"),
        ]),
        ('Dilution  V(t)/V0', [
            ('V0', f"{g('v0_m3',0):.2e} m³  (informational)"),
            ('Regime', f"{g('dilution_id','?')}"),
            ('V(t)/V0', f"{g('dilution_desc','t^0.8 then 1585·exp[...] ')}"),
            ('V(end)/V0', f"{g('V_final',0):.3e}"),
        ]),
        ('Numerics', [
            ('Duration', f"{float(d['max_hours']):.0f} h ({float(d['max_hours'])/24:.1f} d)"),
            ('Size bins', f"{int(d['nbins'])} (TOMAS, 1.7 nm – 17.5 µm)"),
            ('Time steps', f"{len(d['t_seconds'])}"),
            ('dt schedule', f"{g('dt_schedule_str', 'dt = 1/10/60 s')}"),
            ('Processes', 'SO2 chem → nucleation → coagulation → condensation → dilution'),
        ]),
    ]

    rows = []
    for title, items in sections:
        rows.append(('§ ' + title, ''))
        rows.extend(items)

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.2))
    ax.axis('off')
    ax.set_title(f'Input parameters — Scenario {sid}: {name}',
                 fontsize=13, fontweight='bold', pad=16)

    tbl = ax.table(cellText=rows, colWidths=[0.34, 0.66], loc='center',
                   cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#DDDDDD')
        label = rows[r][0]
        if label.startswith('§'):
            cell.set_facecolor('#1565C0')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif c == 0:
            cell.get_text().set_fontweight('bold')
            cell.set_facecolor('#F5F7FA')

    out = os.path.join(figdir, 'parameters.png')
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
    outs.append(plot_parameter_summary(d, figdir))
    outs.append(plot_dilution_trend(d, figdir))
    outs.append(plot_gas_timeseries(d, figdir))
    outs.append(plot_banana(d, figdir, 'dN', 'banana_dN.png'))
    outs.append(plot_banana(d, figdir, 'dA', 'banana_dA.png'))
    outs.append(plot_banana(d, figdir, 'dV', 'banana_dV.png'))
    outs.append(_plot_sizedist(d, figdir, 'dN', 'dN/dlogDp [#/cm³]', 'sizedist_dN.png'))
    outs.append(_plot_sizedist(d, figdir, 'dA', 'dA/dlogDp [m²/cm³]', 'sizedist_dA.png'))
    outs.append(_plot_sizedist(d, figdir, 'dV', 'dV/dlogDp [µm³/cm³]', 'sizedist_dV.png'))
    outs.append(plot_ntotal(d, figdir))
    from .rf_efficiency import (plot_rf_efficiency, plot_rf_and_radius,
                                plot_sulfur_normalized)
    outs.append(plot_rf_efficiency(d, figdir))
    outs.append(plot_rf_and_radius(d, figdir))
    outs.append(plot_sulfur_normalized(d, figdir))
    print('Saved figures:')
    for o in outs:
        print(f'  {o}')
    return outs


if __name__ == '__main__':
    import argparse
    from .run_marianna_dilution import SCENARIOS
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='B1-D2', choices=sorted(SCENARIOS))
    ap.add_argument('--npz', default=None, help='Explicit NPZ path (overrides --scenario)')
    args = ap.parse_args()
    npz = args.npz or SCENARIOS[args.scenario].npz
    plot_all(npz)
