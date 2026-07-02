"""Map tabulated dN/dlogDp distributions onto TOMAS 40-bin and 80-bin grids.

Two distributions are defined:
  dist1  — nucleation-mode dominated (peak ~45 nm, N~977 cm⁻³)
  dist2  — accumulation-mode dominated (peak ~150 nm, N~? cm⁻³)

Usage::

    python -m experimental_case.background_aerosol_distribution
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad

from tomas_jax.core.config import (
    NBINS, ICOMP, SRTSO4, PI,
    make_grid, make_grid_80bin, XK0,
)

# =========================================================================
# Tabulated distributions
# =========================================================================

# Distribution 1 — nucleation-mode dominated (peak at ~45 nm)
_DATA1 = np.array([
    [0.0015,   0   ],
    [0.002,    10  ],
    [0.0023,   60  ],
    [0.0026,   110 ],
    [0.003,    140 ],
    [0.0035,   170 ],
    [0.004,    200 ],
    [0.0045,   240 ],
    [0.005,    270 ],
    [0.006,    310 ],
    [0.007,    350 ],
    [0.008,    390 ],
    [0.009,    440 ],
    [0.01,     490 ],
    [0.0115,   560 ],
    [0.013,    620 ],
    [0.015,    680 ],
    [0.017,    730 ],
    [0.02,     780 ],
    [0.023,    830 ],
    [0.026,    880 ],
    [0.03,     940 ],
    [0.035,    980 ],
    [0.04,     1010],
    [0.045,    1015],   # peak
    [0.05,     950 ],
    [0.06,     750 ],
    [0.075,    580 ],
    [0.09,     390 ],
    [0.1,      250 ],
    [0.12,     180 ],
    [0.15,     130 ],
    [0.18,     90  ],
    [0.2,      60  ],
    [0.25,     40  ],
    [0.3,      30  ],
    [0.4,      20  ],
    [0.5,      15  ],
    [0.6,      10  ],
    [0.8,      5   ],
    [1.0,      2   ],
    [1.5,      0   ],
    [2.0,      0   ],
])

# Distribution 2 — accumulation-mode dominated (peak at ~150 nm)
_DATA2 = np.array([
    [0.0015,   5  ],
    [0.002,    5  ],
    [0.003,    5  ],
    [0.005,    5  ],
    [0.01,     5  ],
    [0.015,    5  ],
    [0.02,     5  ],
    [0.03,     10 ],
    [0.04,     20 ],
    [0.05,     30 ],
    [0.06,     40 ],
    [0.07,     50 ],
    [0.08,     60 ],
    [0.09,     70 ],
    [0.1,      80 ],
    [0.12,     85 ],
    [0.15,     95 ],   # peak
    [0.18,     85 ],
    [0.2,      70 ],
    [0.25,     45 ],
    [0.3,      30 ],
    [0.4,      15 ],
    [0.5,      10 ],
    [0.6,      5  ],
    [0.8,      2  ],
    [1.0,      0  ],
    [1.5,      0  ],
    [2.0,      0  ],
])

# Distribution 3 — Marianna observed size distribution, red OPEN circles
# "observation, 220-230 ppbv" from Fig. S4 (CARMA box-model study; the figure
# reports dN/dlogDp in cm^-3 STP). Digitized by eye from the provided figure.
# An aged/sparse accumulation mode: near-zero below ~0.05 um, broad peak
# ~94 cm^-3 STP near Dp ~0.12 um, tailing to 0 by ~0.6 um.
# NOTE: eyeball digitization — VERIFY against the cross-check plot
# (background_aerosol_distribution.py CLI) before trusting absolute N.
_DATA3 = np.array([
    [0.003,   0   ],
    [0.010,   1   ],
    [0.020,   1   ],
    [0.030,   2   ],
    [0.040,   4   ],
    [0.050,   8   ],
    [0.060,   18  ],
    [0.070,   35  ],
    [0.080,   55  ],
    [0.090,   72  ],
    [0.100,   85  ],
    [0.110,   92  ],
    [0.120,   94  ],   # peak
    [0.130,   92  ],
    [0.150,   82  ],
    [0.180,   58  ],
    [0.200,   42  ],
    [0.250,   22  ],
    [0.300,   11  ],
    [0.400,   4   ],
    [0.500,   1.5 ],
    [0.600,   0.5 ],
    [0.800,   0   ],
    [1.000,   0   ],
])

# Distribution 4 — Marianna observed size distribution, red OPEN DIAMONDS
# "observation, 310-320 ppbv" from the same Fig. S4 (dN/dlogDp in cm^-3 STP).
# Younger air than the red circles: a small-particle plateau (~50-70 cm^-3 STP,
# 5-30 nm) plus a mode peaking ~320 cm^-3 STP near Dp ~0.065-0.07 um.
# NOTE: eyeball digitization (same caveat as _DATA3) — verify with the CLI plot.
_DATA4 = np.array([
    [0.003,   20  ],
    [0.004,   32  ],
    [0.005,   42  ],
    [0.006,   48  ],
    [0.008,   55  ],
    [0.010,   58  ],
    [0.013,   60  ],
    [0.016,   62  ],
    [0.020,   65  ],
    [0.025,   68  ],
    [0.030,   72  ],
    [0.035,   85  ],
    [0.040,   105 ],
    [0.050,   180 ],
    [0.060,   290 ],
    [0.068,   320 ],   # peak
    [0.080,   285 ],
    [0.090,   245 ],
    [0.100,   205 ],
    [0.120,   150 ],
    [0.150,   105 ],
    [0.180,   80  ],
    [0.200,   65  ],
    [0.250,   40  ],
    [0.300,   25  ],
    [0.400,   11  ],
    [0.500,   5   ],
    [0.600,   2   ],
    [0.800,   0.5 ],
    [1.000,   0   ],
])

_DIST_META = {
    'tabulated':  {'data': _DATA1, 'label': 'Dist 1 (nucleation-mode, peak 45 nm)',
                   'color1': '#1E88E5', 'color2': '#E53935'},
    'tabulated2': {'data': _DATA2, 'label': 'Dist 2 (accumulation-mode, peak 150 nm)',
                   'color1': '#43A047', 'color2': '#FB8C00'},
    'redcircles': {'data': _DATA3, 'label': 'Marianna obs (red circles, 220-230 ppbv)',
                   'color1': '#E53935', 'color2': '#1565C0'},
    'reddiamonds': {'data': _DATA4, 'label': 'Marianna obs (red diamonds, 310-320 ppbv)',
                    'color1': '#8E24AA', 'color2': '#00897B'},
}

# =========================================================================
# Constants
# =========================================================================
DENS_SO4 = 1770.0    # kg/m³
BOXVOL   = 1.0e6     # cm³ (1 m³)

KB       = 1.380649e-23   # J/K  Boltzmann
T_STP    = 273.15         # K    (0 °C)
P_STP    = 101325.0       # Pa   (1 atm)


def stp_to_ambient_factor(temp, pres):
    """Multiplicative factor to convert a number concentration reported per
    cm³ STP into a number concentration per cm³ at ambient (temp, pres).

    n_ambient/n_STP = (pres/P_STP) * (T_STP/temp).  At 210 K / 5500 Pa this
    is ~0.0706, i.e. ambient air is much thinner than STP so the actual
    ambient concentration is lower than the STP-normalised value.
    """
    return (pres / P_STP) * (T_STP / temp)


# =========================================================================
# Interpolator factory
# =========================================================================

def _make_interp(data):
    dp_um    = data[:, 0]
    dndlogdp = data[:, 1]
    return interp1d(
        np.log10(dp_um), dndlogdp,
        kind='linear',
        bounds_error=False,
        fill_value=0.0,
    )

_INTERPS = {k: _make_interp(v['data']) for k, v in _DIST_META.items()}


def dndlogdp_at(dp_um, dist='tabulated'):
    """dN/dlogDp [cm⁻³] at diameter dp_um [µm] for the chosen distribution."""
    interp = _INTERPS[dist]
    return np.maximum(0.0, interp(np.log10(np.asarray(dp_um, dtype=float))))


# =========================================================================
# Map onto a TOMAS bin grid
# =========================================================================

def _xk_to_dp_um(xk_np):
    return ((6.0 * xk_np) / (DENS_SO4 * PI)) ** (1.0 / 3.0) * 1e6


def map_to_grid(xk_np, boxvol=BOXVOL, dist='tabulated'):
    """Integrate a tabulated distribution onto a TOMAS bin grid."""
    interp = _INTERPS[dist]
    nbins = len(xk_np) - 1
    dp_edges_um  = _xk_to_dp_um(xk_np)
    log_dp_edges = np.log10(dp_edges_um)

    Nk_cm3 = np.zeros(nbins)
    for k in range(nbins):
        val, _ = quad(
            lambda logdp: float(interp(logdp)),
            log_dp_edges[k], log_dp_edges[k + 1],
            limit=50,
        )
        Nk_cm3[k] = max(0.0, val)

    dlogDp    = log_dp_edges[1:] - log_dp_edges[:-1]
    dp_mid_um = 10.0 ** (0.5 * (log_dp_edges[:-1] + log_dp_edges[1:]))
    m_mid     = np.sqrt(xk_np[:-1] * xk_np[1:])

    Nk = Nk_cm3 * boxvol
    Mk = np.zeros((nbins, ICOMP))
    Mk[:, SRTSO4] = Nk * m_mid

    return Nk, Mk, Nk_cm3, dp_mid_um, dlogDp


# =========================================================================
# Comparison plot (both distributions)
# =========================================================================

def plot_distribution(outdir=None):
    """Plot both distributions on the same axes with 40-bin mapping."""
    xk40 = np.array(make_grid(40, XK0, 2.0))
    dp_fine = np.logspace(np.log10(0.001), np.log10(2.0), 500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Background aerosol size distributions — TOMAS 40-bin mapping',
                 fontsize=12, fontweight='bold')

    for dist_key, meta in _DIST_META.items():
        data     = meta['data']
        dp_um    = data[:, 0]
        dndlogdp = data[:, 1]
        c1, c2   = meta['color1'], meta['color2']
        lbl      = meta['label']

        _, _, Nk40, dp40, dlogDp40 = map_to_grid(xk40, dist=dist_key)
        N_orig = float(np.trapezoid(dndlogdp, np.log10(dp_um)))
        N_40   = float(Nk40.sum())

        dn_fine = dndlogdp_at(dp_fine, dist=dist_key)

        for ax in axes:
            ax.scatter(dp_um, dndlogdp, s=20, color=c1, zorder=5, alpha=0.8)
            ax.plot(dp_fine, dn_fine, color=c1, lw=1.5, ls='--', alpha=0.7)
            ax.step(dp40, Nk40 / dlogDp40, where='mid', color=c2, lw=2.0,
                    label=f'{lbl}\n  N_orig={N_orig:.0f}, N_40bin={N_40:.0f} cm⁻³')

        print(f'{dist_key}: N_orig={N_orig:.1f}, N_40bin={N_40:.1f} cm⁻³')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Dp [µm]')
        ax.set_ylabel('dN/dlogDp [cm⁻³]')
        ax.set_xlim(8e-4, 3.0)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.25, which='both')
        ax.legend(fontsize=8, frameon=False)

    axes[0].set_title('Linear y-axis')
    axes[1].set_title('Log y-axis')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1, 2000)

    fig.tight_layout()

    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 'background_aerosol_distribution.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
    return out


# =========================================================================
# Public accessor
# =========================================================================

def get_initial_state(nbins=40, boxvol=BOXVOL, dist='tabulated',
                      to_ambient=False, temp=None, pres=None):
    """Return Nk and Mk for the chosen distribution mapped onto nbins bins.

    Parameters
    ----------
    nbins : 40 or 80
    boxvol : cm³
    dist : 'tabulated', 'tabulated2', or 'redcircles'
    to_ambient : if True, the tabulated dN/dlogDp (assumed reported per cm³ STP)
        is converted to per cm³ at ambient (temp, pres) by multiplying Nk and
        Mk by ``stp_to_ambient_factor(temp, pres)``. Requires temp and pres.
    temp, pres : ambient temperature [K] and pressure [Pa] (only used when
        ``to_ambient=True``).
    """
    if nbins == 80:
        xk = np.array(make_grid_80bin())
    else:
        xk = np.array(make_grid(nbins, XK0, 2.0))
    Nk, Mk, _, _, _ = map_to_grid(xk, boxvol, dist=dist)
    if to_ambient:
        if temp is None or pres is None:
            raise ValueError("to_ambient=True requires temp and pres")
        f = stp_to_ambient_factor(temp, pres)
        Nk = Nk * f
        Mk = Mk * f
    return Nk, Mk


# =========================================================================
# Red-circles cross-check plot (STP vs ambient)
# =========================================================================

def plot_redcircles_crosscheck(temp=210.0, pres=5500.0, nbins=40, outdir=None):
    """Cross-check the digitized 'redcircles' distribution against the figure.

    Plots, side by side (linear-y and log-y):
      - digitized dN/dlogDp points (STP)        [scatter]
      - fine linear interpolation (STP)          [dashed]
      - 40-bin TOMAS mapping (STP)               [step]
      - 40-bin TOMAS mapping (ambient = STP*f)   [step]
    so the user can confirm the extraction matches the original figure (STP)
    and see the ambient values actually used in the simulation.
    """
    f = stp_to_ambient_factor(temp, pres)
    data = _DIST_META['redcircles']['data']
    dp_um, dndlogdp = data[:, 0], data[:, 1]

    if nbins == 80:
        xk = np.array(make_grid_80bin())
    else:
        xk = np.array(make_grid(nbins, XK0, 2.0))
    _, _, Nk_stp, dp40, dlogDp40 = map_to_grid(xk, dist='redcircles')
    N_orig_stp = float(np.trapezoid(dndlogdp, np.log10(dp_um)))
    N_40_stp   = float(Nk_stp.sum())

    dp_fine = np.logspace(np.log10(0.003), np.log10(2.0), 500)
    dn_fine = dndlogdp_at(dp_fine, dist='redcircles')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Red-circles (obs 220-230 ppbv) extraction cross-check\n'
        f'STP->ambient factor f = (P/Pstp)(Tstp/T) = {f:.4f}  '
        f'(T={temp:.0f}K, P={pres/100:.0f}hPa)',
        fontsize=11, fontweight='bold')

    for ax in axes:
        ax.scatter(dp_um, dndlogdp, s=28, color='#E53935', zorder=5,
                   label=f'digitized points (STP), N={N_orig_stp:.0f} cm⁻³')
        ax.plot(dp_fine, dn_fine, color='#E53935', lw=1.2, ls='--', alpha=0.6,
                label='linear interp (STP)')
        ax.step(dp40, Nk_stp / dlogDp40, where='mid', color='#1565C0', lw=2.0,
                label=f'40-bin map (STP), N={N_40_stp:.0f} cm⁻³')
        ax.step(dp40, (Nk_stp * f) / dlogDp40, where='mid', color='#2E7D32',
                lw=2.0, label=f'40-bin map (AMBIENT), N={N_40_stp*f:.1f} cm⁻³')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Dp [µm]')
        ax.set_ylabel('dN/dlogDp [cm⁻³]')
        ax.set_xlim(3e-3, 2.0)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.25, which='both')
        ax.legend(fontsize=8, frameon=False)

    axes[0].set_title('Linear y-axis')
    axes[1].set_title('Log y-axis')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-2, 200)

    fig.tight_layout()
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 'redcircles_crosscheck.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'redcircles: N_STP(orig)={N_orig_stp:.1f}, N_STP(40bin)={N_40_stp:.1f}, '
          f'N_ambient(40bin)={N_40_stp*f:.2f} cm⁻³')
    print(f'Saved: {out}')
    return out


# =========================================================================
# Fig. S4-style reproduction (matches the original screenshot's framing)
# =========================================================================

def plot_redcircles_figure_style(temp=210.0, pres=5500.0, nbins=40,
                                 ylim_stp=(0, 1000), outdir=None):
    """Reproduce the digitized red-circles distribution in the same framing as
    the original figure (Fig. S4) so it can be overlaid for confirmation.

    Two panels, both: Diameter (µm) on a log x-axis, dN/dlogDp on a linear
    y-axis, digitized observation as red OPEN circles connected by a thin red
    line (the figure's "observation" style), plus the 40-bin TOMAS mapping.
      - Left : STP  (y-limits match the screenshot, default 0-1000)
      - Right: AMBIENT (= STP * f), auto-scaled to its own magnitude.
    """
    f = stp_to_ambient_factor(temp, pres)
    data = _DIST_META['redcircles']['data']
    dp_um, dndlogdp = data[:, 0], data[:, 1]

    if nbins == 80:
        xk = np.array(make_grid_80bin())
    else:
        xk = np.array(make_grid(nbins, XK0, 2.0))
    _, _, Nk_stp, dp40_um, dlogDp40 = map_to_grid(xk, dist='redcircles')
    dn40_stp = Nk_stp / dlogDp40
    N_stp = float(Nk_stp.sum())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        'Background size distribution (red circles, obs 220-230 ppbv) '
        '— reproduction of Fig. S4 framing',
        fontsize=12, fontweight='bold')

    panels = [
        (axes[0], 1.0,  'dN/dlogDp (cm$^{-3}$ STP)',      f'STP   (N={N_stp:.0f} cm⁻³)',        ylim_stp),
        (axes[1], f,    'dN/dlogDp (cm$^{-3}$ ambient)',  f'AMBIENT  (×{f:.4f}, N={N_stp*f:.2f} cm⁻³)', None),
    ]
    for ax, scale, ylabel, title, ylim in panels:
        # Observation: red open circles + thin connecting line (Fig. S4 style)
        ax.plot(dp_um, dndlogdp * scale, color='#E53935', lw=1.0, zorder=4)
        ax.plot(dp_um, dndlogdp * scale, marker='o', ls='none',
                markerfacecolor='none', markeredgecolor='#E53935',
                markersize=7, markeredgewidth=1.3, zorder=5,
                label='observation, 220-230 ppbv')
        # 40-bin TOMAS mapping
        ax.step(dp40_um, dn40_stp * scale, where='mid', color='#1565C0',
                lw=1.8, alpha=0.9, label='TOMAS 40-bin mapping')
        ax.set_xscale('log'); ax.set_xlim(3e-3, 3.0)
        ax.set_xlabel('Diameter (µm)'); ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, which='both')
        ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 'redcircles_figure_style.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'redcircles (Fig.S4 style): N_STP={N_stp:.1f} cm⁻³, '
          f'N_ambient={N_stp*f:.2f} cm⁻³ (f={f:.4f})')
    print(f'Saved: {out}')
    return out


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--redcircles', action='store_true',
                    help='Plot the red-circles extraction cross-check (STP vs ambient)')
    ap.add_argument('--figure-style', action='store_true',
                    help='Reproduce the Fig. S4 framing (STP + ambient panels)')
    ap.add_argument('--temp', type=float, default=210.0)
    ap.add_argument('--pres', type=float, default=5500.0)
    args = ap.parse_args()
    if args.figure_style:
        plot_redcircles_figure_style(temp=args.temp, pres=args.pres)
    elif args.redcircles:
        plot_redcircles_crosscheck(temp=args.temp, pres=args.pres)
    else:
        plot_distribution()
