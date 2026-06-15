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

_DIST_META = {
    'tabulated':  {'data': _DATA1, 'label': 'Dist 1 (nucleation-mode, peak 45 nm)',
                   'color1': '#1E88E5', 'color2': '#E53935'},
    'tabulated2': {'data': _DATA2, 'label': 'Dist 2 (accumulation-mode, peak 150 nm)',
                   'color1': '#43A047', 'color2': '#FB8C00'},
}

# =========================================================================
# Constants
# =========================================================================
DENS_SO4 = 1770.0    # kg/m³
BOXVOL   = 1.0e6     # cm³ (1 m³)


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
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
    return out


# =========================================================================
# Public accessor
# =========================================================================

def get_initial_state(nbins=40, boxvol=BOXVOL, dist='tabulated'):
    """Return Nk and Mk for the chosen distribution mapped onto nbins bins.

    Parameters
    ----------
    nbins : 40 or 80
    boxvol : cm³
    dist : 'tabulated' or 'tabulated2'
    """
    if nbins == 80:
        xk = np.array(make_grid_80bin())
    else:
        xk = np.array(make_grid(nbins, XK0, 2.0))
    Nk, Mk, _, _, _ = map_to_grid(xk, boxvol, dist=dist)
    return Nk, Mk


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    plot_distribution()
