"""Radiative forcing timeseries and RF-banana plots for SAI dilution scenarios.

Computes per-bin and total direct SW RF from the TOMAS-JAX size distributions,
assuming all aerosol mass is SO4 (H2SO4/H2O solution at stratospheric T/RH).

Physics:
    - Mie properties precomputed once per grid using sulfate refractive index
    - RF from Chylek & Wong (1995) / Pierce et al. (2010):
        RF = -(S0/4) * T^2 * (1-fc) * (1-A)^2 * 2*beta * tau
    - Pierce SI parameters: T_atm=1.0, albedo=0.15, cloud_fraction=0.6
    - column_area = boxvol_m3 / layer_depth_m (SAI plume layer depth ~500 m)

Two figures per scenario:
    1. rf_total_<name>.png  — total RF [W/m²] vs time
    2. rf_banana_<name>.png — dRF/dlogDp [W/m²] vs Dp vs time (pcolormesh)

Two subgroup composites per subgroup:
    3. subgroup_rf_total_sg<X>.png  — 3×2 total RF grid (raw|fit)
    4. subgroup_rf_banana_sg<X>.png — 3×2 RF banana grid (raw|fit)

Usage::

    python -m experimental_case.plot_radiative_forcing
    python -m experimental_case.plot_radiative_forcing --outdir results/dilution_scenarios_tabulated2_zero_h2so4
    python -m experimental_case.plot_radiative_forcing --plot-only
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from tomas_jax.core.config import make_grid, make_grid_80bin, XK0, PI
from tomas_jax.physics.radiative_forcing import (
    precompute_mie_properties, compute_rf,
    TATM_STRATOSPHERIC, CLOUD_FRACTION_DEFAULT, ALBEDO_SURFACE_CLEARSKY,
    SOLAR_CONSTANT_PIERCE, DENSITY_H2SO4_SOLUTION, WAVELENGTH_PIERCE,
    REFINDEX_SULFATE,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experimental_case.run_dilution_scenarios import (
    SUBGROUPS, NORMS, VARIANTS, all_scenarios,
    scenario_name, load_scenario,
    _DEFAULT_DILUTION_DIR,
)

# =========================================================================
# RF constants (Pierce et al. 2010 SI — stratospheric)
# =========================================================================
BOXVOL_M3     = 1.0          # m³ (BOXVOL = 1e6 cm³ = 1 m³)
LAYER_DEPTH_M = 500.0        # m  — assumed plume layer depth
COLUMN_AREA   = BOXVOL_M3 / LAYER_DEPTH_M   # m²

_DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(__file__), 'results', 'dilution_scenarios_tabulated_zero_h2so4'
)

# =========================================================================
# Mie precomputation cache (keyed by nbins)
# =========================================================================
_MIE_CACHE = {}

def _get_mie(nbins):
    if nbins not in _MIE_CACHE:
        xk = np.array(make_grid_80bin() if nbins == 80
                      else make_grid(nbins, XK0, 2.0))
        print(f'  Precomputing Mie properties for {nbins}-bin grid...', end=' ', flush=True)
        mie = precompute_mie_properties(
            xk,
            density=DENSITY_H2SO4_SOLUTION,
            wavelength=WAVELENGTH_PIERCE,
            refindex=REFINDEX_SULFATE,
            global_avg_upscatter=True,
        )
        _MIE_CACHE[nbins] = (mie, xk)
        print('done.')
    return _MIE_CACHE[nbins]


# =========================================================================
# Grid geometry helpers (same as run_dilution_scenarios)
# =========================================================================

def _grid_geometry(nbins):
    xk_np = np.array(make_grid_80bin() if nbins == 80
                     else make_grid(nbins, XK0, 2.0))
    m_mid   = np.sqrt(xk_np[:-1] * xk_np[1:])
    dp_m    = (6.0 * m_mid / (DENSITY_H2SO4_SOLUTION * PI)) ** (1.0 / 3.0)
    dp_nm   = dp_m * 1e9
    dp_edges = (6.0 * xk_np / (DENSITY_H2SO4_SOLUTION * PI)) ** (1.0 / 3.0)
    dlogDp  = np.log10(dp_edges[1:] / dp_edges[:-1])
    return dp_nm, dlogDp


# =========================================================================
# Per-scenario RF computation
# =========================================================================

def compute_rf_timeseries(d):
    """Compute RF timeseries from a loaded scenario dict.

    Returns
    -------
    rf_total  : ndarray (nsteps,)  — total RF [W/m²]
    rf_per_bin: ndarray (nsteps, nbins) — per-bin RF [W/m²]
    drf_dlogdp: ndarray (nsteps, nbins) — dRF/dlogDp [W/m²]
    t_h       : ndarray (nsteps,)  — time [hours]
    dp_nm     : ndarray (nbins,)   — bin-centre diameter [nm]
    """
    nbins = int(d['nbins'])
    mie, _ = _get_mie(nbins)
    dp_nm, dlogDp = _grid_geometry(nbins)

    Nk_all = d['Nk_every']            # (nsteps, nbins)
    nsteps = Nk_all.shape[0]

    rf_total   = np.zeros(nsteps)
    rf_per_bin = np.zeros((nsteps, nbins))

    for i in range(nsteps):
        rt, rpb = compute_rf(
            Nk_all[i],
            mie,
            COLUMN_AREA,
            solar_constant=SOLAR_CONSTANT_PIERCE,
            Tatm=TATM_STRATOSPHERIC,
            albedo=ALBEDO_SURFACE_CLEARSKY,
            cloud_fraction=CLOUD_FRACTION_DEFAULT,
        )
        rf_total[i]    = float(rt)
        rf_per_bin[i]  = np.array(rpb)

    drf_dlogdp = rf_per_bin / dlogDp[np.newaxis, :]
    t_h = d['t_seconds'] / 3600.0
    return rf_total, rf_per_bin, drf_dlogdp, t_h, dp_nm


# =========================================================================
# RF banana helper
# =========================================================================

def _rf_banana_norm(drf_dlogdp):
    """Symmetric log norm centred on 0 for RF banana."""
    vmax = max(abs(drf_dlogdp).max(), 1e-6)
    return mcolors.SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax)


# =========================================================================
# Per-scenario plots
# =========================================================================

def plot_rf_scenario(d, sg, norm, variant, outdir, precomputed=None):
    """RF timeseries + RF banana for one scenario."""
    name = scenario_name(sg, norm, variant)

    if precomputed is None:
        rf_total, _, drf_dlogdp, t_h, dp_nm = compute_rf_timeseries(d)
    else:
        rf_total, _, drf_dlogdp, t_h, dp_nm = precomputed

    figs = []

    # — 1. Total RF timeseries —
    sns.set_theme(style='ticks', font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f'Direct SW radiative forcing — {name}\n'
                 f'(T={216}K, P=55hPa, Pierce et al. 2010 params)',
                 fontsize=11, fontweight='bold')

    ax.plot(t_h, rf_total, color='#1565C0', lw=1.8)
    ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
    ax.fill_between(t_h, rf_total, 0,
                    where=(rf_total < 0), alpha=0.15, color='#1565C0',
                    label='Cooling')
    ax.fill_between(t_h, rf_total, 0,
                    where=(rf_total > 0), alpha=0.15, color='#E53935',
                    label='Warming')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('RF [W/m²]')
    ax.set_title('Total direct SW RF')
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()

    out = os.path.join(outdir, 'figs', f'rf_total_{name}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figs.append(out)
    sns.reset_defaults()

    # — 2. RF banana —
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor('#f5f5f5')

    norm_obj = _rf_banana_norm(drf_dlogdp)
    pcm = ax.pcolormesh(t_h, dp_nm, drf_dlogdp.T,
                        norm=norm_obj, cmap='RdBu_r', shading='nearest')

    ax.set_yscale('log')
    ax.set_ylim(1, 2e4)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Dp [nm]')
    ax.set_title(f'dRF/dlogDp [W/m²] — {name}', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='out')

    # RF optimal size band
    ax.axhspan(400, 440, color='gold', alpha=0.25, zorder=2)
    for dp in (400, 440):
        ax.axhline(dp, color='goldenrod', lw=1.0, ls='--', alpha=0.8, zorder=3)

    cbar = fig.colorbar(pcm, ax=ax, label='dRF/dlogDp [W/m²]', pad=0.01)
    fig.tight_layout()

    out = os.path.join(outdir, 'figs', f'rf_banana_{name}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figs.append(out)

    return figs


# =========================================================================
# Subgroup composites
# =========================================================================

def plot_subgroup_rf_total(sg, outdir):
    """3×2 total RF timeseries grid."""
    sns.set_theme(style='ticks', font_scale=1.0)
    norm_labels = {'first_hour_mean': 'First hour mean',
                   'first_value':     'First value',
                   'smoothed':        'Smoothed'}

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f'Total direct SW RF — sg{sg}  (Pierce et al. 2010 params)',
                 fontsize=13, fontweight='bold')

    for row, norm in enumerate(NORMS):
        for col, variant in enumerate(VARIANTS):
            ax = axes[row, col]
            try:
                d = load_scenario(sg, norm, variant, outdir)
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                sns.despine(ax=ax)
                continue

            rf_total, _, _, t_h, _ = compute_rf_timeseries(d)

            ax.plot(t_h, rf_total, color='#1565C0', lw=1.5)
            ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
            ax.fill_between(t_h, rf_total, 0,
                            where=(rf_total < 0), alpha=0.12, color='#1565C0')
            ax.fill_between(t_h, rf_total, 0,
                            where=(rf_total > 0), alpha=0.12, color='#E53935')

            sns.despine(ax=ax)
            ax.grid(True, alpha=0.2)

            if col == 0:
                ax.set_ylabel(f'{norm_labels[norm]}\nRF [W/m²]', fontsize=9)
            if row == 2:
                ax.set_xlabel('Time [h]')
            ax.set_title(variant, fontsize=10)

    fig.tight_layout()
    out = os.path.join(outdir, 'figs', f'subgroup_rf_total_sg{sg}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    sns.reset_defaults()
    return out


def plot_subgroup_rf_banana(sg, outdir):
    """3×2 RF banana grid."""
    norm_labels = {'first_hour_mean': 'First hour mean',
                   'first_value':     'First value',
                   'smoothed':        'Smoothed'}

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f'dRF/dlogDp [W/m²] — sg{sg}  (Pierce et al. 2010 params)',
                 fontsize=13, fontweight='bold')

    # First pass: collect global vmax for consistent color scale
    all_data = {}
    vmax = 1e-10
    for norm in NORMS:
        for variant in VARIANTS:
            try:
                d = load_scenario(sg, norm, variant, outdir)
                _, _, drf, t_h, dp_nm = compute_rf_timeseries(d)
                all_data[(norm, variant)] = (drf, t_h, dp_nm)
                vmax = max(vmax, abs(drf).max())
            except FileNotFoundError:
                pass

    norm_obj = mcolors.SymLogNorm(linthresh=vmax * 1e-3,
                                   vmin=-vmax, vmax=vmax)
    pcm = None
    for row, norm in enumerate(NORMS):
        for col, variant in enumerate(VARIANTS):
            ax = axes[row, col]
            ax.set_facecolor('#f5f5f5')

            if (norm, variant) not in all_data:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                continue

            drf, t_h, dp_nm = all_data[(norm, variant)]
            pcm = ax.pcolormesh(t_h, dp_nm, drf.T,
                                norm=norm_obj, cmap='RdBu_r', shading='nearest')
            ax.set_yscale('log')
            ax.set_ylim(1, 2e4)

            ax.axhspan(400, 440, color='gold', alpha=0.25, zorder=2)
            for dp in (400, 440):
                ax.axhline(dp, color='goldenrod', lw=0.9, ls='--',
                           alpha=0.8, zorder=3)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(which='both', direction='out')

            if col == 0:
                ax.set_ylabel(f'{norm_labels[norm]}\nDp [nm]', fontsize=9)
            if row == 2:
                ax.set_xlabel('Time [h]')
            ax.set_title(variant, fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    if pcm is not None:
        cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(pcm, cax=cax, label='dRF/dlogDp [W/m²]')

    out = os.path.join(outdir, 'figs', f'subgroup_rf_banana_sg{sg}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# =========================================================================
# Run all
# =========================================================================

def plot_all_rf(outdir=_DEFAULT_OUTDIR):
    print(f'  RF plots → {os.path.join(outdir, "figs")}/...')

    # Per-scenario
    for sg, norm, variant in all_scenarios():
        name = scenario_name(sg, norm, variant)
        try:
            d = load_scenario(sg, norm, variant, outdir)
        except FileNotFoundError:
            print(f'    {name}: no data, skipped')
            continue
        figs = plot_rf_scenario(d, sg, norm, variant, outdir)
        print(f'    {name}: {len(figs)} RF figures')

    # Subgroup composites
    for sg in SUBGROUPS:
        out = plot_subgroup_rf_total(sg, outdir)
        print(f'    subgroup_rf_total_sg{sg}: done')
        out = plot_subgroup_rf_banana(sg, outdir)
        print(f'    subgroup_rf_banana_sg{sg}: done')

    print(f'  RF plots complete.')


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Radiative forcing plots for SAI dilution scenarios')
    parser.add_argument('--outdir', default=_DEFAULT_OUTDIR,
                        help='Scenario results directory (contains NPZ files)')
    args = parser.parse_args()
    plot_all_rf(args.outdir)
