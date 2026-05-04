"""Radiative forcing diagnostic plots.

Generates 3 figures:
    1. Scattering efficiency vs radius (Pierce et al. 2010 Figure 1),
       solar-spectrum-weighted Mie, Tabazadeh composition at multiple RH
    2. Upscatter fraction vs asymmetry parameter (Wiscombe & Grams 1976)
    3. H2SO4/H2O equilibrium composition vs temperature (Tabazadeh et al. 1997)

Usage:
    python -m benchmarks.python.plot_radiative_forcing [--outdir DIR] [--temp 220] [--rh 5]
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tomas_jax.core.config import make_grid, XK0, NBINS, ICOMP, PI
from tomas_jax.physics.bhmie import bhmie
from tomas_jax.physics.radiative_forcing import (
    precompute_mie_properties,
    compute_rf,
    compute_optical_depth,
    compute_mass_scattering_efficiency,
    compute_rf_efficiency,
    upscatter_fraction,
    scattering_efficiency_vs_radius,
    h2so4_equilibrium_wt,
    h2so4_solution_density,
    DENSITY_SULFATE,
    WAVELENGTH_DEFAULT,
    REFINDEX_SULFATE,
    SOLAR_CONSTANT,
    TATM_DEFAULT,
    ALBEDO_GLOBAL_AVG,
    # Pierce et al. (2010) SI parameters
    SOLAR_CONSTANT_PIERCE,
    TATM_STRATOSPHERIC,
    CLOUD_FRACTION_DEFAULT,
    ALBEDO_SURFACE_CLEARSKY,
    DENSITY_H2SO4_SOLUTION,
    WAVELENGTH_PIERCE,
    H2SO4_WEIGHT_FRACTION,
)


def make_lognormal_Nk(xk, N_total, gmd, gsd, density=DENSITY_SULFATE):
    """Create a lognormal number distribution on TOMAS bins.

    Args:
        xk: Bin boundaries [kg], shape (nbins+1,).
        N_total: Total number [#/cell].
        gmd: Geometric mean diameter [m].
        gsd: Geometric standard deviation (dimensionless).
        density: Particle density [kg/m³].

    Returns:
        Nk: Number per bin [#/cell], shape (nbins,).
    """
    xk_np = np.asarray(xk, dtype=np.float64)
    nbins = len(xk_np) - 1

    # Geometric-mean mass per bin
    m_geo = np.sqrt(xk_np[:-1] * xk_np[1:])
    # Corresponding diameter
    d_geo = 2.0 * (3.0 * m_geo / (4.0 * PI * density))**(1.0 / 3.0)

    # Lognormal dN/dlog(d)
    ln_gsd = np.log(gsd)
    dNdlogd = (N_total / (np.sqrt(2 * PI) * ln_gsd)
               * np.exp(-0.5 * (np.log(d_geo / gmd) / ln_gsd)**2))

    # Bin widths in log(d) space
    d_lo = 2.0 * (3.0 * xk_np[:-1] / (4.0 * PI * density))**(1.0 / 3.0)
    d_hi = 2.0 * (3.0 * xk_np[1:] / (4.0 * PI * density))**(1.0 / 3.0)
    dlogd = np.log10(d_hi) - np.log10(d_lo)

    Nk = dNdlogd * dlogd
    return np.maximum(Nk, 0.0)


def plot_scattering_efficiency_vs_radius(fig, ax, temp=220.0, n_wavelengths=30):
    """Figure 1: Pierce et al. 2010 Figure 1 reproduction.

    Uses Pierce's SI parameters: T_a=1.0, A=0.6, R=0.15, S₀=1370.
    Solar-spectrum-weighted Mie scattering (Planck 5778K, 300-2500nm).
    Composition computed from Tabazadeh et al. (1997) at the specified
    temperature for several RH values.
    """
    n_radii = 80
    EARTH_AREA = 5.1e14  # m²

    # RH values to show (stratospheric range)
    rh_values = [2, 5, 10, 20]
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    styles = ['-', '-', '--', ':']

    for rh, color, ls in zip(rh_values, colors, styles):
        wt = h2so4_equilibrium_wt(temp, rh)
        rho = h2so4_solution_density(wt)
        wt_frac = wt / 100.0

        pierce_kw = dict(
            solar_constant=SOLAR_CONSTANT_PIERCE,
            Tatm=TATM_STRATOSPHERIC,
            albedo=ALBEDO_SURFACE_CLEARSKY,
            cloud_fraction=CLOUD_FRACTION_DEFAULT,
            density=rho,
        )

        print(f"  RH={rh}%: wt={wt:.1f}%, rho={rho:.0f} kg/m3")
        radii, rf = scattering_efficiency_vs_radius(
            n_radii=n_radii, r_min=5e-9, r_max=10e-6,
            spectral=True, n_wavelengths=n_wavelengths, **pierce_kw,
        )

        # Convert to W/m² per Mt-S using the self-consistent wt%
        MW_RATIO_WET = 1.0 / (wt_frac * 32.0 / 98.0)
        g_per_m2_per_MtS = (1e9 * MW_RATIO_WET * 1e3) / EARTH_AREA
        cool = -rf * 1e-3 * g_per_m2_per_MtS

        peak_val = cool.max()
        peak_r = radii[np.argmax(cool)] * 1e6
        ax.semilogx(radii * 1e6, cool, color=color, linestyle=ls, linewidth=2,
                    label=f'RH={rh}% ({wt:.0f} wt%, $\\rho$={rho:.0f} kg/m$^3$, '
                          f'peak={peak_val:.2f} at r={peak_r:.2f} $\\mu$m)')

    ax.set_xlabel('Particle radius [$\\mu$m]')
    ax.set_ylabel('Scattering cooling efficiency\n[W m$^{-2}$ per Mt-S]')
    ax.set_title(f'RF per unit sulfur burden vs particle size\n'
                 f'(Solar-spectrum-weighted Mie, T={temp:.0f} K)')
    ax.set_xlim(5e-3, 10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Assumptions box at bottom
    assumptions = (
        f'RF = $-$(S$_0$/4) $\\cdot$ T$_a^2$ $\\cdot$ (1$-$A) $\\cdot$ (1$-$R)$^2$ '
        f'$\\cdot$ 2$\\beta\\tau$   (Chylek & Wong 1995 / Pierce et al. 2010 SI)\n'
        f'S$_0$ = {SOLAR_CONSTANT_PIERCE:.0f} W/m$^2$ (solar constant)  |  '
        f'T$_a$ = {TATM_STRATOSPHERIC} (atmospheric transmittance above aerosol; '
        f'stratosphere: nothing above)  |  '
        f'A = {CLOUD_FRACTION_DEFAULT} (cloud fraction)\n'
        f'R = {ALBEDO_SURFACE_CLEARSKY} (clear-sky surface albedo)  |  '
        f'n = 1.4 + 1e-8i (refractive index, ~65 wt% H$_2$SO$_4$)  |  '
        f'Mie spectrum: Planck 5778 K, 300\u20132500 nm, {n_wavelengths} bands\n'
        f'Composition: Tabazadeh et al. (1997) binary H$_2$SO$_4$/H$_2$O '
        f'equilibrium at T = {temp:.0f} K  |  '
        f'Density: CRC Handbook (25$^\\circ$C)'
    )
    fig.text(0.5, -0.01, assumptions, ha='center', va='top', fontsize=6.5,
             wrap=True,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))


def plot_mie_properties(ax1, mie):
    """Figure 2: Qsca, Qext, gsca vs TOMAS bin radius."""
    r_um = mie.radii * 1e6

    color_sca = 'tab:blue'
    color_ext = 'tab:red'
    ax1.semilogx(r_um, mie.Qsca, '-', color=color_sca, linewidth=2, label='$Q_{sca}$')
    ax1.semilogx(r_um, mie.Qext, '--', color=color_ext, linewidth=1.5, label='$Q_{ext}$')
    ax1.set_xlabel('Particle radius [μm]')
    ax1.set_ylabel('Efficiency')
    ax1.set_title('Mie scattering properties per TOMAS bin\n(λ = 550 nm, n = 1.4)')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_g = 'tab:green'
    ax2.semilogx(r_um, mie.gsca, '-.', color=color_g, linewidth=1.5, label='$g$ (asymmetry)')
    ax2.set_ylabel('Asymmetry parameter $g$', color=color_g)
    ax2.tick_params(axis='y', labelcolor=color_g)
    ax2.set_ylim(-0.1, 1.0)
    ax2.legend(loc='right', fontsize=10)


def plot_upscatter_vs_g(ax):
    """Figure 3: Upscatter fraction vs asymmetry parameter at several SZAs."""
    g_values = np.linspace(0, 0.95, 40)
    sza_list = [20, 40, 60, 80]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sza_list)))

    print("  Computing upscatter fractions...")
    for sza_deg, color in zip(sza_list, colors):
        beta = [upscatter_fraction(g, np.radians(sza_deg)) for g in g_values]
        ax.plot(g_values, beta, '-', color=color, linewidth=1.5,
                label=f'SZA = {sza_deg}°')

    ax.set_xlabel('Asymmetry parameter $g$')
    ax.set_ylabel('Upscatter fraction β')
    ax.set_title('Upscatter fraction vs asymmetry parameter\n(Wiscombe & Grams 1976)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.95)
    ax.set_ylim(0, 0.6)


def plot_rf_per_bin(ax, mie, xk, wt_frac, rho_wet):
    """Figure 4: RF per bin for sample lognormal distributions.

    Uses column_area = 1 m², so Nk represents column number density [#/m²].
    Uses Pierce SI parameters for stratospheric aerosol.
    """
    xk_np = np.asarray(xk)
    nbins = len(xk_np) - 1
    r_um = mie.radii * 1e6

    column_area = 1.0  # m²

    # Fixed sulfate burden: 5 mg-S/m²
    burden_wet_kg_m2 = 5e-6 / (wt_frac * 32.0 / 98.0)

    gmds = [0.05e-6, 0.2e-6, 1.0e-6]
    labels = ['Small (GMD=50nm)', 'Optimal (GMD=200nm)', 'Large (GMD=1$\\mu$m)']
    colors = ['tab:blue', 'tab:green', 'tab:red']

    for gmd, label, color in zip(gmds, labels, colors):
        m_mean = (4.0 / 3.0) * PI * rho_wet * (gmd / 2.0)**3
        N_col_total = burden_wet_kg_m2 / m_mean

        Nk = make_lognormal_Nk(xk, N_col_total, gmd, 1.6, density=rho_wet)
        rf_total, rf_per_bin = compute_rf(
            Nk, mie, column_area,
            solar_constant=SOLAR_CONSTANT_PIERCE,
            Tatm=TATM_STRATOSPHERIC,
            albedo=ALBEDO_SURFACE_CLEARSKY,
            cloud_fraction=CLOUD_FRACTION_DEFAULT,
        )
        ax.semilogx(r_um, -rf_per_bin, '-o', color=color,
                     linewidth=1.5, markersize=3,
                     label=f"{label} (RF={rf_total:+.2f} W/m$^2$)")

    ax.set_xlabel('Particle radius [$\\mu$m]')
    ax.set_ylabel('Cooling per bin [W m$^{-2}$]')
    ax.set_title(f'RF contribution per TOMAS bin\n'
                 f'(lognormal, 5 mg-S m$^{{-2}}$, {wt_frac*100:.0f} wt% H$_2$SO$_4$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(r_um[0], r_um[-1])


def plot_composition_vs_temperature(ax):
    """Figure 5: H₂SO₄ weight percent vs temperature for several RH values.

    Computed using Tabazadeh et al. (1997) parameterization.
    """
    temps = np.linspace(185, 260, 80)
    rh_values = [1, 2, 5, 10, 20, 50]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(rh_values)))

    for rh, color in zip(rh_values, colors):
        wt = [h2so4_equilibrium_wt(T, rh) for T in temps]
        ax.plot(temps, wt, '-', color=color, linewidth=1.8,
                label=f'RH = {rh}%')

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('H$_2$SO$_4$ weight percent [%]')
    ax.set_title('Equilibrium H$_2$SO$_4$/H$_2$O composition\n'
                 '(Tabazadeh et al. 1997)')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(185, 260)
    ax.set_ylim(10, 80)
    ax.axvline(220, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('Strat. T', xy=(220, 12), fontsize=8, color='gray', ha='center')


def main():
    parser = argparse.ArgumentParser(description='Radiative forcing diagnostic plots')
    parser.add_argument('--outdir', default='benchmarks/results/radiative_forcing',
                        help='Output directory for figures')
    parser.add_argument('--temp', type=float, default=220.0,
                        help='Temperature for composition calculation [K]')
    parser.add_argument('--rh', type=float, default=5.0,
                        help='Relative humidity for Mie/RF calculations [%%]')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    T_strat = args.temp
    RH_strat = args.rh

    print(f"Tabazadeh (1997) equilibrium at T={T_strat:.0f}K, RH={RH_strat:.0f}%:")
    wt = h2so4_equilibrium_wt(T_strat, RH_strat)
    rho = h2so4_solution_density(wt)
    print(f"  {wt:.1f} wt% H2SO4, rho={rho:.0f} kg/m3")

    # Figure 1: Scattering efficiency vs radius (spectral, 50 bands)
    print("\nFigure 1: Scattering efficiency vs radius (spectral, Tabazadeh)...")
    fig1, ax1 = plt.subplots(1, 1, figsize=(11, 7))
    plot_scattering_efficiency_vs_radius(fig1, ax1, temp=T_strat, n_wavelengths=50)
    fig1.tight_layout(rect=[0, 0.06, 1, 1])
    fig1.savefig(os.path.join(args.outdir, 'fig1_scattering_efficiency_vs_radius.png'),
                 dpi=150, bbox_inches='tight')
    print(f"  Saved: {args.outdir}/fig1_scattering_efficiency_vs_radius.png")

    # Figure 2: Upscatter fraction vs asymmetry parameter
    print("\nFigure 2: Upscatter fraction vs g...")
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    plot_upscatter_vs_g(ax2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, 'fig2_upscatter_fraction.png'), dpi=150)
    print(f"  Saved: {args.outdir}/fig2_upscatter_fraction.png")

    # Figure 3: Composition vs temperature (Tabazadeh parameterization)
    print("\nFigure 3: H2SO4/H2O composition vs temperature...")
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    plot_composition_vs_temperature(ax3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.outdir, 'fig3_h2so4_composition.png'), dpi=150)
    print(f"  Saved: {args.outdir}/fig3_h2so4_composition.png")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Pierce SI: S0={SOLAR_CONSTANT_PIERCE:.0f} W/m2, T_a={TATM_STRATOSPHERIC}, "
          f"A={CLOUD_FRACTION_DEFAULT}, R={ALBEDO_SURFACE_CLEARSKY}")
    print(f"  Spectral Mie: Planck 5778K, 300-2500nm, 50 bands (<1% of converged)")
    print(f"  Composition: Tabazadeh (1997), T={T_strat:.0f}K")
    for rh in [2, 5, 10, 20]:
        wt = h2so4_equilibrium_wt(T_strat, rh)
        rho = h2so4_solution_density(wt)
        print(f"    RH={rh:2d}%: {wt:.1f} wt% H2SO4, rho={rho:.0f} kg/m3")

    plt.close('all')
    print(f"\nAll figures saved to {args.outdir}/")


if __name__ == '__main__':
    main()
