"""PPM vs TFL Condensation Comparison Plots.

Generates four figures:
1. ppm_vs_tfl_distributions.png — Size distributions (5 panels)
2. ppm_vs_tfl_mass.png — Mass distributions (5 panels)
3. ppm_conservation.png — Conservation diagnostics
4. ppm_numerical_diffusion.png — Numerical diffusion comparison (Case C)

Usage:
    python -m benchmarks.python.plot_ppm_comparison [--data-dir PATH] [--output-dir PATH]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .utils import load_csv
from .test_cases import CASES, CASE_KEYS, init_lognormal

from tomas_jax.core.config import (
    SRTSO4, ICOMP, ICOMP_NODIAG, IDIAG, N_GAS_SPECIES, NBINS,
)
from tomas_jax.physics.condensation_sink import calc_condensation_sink
from tomas_jax.physics.ezcond import ezcond
from tomas_jax.physics.ezcond_ppm import ezcond_ppm
from tomas_jax.physics.nh3_equilibrium import eznh3eqm
from tomas_jax.physics.water_equilibrium import calc_equilibrium_water
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics.condensation_ppm import (
    ppm_condensation_step, compute_substeps, compute_wr,
)

CASE_COLORS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c',
               'D': '#d62728', 'E': '#9467bd'}
CASE_LABELS = {k: f"Case {k}: {v['name']}" for k, v in CASES.items()}


def compute_bin_diameters(xk):
    """Compute geometric mean diameter [um] per bin from xk [kg]."""
    pi = 3.141592654
    dens = 1770.0
    Dp = np.zeros(NBINS)
    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk[k]) / (dens * pi)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk[k + 1]) / (dens * pi)) ** 0.3333
        Dp[k] = np.sqrt(Dl * Dh)
    return Dp


def _run_post_ezcond(Nk_out, Mk_out, Gc, rh, xk):
    """Apply NH3 eq, water eq, MNFIX after ezcond."""
    Gc_jax = jnp.array(Gc)
    Mk_jax = jnp.array(Mk_out)
    Gc_jax, Mk_jax = eznh3eqm(Gc_jax, Mk_jax)
    Mk_jax = calc_equilibrium_water(Mk_jax, rh)
    Nk_jax, Mk_jax = mnfix_jax(
        jnp.array(Nk_out), Mk_jax, jnp.array(xk), ICOMP_NODIAG
    )
    return np.array(Nk_jax), np.array(Mk_jax)


def _run_both_methods(case_key, xk):
    """Run TFL and PPM for a given test case, return results dict."""
    case = CASES[case_key]
    temp, pres = case['temp'], case['pres']
    boxvol = 1.0e6
    rh, alpha = case['rh'], 1.0

    Nk_init, Mk_init = init_lognormal(case_key, xk)
    Gc = np.zeros(N_GAS_SPECIES)
    Gc[SRTSO4] = case['Gc_so4']

    CS_jax, sf_jax = calc_condensation_sink(
        jnp.array(Nk_init), jnp.array(Mk_init),
        temp, pres, boxvol, accommodation_coeff=alpha
    )
    mcond = Gc[SRTSO4] * (1.0 - np.exp(-float(CS_jax) * 60.0))
    sinkfrac = np.array(sf_jax)

    # TFL
    Nk_tfl, Mk_tfl = ezcond(
        Nk_init.copy(), Mk_init.copy(), mcond, SRTSO4,
        xk, temp, pres, boxvol, alpha
    )
    Nk_tfl, Mk_tfl = _run_post_ezcond(Nk_tfl, Mk_tfl, Gc, rh, xk)

    # PPM
    Nk_ppm, Mk_ppm = ezcond_ppm(
        Nk_init.copy(), Mk_init.copy(), mcond, SRTSO4,
        xk, temp, pres, boxvol, alpha
    )
    Nk_ppm, Mk_ppm = _run_post_ezcond(Nk_ppm, Mk_ppm, Gc, rh, xk)

    # CFL substeps
    tau = np.zeros(NBINS)
    totsinkfrac = np.sum(sinkfrac)
    tdt = 2.0 / 3.0
    for k in range(NBINS):
        mpo = np.sum(Mk_init[k, :ICOMP_NODIAG])
        mpw = np.sum(Mk_init[k, :])
        WR = mpw / mpo if mpo > 0 else 1.0
        if Nk_init[k] > 0 and totsinkfrac > 0:
            maddp = mcond * sinkfrac[k] / totsinkfrac / Nk_init[k]
            mpw_pp = mpw / Nk_init[k]
            tau[k] = 1.5 * ((mpw_pp + maddp * WR) ** tdt - mpw_pp ** tdt)

    WR_arr = np.array(compute_wr(jnp.array(Mk_init), ICOMP_NODIAG))
    n_sub = int(compute_substeps(jnp.array(xk), jnp.array(tau), jnp.array(WR_arr), 1.0))

    return {
        'Nk_init': Nk_init, 'Mk_init': Mk_init,
        'Nk_tfl': Nk_tfl, 'Mk_tfl': Mk_tfl,
        'Nk_ppm': Nk_ppm, 'Mk_ppm': Mk_ppm,
        'mcond': mcond, 'n_sub': n_sub,
    }


def _fwhm_bins(Nk):
    """Compute FWHM in bins."""
    peak = np.max(Nk)
    above = Nk >= peak / 2.0
    if not np.any(above):
        return 0
    indices = np.where(above)[0]
    return int(indices[-1] - indices[0] + 1)


def plot_distributions(data_dir, output_dir, xk, all_data):
    """Figure 1: Size distributions — Fortran, TFL, PPM."""
    Dp = compute_bin_diameters(xk)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('PPM vs TFL: Number Distributions', fontsize=14)

    for idx, case_key in enumerate(CASE_KEYS):
        ax = axes[idx]
        d = all_data[case_key]

        # Try loading Fortran reference
        try:
            Nk_f = load_csv(f"{data_dir}/level12_case{case_key}_Nk.csv").flatten()
            ax.semilogy(Dp, Nk_f, 'o', color='gray', label='Fortran',
                       markersize=3, alpha=0.7)
        except FileNotFoundError:
            pass

        ax.semilogy(Dp, d['Nk_tfl'], '--', color='blue', label='TFL', linewidth=1.5)
        ax.semilogy(Dp, d['Nk_ppm'], '-', color='red', label='PPM', linewidth=1.5)
        ax.semilogy(Dp, d['Nk_init'], ':', color='gray', label='Initial',
                   alpha=0.5, linewidth=1)

        ax.set_xscale('log')
        ax.set_xlabel('Dp [um]')
        if idx == 0:
            ax.set_ylabel('Nk [#/cell]')
        ax.set_title(CASE_LABELS[case_key], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'ppm_vs_tfl_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_mass(data_dir, output_dir, xk, all_data):
    """Figure 2: Mass distributions and relative error."""
    Dp = compute_bin_diameters(xk)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('PPM vs TFL: SO4 Mass Distributions', fontsize=14)

    for idx, case_key in enumerate(CASE_KEYS):
        d = all_data[case_key]

        # Top: mass distributions
        ax = axes[0, idx]
        try:
            Mk_f = load_csv(f"{data_dir}/level12_case{case_key}_Mk.csv")
            ax.semilogy(Dp, Mk_f[:, SRTSO4], 'o', color='gray',
                       label='Fortran', markersize=3, alpha=0.7)
        except FileNotFoundError:
            pass

        ax.semilogy(Dp, d['Mk_tfl'][:, SRTSO4], '--', color='blue',
                   label='TFL', linewidth=1.5)
        ax.semilogy(Dp, d['Mk_ppm'][:, SRTSO4], '-', color='red',
                   label='PPM', linewidth=1.5)

        ax.set_xscale('log')
        if idx == 0:
            ax.set_ylabel('Mk[SO4] [kg/cell]')
        ax.set_title(CASE_LABELS[case_key], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Bottom: relative error wrt Fortran
        ax = axes[1, idx]
        try:
            Mk_f = load_csv(f"{data_dir}/level12_case{case_key}_Mk.csv")
            denom = np.maximum(np.abs(Mk_f[:, SRTSO4]), 1e-30)
            err_tfl = np.abs(d['Mk_tfl'][:, SRTSO4] - Mk_f[:, SRTSO4]) / denom
            err_ppm = np.abs(d['Mk_ppm'][:, SRTSO4] - Mk_f[:, SRTSO4]) / denom
            ax.semilogy(Dp, err_tfl, '--', color='blue', label='TFL err', linewidth=1)
            ax.semilogy(Dp, err_ppm, '-', color='red', label='PPM err', linewidth=1)
            ax.axhline(y=1e-2, color='gray', linestyle=':', alpha=0.5)
        except FileNotFoundError:
            # Just show PPM vs TFL error
            denom = np.maximum(np.abs(d['Mk_tfl'][:, SRTSO4]), 1e-30)
            err = np.abs(d['Mk_ppm'][:, SRTSO4] - d['Mk_tfl'][:, SRTSO4]) / denom
            ax.semilogy(Dp, err, '-', color='purple', label='PPM vs TFL')

        ax.set_xscale('log')
        if idx == 0:
            ax.set_ylabel('Relative Error')
        ax.set_xlabel('Dp [um]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'ppm_vs_tfl_mass.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_conservation(output_dir, xk, all_data):
    """Figure 3: Conservation diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPM vs TFL: Conservation & Performance', fontsize=14)

    # Panel 1: Number conservation
    ax = axes[0, 0]
    n_errs_tfl = []
    n_errs_ppm = []
    for ck in CASE_KEYS:
        d = all_data[ck]
        N0 = np.sum(d['Nk_init'])
        n_errs_tfl.append(abs(np.sum(d['Nk_tfl']) - N0) / N0)
        n_errs_ppm.append(abs(np.sum(d['Nk_ppm']) - N0) / N0)
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w/2, n_errs_tfl, w, label='TFL', color='blue', alpha=0.7)
    ax.bar(x + w/2, n_errs_ppm, w, label='PPM', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}" for k in CASE_KEYS])
    ax.set_ylabel('Number Conservation Error')
    ax.set_title('Number Conservation')
    if any(v > 0 for v in n_errs_tfl + n_errs_ppm):
        ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Mass conservation
    ax = axes[0, 1]
    m_errs_tfl = []
    m_errs_ppm = []
    for ck in CASE_KEYS:
        d = all_data[ck]
        M0 = np.sum(d['Mk_init'][:, SRTSO4])
        mc = d['mcond']
        m_errs_tfl.append(abs(np.sum(d['Mk_tfl'][:, SRTSO4]) - M0 - mc) / max(mc, 1e-30))
        m_errs_ppm.append(abs(np.sum(d['Mk_ppm'][:, SRTSO4]) - M0 - mc) / max(mc, 1e-30))
    ax.bar(x - w/2, m_errs_tfl, w, label='TFL', color='blue', alpha=0.7)
    ax.bar(x + w/2, m_errs_ppm, w, label='PPM', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}" for k in CASE_KEYS])
    ax.set_ylabel('Mass Conservation Error')
    ax.set_title('SO4 Mass Conservation')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: CFL substeps
    ax = axes[1, 0]
    n_subs = [all_data[ck]['n_sub'] for ck in CASE_KEYS]
    ax.bar(x, n_subs, color=[CASE_COLORS[k] for k in CASE_KEYS])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n{CASES[k]['name']}" for k in CASE_KEYS], fontsize=8)
    ax.set_ylabel('CFL Substeps')
    ax.set_title('PPM CFL Substeps per Case')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: FWHM comparison
    ax = axes[1, 1]
    fwhm_tfl = [_fwhm_bins(all_data[ck]['Nk_tfl']) for ck in CASE_KEYS]
    fwhm_ppm = [_fwhm_bins(all_data[ck]['Nk_ppm']) for ck in CASE_KEYS]
    ax.bar(x - w/2, fwhm_tfl, w, label='TFL', color='blue', alpha=0.7)
    ax.bar(x + w/2, fwhm_ppm, w, label='PPM', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}" for k in CASE_KEYS])
    ax.set_ylabel('FWHM [bins]')
    ax.set_title('Distribution Width')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'ppm_conservation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_numerical_diffusion(output_dir, xk, all_data):
    """Figure 4: Numerical diffusion comparison (Case C)."""
    Dp = compute_bin_diameters(xk)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Numerical Diffusion: PPM vs TFL (Case C — nucleation mode)', fontsize=13)

    d = all_data['C']

    # Normalize to peak=1 for shape comparison
    Nk_tfl_norm = d['Nk_tfl'] / np.max(d['Nk_tfl'])
    Nk_ppm_norm = d['Nk_ppm'] / np.max(d['Nk_ppm'])
    Nk_init_norm = d['Nk_init'] / np.max(d['Nk_init'])

    ax.plot(Dp, Nk_init_norm, ':', color='gray', label='Initial', linewidth=1)
    ax.plot(Dp, Nk_tfl_norm, '--', color='blue', label='TFL (1st order)',
            linewidth=2)
    ax.plot(Dp, Nk_ppm_norm, '-', color='red', label='PPM (3rd order)',
            linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel('Dp [um]')
    ax.set_ylabel('Normalized Nk')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # FWHM annotations
    fwhm_tfl = _fwhm_bins(d['Nk_tfl'])
    fwhm_ppm = _fwhm_bins(d['Nk_ppm'])
    ax.text(0.95, 0.95, f'FWHM: TFL={fwhm_tfl} bins, PPM={fwhm_ppm} bins',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(output_dir, 'ppm_numerical_diffusion.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PPM vs TFL comparison plots"
    )
    parser.add_argument(
        '--data-dir',
        default=os.path.join(os.path.dirname(__file__), '..', 'fortran', 'output'),
        help='Path to Fortran benchmark output directory'
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.join(os.path.dirname(__file__), '..', 'results'),
        help='Path to write plot images'
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Generating PPM vs TFL comparison plots...")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")

    # Load xk
    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    # Run both methods for all cases
    print("  Running TFL and PPM for all cases...")
    all_data = {}
    for case_key in CASE_KEYS:
        all_data[case_key] = _run_both_methods(case_key, xk)
        print(f"    Case {case_key} done")

    # Generate plots
    plot_distributions(data_dir, output_dir, xk, all_data)
    plot_mass(data_dir, output_dir, xk, all_data)
    plot_conservation(output_dir, xk, all_data)
    plot_numerical_diffusion(output_dir, xk, all_data)

    print("\nAll PPM comparison plots generated!")


if __name__ == '__main__':
    main()
