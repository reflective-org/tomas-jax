"""Visualization of Fortran vs JAX condensation benchmarks.

Generates comparison plots for Levels 11-13 across all 5 test cases.

Usage:
    python -m benchmarks.python.plot_condensation [--data-dir PATH] [--output-dir PATH]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# JAX setup
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .utils import load_csv
from .test_cases import CASES, CASE_KEYS, init_lognormal

from tomas_jax.physics.gas_properties import (
    calc_gas_diffusivity, calc_mean_molecular_speed, calc_mean_free_path,
)
from tomas_jax.physics.condensation_sink import calc_condensation_sink
from tomas_jax.core.config import (
    SRTSO4, SRTNH4, SRTH2O, ICOMP_NODIAG, N_GAS_SPECIES,
    NBINS_LEGACY as NBINS,
)
from tomas_jax.physics.ezcond import ezcond
from tomas_jax.physics.ezcond_ppm import ezcond_ppm
from tomas_jax.physics.nh3_equilibrium import eznh3eqm
from tomas_jax.physics.water_equilibrium import calc_equilibrium_water
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.solvers.diffrax import diffrax_step
from tomas_jax.solvers.condensation import condensation_step

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


def plot_level11(data_dir, output_dir, xk):
    """Level 11: Gas properties + condensation sink comparison."""
    Dp = compute_bin_diameters(xk)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Level 11: Condensation Sink — Fortran vs JAX', fontsize=14)

    # Top row: sinkfrac per case
    for idx, case_key in enumerate(CASE_KEYS):
        case = CASES[case_key]
        ax = axes[0, idx] if idx < 3 else axes[1, idx - 3]

        # Fortran
        condsink = load_csv(f"{data_dir}/level11_case{case_key}_condsink.csv").flatten()
        CS_f = condsink[0]
        sf_f = condsink[1:]

        # JAX
        Nk, Mk = init_lognormal(case_key, xk)
        CS_j, sf_j = calc_condensation_sink(
            jnp.array(Nk), jnp.array(Mk),
            case['temp'], case['pres'], 1.0e6,
            alpha=1.0
        )
        sf_j = np.array(sf_j)

        ax.semilogy(Dp, sf_f, 'o-', color=CASE_COLORS[case_key],
                     label=f'Fortran (CS={CS_f:.3e})', markersize=3, alpha=0.7)
        ax.semilogy(Dp, sf_j, 's--', color=CASE_COLORS[case_key],
                     label=f'JAX (CS={float(CS_j):.3e})', markersize=3, alpha=0.7)
        ax.set_xscale('log')
        ax.set_xlabel('Dp [μm]')
        ax.set_ylabel('Sink fraction')
        ax.set_title(CASE_LABELS[case_key])
        ax.legend(fontsize=8)
        ax.set_ylim(1e-10, 1)
        ax.grid(True, alpha=0.3)

    # Fill remaining subplot with relative error summary
    ax = axes[1, 2]
    cs_errors = []
    for case_key in CASE_KEYS:
        condsink = load_csv(f"{data_dir}/level11_case{case_key}_condsink.csv").flatten()
        CS_f = condsink[0]
        Nk, Mk = init_lognormal(case_key, xk)
        CS_j, _ = calc_condensation_sink(
            jnp.array(Nk), jnp.array(Mk),
            CASES[case_key]['temp'], CASES[case_key]['pres'], 1.0e6)
        cs_errors.append(abs(float(CS_j) - CS_f) / CS_f)

    ax.barh(range(5), cs_errors, color=[CASE_COLORS[k] for k in CASE_KEYS])
    ax.set_yticks(range(5))
    ax.set_yticklabels([CASE_LABELS[k] for k in CASE_KEYS], fontsize=8)
    ax.set_xlabel('CS Relative Error')
    ax.set_title('CS: Fortran vs JAX')
    ax.axvline(x=0.2, color='r', linestyle='--', alpha=0.5, label='Tolerance (20%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'level11_condsink.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_level12(data_dir, output_dir, xk):
    """Level 12: Isolated condensation comparison."""
    Dp = compute_bin_diameters(xk)

    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle('Level 12: Isolated Condensation — Fortran vs JAX', fontsize=14)

    for idx, case_key in enumerate(CASE_KEYS):
        case = CASES[case_key]

        # Fortran reference
        Nk_f = load_csv(f"{data_dir}/level12_case{case_key}_Nk.csv").flatten()
        Mk_f = load_csv(f"{data_dir}/level12_case{case_key}_Mk.csv")

        # JAX computation
        Nk, Mk = init_lognormal(case_key, xk)
        Gc = np.zeros(N_GAS_SPECIES)
        Gc[SRTSO4] = case['Gc_so4']

        CS_j, _ = calc_condensation_sink(
            jnp.array(Nk), jnp.array(Mk),
            case['temp'], case['pres'], 1.0e6)
        mcond = Gc[SRTSO4] * (1.0 - np.exp(-float(CS_j) * 60.0))

        Nk_j, Mk_j = ezcond(Nk, Mk, mcond, SRTSO4, xk,
                              case['temp'], case['pres'], 1.0e6, 1.0)

        Gc_jax = jnp.array(Gc)
        Mk_jax = jnp.array(Mk_j)
        Gc_jax, Mk_jax = eznh3eqm(Gc_jax, Mk_jax)
        Mk_jax = calc_equilibrium_water(Mk_jax, case['rh'])
        Nk_jax, Mk_jax = mnfix_jax(jnp.array(Nk_j), Mk_jax, jnp.array(xk), ICOMP_NODIAG)
        Nk_j = np.array(Nk_jax)
        Mk_j = np.array(Mk_jax)

        # Also run PPM for comparison
        try:
            Nk_ppm, Mk_ppm = ezcond_ppm(Nk, Mk, mcond, SRTSO4, xk,
                                          case['temp'], case['pres'], 1.0e6, 1.0)
            Gc_ppm = jnp.array(Gc)
            Mk_ppm_jax = jnp.array(Mk_ppm)
            Gc_ppm, Mk_ppm_jax = eznh3eqm(Gc_ppm, Mk_ppm_jax)
            Mk_ppm_jax = calc_equilibrium_water(Mk_ppm_jax, case['rh'])
            Nk_ppm_jax, Mk_ppm_jax = mnfix_jax(jnp.array(Nk_ppm), Mk_ppm_jax,
                                                  jnp.array(xk), ICOMP_NODIAG)
            Nk_ppm = np.array(Nk_ppm_jax)
            Mk_ppm = np.array(Mk_ppm_jax)
            has_ppm = True
        except Exception:
            has_ppm = False

        # Row 0: Number distribution
        ax = axes[0, idx]
        ax.semilogy(Dp, Nk_f, 'o-', color='blue', label='Fortran', markersize=3)
        ax.semilogy(Dp, Nk_j, 's--', color='red', label='TFL', markersize=3)
        if has_ppm:
            ax.semilogy(Dp, Nk_ppm, '^-', color='green', label='PPM',
                       markersize=3, alpha=0.8)
        ax.set_xscale('log')
        ax.set_title(CASE_LABELS[case_key], fontsize=9)
        ax.set_ylabel('Nk [#/cell]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 1: SO4 mass distribution
        ax = axes[1, idx]
        ax.semilogy(Dp, Mk_f[:, SRTSO4], 'o-', color='blue', markersize=3)
        ax.semilogy(Dp, Mk_j[:, SRTSO4], 's--', color='red', markersize=3)
        if has_ppm:
            ax.semilogy(Dp, Mk_ppm[:, SRTSO4], '^-', color='green',
                       markersize=3, alpha=0.8)
        ax.set_xscale('log')
        ax.set_ylabel('Mk[SO4] [kg/cell]')
        ax.grid(True, alpha=0.3)

        # Row 2: Relative error in Mk (SO4)
        ax = axes[2, idx]
        denom = np.maximum(np.abs(Mk_f[:, SRTSO4]), 1e-30)
        rel_err_tfl = np.abs(Mk_j[:, SRTSO4] - Mk_f[:, SRTSO4]) / denom
        ax.semilogy(Dp, rel_err_tfl, 'x-', color='red', markersize=4,
                   label='TFL err')
        if has_ppm:
            rel_err_ppm = np.abs(Mk_ppm[:, SRTSO4] - Mk_f[:, SRTSO4]) / denom
            ax.semilogy(Dp, rel_err_ppm, '^-', color='green', markersize=4,
                       label='PPM err')
        ax.axhline(y=1e-2, color='gray', linestyle='--', alpha=0.5, label='1% tol')
        ax.set_xscale('log')
        ax.set_ylabel('Rel Error (Mk SO4)')
        ax.set_xlabel('Dp [um]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'level12_condensation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_level13(data_dir, output_dir, xk):
    """Level 13: Combined coag+cond comparison."""
    Dp = compute_bin_diameters(xk)

    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle('Level 13: Combined Coagulation + Condensation — Fortran vs JAX',
                 fontsize=14)

    for idx, case_key in enumerate(CASE_KEYS):
        case = CASES[case_key]

        # Fortran reference
        Nk_f = load_csv(f"{data_dir}/level13_case{case_key}_Nk.csv").flatten()
        Mk_f = load_csv(f"{data_dir}/level13_case{case_key}_Mk.csv")
        Gc_f = load_csv(f"{data_dir}/level13_case{case_key}_Gc.csv").flatten()

        # JAX computation
        Nk, Mk = init_lognormal(case_key, xk)
        Gc = np.zeros(N_GAS_SPECIES)
        Gc[SRTSO4] = case['Gc_so4']

        Nk_jax = jnp.array(Nk)
        Mk_jax = jnp.array(Mk)
        xk_jax = jnp.array(xk)
        Gc_jax = jnp.array(Gc)

        try:
            Nk_coag, Mk_coag = diffrax_step(
                Nk_jax, Mk_jax, xk_jax,
                case['temp'], case['pres'], 1.0e6, 60.0,
                icomp_nodiag=ICOMP_NODIAG)
            Nk_final, Mk_final, Gc_final = condensation_step(
                Nk_coag, Mk_coag, Gc_jax, xk_jax,
                case['temp'], case['pres'], 1.0e6,
                case['rh'], 1.0, 60.0)
            Nk_j = np.array(Nk_final)
            Mk_j = np.array(Mk_final)
        except Exception as e:
            print(f"  Case {case_key} failed: {e}")
            continue

        # Initial state for reference
        Nk_init, _ = init_lognormal(case_key, xk)

        # Row 0: Number distribution (initial, Fortran, JAX)
        ax = axes[0, idx]
        ax.semilogy(Dp, Nk_init, '-', color='gray', label='Initial', alpha=0.5)
        ax.semilogy(Dp, Nk_f, 'o-', color='blue', label='Fortran', markersize=3)
        ax.semilogy(Dp, Nk_j, 's--', color='red', label='JAX', markersize=3)
        ax.set_xscale('log')
        ax.set_title(CASE_LABELS[case_key], fontsize=9)
        ax.set_ylabel('Nk [#/cell]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 1: SO4 mass distribution
        ax = axes[1, idx]
        Mk_init_plot = init_lognormal(case_key, xk)[1]
        ax.semilogy(Dp, Mk_init_plot[:, SRTSO4], '-', color='gray', alpha=0.5)
        ax.semilogy(Dp, Mk_f[:, SRTSO4], 'o-', color='blue', markersize=3)
        ax.semilogy(Dp, Mk_j[:, SRTSO4], 's--', color='red', markersize=3)
        ax.set_xscale('log')
        ax.set_ylabel('Mk[SO4] [kg/cell]')
        ax.grid(True, alpha=0.3)

        # Row 2: Relative error
        ax = axes[2, idx]
        denom_nk = np.maximum(np.abs(Nk_f), 1e-2)
        denom_mk = np.maximum(np.abs(Mk_f[:, SRTSO4]), 1e-30)
        rel_nk = np.abs(Nk_j - Nk_f) / denom_nk
        rel_mk = np.abs(Mk_j[:, SRTSO4] - Mk_f[:, SRTSO4]) / denom_mk
        ax.semilogy(Dp, rel_nk, 'x-', color='orange', markersize=4, label='Nk err')
        ax.semilogy(Dp, rel_mk, 'x-', color='green', markersize=4, label='Mk err')
        ax.axhline(y=5e-2, color='r', linestyle='--', alpha=0.5, label='5% tol')
        ax.set_xscale('log')
        ax.set_ylabel('Relative Error')
        ax.set_xlabel('Dp [μm]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'level13_coagcond.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_summary(data_dir, output_dir, xk):
    """Summary plot: mass conservation and overall error across cases."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Condensation Benchmark Summary', fontsize=14)

    # Panel 1: CS relative error by case
    ax = axes[0]
    cs_errors = {}
    for case_key in CASE_KEYS:
        case = CASES[case_key]
        condsink = load_csv(f"{data_dir}/level11_case{case_key}_condsink.csv").flatten()
        CS_f = condsink[0]
        Nk, Mk = init_lognormal(case_key, xk)
        CS_j, _ = calc_condensation_sink(
            jnp.array(Nk), jnp.array(Mk),
            case['temp'], case['pres'], 1.0e6)
        cs_errors[case_key] = abs(float(CS_j) - CS_f) / CS_f
    bars = ax.bar(range(5), [cs_errors[k] for k in CASE_KEYS],
                  color=[CASE_COLORS[k] for k in CASE_KEYS])
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{k}\n{CASES[k]['name']}" for k in CASE_KEYS], fontsize=8)
    ax.set_ylabel('CS Relative Error')
    ax.set_title('Level 11: Condensation Sink')
    ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Tol=20%')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Level 12 Mk max rel error by case
    ax = axes[1]
    mk_errors = {}
    for case_key in CASE_KEYS:
        case = CASES[case_key]
        Mk_f = load_csv(f"{data_dir}/level12_case{case_key}_Mk.csv")
        Nk, Mk = init_lognormal(case_key, xk)
        Gc = np.zeros(N_GAS_SPECIES)
        Gc[SRTSO4] = case['Gc_so4']
        CS_j, _ = calc_condensation_sink(
            jnp.array(Nk), jnp.array(Mk),
            case['temp'], case['pres'], 1.0e6)
        mcond = Gc[SRTSO4] * (1.0 - np.exp(-float(CS_j) * 60.0))
        Nk_j, Mk_j = ezcond(Nk, Mk, mcond, SRTSO4, xk,
                              case['temp'], case['pres'], 1.0e6, 1.0)
        Gc_jax = jnp.array(Gc)
        Mk_jax = jnp.array(Mk_j)
        Gc_jax, Mk_jax = eznh3eqm(Gc_jax, Mk_jax)
        Mk_jax = calc_equilibrium_water(Mk_jax, case['rh'])
        _, Mk_jax = mnfix_jax(jnp.array(Nk_j), Mk_jax, jnp.array(xk), ICOMP_NODIAG)
        Mk_j = np.array(Mk_jax)
        denom = np.maximum(np.abs(Mk_f), 1e-30)
        rel = np.abs(Mk_j - Mk_f) / denom
        mk_errors[case_key] = np.max(rel)
    bars = ax.bar(range(5), [mk_errors[k] for k in CASE_KEYS],
                  color=[CASE_COLORS[k] for k in CASE_KEYS])
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{k}\n{CASES[k]['name']}" for k in CASE_KEYS], fontsize=8)
    ax.set_ylabel('Mk Max Relative Error')
    ax.set_title('Level 12: Isolated Condensation')
    ax.axhline(y=1e-2, color='r', linestyle='--', alpha=0.5, label='Tol=1%')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Level 13 mass conservation
    ax = axes[2]
    mass_errs_f = []
    mass_errs_j = []
    for case_key in CASE_KEYS:
        case = CASES[case_key]
        Nk_init, Mk_init = init_lognormal(case_key, xk)
        M_init = np.sum(Mk_init[:, SRTSO4]) + case['Gc_so4']

        Mk_f = load_csv(f"{data_dir}/level13_case{case_key}_Mk.csv")
        Gc_f = load_csv(f"{data_dir}/level13_case{case_key}_Gc.csv").flatten()
        M_f = np.sum(Mk_f[:, SRTSO4]) + Gc_f[SRTSO4]
        mass_errs_f.append(abs(M_f - M_init) / M_init)

        # JAX
        Nk_jax = jnp.array(Nk_init)
        Mk_jax = jnp.array(Mk_init)
        xk_jax = jnp.array(xk)
        Gc_np = np.zeros(N_GAS_SPECIES)
        Gc_np[SRTSO4] = case['Gc_so4']
        try:
            Nk_c, Mk_c = diffrax_step(
                Nk_jax, Mk_jax, xk_jax,
                case['temp'], case['pres'], 1.0e6, 60.0,
                icomp_nodiag=ICOMP_NODIAG)
            Nk_out, Mk_out, Gc_out = condensation_step(
                Nk_c, Mk_c, jnp.array(Gc_np), xk_jax,
                case['temp'], case['pres'], 1.0e6, case['rh'], 1.0, 60.0)
            M_j = float(jnp.sum(Mk_out[:, SRTSO4])) + float(Gc_out[SRTSO4])
            mass_errs_j.append(abs(M_j - M_init) / M_init)
        except:
            mass_errs_j.append(0)

    x = np.arange(5)
    w = 0.35
    ax.bar(x - w / 2, mass_errs_f, w, label='Fortran',
           color=[CASE_COLORS[k] for k in CASE_KEYS], alpha=0.6)
    ax.bar(x + w / 2, mass_errs_j, w, label='JAX',
           color=[CASE_COLORS[k] for k in CASE_KEYS], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n{CASES[k]['name']}" for k in CASE_KEYS], fontsize=8)
    ax.set_ylabel('SO4 Mass Conservation Error')
    ax.set_title('Level 13: Combined Coag+Cond')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'condensation_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate condensation benchmark comparison plots"
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

    print("Generating condensation benchmark plots...")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")

    # Load xk (shared across all levels)
    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    plot_level11(data_dir, output_dir, xk)
    plot_level12(data_dir, output_dir, xk)
    plot_level13(data_dir, output_dir, xk)
    plot_summary(data_dir, output_dir, xk)

    print("\nAll plots generated!")


if __name__ == '__main__':
    main()
