"""Banana plot: Fortran vs JAX nucleation+coagulation+condensation comparison.

Loads hourly Fortran output for S01 "full" mode and runs JAX full mode,
then plots side-by-side banana plots (time vs diameter heatmaps of dN/dlogDp).

Usage::

    python -m benchmarks.python.plot_nucleation_banana
    python -m benchmarks.python.plot_nucleation_banana --scenario 1
    python -m benchmarks.python.plot_nucleation_banana --scenario 1 --method tfl_jit
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- Fortran float fixer (handles missing 'E' in exponents) ---
import re
_FORTRAN_RE = re.compile(r'(\d)([+-])(\d)')

def _fix_fortran_float(s):
    return _FORTRAN_RE.sub(r'\1E\2\3', s.strip())


# --- Load Fortran hourly output ---
FORTRAN_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                           'tomas_fortran', 'output', '24h')
XK0_STANDARD = 1.5625e-23  # 1e-21 * 2^(-6), standard TOMAS grid start


def load_fortran_hourly(scenario_id, mode='full', fortran_dir=None):
    """Load 24 hourly Fortran snapshots for a given scenario and mode.

    Returns dict with Nk(24,36), Mk(24,36,44), Gc(24,43), xk(37).
    """
    if fortran_dir is None:
        fortran_dir = FORTRAN_DIR

    prefix = f's{scenario_id:02d}_{mode}'
    nhours = 24

    # Standard TOMAS grid (36 bins)
    nbins = 36
    icomp = 44
    xk = np.array([XK0_STANDARD * 2.0**k for k in range(nbins + 1)])

    Nk_all = np.zeros((nhours, nbins))
    Mk_all = np.zeros((nhours, nbins, icomp))
    Gc_all = np.zeros((nhours, icomp - 1))

    for ihour in range(1, nhours + 1):
        nk_path = os.path.join(fortran_dir, f'{prefix}_hour{ihour:02d}_Nk.csv')
        mk_path = os.path.join(fortran_dir, f'{prefix}_hour{ihour:02d}_Mk.csv')
        gc_path = os.path.join(fortran_dir, f'{prefix}_hour{ihour:02d}_Gc.csv')

        if not os.path.exists(nk_path):
            print(f"  Missing: {nk_path}")
            return None

        Nk_all[ihour-1] = np.array([float(_fix_fortran_float(l))
                                     for l in open(nk_path).readlines()])

        Mk_rows = []
        for line in open(mk_path).readlines():
            row = [float(_fix_fortran_float(v)) for v in line.split(',')]
            Mk_rows.append(row)
        Mk_all[ihour-1] = np.array(Mk_rows)

        Gc_all[ihour-1] = np.array([float(_fix_fortran_float(l))
                                     for l in open(gc_path).readlines()])

    return {'Nk': Nk_all, 'Mk': Mk_all, 'Gc': Gc_all, 'xk': xk}


# --- JAX run ---
def run_jax_scenario(scenario_id, method='ppm_jit'):
    """Run JAX full mode for a scenario, return hourly snapshots."""
    from benchmarks.python.run_24h_scenarios import run_scenario
    from benchmarks.python.scenarios import get_scenarios

    scenarios = get_scenarios(n=50, seed=42)
    scenario = scenarios[scenario_id - 1]

    print(f"  Running JAX S{scenario_id:02d} full mode ({method})...")
    result = run_scenario(scenario, mode='full', method=method, verbose=True)
    return result


# --- Banana plot ---
def compute_dNdlogDp(Nk, xk, boxvol=1e6, density=1770.0):
    """Compute dN/dlogDp [#/cm³] from Nk [#/cell] and xk [kg]."""
    pi = np.pi
    nbins = Nk.shape[-1]
    Dp_lo = 1e6 * ((6.0 * xk[:nbins]) / (density * pi))**(1.0/3.0)   # μm
    Dp_hi = 1e6 * ((6.0 * xk[1:nbins+1]) / (density * pi))**(1.0/3.0)
    Dp_mid = np.sqrt(Dp_lo * Dp_hi)  # geometric mean
    dlogDp = np.log10(Dp_hi) - np.log10(Dp_lo)
    # Convert #/cell to #/cm³
    Nk_cm3 = Nk / boxvol
    dNdlogDp = Nk_cm3 / dlogDp
    return Dp_mid, dNdlogDp


def plot_banana(fortran_data, jax_data, jax_xk, scenario_id, method,
                output_dir=None):
    """Plot side-by-side banana plots (Fortran vs JAX)."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'results',
                                  'nucleation_benchmark')
    os.makedirs(output_dir, exist_ok=True)

    hours = np.arange(1, 25)

    # Compute dN/dlogDp for all hours
    f_xk = fortran_data['xk']
    f_Dp, _ = compute_dNdlogDp(fortran_data['Nk'][0], f_xk)
    j_Dp, _ = compute_dNdlogDp(jax_data['Nk'][0], jax_xk)

    # Build 2D arrays: (nhours, nbins)
    f_dN = np.zeros((24, len(f_Dp)))
    j_dN = np.zeros((24, len(j_Dp)))
    for ih in range(24):
        _, f_dN[ih] = compute_dNdlogDp(fortran_data['Nk'][ih], f_xk)
        _, j_dN[ih] = compute_dNdlogDp(jax_data['Nk'][ih], jax_xk)

    # Clamp tiny values for log color scale
    f_dN = np.maximum(f_dN, 1.0)
    j_dN = np.maximum(j_dN, 1.0)

    # Shared color limits
    vmin = 1.0
    vmax = max(np.max(f_dN), np.max(j_dN))

    # Dark theme
    BG = '#1a1a2e'
    TEXT = '#e0e0e0'

    fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(17, 6), facecolor=BG,
                                         gridspec_kw={'width_ratios': [1, 1, 0.05]})
    fig.suptitle(f'Nucleation Banana Plot — S{scenario_id:02d} Full Mode',
                 fontsize=14, color=TEXT, fontweight='bold')

    for ax, dN, Dp, title, show_ylabel in [
            (ax1, f_dN, f_Dp, 'Fortran (36 bins)', True),
            (ax2, j_dN, j_Dp, f'JAX {method} ({len(j_Dp)} bins)', False)]:
        ax.set_facecolor(BG)
        im = ax.pcolormesh(hours, Dp * 1e3, dN.T,  # Dp in nm
                           norm=LogNorm(vmin=vmin, vmax=vmax),
                           cmap='inferno', shading='nearest')
        ax.set_yscale('log')
        if show_ylabel:
            ax.set_ylabel('Dp [nm]', color=TEXT, fontsize=11)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Hour', color=TEXT, fontsize=11)
        ax.set_title(title, color=TEXT, fontsize=12)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_color(TEXT)

    cbar = fig.colorbar(im, cax=cax, label='dN/dlogDp [#/cm³]')
    cbar.ax.yaxis.label.set_color(TEXT)
    cbar.ax.tick_params(colors=TEXT)

    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.92)
    fname = os.path.join(output_dir,
                         f'nucleation_banana_s{scenario_id:02d}_{method}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {fname}")

    # --- Also plot N_tot and M_dry timeseries comparison ---
    fig2, (ax_n, ax_m) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig2.suptitle(f'S{scenario_id:02d} Full Mode — N & M Timeseries',
                  fontsize=13, color=TEXT, fontweight='bold')

    # N_tot
    f_Ntot = np.sum(fortran_data['Nk'], axis=1) / 1e6  # per cm³
    j_Ntot = np.sum(jax_data['Nk'], axis=1) / 1e6
    ax_n.set_facecolor(BG)
    ax_n.plot(hours, f_Ntot, 'o-', color='#2CA02C', lw=1.5, ms=4,
              label='Fortran')
    ax_n.plot(hours, j_Ntot, 's--', color='#FF6B35', lw=1.5, ms=4,
              label=f'JAX {method}')
    ax_n.set_xlabel('Hour', color=TEXT)
    ax_n.set_ylabel('N_tot [#/cm³]', color=TEXT)
    ax_n.set_title('Total Number', color=TEXT)
    ax_n.legend(facecolor='#2a2a3e', edgecolor=TEXT, labelcolor=TEXT)
    ax_n.tick_params(colors=TEXT)
    for spine in ax_n.spines.values():
        spine.set_color(TEXT)

    # M_dry
    SRTH2O = 43
    f_Mdry = np.sum(fortran_data['Mk'][:, :, :SRTH2O], axis=(1, 2))
    j_Mdry = np.sum(jax_data['Mk'][:, :, :SRTH2O], axis=(1, 2))
    ax_m.set_facecolor(BG)
    ax_m.plot(hours, f_Mdry, 'o-', color='#2CA02C', lw=1.5, ms=4,
              label='Fortran')
    ax_m.plot(hours, j_Mdry, 's--', color='#FF6B35', lw=1.5, ms=4,
              label=f'JAX {method}')
    ax_m.set_xlabel('Hour', color=TEXT)
    ax_m.set_ylabel('M_dry [kg/cell]', color=TEXT)
    ax_m.set_title('Total Dry Mass', color=TEXT)
    ax_m.legend(facecolor='#2a2a3e', edgecolor=TEXT, labelcolor=TEXT)
    ax_m.tick_params(colors=TEXT)
    for spine in ax_m.spines.values():
        spine.set_color(TEXT)

    plt.tight_layout()
    fname2 = os.path.join(output_dir,
                          f'nucleation_timeseries_s{scenario_id:02d}_{method}.png')
    fig2.savefig(fname2, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig2)
    print(f"  Saved: {fname2}")

    # --- Print quantitative comparison ---
    print(f"\n  === S{scenario_id:02d} Full Mode Comparison ===")
    print(f"  {'Hour':>4s}  {'F_Ntot':>12s}  {'J_Ntot':>12s}  {'N_err%':>8s}"
          f"  {'F_Mdry':>12s}  {'J_Mdry':>12s}  {'M_err%':>8s}")
    for ih in range(24):
        n_err = 100.0 * abs(j_Ntot[ih] - f_Ntot[ih]) / max(f_Ntot[ih], 1e-30)
        m_err = 100.0 * abs(j_Mdry[ih] - f_Mdry[ih]) / max(f_Mdry[ih], 1e-30)
        print(f"  {ih+1:4d}  {f_Ntot[ih]:12.4e}  {j_Ntot[ih]:12.4e}  {n_err:7.2f}%"
              f"  {f_Mdry[ih]:12.4e}  {j_Mdry[ih]:12.4e}  {m_err:7.2f}%")

    # Final size distribution comparison (hour 24)
    fig3, (ax_sd_n, ax_sd_m) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig3.suptitle(f'S{scenario_id:02d} Full Mode — Final Size Distribution (Hour 24)',
                  fontsize=13, color=TEXT, fontweight='bold')

    # dN/dlogDp
    _, f_dN24 = compute_dNdlogDp(fortran_data['Nk'][-1], f_xk)
    _, j_dN24 = compute_dNdlogDp(jax_data['Nk'][-1], jax_xk)
    ax_sd_n.set_facecolor(BG)
    ax_sd_n.plot(f_Dp * 1e3, f_dN24, '-', color='#2CA02C', lw=2,
                 label='Fortran')
    ax_sd_n.plot(j_Dp * 1e3, j_dN24, '--', color='#FF6B35', lw=2,
                 label=f'JAX {method}')
    ax_sd_n.set_xscale('log')
    ax_sd_n.set_xlabel('Dp [nm]', color=TEXT)
    ax_sd_n.set_ylabel('dN/dlogDp [#/cm³]', color=TEXT)
    ax_sd_n.set_title('Number Distribution', color=TEXT)
    ax_sd_n.legend(facecolor='#2a2a3e', edgecolor=TEXT, labelcolor=TEXT)
    ax_sd_n.tick_params(colors=TEXT)
    for spine in ax_sd_n.spines.values():
        spine.set_color(TEXT)

    # dM/dlogDp
    pi = np.pi
    density = 1770.0
    def compute_dMdlogDp(Mk, xk, boxvol=1e6):
        nbins_loc = Mk.shape[0]
        Dp_lo = 1e6 * ((6.0 * xk[:nbins_loc]) / (density * pi))**(1.0/3.0)
        Dp_hi = 1e6 * ((6.0 * xk[1:nbins_loc+1]) / (density * pi))**(1.0/3.0)
        Dp_mid = np.sqrt(Dp_lo * Dp_hi)
        dlogDp = np.log10(Dp_hi) - np.log10(Dp_lo)
        M_dry = np.sum(Mk[:, :SRTH2O], axis=1) / boxvol  # kg/cm³
        return Dp_mid, M_dry / dlogDp

    f_Dp_m, f_dM24 = compute_dMdlogDp(fortran_data['Mk'][-1], f_xk)
    j_Dp_m, j_dM24 = compute_dMdlogDp(jax_data['Mk'][-1], jax_xk)
    ax_sd_m.set_facecolor(BG)
    ax_sd_m.plot(f_Dp_m * 1e3, f_dM24, '-', color='#2CA02C', lw=2,
                 label='Fortran')
    ax_sd_m.plot(j_Dp_m * 1e3, j_dM24, '--', color='#FF6B35', lw=2,
                 label=f'JAX {method}')
    ax_sd_m.set_xscale('log')
    ax_sd_m.set_xlabel('Dp [nm]', color=TEXT)
    ax_sd_m.set_ylabel('dM/dlogDp [kg/cm³]', color=TEXT)
    ax_sd_m.set_title('Mass Distribution', color=TEXT)
    ax_sd_m.legend(facecolor='#2a2a3e', edgecolor=TEXT, labelcolor=TEXT)
    ax_sd_m.tick_params(colors=TEXT)
    for spine in ax_sd_m.spines.values():
        spine.set_color(TEXT)

    plt.tight_layout()
    fname3 = os.path.join(output_dir,
                          f'nucleation_sizedist_s{scenario_id:02d}_{method}.png')
    fig3.savefig(fname3, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig3)
    print(f"  Saved: {fname3}")


def main():
    parser = argparse.ArgumentParser(
        description='Banana plot: Fortran vs JAX nucleation comparison')
    parser.add_argument('--scenario', type=int, default=1,
                        help='Scenario ID (default: 1)')
    parser.add_argument('--method', type=str, default='ppm_jit',
                        choices=['ppm_jit', 'tfl_jit'],
                        help='JAX condensation method (default: ppm_jit)')
    args = parser.parse_args()

    sid = args.scenario
    method = args.method

    print(f"=== Nucleation Banana Plot: S{sid:02d}, method={method} ===\n")

    # Load Fortran
    print("--- Loading Fortran full-mode hourly output ---")
    fortran = load_fortran_hourly(sid, mode='full')
    if fortran is None:
        print("ERROR: Fortran output not found. Run Fortran benchmark_24h first.")
        sys.exit(1)
    print(f"  Fortran loaded: {fortran['Nk'].shape}")

    # Run JAX
    print("\n--- Running JAX full mode ---")
    jax_result = run_jax_scenario(sid, method=method)
    jax_xk = np.array(jax_result['scenario_params'].get('xk',
                       np.array([XK0_STANDARD * 2.0**k for k in range(37)])))

    # Need xk from the JAX side — get it from config
    from tomas_jax.core.config import make_grid
    jax_xk = np.array(make_grid())

    print(f"  JAX loaded: Nk={jax_result['Nk'].shape}")

    # Plot
    print("\n--- Generating plots ---")
    plot_banana(fortran, jax_result, jax_xk, sid, method)

    print("\nDone!")


if __name__ == '__main__':
    main()
