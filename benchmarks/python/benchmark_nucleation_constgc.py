"""Constant-gas nucleation + coagulation + condensation benchmark.

Compares JAX (PPM, 40/80 bins) vs Fortran (TFL, 36 bins standard grid)
with constant H2SO4 = 1e7 molec/cm3 over 24 hours.

Usage::

    source .venv/bin/activate
    python -m benchmarks.python.benchmark_nucleation_constgc
    python -m benchmarks.python.benchmark_nucleation_constgc --fortran
    python -m benchmarks.python.benchmark_nucleation_constgc --dt 30
"""
import os
import sys
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from tomas_jax.core.config import (
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO,
    make_grid,
)
from tomas_jax.solvers.condensation import full_step_jax


# ---------------------------------------------------------------------------
# Physical constants (match Fortran constgc harness)
# ---------------------------------------------------------------------------
PI = 3.141592654
DENS_INIT = 1770.0      # kg/m3 (sulfate density)
NEPS = 1.0e-3
BOXVOL = 1.0e6           # cm3
XK0_STANDARD = 1.0e-21 * 2.0**(-6)  # standard TOMAS Mo

# Nucleation parameters (match Fortran 24h harness)
NUC_ORG_CONC = 1e7       # organic vapor [molec/cm3]
NUC_NH3_CONC = 1e9       # NH3 [molec/cm3]
NUC_FION = 3.0           # ion formation rate [pairs/cm3/s]
NUC_FN_SCALE = 1.0       # scaling factor

# Grid configs: (label, nbins, doubling_factor)
GRID_CONFIGS = [
    ("40 bins (×2)",      40, 2.0),
    ("80 bins (×√2)",     80, 2.0**0.5),
]

# Fortran output directory
FORTRAN_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                           'tomas_fortran', 'output', 'constgc')


# ---------------------------------------------------------------------------
# Scan function: constant-gas full mode (nucl + coag + cond with PPM)
# ---------------------------------------------------------------------------
def _run_full_scan_ppm(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                       nsteps, org_conc, nh3_conc, fion,
                       enable_organic=1.0):
    """Nucl + Coag + PPM condensation scan with Gc reset each step.

    Returns (Nk_f, Mk_f, Gc_f, hist, Nk_history) where:
        hist:       (nsteps, 3) — [N_tot, M_dry, Gc_SO4] per step
        Nk_history: (nsteps, nbins) — full Nk at each step (for banana plot)
    """
    @partial(jax.jit, static_argnums=(10,))
    def _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps,
              org_conc, nh3_conc, fion, enable_organic):
        Gc0 = Gc.copy()

        def step(carry, _):
            Nk_c, Mk_c, Gc_c = carry
            Nk_c, Mk_c, _ = full_step_jax(
                Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
                org_conc, nh3_conc, fion,
                enable_organic=enable_organic, enable_inorganic=1.0,
                fn_scale=1.0, use_tfl=0.0)
            scalar_diag = jnp.array([jnp.sum(Nk_c),
                                      jnp.sum(Mk_c[:, :SRTH2O]),
                                      Gc0[SRTSO4]])
            return (Nk_c, Mk_c, Gc0), (scalar_diag, Nk_c)

        (Nk_f, Mk_f, Gc_f), (hist, Nk_history) = jax.lax.scan(
            step, (Nk, Mk, Gc), None, length=nsteps)
        return Nk_f, Mk_f, Gc_f, hist, Nk_history

    return _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps,
                 org_conc, nh3_conc, fion, enable_organic)


# ---------------------------------------------------------------------------
# Initialization (matches Fortran constgc harness)
# ---------------------------------------------------------------------------
def init_lognormal(N_total, Dp_gmd_um, sigma_gsd, xk):
    """Initialize lognormal distribution on an arbitrary grid."""
    xk_np = np.array(xk)
    nbins = len(xk_np) - 1
    Nk = np.zeros(nbins)
    Mk = np.zeros((nbins, ICOMP))

    for k in range(nbins):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dk = np.sqrt(Dl * Dh)
        dN = ((N_total * BOXVOL)
              / (np.sqrt(2.0 * PI) * Dk * np.log(sigma_gsd))
              * np.exp(-(np.log(Dk / Dp_gmd_um) ** 2
                         / (2.0 * np.log(sigma_gsd) ** 2)))
              * (Dh - Dl))
        Nk[k] = dN
        Mk[k, SRTSO4] = dN * np.sqrt(xk_np[k] * xk_np[k + 1])

    for k in range(nbins):
        if Nk[k] < NEPS:
            Nk[k] = NEPS
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = NEPS * 1.4 * xk_np[k]

    return Nk, Mk


# ---------------------------------------------------------------------------
# Fortran results loader
# ---------------------------------------------------------------------------
def _fix_fortran_float(s):
    """Fix Fortran exponents missing 'E' (e.g., '0.1234+05' -> '0.1234E+05')."""
    import re
    return re.sub(r'(\d)([\+\-])(\d)', r'\1E\2\3', s.strip())


def load_fortran_results(dt_val, fortran_dir=None):
    """Load Fortran full-mode Nk/Mk/xk output + hourly Nk for a given dt."""
    if fortran_dir is None:
        fortran_dir = FORTRAN_DIR
    dt_int = int(dt_val)

    prefix = 'constgc_full'
    nk_path = os.path.join(fortran_dir, f'{prefix}_dt{dt_int:02d}_final_Nk.csv')
    mk_path = os.path.join(fortran_dir, f'{prefix}_dt{dt_int:02d}_final_Mk.csv')
    xk_path = os.path.join(fortran_dir, 'constgc_xk.csv')

    if not os.path.exists(nk_path):
        print(f"  [Fortran] Not found: {nk_path}")
        return None

    Nk = np.array([float(_fix_fortran_float(line))
                    for line in open(nk_path).readlines()])
    Mk = []
    for line in open(mk_path).readlines():
        row = [float(_fix_fortran_float(v)) for v in line.split(',')]
        Mk.append(row)
    Mk = np.array(Mk)

    if os.path.exists(xk_path):
        xk = np.array([float(_fix_fortran_float(line))
                        for line in open(xk_path).readlines()])
    else:
        xk = np.array([XK0_STANDARD * 2.0**k for k in range(37)])

    # Load hourly Nk snapshots (hours 1-24)
    hourly_Nk = []
    for h in range(1, 25):
        hpath = os.path.join(fortran_dir,
                             f'{prefix}_dt{dt_int:02d}_hour{h:02d}_Nk.csv')
        if os.path.exists(hpath):
            nk_h = np.array([float(_fix_fortran_float(line))
                             for line in open(hpath).readlines()])
            hourly_Nk.append(nk_h)
    if hourly_Nk:
        hourly_Nk = np.array(hourly_Nk)  # (24, nbins)
    else:
        hourly_Nk = None

    return {'Nk': Nk, 'Mk': Mk, 'xk': xk, 'hourly_Nk': hourly_Nk}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _bin_diameters(xk):
    xk_np = np.array(xk)
    Dp_lo = 1e9 * ((6.0 * xk_np[:-1]) / (DENS_INIT * np.pi)) ** (1.0 / 3.0)
    Dp_hi = 1e9 * ((6.0 * xk_np[1:]) / (DENS_INIT * np.pi)) ** (1.0 / 3.0)
    Dp_mid = np.sqrt(Dp_lo * Dp_hi)
    dlogDp = np.log10(Dp_hi / Dp_lo)
    return Dp_mid, dlogDp


def compute_dNdlogDp(Nk, xk):
    Dp_mid, dlogDp = _bin_diameters(xk)
    dNdlogDp = (np.array(Nk) / BOXVOL) / np.maximum(dlogDp, 1e-30)
    return Dp_mid, dNdlogDp


def compute_dMdlogDp(Mk, xk):
    Dp_mid, dlogDp = _bin_diameters(xk)
    Mk_np = np.array(Mk)
    M_dry = np.sum(Mk_np[:, :SRTH2O], axis=1)
    dMdlogDp = (M_dry * 1e9) / np.maximum(dlogDp, 1e-30)
    return Dp_mid, dMdlogDp


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(dt_val=60.0, n_total=1e4, gmd_um=0.02, gsd=1.6,
                  temp_val=298.0, pres_val=101325.0, rh_val=0.30,
                  h2so4_molec_cm3=1e7, include_fortran=False,
                  make_plots=True, enable_organic_nuc=True):
    """Run nucleation + coag + cond benchmark with constant H2SO4."""

    nsteps = int(86400.0 / dt_val)
    h2so4_kg = h2so4_molec_cm3 * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO

    print("=" * 60)
    print("Constant-Gas Full Benchmark (Nucl + Coag + PPM Cond)")
    print("=" * 60)
    print(f"  N = {n_total:.0e} #/cm3, GMD = {gmd_um} um, GSD = {gsd}")
    print(f"  T = {temp_val} K, P = {pres_val} Pa, RH = {rh_val}")
    print(f"  H2SO4 = {h2so4_molec_cm3:.0e} molec/cm3 (held constant)")
    org_flag = 1.0 if enable_organic_nuc else 0.0
    nuc_schemes = []
    if enable_organic_nuc:
        nuc_schemes.append(f"Riccobono (org={NUC_ORG_CONC:.0e})")
    nuc_schemes.append(f"Dunne (NH3={NUC_NH3_CONC:.0e}, fion={NUC_FION})")
    print(f"  Nucleation: {' + '.join(nuc_schemes)}")
    print(f"  dt = {dt_val}s, nsteps = {nsteps} (24h)")
    print()

    results = {}

    for label, nbins, factor in GRID_CONFIGS:
        print(f"--- {label} ---")
        xk0 = (np.pi / 6.0) * (1.7e-9)**3 * DENS_INIT  # 1.7nm start
        xk = make_grid(nbins=nbins, xk0=xk0, doubling_factor=factor)

        Nk, Mk = init_lognormal(n_total, gmd_um, gsd, xk)
        Gc = np.zeros(N_GAS_SPECIES)
        Gc[SRTSO4] = h2so4_kg

        Nk_jax = jnp.array(Nk)
        Mk_jax = jnp.array(Mk)
        Gc_jax = jnp.array(Gc)
        xk_jax = jnp.array(xk)

        temp = jnp.array(temp_val)
        pres = jnp.array(pres_val)
        boxvol = jnp.array(float(BOXVOL))
        rh = jnp.array(rh_val)
        alpha = jnp.array(1.0)
        dt = jnp.array(dt_val)

        M_aer_init = float(np.sum(Mk[:, :SRTH2O]))
        N_init = float(np.sum(Nk))

        entry = {
            'label': label, 'nbins': nbins, 'factor': factor,
            'xk': np.array(xk),
            'Nk_init': Nk.copy(), 'Mk_init': Mk.copy(),
            'Gc_init': Gc.copy(),
            'N_init': N_init,
            'M_aer_init': M_aer_init,
        }

        # Compile
        print(f"  PPM (full): compiling...", end="", flush=True)
        _ = _run_full_scan_ppm(
            Nk_jax, Mk_jax, Gc_jax, xk_jax,
            temp, pres, boxvol, rh, alpha, dt, nsteps,
            jnp.float64(NUC_ORG_CONC), jnp.float64(NUC_NH3_CONC),
            jnp.float64(NUC_FION), enable_organic=org_flag)
        jax.block_until_ready(_)

        # Run
        print(f" running...", end="", flush=True)
        t0 = time.perf_counter()
        Nk_f, Mk_f, Gc_f, hist, Nk_history = _run_full_scan_ppm(
            Nk_jax, Mk_jax, Gc_jax, xk_jax,
            temp, pres, boxvol, rh, alpha, dt, nsteps,
            jnp.float64(NUC_ORG_CONC), jnp.float64(NUC_NH3_CONC),
            jnp.float64(NUC_FION), enable_organic=org_flag)
        jax.block_until_ready(Nk_f)
        wall_time = time.perf_counter() - t0

        M_aer_f = float(jnp.sum(Mk_f[:, :SRTH2O]))
        N_f = float(jnp.sum(Nk_f))

        entry['Nk_final'] = np.array(Nk_f)
        entry['Mk_final'] = np.array(Mk_f)
        entry['Gc_final'] = np.array(Gc_f)
        entry['hist'] = np.array(hist)
        entry['Nk_history'] = np.array(Nk_history)  # (nsteps, nbins)
        entry['wall_time'] = wall_time
        entry['N_final'] = N_f
        entry['M_aer_final'] = M_aer_f

        print(f" {wall_time:.2f}s")
        print(f"    N: {N_init:.4e} → {N_f:.4e}  "
              f"(+{(N_f - N_init) / max(N_init, 1e-30) * 100:.1f}%)")
        print(f"    M_dry: {M_aer_init * 1e9:.4f} → {M_aer_f * 1e9:.4f} μg/m³  "
              f"(+{(M_aer_f - M_aer_init) / max(M_aer_init, 1e-30) * 100:.1f}%)")

        results[nbins] = entry

    # --- Load Fortran ---
    fortran_data = None
    if include_fortran:
        print(f"\n--- Loading Fortran full-mode results (36 bins, standard grid) ---")
        fortran_data = load_fortran_results(dt_val)
        if fortran_data is not None:
            f_Nk = fortran_data['Nk']
            f_Mk = fortran_data['Mk']
            f_M_dry = float(np.sum(f_Mk[:, :SRTH2O]))
            f_N = float(np.sum(f_Nk))
            print(f"  Fortran: N_tot = {f_N:.4e}, M_dry = {f_M_dry * 1e9:.4f} μg/m³")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Config':<22s} {'Time':>8s} {'N_final':>12s} {'M_dry [μg/m³]':>14s}")
    print("-" * 60)
    for nbins_key in sorted(results.keys()):
        r = results[nbins_key]
        print(f"{r['label']:<22s} {r['wall_time']:>7.2f}s {r['N_final']:>12.4e} "
              f"{r['M_aer_final'] * 1e9:>14.4f}")
    if fortran_data is not None:
        f_N = float(np.sum(fortran_data['Nk']))
        f_M = float(np.sum(fortran_data['Mk'][:, :SRTH2O]))
        print(f"{'Fortran (36 bins)':<22s} {'—':>8s} {f_N:>12.4e} {f_M * 1e9:>14.4f}")

    # --- Plots ---
    if make_plots:
        plot_benchmark(results, fortran_data, dt_val, n_total, gmd_um, gsd,
                       temp_val, pres_val, h2so4_molec_cm3)

    return results, fortran_data


def plot_benchmark(results, fortran_data, dt_val, n_total, gmd_um, gsd,
                   temp_val, pres_val, h2so4_molec_cm3):
    """Generate benchmark comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'nucleation_constgc')
    os.makedirs(outdir, exist_ok=True)

    sorted_keys = sorted(results.keys())

    # Style
    _TEXT = '#333333'
    _GRID = '#E0E0E0'
    _ANNOT = '#666666'
    _FORTRAN_C = '#2CA02C'
    _PALETTE = {40: '#1A1A2E', 80: '#C44E52', 160: '#4C72B0'}
    _LW = {40: 2.2, 80: 1.8, 160: 1.5}

    def _color(n): return _PALETTE.get(n, '#555555')
    def _lw(n): return _LW.get(n, 1.5)

    info_text = (
        f"Aerosol: N₀ = {n_total:.0e} #/cm³, GMD = {gmd_um} μm, GSD = {gsd}\n"
        f"Environment: T = {temp_val:.0f} K, P = {pres_val:.0f} Pa\n"
        f"Gas: H₂SO₄ = {h2so4_molec_cm3:.0e} molec/cm³ (constant)  |  "
        f"Nucleation: org = {NUC_ORG_CONC:.0e}, NH₃ = {NUC_NH3_CONC:.0e}\n"
        f"dt = {dt_val:.0f}s, 24h  |  Mode: Nucl + Coag + PPM Cond"
    )

    def _style_ax(ax, xlabel='', ylabel='', title=''):
        ax.set_xscale('log')
        ax.minorticks_on()
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(axis='both', which='major', length=4, width=0.6,
                       labelsize=9, colors=_TEXT)
        ax.tick_params(axis='both', which='minor', length=2, width=0.4,
                       colors=_TEXT)
        ax.yaxis.grid(True, which='major', color=_GRID, linewidth=0.6)
        ax.xaxis.grid(True, which='major', color=_GRID, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        if xlabel: ax.set_xlabel(xlabel, fontsize=10, color=_TEXT, labelpad=8)
        if ylabel: ax.set_ylabel(ylabel, fontsize=10, color=_TEXT, labelpad=8)
        if title: ax.set_title(title, fontsize=12, color=_TEXT, fontweight='bold',
                               loc='left', pad=10)

    # =====================================================================
    # Figure 1: dN/dlogDp — initial vs final (2 panels)
    # =====================================================================
    fig, (ax_init, ax_final) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle('Nucleation Benchmark — Number Size Distribution',
                 fontsize=13, color=_TEXT, fontweight='bold', x=0.02, ha='left')

    for nbins_key in sorted_keys:
        r = results[nbins_key]
        Dp, dN_init = compute_dNdlogDp(r['Nk_init'], r['xk'])
        ax_init.plot(Dp, dN_init, color=_color(nbins_key), lw=_lw(nbins_key),
                     label=r['label'])
        Dp, dN_final = compute_dNdlogDp(r['Nk_final'], r['xk'])
        ax_final.plot(Dp, dN_final, color=_color(nbins_key), lw=_lw(nbins_key),
                      label=r['label'])

    if fortran_data is not None:
        Dp_f, dN_f = compute_dNdlogDp(fortran_data['Nk'], fortran_data['xk'])
        ax_final.plot(Dp_f, dN_f, color=_FORTRAN_C, lw=2.0, ls=':',
                      label='Fortran (36 bins)')

    _style_ax(ax_init, 'Dp [nm]', 'dN/dlogDp [#/cm³]', 'Initial')
    _style_ax(ax_final, 'Dp [nm]', '', 'After 24h — PPM')
    ax_init.legend(fontsize=8.5, frameon=False)
    ax_final.legend(fontsize=8.5, frameon=False)

    fig.text(0.5, -0.02, info_text, ha='center', va='top', fontsize=7,
             color=_ANNOT, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7F7F7',
                       edgecolor='#E0E0E0', linewidth=0.5))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path1 = os.path.join(outdir, f'full_constgc_dt{int(dt_val)}_dN.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")
    plt.close(fig)

    # =====================================================================
    # Figure 2: dM/dlogDp — initial vs final
    # =====================================================================
    fig, (ax_init, ax_final) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle('Nucleation Benchmark — Mass Size Distribution',
                 fontsize=13, color=_TEXT, fontweight='bold', x=0.02, ha='left')

    for nbins_key in sorted_keys:
        r = results[nbins_key]
        Dp, dM_init = compute_dMdlogDp(r['Mk_init'], r['xk'])
        ax_init.plot(Dp, dM_init, color=_color(nbins_key), lw=_lw(nbins_key),
                     label=r['label'])
        Dp, dM_final = compute_dMdlogDp(r['Mk_final'], r['xk'])
        ax_final.plot(Dp, dM_final, color=_color(nbins_key), lw=_lw(nbins_key),
                      label=r['label'])

    if fortran_data is not None:
        Dp_f, dM_f = compute_dMdlogDp(fortran_data['Mk'], fortran_data['xk'])
        ax_final.plot(Dp_f, dM_f, color=_FORTRAN_C, lw=2.0, ls=':',
                      label='Fortran (36 bins)')

    _style_ax(ax_init, 'Dp [nm]', 'dM/dlogDp [μg/m³]', 'Initial')
    _style_ax(ax_final, 'Dp [nm]', '', 'After 24h — PPM')
    ax_init.legend(fontsize=8.5, frameon=False)
    ax_final.legend(fontsize=8.5, frameon=False)

    fig.text(0.5, -0.02, info_text, ha='center', va='top', fontsize=7,
             color=_ANNOT, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7F7F7',
                       edgecolor='#E0E0E0', linewidth=0.5))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path2 = os.path.join(outdir, f'full_constgc_dt{int(dt_val)}_dM.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path2}")
    plt.close(fig)

    # =====================================================================
    # Figure 3: N and M time series (from scan history)
    # =====================================================================
    fig, (ax_n, ax_m) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Nucleation Benchmark — Time Evolution',
                 fontsize=13, color=_TEXT, fontweight='bold', x=0.02, ha='left')

    for nbins_key in sorted_keys:
        r = results[nbins_key]
        hist = r['hist']  # (nsteps, 3): N_tot, M_dry, Gc_SO4
        hours = np.arange(1, len(hist) + 1) * dt_val / 3600.0
        ax_n.plot(hours, hist[:, 0] / BOXVOL, color=_color(nbins_key),
                  lw=_lw(nbins_key), label=r['label'])
        ax_m.plot(hours, hist[:, 1] * 1e9, color=_color(nbins_key),
                  lw=_lw(nbins_key), label=r['label'])

    for ax in (ax_n, ax_m):
        ax.set_xscale('linear')
        ax.minorticks_on()
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(axis='both', which='major', length=4, width=0.6,
                       labelsize=9, colors=_TEXT)
        ax.yaxis.grid(True, which='major', color=_GRID, linewidth=0.6)
        ax.xaxis.grid(True, which='major', color=_GRID, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel('Time [hours]', fontsize=10, color=_TEXT)

    ax_n.set_ylabel('N_total [#/cm³]', fontsize=10, color=_TEXT)
    ax_n.set_title('Total Number', fontsize=12, color=_TEXT, fontweight='bold',
                   loc='left')
    ax_m.set_ylabel('M_dry [μg/m³]', fontsize=10, color=_TEXT)
    ax_m.set_title('Total Dry Mass', fontsize=12, color=_TEXT, fontweight='bold',
                   loc='left')
    ax_n.legend(fontsize=8.5, frameon=False)
    ax_m.legend(fontsize=8.5, frameon=False)

    fig.text(0.5, -0.02, info_text, ha='center', va='top', fontsize=7,
             color=_ANNOT, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7F7F7',
                       edgecolor='#E0E0E0', linewidth=0.5))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path3 = os.path.join(outdir, f'full_constgc_dt{int(dt_val)}_timeseries.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path3}")
    plt.close(fig)

    # =====================================================================
    # Figure 4: Banana plot — 3 panels (JAX 40, JAX 80, Fortran 36)
    # =====================================================================
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec

    # Build banana data for each panel: (hours_array, Dp_mid, dNdlogDp_2D)
    panels = []
    for nbins_key in sorted_keys:
        r = results[nbins_key]
        Nk_hist = r['Nk_history']  # (nsteps, nbins)
        xk_np = r['xk']
        Dp_mid, dlogDp = _bin_diameters(xk_np)

        # Subsample to hourly (every steps_per_hour steps)
        steps_per_hour = int(3600.0 / dt_val)
        hourly_idx = np.arange(steps_per_hour - 1, Nk_hist.shape[0], steps_per_hour)
        Nk_hourly = Nk_hist[hourly_idx]  # (24, nbins)
        hours = (hourly_idx + 1) * dt_val / 3600.0

        dNdlogDp_2D = (Nk_hourly / BOXVOL) / np.maximum(dlogDp[None, :], 1e-30)
        panels.append((r['label'], hours, Dp_mid, dNdlogDp_2D))

    # Add Fortran panel
    if fortran_data is not None and fortran_data.get('hourly_Nk') is not None:
        f_xk = fortran_data['xk']
        f_Dp_mid, f_dlogDp = _bin_diameters(f_xk)
        f_hourly_Nk = fortran_data['hourly_Nk']  # (24, 36)
        f_hours = np.arange(1, 25, dtype=float)
        f_dNdlogDp_2D = (f_hourly_Nk / BOXVOL) / np.maximum(f_dlogDp[None, :], 1e-30)
        panels.append(('Fortran (36 bins)', f_hours, f_Dp_mid, f_dNdlogDp_2D))

    n_panels = len(panels)
    if n_panels > 0:
        # Find global color range
        all_vals = np.concatenate([p[3].ravel() for p in panels])
        all_vals = all_vals[all_vals > 0]
        vmin = max(np.percentile(all_vals, 1), 1.0)
        vmax = np.percentile(all_vals, 99.9)

        # Create figure with gridspec: n_panels data + 1 colorbar
        fig = plt.figure(figsize=(5.5 * n_panels + 0.8, 5.5))
        gs = gridspec.GridSpec(1, n_panels + 1,
                               width_ratios=[1] * n_panels + [0.04],
                               wspace=0.25)

        fig.suptitle('Nucleation Banana Plot — dN/dlogDp [#/cm³]',
                     fontsize=13, color=_TEXT, fontweight='bold',
                     x=0.02, ha='left', y=0.98)

        im = None
        for i, (label, hours, Dp_mid, data) in enumerate(panels):
            ax = fig.add_subplot(gs[0, i])
            im = ax.pcolormesh(hours, Dp_mid, data.T,
                               norm=LogNorm(vmin=vmin, vmax=vmax),
                               cmap='inferno', shading='nearest')
            ax.set_yscale('log')
            ax.set_xlabel('Time [hours]', fontsize=10, color=_TEXT)
            if i == 0:
                ax.set_ylabel('Dp [nm]', fontsize=10, color=_TEXT)
            ax.set_title(label, fontsize=11, color=_TEXT, fontweight='bold',
                         loc='left', pad=8)
            ax.tick_params(axis='both', which='major', labelsize=9, colors=_TEXT)
            for sp in ax.spines.values():
                sp.set_visible(False)

        # Colorbar
        cax = fig.add_subplot(gs[0, n_panels])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('dN/dlogDp [#/cm³]', fontsize=9, color=_TEXT)
        cb.ax.tick_params(labelsize=8, colors=_TEXT)

        fig.text(0.5, -0.02, info_text, ha='center', va='top', fontsize=7,
                 color=_ANNOT, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7F7F7',
                           edgecolor='#E0E0E0', linewidth=0.5))
        fig.tight_layout(rect=[0, 0.06, 1, 0.93])
        path4 = os.path.join(outdir, f'full_constgc_dt{int(dt_val)}_banana.png')
        fig.savefig(path4, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path4}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Constant-gas nucleation + coagulation + condensation benchmark')
    parser.add_argument('--dt', type=float, default=60.0,
                        help='Timestep [s] (default: 60)')
    parser.add_argument('--n-total', type=float, default=1e4,
                        help='N_total [#/cm3] (default: 1e4)')
    parser.add_argument('--gmd', type=float, default=0.02,
                        help='GMD [um] (default: 0.02)')
    parser.add_argument('--gsd', type=float, default=1.6,
                        help='GSD (default: 1.6)')
    parser.add_argument('--temp', type=float, default=298.0,
                        help='Temperature [K] (default: 298)')
    parser.add_argument('--pres', type=float, default=101325.0,
                        help='Pressure [Pa] (default: 101325)')
    parser.add_argument('--h2so4', type=float, default=1e7,
                        help='H2SO4 [molec/cm3] (default: 1e7)')
    parser.add_argument('--fortran', action='store_true',
                        help='Load and overlay Fortran results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--no-organic-nuc', action='store_true',
                        help='Disable Riccobono organic nucleation (Dunne only)')
    args = parser.parse_args()

    run_benchmark(
        dt_val=args.dt,
        n_total=args.n_total,
        gmd_um=args.gmd,
        gsd=args.gsd,
        temp_val=args.temp,
        pres_val=args.pres,
        h2so4_molec_cm3=args.h2so4,
        include_fortran=args.fortran,
        make_plots=not args.no_plots,
        enable_organic_nuc=not args.no_organic_nuc,
    )


if __name__ == '__main__':
    main()
