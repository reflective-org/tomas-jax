"""SAI box model — 96h simulation of aircraft SO2 release in the stratosphere.

Runs a 4-day (96h) box-model simulation with all processes (SO2 chemistry,
nucleation, coagulation, condensation, dilution) starting from a fresh SO2
plume at stratospheric conditions (216K, 55 hPa).

Three dilution cases:
    baseline  — No dilution (closed box)
    vol_2x    — Volume doubles in 24h (kdil = ln(2)/86400 = 8.02e-6 s⁻¹)
    vol_178x  — Volume +78% in 24h (kdil = ln(1.78)/86400 = 6.68e-6 s⁻¹)

Initial SO2 concentration computed from plume_concentration.py (~2e13 molec/cm³).

Figures:
    1. N_total timeseries (absolute + ratio with tracer)
    2. M_dry timeseries (absolute + ratio with tracer)
    3. dN/dlogDp snapshots at 12h, 24h, 48h, 72h, 96h
    4. dM/dlogDp snapshots at same times
    5. Banana plots (one panel per case)
    6. Gas-phase SO2 + H2SO4 timeseries

Usage::

    python -m experimental_case.run_sai_simulation              # run + plot
    python -m experimental_case.run_sai_simulation --plot-only  # replot from NPZ
"""
import os
import sys
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTSO2, SRTH2O,
    MW_H2SO4, MW_SO2, KB, AVOGADRO, PI,
    make_grid, XK0,
)
from tomas_jax.solvers.condensation import make_step
from tomas_jax.physics.dilution import dilute_tracer

from .plume_concentration import plume_so2_concentration

# =========================================================================
# Constants
# =========================================================================
DENS_INIT = 1770.0  # kg/m3 (sulfate density for initial size distribution)
BOXVOL = 1.0e6      # cm3
DT = 60.0           # seconds per step
ALPHA = 1.0         # accommodation coefficient

NHOURS = 96         # 4-day simulation
NSTEPS = NHOURS * 60  # 5760 steps

# Stratospheric conditions
TEMP = 216.0        # K
PRES = 5500.0       # Pa (55 hPa)
RH = 0.01           # very dry stratosphere

# Initial aerosol: sparse background
N_BG = 5.0          # #/cm3
GMD_BG = 80e-9      # 80 nm geometric mean diameter
GSD_BG = 1.6        # geometric standard deviation

# Initial gas phase
H2SO4_INIT = 1e5    # molec/cm3 (low stratospheric background)

# OH for SO2 oxidation
OH_CONC = 1e6       # molec/cm3

# Nucleation precursors (stratospheric)
NUC_ORG = jnp.float64(1e7)   # organic vapor [molec/cm3]
NUC_NH3 = jnp.float64(1e9)   # NH3 [pptv]
NUC_FION = jnp.float64(3.0)  # ion-pair production rate [cm-3 s-1]

# =========================================================================
# Three dilution cases
# =========================================================================
CASES = {
    'baseline': {
        'kdil': 0.0,
        'label': 'No dilution (closed box)',
        'color': '#333333', 'ls': '-',
    },
    'vol_2x': {
        'kdil': np.log(2.0) / 86400.0,   # 8.02e-6 s^-1
        'label': r'Vol 2$\times$ in 24h ($k_{dil}$=8.0e-6, $\tau$≈34.7h)',
        'color': '#2196F3', 'ls': '-',
    },
    'vol_178x': {
        'kdil': np.log(1.78) / 86400.0,  # 6.68e-6 s^-1
        'label': r'Vol 1.78$\times$ in 24h ($k_{dil}$=6.7e-6, $\tau$≈41.6h)',
        'color': '#4CAF50', 'ls': '-',
    },
}
CASE_ORDER = ['baseline', 'vol_2x', 'vol_178x']

# =========================================================================
# Output directory
# =========================================================================
_OUTDIR = os.path.join(os.path.dirname(__file__), 'results')


def _get_outdir():
    return _OUTDIR


# =========================================================================
# Initialization
# =========================================================================

def _get_initial_so2_molec_cm3():
    """Compute initial SO2 from plume calculator."""
    result = plume_so2_concentration(
        release_rate_ton_per_min=1.0,
        aircraft_speed_m_per_s=250.0,
        plume_radius_m=100.0,
        temp_K=TEMP,
        pres_Pa=PRES,
    )
    return result['molec_per_cm3']


def make_initial_state(nbins, xk):
    """Create initial Nk, Mk, Gc for SAI scenario.

    Background aerosol: lognormal at 80nm, 5 #/cm³.
    Gas: SO2 from plume calculator, H2SO4 at 1e5 molec/cm³.
    """
    xk_np = np.array(xk)

    Nk = np.zeros(nbins)
    Mk = np.zeros((nbins, ICOMP))

    for k in range(nbins):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dk = np.sqrt(Dl * Dh)
        gmd_um = GMD_BG * 1e6

        np_init = ((N_BG * BOXVOL)
                   / (np.sqrt(2.0 * PI) * Dk * np.log(GSD_BG))
                   * np.exp(-(np.log(Dk / gmd_um) ** 2
                              / (2.0 * np.log(GSD_BG) ** 2)))
                   * (Dh - Dl))

        Nk[k] = max(np_init, 1e-3)
        Mk[k, SRTSO4] = Nk[k] * np.sqrt(xk_np[k] * xk_np[k + 1])

    # Gas phase
    Gc = np.zeros(N_GAS_SPECIES)
    so2_init = _get_initial_so2_molec_cm3()

    Gc[SRTSO4] = H2SO4_INIT * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO
    Gc[SRTSO2] = so2_init * BOXVOL * (MW_SO2 / 1000.0) / AVOGADRO

    return jnp.array(Nk), jnp.array(Mk), jnp.array(Gc), so2_init


# =========================================================================
# Single scenario runner
# =========================================================================

def run_scenario(case_key, nbins, xk, verbose=True):
    """Run one SAI scenario for 96h.

    Returns dict with every-minute timeseries.
    """
    case = CASES[case_key]
    kdil = case['kdil']

    temp = jnp.float64(TEMP)
    pres = jnp.float64(PRES)
    rh = jnp.float64(RH)
    alpha = jnp.float64(ALPHA)
    boxvol = jnp.float64(BOXVOL)
    dt = jnp.float64(DT)

    Nk, Mk, Gc, so2_init = make_initial_state(nbins, xk)

    # Background: clean air (zeros)
    Nk_bg = jnp.zeros_like(Nk)
    Mk_bg = jnp.zeros_like(Mk)
    Gc_bg = jnp.zeros_like(Gc)

    # Build process list
    processes = ['so2_chemistry', 'nucleation', 'coagulation', 'condensation']
    if kdil > 0:
        processes.append('dilution')

    step_fn = make_step(processes, cond_method='ppm_jit')
    step_fn_jit = jax.jit(step_fn)

    # kwargs
    oh_conc = jnp.float64(OH_CONC)
    kw = dict(
        oh_conc=oh_conc,
        org_conc=NUC_ORG, nh3_conc=NUC_NH3, fion=NUC_FION,
    )
    if kdil > 0:
        kw.update(
            kdil=jnp.float64(kdil),
            Nk_bg=Nk_bg, Mk_bg=Mk_bg, Gc_bg=Gc_bg,
        )

    # Warmup JIT
    if verbose:
        print(f"  {case_key}: JIT warmup...", end=' ', flush=True)
    _ = step_fn_jit(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, **kw)
    if verbose:
        print("done.", flush=True)

    # Storage — every minute
    Nk_every = np.zeros((NSTEPS, nbins))
    Mk_dry_every = np.zeros((NSTEPS, nbins))
    N_tot_every = np.zeros(NSTEPS)
    M_dry_every = np.zeros(NSTEPS)
    Gc_SO2_every = np.zeros(NSTEPS)
    Gc_SO4_every = np.zeros(NSTEPS)
    tracer_every = np.zeros(NSTEPS)

    tracer = jnp.float64(1.0)

    so2_to_molec = AVOGADRO / (MW_SO2 / 1000.0) / BOXVOL
    so4_to_molec = AVOGADRO / (MW_H2SO4 / 1000.0) / BOXVOL

    t0 = time.time()

    for i in range(NSTEPS):
        # Record state BEFORE stepping
        Nk_np = np.array(Nk)
        Mk_np = np.array(Mk)
        Gc_np = np.array(Gc)

        Nk_every[i] = Nk_np
        Mk_dry_every[i] = np.sum(Mk_np[:, :SRTH2O], axis=1)
        N_tot_every[i] = float(jnp.sum(Nk))
        M_dry_every[i] = float(jnp.sum(Mk[:, :SRTH2O]))
        Gc_SO2_every[i] = Gc_np[SRTSO2] * so2_to_molec
        Gc_SO4_every[i] = Gc_np[SRTSO4] * so4_to_molec
        tracer_every[i] = float(tracer)

        # Progress
        if verbose and i > 0 and i % 1440 == 0:
            elapsed = time.time() - t0
            print(f"    t={i//60}h  N={N_tot_every[i]/BOXVOL:.2e}/cm3  "
                  f"SO2={Gc_SO2_every[i]:.2e}  elapsed={elapsed:.1f}s")

        # Step
        Nk, Mk, Gc = step_fn_jit(Nk, Mk, Gc, xk, temp, pres, boxvol,
                                   rh, alpha, dt, **kw)

        # Update passive tracer
        if kdil > 0:
            tracer = dilute_tracer(tracer, DT, kdil)

    wall_time = time.time() - t0

    result = {
        'Nk_every': Nk_every,
        'Mk_dry_every': Mk_dry_every,
        'N_tot_every': N_tot_every,
        'M_dry_every': M_dry_every,
        'Gc_SO2_molec': Gc_SO2_every,
        'Gc_SO4_molec': Gc_SO4_every,
        'tracer_every': tracer_every,
        'wall_time_s': wall_time,
        'case': case_key,
        'kdil': kdil,
        'nbins': nbins,
        'nhours': NHOURS,
        'nsteps': NSTEPS,
        'so2_init_molec_cm3': so2_init,
    }

    if verbose:
        tau_str = f"τ={1.0/kdil/3600:.1f}h" if kdil > 0 else "no dil"
        print(f"  {case_key:<10s} | kdil={kdil:.2e} ({tau_str:<12s}) | "
              f"{wall_time:6.1f}s | N_final={N_tot_every[-1]/BOXVOL:.2e}/cm3")

    return result


# =========================================================================
# Run all cases
# =========================================================================

def run_all(verbose=True):
    """Run all 3 dilution cases and save NPZ files."""
    nbins = NBINS
    xk = make_grid(nbins, XK0, 2.0)
    outdir = _get_outdir()
    os.makedirs(outdir, exist_ok=True)

    so2_init = _get_initial_so2_molec_cm3()

    print("=" * 72)
    print(f"SAI Box Model — {NHOURS}h simulation")
    print(f"  Conditions: T={TEMP}K, P={PRES}Pa, RH={RH}")
    print(f"  Initial SO2: {so2_init:.2e} molec/cm3 ({so2_init/1e13:.1f}×10¹³)")
    print(f"  Initial H2SO4: {H2SO4_INIT:.0e} molec/cm3")
    print(f"  Background aerosol: {N_BG} #/cm3, GMD={GMD_BG*1e9:.0f}nm")
    print(f"  OH: {OH_CONC:.0e} molec/cm3")
    print(f"  Grid: {nbins} bins")
    print(f"  Cases: {CASE_ORDER}")
    print("=" * 72)

    for case_key in CASE_ORDER:
        res = run_scenario(case_key, nbins, xk, verbose)

        fname = f"sai_{case_key}_{nbins}bin.npz"
        np.savez_compressed(
            os.path.join(outdir, fname),
            **{k: v for k, v in res.items()
               if isinstance(v, (np.ndarray, np.floating, float, int))},
        )

    print("-" * 72)
    print(f"Results saved to {outdir}")


# =========================================================================
# Data loading and plotting helpers
# =========================================================================

def _load(case_key, nbins=NBINS):
    """Load NPZ for one case."""
    fname = f"sai_{case_key}_{nbins}bin.npz"
    outdir = _get_outdir()
    path = os.path.join(outdir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}. Run simulations first "
            f"(python -m experimental_case.run_sai_simulation).")
    return dict(np.load(path, allow_pickle=True))


def _time_hours():
    return np.arange(NSTEPS) / 60.0


def _dp_midpoints(xk_np):
    """Bin-center diameters [m] from bin-edge masses."""
    m_mid = np.sqrt(xk_np[:-1] * xk_np[1:])
    return np.cbrt(m_mid / DENS_INIT * (6.0 / PI))


def _dlogDp(xk_np):
    """Delta-log10(Dp) for each bin."""
    dp_edges = np.cbrt(xk_np / DENS_INIT * (6.0 / PI))
    return np.log10(dp_edges[1:] / dp_edges[:-1])


# =========================================================================
# Figure 1: N_total timeseries
# =========================================================================

def plot_fig1_ntotal():
    """Fig 1: N_total vs time (absolute + ratio with tracer)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 1: Total Particle Number — SAI {NHOURS}h '
                 f'(T={TEMP}K, P={PRES/100:.0f}hPa)',
                 fontsize=13, fontweight='bold')

    t_hr = _time_hours()

    for case_key in CASE_ORDER:
        d = _load(case_key)
        c = CASES[case_key]
        N_cm3 = d['N_tot_every'] / BOXVOL

        ax1.semilogy(t_hr, N_cm3, color=c['color'], ls=c['ls'],
                     lw=2, label=c['label'])

        N_ratio = d['N_tot_every'] / d['N_tot_every'][0]
        ax2.semilogy(t_hr, N_ratio, color=c['color'], ls=c['ls'],
                     lw=2, label=c['label'])

        if float(d['kdil']) > 0:
            ax2.plot(t_hr, d['tracer_every'], color=c['color'], ls='--',
                     lw=1.5, alpha=0.7, label=f"Tracer ({case_key})")

    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('N_total [#/cm³]')
    ax1.set_title('Absolute')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, which='both')

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('N(t) / N(0)')
    ax2.set_title('Ratio to initial (dashed = passive tracer)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    return fig


# =========================================================================
# Figure 1b: N_total timeseries (linear y-axis)
# =========================================================================

def plot_fig1b_ntotal_linear():
    """Fig 1b: N_total vs time (linear y-axis)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 1b: Total Particle Number (linear) — SAI {NHOURS}h '
                 f'(T={TEMP}K, P={PRES/100:.0f}hPa)',
                 fontsize=13, fontweight='bold')

    t_hr = _time_hours()

    for case_key in CASE_ORDER:
        d = _load(case_key)
        c = CASES[case_key]
        N_cm3 = d['N_tot_every'] / BOXVOL

        ax1.plot(t_hr, N_cm3, color=c['color'], ls=c['ls'],
                 lw=2, label=c['label'])

        N_ratio = d['N_tot_every'] / d['N_tot_every'][0]
        ax2.plot(t_hr, N_ratio, color=c['color'], ls=c['ls'],
                 lw=2, label=c['label'])

        if float(d['kdil']) > 0:
            ax2.plot(t_hr, d['tracer_every'], color=c['color'], ls='--',
                     lw=1.5, alpha=0.7, label=f"Tracer ({case_key})")

    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('N_total [#/cm³]')
    ax1.set_title('Absolute')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('N(t) / N(0)')
    ax2.set_title('Ratio to initial (dashed = passive tracer)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =========================================================================
# Figure 2: M_dry timeseries
# =========================================================================

def plot_fig2_mdry():
    """Fig 2: M_dry vs time (absolute + ratio with tracer)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 2: Dry Aerosol Mass — SAI {NHOURS}h',
                 fontsize=13, fontweight='bold')

    t_hr = _time_hours()
    # kg/cell → µg/m³  (BOXVOL = 1e6 cm³ = 1 m³, so ×1e9 for kg→µg)
    kg_to_ug_m3 = 1e9 / (BOXVOL * 1e-6)

    for case_key in CASE_ORDER:
        d = _load(case_key)
        c = CASES[case_key]

        M_ug = d['M_dry_every'] * kg_to_ug_m3
        ax1.semilogy(t_hr, M_ug, color=c['color'], ls=c['ls'],
                     lw=2, label=c['label'])

        M_ratio = d['M_dry_every'] / d['M_dry_every'][0]
        ax2.plot(t_hr, M_ratio, color=c['color'], ls=c['ls'],
                 lw=2, label=c['label'])

        if float(d['kdil']) > 0:
            ax2.plot(t_hr, d['tracer_every'], color=c['color'], ls='--',
                     lw=1.5, alpha=0.7, label=f"Tracer ({case_key})")

    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel(r'M$_{dry}$ [$\mu$g/m³]')
    ax1.set_title('Absolute')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, which='both')

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('M(t) / M(0)')
    ax2.set_title('Ratio to initial (dashed = passive tracer)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =========================================================================
# Figure 2b: M_dry timeseries (linear y-axis)
# =========================================================================

def plot_fig2b_mdry_linear():
    """Fig 2b: M_dry vs time (linear y-axis)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 2b: Dry Aerosol Mass (linear) — SAI {NHOURS}h',
                 fontsize=13, fontweight='bold')

    t_hr = _time_hours()
    kg_to_ug_m3 = 1e9 / (BOXVOL * 1e-6)

    for case_key in CASE_ORDER:
        d = _load(case_key)
        c = CASES[case_key]

        M_ug = d['M_dry_every'] * kg_to_ug_m3
        ax1.plot(t_hr, M_ug, color=c['color'], ls=c['ls'],
                 lw=2, label=c['label'])

        M_ratio = d['M_dry_every'] / d['M_dry_every'][0]
        ax2.plot(t_hr, M_ratio, color=c['color'], ls=c['ls'],
                 lw=2, label=c['label'])

        if float(d['kdil']) > 0:
            ax2.plot(t_hr, d['tracer_every'], color=c['color'], ls='--',
                     lw=1.5, alpha=0.7, label=f"Tracer ({case_key})")

    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel(r'M$_{dry}$ [$\mu$g/m³]')
    ax1.set_title('Absolute')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('M(t) / M(0)')
    ax2.set_title('Ratio to initial (dashed = passive tracer)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =========================================================================
# Figure 3: dN/dlogDp snapshots
# =========================================================================

def plot_fig3_dndlogdp():
    """Fig 3: dN/dlogDp at 12h, 24h, 48h, 72h, 96h (2×3 grid)."""
    import matplotlib.pyplot as plt

    xk = np.array(make_grid(NBINS, XK0, 2.0))
    dp_nm = _dp_midpoints(xk) * 1e9
    dlogDp = _dlogDp(xk)

    snapshot_hours = [12, 24, 48, 72, 96]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Fig 3: dN/dlogDp — SAI {NHOURS}h',
                 fontsize=13, fontweight='bold')

    for idx, hr in enumerate(snapshot_hours):
        ax = axes.flat[idx]
        step_idx = min(hr * 60, NSTEPS - 1)

        for case_key in CASE_ORDER:
            d = _load(case_key)
            c = CASES[case_key]
            Nk = d['Nk_every'][step_idx]
            dNdlogDp = (Nk / BOXVOL) / dlogDp
            ax.loglog(dp_nm, dNdlogDp, color=c['color'], ls=c['ls'],
                      lw=2, label=c['label'])

        ax.set_xlabel('Dp [nm]')
        ax.set_ylabel('dN/dlogDp [#/cm³]')
        ax.set_title(f't = {hr}h')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(1, 2e4)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide 6th panel
    axes.flat[5].set_visible(False)

    plt.tight_layout()
    return fig


# =========================================================================
# Figure 3b: dN/dlogDp snapshots (linear y-axis)
# =========================================================================

def plot_fig3b_dndlogdp_linear():
    """Fig 3b: dN/dlogDp at 12h, 24h, 48h, 72h, 96h (linear y, log x)."""
    import matplotlib.pyplot as plt

    xk = np.array(make_grid(NBINS, XK0, 2.0))
    dp_nm = _dp_midpoints(xk) * 1e9
    dlogDp = _dlogDp(xk)

    snapshot_hours = [12, 24, 48, 72, 96]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Fig 3b: dN/dlogDp (linear) — SAI {NHOURS}h',
                 fontsize=13, fontweight='bold')

    for idx, hr in enumerate(snapshot_hours):
        ax = axes.flat[idx]
        step_idx = min(hr * 60, NSTEPS - 1)

        for case_key in CASE_ORDER:
            d = _load(case_key)
            c = CASES[case_key]
            Nk = d['Nk_every'][step_idx]
            dNdlogDp = (Nk / BOXVOL) / dlogDp
            ax.semilogx(dp_nm, dNdlogDp, color=c['color'], ls=c['ls'],
                        lw=2, label=c['label'])

        ax.set_xlabel('Dp [nm]')
        ax.set_ylabel('dN/dlogDp [#/cm³]')
        ax.set_title(f't = {hr}h')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(1, 2e4)
        ax.set_ylim(bottom=0)
        if idx == 0:
            ax.legend(fontsize=7)

    axes.flat[5].set_visible(False)

    plt.tight_layout()
    return fig


# =========================================================================
# Figure 4: dM/dlogDp snapshots
# =========================================================================

def plot_fig4_dmdlogdp():
    """Fig 4: dM/dlogDp at 12h, 24h, 48h, 72h, 96h (2×3 grid)."""
    import matplotlib.pyplot as plt

    xk = np.array(make_grid(NBINS, XK0, 2.0))
    dp_nm = _dp_midpoints(xk) * 1e9
    dlogDp = _dlogDp(xk)
    kg_to_ug_m3 = 1e9 / (BOXVOL * 1e-6)

    snapshot_hours = [12, 24, 48, 72, 96]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Fig 4: dM/dlogDp — SAI {NHOURS}h',
                 fontsize=13, fontweight='bold')

    for idx, hr in enumerate(snapshot_hours):
        ax = axes.flat[idx]
        step_idx = min(hr * 60, NSTEPS - 1)

        for case_key in CASE_ORDER:
            d = _load(case_key)
            c = CASES[case_key]
            Mk_dry = d['Mk_dry_every'][step_idx]
            dMdlogDp = (Mk_dry * kg_to_ug_m3) / dlogDp
            ax.loglog(dp_nm, dMdlogDp, color=c['color'], ls=c['ls'],
                      lw=2, label=c['label'])

        ax.set_xlabel('Dp [nm]')
        ax.set_ylabel(r'dM/dlogDp [$\mu$g/m³]')
        ax.set_title(f't = {hr}h')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(1, 2e4)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide 6th panel
    axes.flat[5].set_visible(False)

    plt.tight_layout()
    return fig


# =========================================================================
# Figure 4b: dM/dlogDp snapshots (linear y-axis)
# =========================================================================

def plot_fig4b_dmdlogdp_linear():
    """Fig 4b: dM/dlogDp at 12h, 24h, 48h, 72h, 96h (linear y, log x)."""
    import matplotlib.pyplot as plt

    xk = np.array(make_grid(NBINS, XK0, 2.0))
    dp_nm = _dp_midpoints(xk) * 1e9
    dlogDp = _dlogDp(xk)
    kg_to_ug_m3 = 1e9 / (BOXVOL * 1e-6)

    snapshot_hours = [12, 24, 48, 72, 96]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Fig 4b: dM/dlogDp (linear) — SAI {NHOURS}h',
                 fontsize=13, fontweight='bold')

    for idx, hr in enumerate(snapshot_hours):
        ax = axes.flat[idx]
        step_idx = min(hr * 60, NSTEPS - 1)

        for case_key in CASE_ORDER:
            d = _load(case_key)
            c = CASES[case_key]
            Mk_dry = d['Mk_dry_every'][step_idx]
            dMdlogDp = (Mk_dry * kg_to_ug_m3) / dlogDp
            ax.semilogx(dp_nm, dMdlogDp, color=c['color'], ls=c['ls'],
                        lw=2, label=c['label'])

        ax.set_xlabel('Dp [nm]')
        ax.set_ylabel(r'dM/dlogDp [$\mu$g/m³]')
        ax.set_title(f't = {hr}h')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(1, 2e4)
        ax.set_ylim(bottom=0)
        if idx == 0:
            ax.legend(fontsize=7)

    axes.flat[5].set_visible(False)

    plt.tight_layout()
    return fig


# =========================================================================
# Figure 5: Banana plots
# =========================================================================

def plot_fig5_banana():
    """Fig 5: Banana plots (one panel per case)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    xk = np.array(make_grid(NBINS, XK0, 2.0))
    dp_nm = _dp_midpoints(xk) * 1e9
    dlogDp = _dlogDp(xk)
    t_hr = _time_hours()

    ncases = len(CASE_ORDER)
    fig = plt.figure(figsize=(6 * ncases + 1, 5))
    # GridSpec: ncases data columns + 1 narrow colorbar column
    gs = GridSpec(1, ncases + 1, figure=fig,
                  width_ratios=[1] * ncases + [0.04], wspace=0.3)
    fig.suptitle(f'Fig 5: Size Distribution Evolution — SAI {NHOURS}h',
                 fontsize=13, fontweight='bold')

    norm = mcolors.LogNorm(vmin=1e-2, vmax=1e8)
    pcm = None
    for idx, case_key in enumerate(CASE_ORDER):
        ax = fig.add_subplot(gs[0, idx])
        d = _load(case_key)
        c = CASES[case_key]

        Nk_all = d['Nk_every']
        dNdlogDp = (Nk_all / BOXVOL) / dlogDp[np.newaxis, :]

        # Subsample for plotting (every 10 minutes)
        sub = 10
        t_sub = t_hr[::sub]
        data_sub = dNdlogDp[::sub]

        pcm = ax.pcolormesh(
            t_sub, dp_nm, data_sub.T,
            norm=norm, cmap='inferno', shading='nearest',
        )
        ax.set_yscale('log')
        ax.set_ylim(1, 2e4)
        ax.set_xlabel('Time [hours]')
        if idx == 0:
            ax.set_ylabel('Dp [nm]')
        else:
            ax.set_ylabel('')
        ax.set_title(c['label'], fontsize=9)

    cax = fig.add_subplot(gs[0, ncases])
    fig.colorbar(pcm, cax=cax, label='dN/dlogDp [#/cm³]')
    fig.subplots_adjust(top=0.88, bottom=0.12)
    return fig


# =========================================================================
# Figure 6: Gas-phase SO2 + H2SO4
# =========================================================================

def plot_fig6_gas():
    """Fig 6: SO2 timeseries + H2SO4 production rate.

    The instantaneous gas-phase H2SO4 is ~0 because the condensation sink
    from ~10⁷ nucleated particles/cm³ consumes all H2SO4 within each 60s step.
    Instead, we show the H2SO4 production rate = k1(T,P,RH) × [OH] × [SO2],
    which is the physically meaningful quantity (equal to the condensation flux).
    """
    import matplotlib.pyplot as plt
    from tomas_jax.physics.so2_chemistry import calc_k1_so2_oh
    import jax.numpy as jnp

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 6: Gas-Phase SO2 and H2SO4 Production — SAI {NHOURS}h',
                 fontsize=13, fontweight='bold')

    t_hr = _time_hours()

    # Compute k1 for H2SO4 production rate
    k1 = float(calc_k1_so2_oh(jnp.float64(TEMP), jnp.float64(PRES),
                               jnp.float64(RH)))

    for case_key in CASE_ORDER:
        d = _load(case_key)
        c = CASES[case_key]

        # Left panel: SO2
        ax1.semilogy(t_hr, d['Gc_SO2_molec'], color=c['color'], ls=c['ls'],
                     lw=2, label=c['label'])

        # Right panel: H2SO4 production rate = k1 × [OH] × [SO2]
        h2so4_prod = k1 * OH_CONC * d['Gc_SO2_molec']
        ax2.semilogy(t_hr, h2so4_prod, color=c['color'], ls=c['ls'],
                     lw=2, label=c['label'])

    # Pure-dilution reference on SO2
    baseline = _load('baseline')
    so2_init = baseline['Gc_SO2_molec'][0]
    for case_key in CASE_ORDER[1:]:
        c = CASES[case_key]
        tracer_so2 = so2_init * np.exp(-c['kdil'] * np.arange(NSTEPS) * DT)
        ax1.semilogy(t_hr, tracer_so2, color=c['color'], ls=':',
                     lw=1, alpha=0.5, label=f"Pure dilution ({case_key})")

    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('SO2 [molec/cm³]')
    ax1.set_title('SO2 (solid = full model, dotted = dilution only)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, which='both')

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel(r'H$_2$SO$_4$ production [molec/cm³/s]')
    ax2.set_title(r'H$_2$SO$_4$ production rate ($= k_1 \times$ [OH] $\times$ [SO$_2$])')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    return fig


# =========================================================================
# Plot all figures
# =========================================================================

def plot_all():
    """Generate all 6 figures and save as PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outdir = _get_outdir()
    os.makedirs(outdir, exist_ok=True)

    figures = [
        ('fig1_ntotal.png', plot_fig1_ntotal),
        ('fig1b_ntotal_linear.png', plot_fig1b_ntotal_linear),
        ('fig2_mdry.png', plot_fig2_mdry),
        ('fig2b_mdry_linear.png', plot_fig2b_mdry_linear),
        ('fig3_dndlogdp.png', plot_fig3_dndlogdp),
        ('fig3b_dndlogdp_linear.png', plot_fig3b_dndlogdp_linear),
        ('fig4_dmdlogdp.png', plot_fig4_dmdlogdp),
        ('fig4b_dmdlogdp_linear.png', plot_fig4b_dmdlogdp_linear),
        ('fig5_banana.png', plot_fig5_banana),
        ('fig6_gas.png', plot_fig6_gas),
    ]

    for fname, plot_fn in figures:
        print(f"  Plotting {fname}...")
        fig = plot_fn()
        fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"All figures saved to {outdir}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAI Box Model Simulation')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only regenerate plots from existing NPZ files')
    args = parser.parse_args()

    if not args.plot_only:
        run_all()

    plot_all()
