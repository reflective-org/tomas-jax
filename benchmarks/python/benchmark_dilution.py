"""Dilution benchmark: dilution rates + baseline, 24h or 48h.

Runs full-process simulations (SO2 chemistry + nucleation + coagulation +
condensation + dilution) with multiple dilution rates and a no-dilution
baseline, producing 6 comparison figures.

Two scenario profiles:
    surface (default) — Surface conditions (288K, 1atm), 24h,
        kdil = 1e-4 (BL) and 1e-3 (plume)
    stratosphere — Lower stratosphere (216K, 55hPa), 48h, 20 ppb SO2,
        kdil from volume doubling (8.02e-6) and 1.78× growth (6.68e-6)

Each scenario runs with all processes: so2_chemistry → nucleation →
coagulation → condensation → dilution, using make_step() composable API.

Figures:
    1. N_total timeseries (with passive tracer overlay)
    2. M_dry timeseries
    3. Gas-phase SO2 + H2SO4
    4. Size distributions at 0h, 6h, 12h, final
    5. N/M ratios vs passive tracer (dilution attribution)
    6. Banana plots (size-resolved time evolution)

Usage::

    python -m benchmarks.python.benchmark_dilution                        # surface, run + plot
    python -m benchmarks.python.benchmark_dilution --profile stratosphere # strat, run + plot
    python -m benchmarks.python.benchmark_dilution --plot-only            # replot from NPZ
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
    SRTSO4, SRTSO2, SRTH2O, XK0,
    MW_H2SO4, MW_SO2, KB, AVOGADRO, PI,
    make_grid,
)
from tomas_jax.solvers.condensation import make_step
from tomas_jax.physics.dilution import dilute_tracer

# =========================================================================
# Constants
# =========================================================================
DENS_INIT = 1770.0  # kg/m3
BOXVOL = 1.0e6      # cm3
DT = 60.0           # seconds per step
ALPHA = 1.0

# Nucleation parameters (used in all profiles)
NUC_ORG = jnp.float64(1e7)
NUC_NH3 = jnp.float64(1e9)
NUC_FION = jnp.float64(3.0)

# =========================================================================
# Scenario profiles
# =========================================================================

def _ppb_to_molec_cm3(ppb, temp, pres):
    """Convert ppbv to molec/cm3."""
    M_total = pres / (KB * temp) * 1e-6
    return ppb * 1e-9 * M_total


PROFILES = {
    'surface': {
        'label': 'Surface (288K, 1atm)',
        'temp': 288.0, 'pres': 101325.0, 'rh': 0.50,
        'nhours': 24,
        'so2_molec_cm3': 5e10,
        'h2so4_molec_cm3': 1e7,
        'oh_conc': 1e6,
        'n_total': 1e4, 'gmd': 50e-9, 'gsd': 1.6,
        'cases': {
            'baseline': {'kdil': 0.0,
                         'label': 'No dilution',
                         'color': '#333333', 'ls': '-'},
            'moderate': {'kdil': 1e-4,
                         'label': r'$k_{dil}$=1e-4 s⁻¹ (BL, τ≈2.8h)',
                         'color': '#2196F3', 'ls': '-'},
            'strong':   {'kdil': 1e-3,
                         'label': r'$k_{dil}$=1e-3 s⁻¹ (plume, τ≈17min)',
                         'color': '#E91E63', 'ls': '-'},
        },
        'case_order': ['baseline', 'moderate', 'strong'],
    },
    'stratosphere': {
        'label': 'Lower stratosphere (216K, 55hPa)',
        'temp': 216.0, 'pres': 5500.0, 'rh': 0.01,
        'nhours': 48,
        # SO2 = 20 ppb at stratospheric conditions
        'so2_ppb': 20.0,
        'h2so4_molec_cm3': 1e6,  # lower in stratosphere
        'oh_conc': 1e6,
        'n_total': 1e3, 'gmd': 80e-9, 'gsd': 1.6,  # stratospheric background
        'cases': {
            'baseline': {'kdil': 0.0,
                         'label': 'No dilution',
                         'color': '#333333', 'ls': '-'},
            'vol_2x':   {'kdil': np.log(2.0) / 86400.0,   # 8.02e-6 s^-1
                         'label': r'Vol 2× in 24h ($k_{dil}$=8.0e-6, τ≈34.7h)',
                         'color': '#2196F3', 'ls': '-'},
            'vol_178x': {'kdil': np.log(1.78) / 86400.0,  # 6.68e-6 s^-1
                         'label': r'Vol 1.78× in 24h ($k_{dil}$=6.7e-6, τ≈41.6h)',
                         'color': '#4CAF50', 'ls': '-'},
        },
        'case_order': ['baseline', 'vol_2x', 'vol_178x'],
    },
}

# Active profile (set by CLI or run_all)
_PROFILE = None

def _get_profile():
    return PROFILES[_PROFILE]

def _get_outdir():
    return os.path.join(os.path.dirname(__file__), '..', 'results',
                        f'dilution_benchmark_{_PROFILE}')


# =========================================================================
# Initialization
# =========================================================================

def make_initial_state(nbins, xk):
    """Create initial Nk, Mk, Gc from active profile."""
    prof = _get_profile()
    xk_np = np.array(xk)

    n_total = prof['n_total']
    gmd = prof['gmd']
    gsd = prof['gsd']

    Nk = np.zeros(nbins)
    Mk = np.zeros((nbins, ICOMP))

    for k in range(nbins):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dk = np.sqrt(Dl * Dh)
        gmd_um = gmd * 1e6

        np_init = ((n_total * BOXVOL)
                   / (np.sqrt(2.0 * PI) * Dk * np.log(gsd))
                   * np.exp(-(np.log(Dk / gmd_um) ** 2
                              / (2.0 * np.log(gsd) ** 2)))
                   * (Dh - Dl))

        Nk[k] = max(np_init, 1e-3)
        Mk[k, SRTSO4] = Nk[k] * np.sqrt(xk_np[k] * xk_np[k + 1])

    # Gas phase — SO2 can be specified in ppb or molec/cm3
    Gc = np.zeros(N_GAS_SPECIES)
    h2so4_init = prof['h2so4_molec_cm3']
    if 'so2_ppb' in prof:
        so2_init = _ppb_to_molec_cm3(prof['so2_ppb'], prof['temp'], prof['pres'])
    else:
        so2_init = prof['so2_molec_cm3']

    Gc[SRTSO4] = h2so4_init * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO
    Gc[SRTSO2] = so2_init * BOXVOL * (MW_SO2 / 1000.0) / AVOGADRO

    return jnp.array(Nk), jnp.array(Mk), jnp.array(Gc)


# =========================================================================
# Single scenario runner
# =========================================================================

def run_scenario(case_key, nbins, xk, verbose=True):
    """Run one dilution scenario for the active profile duration.

    Returns dict with every-minute and hourly timeseries.
    """
    prof = _get_profile()
    case = prof['cases'][case_key]
    kdil = case['kdil']
    nhours = prof['nhours']
    nsteps = nhours * 60

    temp = jnp.float64(prof['temp'])
    pres = jnp.float64(prof['pres'])
    rh = jnp.float64(prof['rh'])
    alpha = jnp.float64(ALPHA)
    boxvol = jnp.float64(BOXVOL)
    dt = jnp.float64(DT)

    Nk, Mk, Gc = make_initial_state(nbins, xk)

    # Save initial state for background (clean-air: zeros)
    Nk_bg = jnp.zeros_like(Nk)
    Mk_bg = jnp.zeros_like(Mk)
    Gc_bg = jnp.zeros_like(Gc)

    # Build process list
    processes = ['so2_chemistry', 'nucleation', 'coagulation', 'condensation']
    if kdil > 0:
        processes.append('dilution')

    step_fn_jit = make_step(processes, cond_method='ppm_jit')

    # kwargs
    oh_conc = jnp.float64(prof['oh_conc'])
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
    _ = step_fn_jit(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, **kw)

    # Storage
    Nk_every = np.zeros((nsteps, nbins))
    Gc_SO2_every = np.zeros(nsteps)
    Gc_SO4_every = np.zeros(nsteps)
    N_tot_every = np.zeros(nsteps)
    M_dry_every = np.zeros(nsteps)
    tracer_every = np.zeros(nsteps)

    Nk_hourly = np.zeros((nhours, nbins))
    N_tot_hourly = np.zeros(nhours)
    M_dry_hourly = np.zeros(nhours)

    tracer = jnp.float64(1.0)

    so2_to_molec = AVOGADRO / (MW_SO2 / 1000.0) / BOXVOL
    so4_to_molec = AVOGADRO / (MW_H2SO4 / 1000.0) / BOXVOL

    t0 = time.time()

    for i in range(nsteps):
        # Record state BEFORE stepping
        Nk_np = np.array(Nk)
        Gc_np = np.array(Gc)

        Nk_every[i] = Nk_np
        Gc_SO2_every[i] = Gc_np[SRTSO2] * so2_to_molec
        Gc_SO4_every[i] = Gc_np[SRTSO4] * so4_to_molec
        N_tot_every[i] = float(jnp.sum(Nk))
        M_dry_every[i] = float(jnp.sum(Mk[:, :SRTH2O]))
        tracer_every[i] = float(tracer)

        if i % 60 == 0:
            hr = i // 60
            Nk_hourly[hr] = Nk_np
            N_tot_hourly[hr] = N_tot_every[i]
            M_dry_hourly[hr] = M_dry_every[i]

        # Step
        Nk, Mk, Gc = step_fn_jit(Nk, Mk, Gc, xk, temp, pres, boxvol,
                                   rh, alpha, dt, **kw)

        # Update passive tracer
        if kdil > 0:
            tracer = dilute_tracer(tracer, DT, kdil)

    wall_time = time.time() - t0

    result = {
        'Nk_every': Nk_every,
        'Gc_SO2_molec': Gc_SO2_every,
        'Gc_SO4_molec': Gc_SO4_every,
        'N_tot_every': N_tot_every,
        'M_dry_every': M_dry_every,
        'tracer_every': tracer_every,
        'Nk_hourly': Nk_hourly,
        'N_tot_hourly': N_tot_hourly,
        'M_dry_hourly': M_dry_hourly,
        'Nk_final': np.array(Nk),
        'Mk_final': np.array(Mk),
        'wall_time_s': wall_time,
        'case': case_key,
        'kdil': kdil,
        'nbins': nbins,
        'nhours': nhours,
        'nsteps': nsteps,
    }

    if verbose:
        tau_str = f"τ={1.0/kdil:.0f}s" if kdil > 0 else "no dil"
        print(f"  {case_key:<10s} | kdil={kdil:.0e} ({tau_str:<12s}) | "
              f"{wall_time:6.2f}s | N_init={N_tot_every[0]/BOXVOL:.2e}/cm3")

    return result


# =========================================================================
# Run all
# =========================================================================

def run_all(verbose=True):
    """Run all dilution cases for the active profile and save NPZ files."""
    prof = _get_profile()
    nbins = NBINS
    xk = make_grid(nbins, XK0, 2.0)
    outdir = _get_outdir()

    os.makedirs(outdir, exist_ok=True)

    # Compute SO2 for display
    if 'so2_ppb' in prof:
        so2_molec = _ppb_to_molec_cm3(prof['so2_ppb'], prof['temp'], prof['pres'])
        so2_str = f"{prof['so2_ppb']:.0f} ppb ({so2_molec:.2e} molec/cm3)"
    else:
        so2_str = f"{prof['so2_molec_cm3']:.0e} molec/cm3"

    print("=" * 72)
    print(f"Dilution Benchmark — {prof['nhours']}h, {prof['label']}")
    print(f"  Cases: {prof['case_order']}")
    print(f"  SO2:  {so2_str}")
    print(f"  OH:   {prof['oh_conc']:.0e} molec/cm3")
    print(f"  Grid: {nbins} bins, T={prof['temp']}K, "
          f"P={prof['pres']:.0f}Pa, RH={prof['rh']}")
    print("=" * 72)

    for case_key in prof['case_order']:
        res = run_scenario(case_key, nbins, xk, verbose)

        fname = f"dilution_{case_key}_{nbins}bin.npz"
        np.savez_compressed(
            os.path.join(outdir, fname),
            **{k: v for k, v in res.items()
               if isinstance(v, (np.ndarray, np.floating, float, int))},
        )

    print("-" * 72)
    print(f"Results saved to {outdir}")


# =========================================================================
# Plotting helpers
# =========================================================================

def _load(case_key, nbins=NBINS):
    """Load NPZ for one case."""
    fname = f"dilution_{case_key}_{nbins}bin.npz"
    outdir = _get_outdir()
    path = os.path.join(outdir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {fname}. Run simulations first "
            f"(python -m benchmarks.python.benchmark_dilution --profile {_PROFILE}).")
    return dict(np.load(path, allow_pickle=True))


def _time_hours():
    prof = _get_profile()
    nsteps = prof['nhours'] * 60
    return np.arange(nsteps) / 60.0


def _dp_midpoints(xk_np):
    m_mid = np.sqrt(xk_np[:-1] * xk_np[1:])
    return np.cbrt(m_mid / DENS_INIT * (6.0 / PI))


def _dlogDp(xk_np):
    dp_edges = np.cbrt(xk_np / DENS_INIT * (6.0 / PI))
    return np.log10(dp_edges[1:] / dp_edges[:-1])


# =========================================================================
# Figures
# =========================================================================

def plot_fig1_ntotal():
    """Fig 1: N_total timeseries with passive tracer overlay."""
    import matplotlib.pyplot as plt

    prof = _get_profile()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 1: Total Particle Number — {prof["label"]} ({prof["nhours"]}h)',
                 fontsize=13, fontweight='bold')

    t_hr = _time_hours()

    for case_key in prof['case_order']:
        d = _load(case_key)
        c = prof['cases'][case_key]
        N_cm3 = d['N_tot_every'] / BOXVOL
        N_ratio = d['N_tot_every'] / d['N_tot_every'][0]

        ax1.semilogy(t_hr, N_cm3, color=c['color'], ls=c['ls'],
                     lw=2, label=c['label'])
        ax2.plot(t_hr, N_ratio, color=c['color'], ls=c['ls'],
                 lw=2, label=c['label'])

        if float(d['kdil']) > 0:
            ax2.plot(t_hr, d['tracer_every'], color=c['color'], ls='--',
                     lw=1.5, alpha=0.7, label=f"Tracer ({case_key})")

    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('N_total [#/cm³]')
    ax1.set_title('Absolute')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which='both')

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('N(t) / N(0)')
    ax2.set_title('Ratio to initial (dashed = passive tracer)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def plot_fig2_mdry():
    """Fig 2: M_dry timeseries."""
    import matplotlib.pyplot as plt

    prof = _get_profile()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 2: Dry Aerosol Mass — {prof["label"]} ({prof["nhours"]}h)',
                 fontsize=13, fontweight='bold')

    t_hr = _time_hours()

    for case_key in prof['case_order']:
        d = _load(case_key)
        c = prof['cases'][case_key]

        ax1.semilogy(t_hr, d['M_dry_every'], color=c['color'], ls=c['ls'],
                     lw=2, label=c['label'])

        M_ratio = d['M_dry_every'] / d['M_dry_every'][0]
        ax2.plot(t_hr, M_ratio, color=c['color'], ls=c['ls'],
                 lw=2, label=c['label'])

        if float(d['kdil']) > 0:
            ax2.plot(t_hr, d['tracer_every'], color=c['color'], ls='--',
                     lw=1.5, alpha=0.7, label=f"Tracer ({case_key})")

    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('M_dry [kg/cell]')
    ax1.set_title('Absolute')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which='both')

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('M(t) / M(0)')
    ax2.set_title('Ratio to initial (dashed = passive tracer)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fig3_gas():
    """Fig 3: Gas-phase SO2 + H2SO4 timeseries."""
    import matplotlib.pyplot as plt

    prof = _get_profile()
    nsteps = prof['nhours'] * 60
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 3: Gas-Phase Concentrations — {prof["label"]} ({prof["nhours"]}h)',
                 fontsize=13, fontweight='bold')

    t_hr = _time_hours()

    for case_key in prof['case_order']:
        d = _load(case_key)
        c = prof['cases'][case_key]

        ax1.semilogy(t_hr, d['Gc_SO2_molec'], color=c['color'], ls=c['ls'],
                     lw=2, label=c['label'])
        ax2.semilogy(t_hr, d['Gc_SO4_molec'], color=c['color'], ls=c['ls'],
                     lw=2, label=c['label'])

    # Pure-dilution reference for non-baseline cases
    baseline = _load(prof['case_order'][0])
    so2_init = baseline['Gc_SO2_molec'][0]
    for case_key in prof['case_order'][1:]:
        c = prof['cases'][case_key]
        tracer_so2 = so2_init * np.exp(-c['kdil'] * np.arange(nsteps) * DT)
        ax1.semilogy(t_hr, tracer_so2, color=c['color'], ls=':',
                     lw=1, alpha=0.5, label=f"Pure dilution ({case_key})")

    ax1.set_xlabel('Time [hours]')
    ax1.set_ylabel('SO2 [molec/cm³]')
    ax1.set_title('SO2 (solid = full model, dotted = dilution only)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, which='both')

    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('H2SO4 [molec/cm³]')
    ax2.set_title('H2SO4 gas')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    return fig


def plot_fig4_sizedist():
    """Fig 4: Size distributions at 4 time snapshots."""
    import matplotlib.pyplot as plt

    prof = _get_profile()
    nhours = prof['nhours']
    xk = np.array(make_grid(NBINS, XK0, 2.0))
    dp_m = _dp_midpoints(xk)
    dp_nm = dp_m * 1e9
    dlogDp = _dlogDp(xk)

    # Pick 4 evenly spaced snapshots
    h_list = [0, nhours // 4, nhours // 2, nhours - 1]
    h_labels = [f'{h}h' for h in h_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Fig 4: Size Distributions — {prof["label"]}',
                 fontsize=13, fontweight='bold')

    for idx, (hr_idx, hr_label) in enumerate(zip(h_list, h_labels)):
        ax = axes.flat[idx]

        for case_key in prof['case_order']:
            d = _load(case_key)
            c = prof['cases'][case_key]

            step_idx = hr_idx * 60
            Nk = d['Nk_every'][step_idx]
            dNdlogDp = (Nk / BOXVOL) / dlogDp

            ax.loglog(dp_nm, dNdlogDp, color=c['color'], ls=c['ls'],
                      lw=2, label=c['label'])

        ax.set_xlabel('Dp [nm]')
        ax.set_ylabel('dN/dlogDp [#/cm³]')
        ax.set_title(f't = {hr_label}')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(1, 2e4)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_fig5_attribution():
    """Fig 5: Dilution attribution — ratio to no-dilution baseline."""
    import matplotlib.pyplot as plt

    prof = _get_profile()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 5: Dilution Impact — Ratio to Baseline, {prof["label"]}',
                 fontsize=13, fontweight='bold')

    t_hr = _time_hours()

    baseline_key = prof['case_order'][0]
    baseline = _load(baseline_key)
    N_base = baseline['N_tot_every']
    M_base = baseline['M_dry_every']

    for case_key in prof['case_order'][1:]:
        d = _load(case_key)
        c = prof['cases'][case_key]
        tracer = d['tracer_every']

        N_ratio = d['N_tot_every'] / np.maximum(N_base, 1e-30)
        M_ratio = d['M_dry_every'] / np.maximum(M_base, 1e-30)

        ax1.semilogy(t_hr, N_ratio, color=c['color'], lw=2, label=c['label'])
        ax2.semilogy(t_hr, M_ratio, color=c['color'], lw=2, label=c['label'])

        ax1.semilogy(t_hr, tracer, color=c['color'], ls='--', lw=1.5,
                     alpha=0.6, label=f"Tracer ({case_key})")
        ax2.semilogy(t_hr, tracer, color=c['color'], ls='--', lw=1.5,
                     alpha=0.6, label=f"Tracer ({case_key})")

    for ax in [ax1, ax2]:
        ax.set_xlabel('Time [hours]')
        ax.grid(True, alpha=0.3, which='both')

    ax1.set_ylabel('N_diluted / N_baseline')
    ax1.set_title('Number (solid=actual, dashed=passive tracer)')
    ax1.legend(fontsize=7, loc='best')

    ax2.set_ylabel('M_diluted / M_baseline')
    ax2.set_title('Mass (solid=actual, dashed=passive tracer)')
    ax2.legend(fontsize=7, loc='best')

    plt.tight_layout()
    return fig


def plot_fig6_banana():
    """Fig 6: Banana plots (one panel per case)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    prof = _get_profile()
    ncases = len(prof['case_order'])

    xk = np.array(make_grid(NBINS, XK0, 2.0))
    dp_m = _dp_midpoints(xk)
    dp_nm = dp_m * 1e9
    dlogDp = _dlogDp(xk)

    t_hr = _time_hours()

    fig, axes = plt.subplots(1, ncases, figsize=(6 * ncases, 5))
    if ncases == 1:
        axes = [axes]
    fig.suptitle(f'Fig 6: Size Distribution Evolution — {prof["label"]}',
                 fontsize=13, fontweight='bold')

    for idx, case_key in enumerate(prof['case_order']):
        ax = axes[idx]
        d = _load(case_key)
        c = prof['cases'][case_key]

        Nk_all = d['Nk_every']
        dNdlogDp = (Nk_all / BOXVOL) / dlogDp[np.newaxis, :]

        sub = 10
        t_sub = t_hr[::sub]
        data_sub = dNdlogDp[::sub]

        pcm = ax.pcolormesh(
            t_sub, dp_nm, data_sub.T,
            norm=mcolors.LogNorm(vmin=1e-2, vmax=1e6),
            cmap='inferno', shading='nearest',
        )
        ax.set_yscale('log')
        ax.set_ylim(1, 2e4)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Dp [nm]')
        ax.set_title(c['label'])

    fig.colorbar(pcm, ax=axes, label='dN/dlogDp [#/cm³]', shrink=0.8)
    plt.tight_layout()
    return fig


# =========================================================================
# Main
# =========================================================================

def plot_all():
    """Generate all 6 figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outdir = _get_outdir()
    os.makedirs(outdir, exist_ok=True)

    figures = [
        ('fig1_ntotal.png', plot_fig1_ntotal),
        ('fig2_mdry.png', plot_fig2_mdry),
        ('fig3_gas.png', plot_fig3_gas),
        ('fig4_sizedist.png', plot_fig4_sizedist),
        ('fig5_attribution.png', plot_fig5_attribution),
        ('fig6_banana.png', plot_fig6_banana),
    ]

    for fname, plot_fn in figures:
        print(f"  Plotting {fname}...")
        fig = plot_fn()
        fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"All figures saved to {outdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dilution Benchmark')
    parser.add_argument('--profile', choices=list(PROFILES.keys()),
                        default='surface',
                        help='Scenario profile (default: surface)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only regenerate plots from existing NPZ files')
    args = parser.parse_args()

    _PROFILE = args.profile

    if not args.plot_only:
        run_all()

    plot_all()
