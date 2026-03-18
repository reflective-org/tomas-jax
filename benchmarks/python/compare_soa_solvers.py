"""SOA Solver Comparison: Fortran (top-hat) vs Sequential vs Coupled.

3-way comparison of SOA/VBS condensation solvers on 2 scenarios using the
36-bin legacy grid.  Both Python solvers use ``use_ppm=False`` (top-hat
redistribution) to isolate the coupling-algorithm difference from the
redistribution difference.

Scenarios (matching Fortran benchmark_soa.f):
    sA — Pure condensation  (288 K, 1 atm,  N=1e4, GMD=50 nm, GSD=1.6)
    sB — Mixed cond/evap    (270 K, 800 hPa, N=5e3, GMD=80 nm, GSD=1.5)

Solvers:
    Fortran      — top-hat redistribution (soacond.f reference)
    Sequential   — Python Gauss-Seidel VBS loop, use_ppm=False
    Coupled      — Python vectorised fixed-point (Jacobi), use_ppm=False

Figures (9):
    1. dN/dlogDp  (log-log)        4 snapshots x 2 scenarios
    2. dN/dlogDp  (semilog-x)      4 snapshots x 2 scenarios
    3. dM_dry/dlogDp (log-log)     4 snapshots x 2 scenarios
    4. dM_dry/dlogDp (semilog-x)   4 snapshots x 2 scenarios
    5. VBS gas Gc(t)               6 panels x 2 scenarios
    6. VBS particle mass bars      4 snapshots x 2 scenarios
    7. Banana plots                3 cols x 2 scenarios
    8. Totals N(t) & M_dry(t)     2 cols x 2 scenarios
    9. Relative error vs Fortran   4 snapshots x 2 scenarios

Usage::

    python -m benchmarks.python.compare_soa_solvers --run
    python -m benchmarks.python.compare_soa_solvers --plot-only
"""

import os
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp

from benchmarks.python.benchmark_soa import (
    make_initial_state, load_fortran_scenario,
    _make_xk, _make_xk_np, _dp_midpoints, _dlogDp,
    _load_fortran_csv,
    NBINS, BOXVOL, DT, NSTEPS, NHOURS, N_VBS,
    FORTRAN_DIR, SCENARIOS, MO, DENS_INIT, ALPHA,
)
from tomas_jax.core.config import (
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES, SRTSO4, SRTORG1, SRTH2O, PI,
)
from tomas_jax.physics.soa_condensation import soa_condensation_step
from tomas_jax.core.mnfix_jax import mnfix_jax

# =========================================================================
# Output paths
# =========================================================================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results',
                           'soa_comparison')

SCENARIO_ORDER = ['sA', 'sB']

# =========================================================================
# Line styles (presentation-quality, consistent across all figures)
# =========================================================================
LS_FORTRAN = dict(color='#bbbbbb', lw=4.0, ls='-', zorder=1, alpha=0.9)
LS_SEQ = dict(color='#1f77b4', lw=2.5, ls='--', zorder=3)
LS_COUP = dict(color='#d62728', lw=2.5, ls='-', zorder=2)

_SOLVER_DEFS = [
    ('Fortran',     LS_FORTRAN),
    ('Sequential',  LS_SEQ),
    ('Coupled',     LS_COUP),
]


def _conditions_text(sc):
    """Format scenario conditions for text-box annotation."""
    # Convert Gc_org from kg/cell to µg/m³
    # boxvol = 1e6 cm³ = 1e-6 m³
    vol_m3 = BOXVOL * 1e-6
    gc_ugm3 = [g / vol_m3 * 1e9 for g in sc['Gc_org']]
    gc_str = ', '.join(f'{g:.0f}' for g in gc_ugm3)
    return (f"T={sc['temp']:.0f} K, P={sc['pres']/100:.0f} hPa\n"
            f"N={sc['n_total']:.0e} cm$^{{-3}}$\n"
            f"GMD={sc['gmd']*1e9:.0f} nm, GSD={sc['gsd']:.1f}\n"
            f"Init: 91% SO$_4$ + 9% org seed, no H$_2$SO$_4$ gas\n"
            f"Gc_org=[{gc_str}] µg/m³\n"
            f"C*=[0.01, 0.1, 1, 10, 100, 1000] µg/m³")


def _add_conditions(ax, sc):
    """Add a conditions text box to the upper-right of *ax*."""
    ax.text(0.97, 0.97, _conditions_text(sc),
            transform=ax.transAxes, fontsize=8,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='wheat', alpha=0.8))


def _make_legend(fig):
    """Add a single top-right legend with proxy Line2D handles."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], **LS_FORTRAN, label='Fortran'),
        Line2D([0], [0], **LS_SEQ, label='Sequential'),
        Line2D([0], [0], **LS_COUP, label='Coupled'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=11,
               framealpha=0.9)


# =========================================================================
# rcParams for presentation quality
# =========================================================================
def _apply_rcparams():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
    })


# =========================================================================
# Simulation
# =========================================================================

def run_python_solver(label, solver, use_ppm=False, verbose=True):
    """Run one SOA-only scenario with the specified solver.

    Returns dict with Nk (25,NBINS), Mk (25,NBINS,ICOMP),
    Gc (25,N_GAS_SPECIES), Nk_full (NSTEPS+1,NBINS), wall_time_s.
    """
    sc = SCENARIOS[label]
    xk = _make_xk()
    temp = jnp.float64(sc['temp'])
    pres = jnp.float64(sc['pres'])
    rh = jnp.float64(sc['rh'])
    alpha_j = jnp.float64(ALPHA)
    boxvol = jnp.float64(BOXVOL)
    dt = jnp.float64(DT)

    Nk, Mk, Gc = make_initial_state(sc)

    @jax.jit
    def step(Nk, Mk, Gc):
        Nk, Mk, Gc = soa_condensation_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha_j, dt,
            use_ppm=use_ppm, solver=solver,
        )
        Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)
        return Nk, Mk, Gc

    # JIT warmup
    _ = step(Nk, Mk, Gc)

    # Storage: hourly (25 snapshots) + full-resolution Nk (1441 snapshots)
    Nk_hourly = np.zeros((NHOURS + 1, NBINS))
    Mk_hourly = np.zeros((NHOURS + 1, NBINS, ICOMP))
    Gc_hourly = np.zeros((NHOURS + 1, N_GAS_SPECIES))
    Nk_full = np.zeros((NSTEPS + 1, NBINS))

    Nk_hourly[0] = np.array(Nk)
    Mk_hourly[0] = np.array(Mk)
    Gc_hourly[0] = np.array(Gc)
    Nk_full[0] = np.array(Nk)

    t0 = time.time()
    for i in range(NSTEPS):
        Nk, Mk, Gc = step(Nk, Mk, Gc)
        Nk_full[i + 1] = np.array(Nk)
        if (i + 1) % 60 == 0:
            hr = (i + 1) // 60
            Nk_hourly[hr] = np.array(Nk)
            Mk_hourly[hr] = np.array(Mk)
            Gc_hourly[hr] = np.array(Gc)
    wall_time = time.time() - t0

    if verbose:
        N_final = float(jnp.sum(Nk)) / BOXVOL
        print(f"    {solver:12s}  {label}: {wall_time:.2f}s, "
              f"N_total={N_final:.2e}/cm3")

    return {
        'Nk': Nk_hourly,
        'Mk': Mk_hourly,
        'Gc': Gc_hourly,
        'Nk_full': Nk_full,
        'wall_time_s': wall_time,
    }


def run_all(verbose=True):
    """Run both Python solvers for all scenarios, save NPZ files."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print(f"SOA Solver Comparison — {NHOURS}h, {NBINS} bins")
    print("  Solvers: sequential (use_ppm=False), coupled (use_ppm=False)")
    print("  Reference: Fortran soacond.f (top-hat)")
    print("=" * 72)

    for label in SCENARIO_ORDER:
        sc = SCENARIOS[label]
        print(f"\n  Scenario {sc['label']}")

        seq = run_python_solver(label, 'sequential', use_ppm=False, verbose=verbose)
        coup = run_python_solver(label, 'coupled', use_ppm=False, verbose=verbose)

        fname = os.path.join(RESULTS_DIR,
                             f'soa_comparison_{label}_{NBINS}bin.npz')
        np.savez_compressed(
            fname,
            seq_Nk=seq['Nk'], seq_Mk=seq['Mk'], seq_Gc=seq['Gc'],
            seq_Nk_full=seq['Nk_full'],
            seq_wall_time=seq['wall_time_s'],
            coup_Nk=coup['Nk'], coup_Mk=coup['Mk'], coup_Gc=coup['Gc'],
            coup_Nk_full=coup['Nk_full'],
            coup_wall_time=coup['wall_time_s'],
        )

    print("\n" + "-" * 72)
    print(f"Results saved to {RESULTS_DIR}")


# =========================================================================
# Data loaders
# =========================================================================

def _load_fortran_minute_Nk(label):
    """Load minute-resolution Nk from Fortran CSV files.

    Returns array (NSTEPS+1, NBINS) or None if files don't exist.
    """
    # Check first file
    f0 = os.path.join(FORTRAN_DIR, f'{label}_soa_min0000_Nk.csv')
    if not os.path.exists(f0):
        return None

    Nk_full = np.zeros((NSTEPS + 1, NBINS))
    for i in range(NSTEPS + 1):
        fname = os.path.join(FORTRAN_DIR,
                             f'{label}_soa_min{i:04d}_Nk.csv')
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"Missing Fortran minute output: {fname}. "
                "Run: cd tomas_fortran && make benchmark_soa && "
                "./benchmark_soa.exe")
        Nk_full[i] = _load_fortran_csv(fname)
    return Nk_full


def _load_results(label):
    """Load Fortran + both Python solver results.

    Returns (fort, seq, coup) dicts with keys Nk, Mk, Gc[, Nk_full,
    wall_time_s].
    """
    # Fortran (hourly + minute Nk)
    fort = load_fortran_scenario(label)
    fort['Nk_full'] = _load_fortran_minute_Nk(label)

    # Python
    fname = os.path.join(RESULTS_DIR,
                         f'soa_comparison_{label}_{NBINS}bin.npz')
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Missing: {fname}. "
            "Run: python -m benchmarks.python.compare_soa_solvers --run")
    d = dict(np.load(fname, allow_pickle=True))

    seq = {
        'Nk': d['seq_Nk'],
        'Mk': d['seq_Mk'],
        'Gc': d['seq_Gc'],
        'Nk_full': d.get('seq_Nk_full', None),
        'wall_time_s': float(d['seq_wall_time']),
    }
    coup = {
        'Nk': d['coup_Nk'],
        'Mk': d['coup_Mk'],
        'Gc': d['coup_Gc'],
        'Nk_full': d.get('coup_Nk_full', None),
        'wall_time_s': float(d['coup_wall_time']),
    }
    return fort, seq, coup


# =========================================================================
# Derived-quantity helpers
# =========================================================================

def _dN_dlogDp(Nk_snap, dlogDp):
    """dN/dlogDp [#/cm3]."""
    return Nk_snap / BOXVOL / dlogDp


def _dM_dlogDp(Mk_snap, dlogDp):
    """dM_dry/dlogDp [kg/cm3].  Dry mass = species 0..SRTH2O-1."""
    M_dry = np.sum(Mk_snap[:, :SRTH2O], axis=1)
    return M_dry / BOXVOL / dlogDp


# =========================================================================
# Figure 1: dN/dlogDp (log-log) at 0, 6, 12, 24 h
# =========================================================================

def plot_fig1_sizedist_log():
    import matplotlib.pyplot as plt
    _apply_rcparams()

    xk_np = _make_xk_np()
    dp = _dp_midpoints(xk_np) * 1e9
    dlogDp = _dlogDp(xk_np)
    times = [0, 6, 12, 24]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey='row')
    fig.suptitle('Size Distribution (dN/dlogDp) — log-log',
                 fontsize=14, fontweight='bold', y=0.98)

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        fort, seq, coup = _load_results(label)

        for col, hr in enumerate(times):
            ax = axes[row, col]

            # Fortran (thick gray background)
            dN_f = _dN_dlogDp(fort['Nk'][hr], dlogDp)
            ax.loglog(dp, dN_f, **LS_FORTRAN)

            # Sequential
            dN_s = _dN_dlogDp(seq['Nk'][hr], dlogDp)
            ax.loglog(dp, dN_s, **LS_SEQ)

            # Coupled
            dN_c = _dN_dlogDp(coup['Nk'][hr], dlogDp)
            ax.loglog(dp, dN_c, **LS_COUP)

            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\ndN/dlogDp [#/cm³]', fontsize=9)
                _add_conditions(ax, sc)
            if row == 1:
                ax.set_xlabel('Dp [nm]')
            ax.set_xlim(1, 2e4)

    _make_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    return fig


# =========================================================================
# Figure 2: dN/dlogDp (semilog-x, linear y)
# =========================================================================

def plot_fig2_sizedist_linear():
    import matplotlib.pyplot as plt
    _apply_rcparams()

    xk_np = _make_xk_np()
    dp = _dp_midpoints(xk_np) * 1e9
    dlogDp = _dlogDp(xk_np)
    times = [0, 6, 12, 24]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey='row')
    fig.suptitle('Size Distribution (dN/dlogDp) — linear y',
                 fontsize=14, fontweight='bold', y=0.98)

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        fort, seq, coup = _load_results(label)

        for col, hr in enumerate(times):
            ax = axes[row, col]

            dN_f = _dN_dlogDp(fort['Nk'][hr], dlogDp)
            ax.plot(dp, dN_f, **LS_FORTRAN)

            dN_s = _dN_dlogDp(seq['Nk'][hr], dlogDp)
            ax.plot(dp, dN_s, **LS_SEQ)

            dN_c = _dN_dlogDp(coup['Nk'][hr], dlogDp)
            ax.plot(dp, dN_c, **LS_COUP)

            ax.set_xscale('log')
            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\ndN/dlogDp [#/cm³]', fontsize=9)
                _add_conditions(ax, sc)
            if row == 1:
                ax.set_xlabel('Dp [nm]')
            ax.set_xlim(1, 2e4)

    _make_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    return fig


# =========================================================================
# Figure 3: dM_dry/dlogDp (log-log)
# =========================================================================

def plot_fig3_massdist_log():
    import matplotlib.pyplot as plt
    _apply_rcparams()

    xk_np = _make_xk_np()
    dp = _dp_midpoints(xk_np) * 1e9
    dlogDp = _dlogDp(xk_np)
    times = [0, 6, 12, 24]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey='row')
    fig.suptitle('Mass Distribution (dM_dry/dlogDp) — log-log',
                 fontsize=14, fontweight='bold', y=0.98)

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        fort, seq, coup = _load_results(label)

        for col, hr in enumerate(times):
            ax = axes[row, col]

            # Fortran dry mass: species 0..41 (Fortran has 43 species; H2O=42 excluded)
            Mk_f_dry = np.sum(fort['Mk'][hr, :, :42], axis=1)
            dM_f = Mk_f_dry / BOXVOL / dlogDp
            ax.loglog(dp, np.maximum(dM_f, 1e-30), **LS_FORTRAN)

            dM_s = _dM_dlogDp(seq['Mk'][hr], dlogDp)
            ax.loglog(dp, np.maximum(dM_s, 1e-30), **LS_SEQ)

            dM_c = _dM_dlogDp(coup['Mk'][hr], dlogDp)
            ax.loglog(dp, np.maximum(dM_c, 1e-30), **LS_COUP)

            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\ndM_dry/dlogDp [kg/cm³]',
                              fontsize=9)
                _add_conditions(ax, sc)
            if row == 1:
                ax.set_xlabel('Dp [nm]')
            ax.set_xlim(1, 2e4)

    _make_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    return fig


# =========================================================================
# Figure 4: dM_dry/dlogDp (semilog-x, linear y)
# =========================================================================

def plot_fig4_massdist_linear():
    import matplotlib.pyplot as plt
    _apply_rcparams()

    xk_np = _make_xk_np()
    dp = _dp_midpoints(xk_np) * 1e9
    dlogDp = _dlogDp(xk_np)
    times = [0, 6, 12, 24]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey='row')
    fig.suptitle('Mass Distribution (dM_dry/dlogDp) — linear y',
                 fontsize=14, fontweight='bold', y=0.98)

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        fort, seq, coup = _load_results(label)

        for col, hr in enumerate(times):
            ax = axes[row, col]

            Mk_f_dry = np.sum(fort['Mk'][hr, :, :42], axis=1)
            dM_f = Mk_f_dry / BOXVOL / dlogDp
            ax.plot(dp, dM_f, **LS_FORTRAN)

            dM_s = _dM_dlogDp(seq['Mk'][hr], dlogDp)
            ax.plot(dp, dM_s, **LS_SEQ)

            dM_c = _dM_dlogDp(coup['Mk'][hr], dlogDp)
            ax.plot(dp, dM_c, **LS_COUP)

            ax.set_xscale('log')
            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\ndM_dry/dlogDp [kg/cm³]',
                              fontsize=9)
                _add_conditions(ax, sc)
            if row == 1:
                ax.set_xlabel('Dp [nm]')
            ax.set_xlim(1, 2e4)

    _make_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    return fig


# =========================================================================
# Figure 5: VBS gas evolution (6 panels x 2 scenarios)
# =========================================================================

def plot_fig5_gas_evolution():
    import matplotlib.pyplot as plt
    _apply_rcparams()

    hours = np.arange(NHOURS + 1)
    cstar_labels = ['0.01', '0.1', '1', '10', '100', '1000']

    fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharex=True)
    fig.suptitle('VBS Gas-Phase Evolution — Gc(t) per C* bin',
                 fontsize=14, fontweight='bold', y=0.98)

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        fort, seq, coup = _load_results(label)

        for col in range(N_VBS):
            ax = axes[row, col]
            # Fortran index: species 1..6 in 43-element Gc
            f_idx = 1 + col
            # Python index
            p_idx = SRTORG1 + col

            ax.semilogy(hours, fort['Gc'][:, f_idx], **LS_FORTRAN)
            ax.semilogy(hours, seq['Gc'][:, p_idx], **LS_SEQ)
            ax.semilogy(hours, coup['Gc'][:, p_idx], **LS_COUP)

            if row == 0:
                ax.set_title(f"C*={cstar_labels[col]} µg/m³", fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nGc [kg/cell]', fontsize=9)
                _add_conditions(ax, sc)
            if row == 1:
                ax.set_xlabel('Hours')

    _make_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    return fig


# =========================================================================
# Figure 6: VBS particle mass grouped bars
# =========================================================================

def plot_fig6_vbs_particle():
    import matplotlib.pyplot as plt
    _apply_rcparams()

    times = [0, 6, 12, 24]
    cstar_labels = ['0.01', '0.1', '1', '10', '100', '1000']
    x = np.arange(N_VBS)
    w = 0.25

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle('VBS Particle-Phase Mass per Bin',
                 fontsize=14, fontweight='bold', y=0.98)

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        fort, seq, coup = _load_results(label)

        for col, hr in enumerate(times):
            ax = axes[row, col]

            # Fortran
            fort_org = np.array([np.sum(fort['Mk'][hr, :, 1 + j])
                                 for j in range(N_VBS)])
            ax.bar(x - w, fort_org, w, color='#bbbbbb', edgecolor='#666666',
                   label='Fortran')

            # Sequential
            seq_org = np.array([np.sum(seq['Mk'][hr, :, SRTORG1 + j])
                                for j in range(N_VBS)])
            ax.bar(x, seq_org, w, color='#1f77b4', edgecolor='#0d4a7a',
                   label='Sequential')

            # Coupled
            coup_org = np.array([np.sum(coup['Mk'][hr, :, SRTORG1 + j])
                                 for j in range(N_VBS)])
            ax.bar(x + w, coup_org, w, color='#d62728', edgecolor='#8b1a1a',
                   label='Coupled')

            ax.set_xticks(x)
            ax.set_xticklabels(cstar_labels, fontsize=8)
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nMk_org [kg/cell]', fontsize=9)
                _add_conditions(ax, sc)
            if row == 1:
                ax.set_xlabel('C* [µg/m³]')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# =========================================================================
# Figure 7: Banana plots (pcolormesh)
# =========================================================================

def plot_fig7_banana():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    _apply_rcparams()

    xk_np = _make_xk_np()
    dlogDp = _dlogDp(xk_np)

    # Dp edges for pcolormesh [nm]
    dp_edges = np.cbrt(xk_np / DENS_INIT * (6.0 / PI)) * 1e9

    solver_names = ['Fortran', 'Sequential', 'Coupled']

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('Banana Plots — dN/dlogDp evolution (60 s resolution)',
                 fontsize=14, fontweight='bold', y=0.98)

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        fort, seq, coup = _load_results(label)

        # Minute-resolution time edges (shared by all solvers that have it)
        t_min_h = np.arange(NSTEPS + 1) * (DT / 3600.0)
        te_min = np.concatenate([t_min_h - DT / 7200.0,
                                 [t_min_h[-1] + DT / 7200.0]])
        te_hourly = np.arange(NHOURS + 2) - 0.5

        # Build per-solver dN arrays + time edges
        def _pick_full(data):
            if data.get('Nk_full') is not None:
                return (data['Nk_full'] / BOXVOL / dlogDp[np.newaxis, :],
                        te_min)
            return (data['Nk'] / BOXVOL / dlogDp[np.newaxis, :],
                    te_hourly)

        dN_fort, te_fort = _pick_full(fort)
        dN_seq, te_seq = _pick_full(seq)
        dN_coup, te_coup = _pick_full(coup)

        # Shared vmin/vmax from Fortran reference
        vmax = np.max(dN_fort[dN_fort > 0]) if np.any(dN_fort > 0) else 1.0
        vmin = vmax * 1e-4

        all_dN = [dN_fort, dN_seq, dN_coup]
        all_te = [te_fort, te_seq, te_coup]

        for col, (dN, te, sname) in enumerate(
                zip(all_dN, all_te, solver_names)):
            ax = axes[row, col]
            dN_safe = np.maximum(dN, vmin * 0.1)

            pcm = ax.pcolormesh(
                te, dp_edges, dN_safe.T,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cmap='inferno', shading='flat')

            ax.set_yscale('log')
            ax.set_ylim(dp_edges[0], dp_edges[-1])
            ax.set_xlim(0, NHOURS)

            if row == 0:
                ax.set_title(sname, fontsize=12)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nDp [nm]', fontsize=9)
            if row == 1:
                ax.set_xlabel('Time [h]')

        # Shared colorbar per row
        cbar = fig.colorbar(pcm, ax=axes[row, :].tolist(), shrink=0.8,
                            pad=0.02)
        cbar.set_label('dN/dlogDp [#/cm³]', fontsize=9)

    fig.subplots_adjust(left=0.05, right=0.92, top=0.93, bottom=0.08,
                        wspace=0.15, hspace=0.25)
    return fig


# =========================================================================
# Figure 8: Total N(t) and M_dry(t) timeseries
# =========================================================================

def plot_fig8_totals():
    import matplotlib.pyplot as plt
    _apply_rcparams()

    hours = np.arange(NHOURS + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Total Number & Dry Mass Timeseries',
                 fontsize=14, fontweight='bold', y=0.98)

    col_labels = ['N_total [#/cm³]', 'M_dry [kg/cell]']

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        fort, seq, coup = _load_results(label)

        # N_total
        N_f = np.sum(fort['Nk'], axis=1) / BOXVOL
        N_s = np.sum(seq['Nk'], axis=1) / BOXVOL
        N_c = np.sum(coup['Nk'], axis=1) / BOXVOL

        ax = axes[row, 0]
        ax.semilogy(hours, N_f, **LS_FORTRAN)
        ax.semilogy(hours, N_s, **LS_SEQ)
        ax.semilogy(hours, N_c, **LS_COUP)
        if row == 0:
            ax.set_title(col_labels[0], fontsize=11)
        ax.set_ylabel(f'{sc["label"]}', fontsize=9)
        if row == 1:
            ax.set_xlabel('Hours')

        # M_dry
        M_f = np.sum(fort['Mk'][:, :, :42], axis=(1, 2))
        M_s = np.sum(seq['Mk'][:, :, :SRTH2O], axis=(1, 2))
        M_c = np.sum(coup['Mk'][:, :, :SRTH2O], axis=(1, 2))

        ax = axes[row, 1]
        ax.semilogy(hours, M_f, **LS_FORTRAN)
        ax.semilogy(hours, M_s, **LS_SEQ)
        ax.semilogy(hours, M_c, **LS_COUP)
        if row == 0:
            ax.set_title(col_labels[1], fontsize=11)
        if row == 1:
            ax.set_xlabel('Hours')

        # Wall-time annotation
        ax.text(0.03, 0.03,
                f"Wall: seq={seq['wall_time_s']:.2f}s, "
                f"coup={coup['wall_time_s']:.2f}s",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='lightyellow', alpha=0.8))

        # Conditions text on left panel
        if row == 0 or row == 1:
            _add_conditions(axes[row, 0], sc)

    _make_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    return fig


# =========================================================================
# Figure 9: Relative error vs Fortran at 4 snapshots
# =========================================================================

def plot_fig9_relerror():
    import matplotlib.pyplot as plt
    _apply_rcparams()

    xk_np = _make_xk_np()
    dp = _dp_midpoints(xk_np) * 1e9
    dlogDp = _dlogDp(xk_np)
    times = [0, 6, 12, 24]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    fig.suptitle('Per-Bin Relative Error vs Fortran — dN/dlogDp',
                 fontsize=14, fontweight='bold', y=0.98)

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        fort, seq, coup = _load_results(label)

        for col, hr in enumerate(times):
            ax = axes[row, col]

            dN_f = _dN_dlogDp(fort['Nk'][hr], dlogDp)
            dN_s = _dN_dlogDp(seq['Nk'][hr], dlogDp)
            dN_c = _dN_dlogDp(coup['Nk'][hr], dlogDp)

            # Threshold: only where Fortran has significant signal
            threshold = np.max(dN_f) * 1e-4
            denom = np.maximum(np.abs(dN_f), threshold)
            active = dN_f > threshold

            err_s = np.full(NBINS, np.nan)
            err_c = np.full(NBINS, np.nan)
            err_s[active] = (dN_s[active] - dN_f[active]) / denom[active]
            err_c[active] = (dN_c[active] - dN_f[active]) / denom[active]

            ax.semilogx(dp, err_s * 100, color=LS_SEQ['color'],
                        ls='--', lw=2, marker='o', ms=3, label='Sequential')
            ax.semilogx(dp, err_c * 100, color=LS_COUP['color'],
                        ls='-', lw=2, marker='s', ms=3, label='Coupled')

            # Reference lines
            ax.axhline(0, color='k', ls='-', lw=0.5)
            for pct in [10, -10, 50, -50]:
                ax.axhline(pct, color='gray', ls=':', lw=0.5, alpha=0.5)

            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nRelative error [%]', fontsize=9)
                _add_conditions(ax, sc)
            if row == 1:
                ax.set_xlabel('Dp [nm]')
            ax.set_xlim(1, 2e4)

    # Legend for error plots (sequential + coupled only)
    from matplotlib.lines import Line2D
    err_handles = [
        Line2D([0], [0], color=LS_SEQ['color'], ls='--', lw=2,
               marker='o', ms=3, label='Sequential'),
        Line2D([0], [0], color=LS_COUP['color'], ls='-', lw=2,
               marker='s', ms=3, label='Coupled'),
    ]
    fig.legend(handles=err_handles, loc='upper right', fontsize=11,
               framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    return fig


# =========================================================================
# Main
# =========================================================================

def plot_all():
    """Generate all 9 figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    figures = [
        ('fig1_sizedist_log.png',     plot_fig1_sizedist_log),
        ('fig2_sizedist_linear.png',  plot_fig2_sizedist_linear),
        ('fig3_massdist_log.png',     plot_fig3_massdist_log),
        ('fig4_massdist_linear.png',  plot_fig4_massdist_linear),
        ('fig5_gas_evolution.png',    plot_fig5_gas_evolution),
        ('fig6_vbs_particle.png',     plot_fig6_vbs_particle),
        ('fig7_banana.png',           plot_fig7_banana),
        ('fig8_totals.png',           plot_fig8_totals),
        ('fig9_relerror.png',         plot_fig9_relerror),
    ]

    for fname, plot_fn in figures:
        print(f"  Plotting {fname}...")
        fig = plot_fn()
        fig.savefig(os.path.join(RESULTS_DIR, fname),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"\nAll 9 figures saved to {RESULTS_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SOA Solver Comparison: Fortran vs Sequential vs Coupled')
    parser.add_argument('--run', action='store_true',
                        help='Run both Python solvers and save NPZ')
    parser.add_argument('--plot-only', action='store_true',
                        help='Regenerate plots from saved NPZ data')
    args = parser.parse_args()

    if args.run:
        run_all()

    if args.plot_only or args.run:
        plot_all()

    if not args.run and not args.plot_only:
        parser.print_help()
