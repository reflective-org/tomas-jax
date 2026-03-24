"""SOA/VBS benchmark: Fortran soacond.f (top-hat) vs JAX (PPM).

SOA-only benchmark: organic VBS condensation onto inert sulfate particles.
No H2SO4 condensation — only soa_condensation_step() + mnfix_jax().

Both Fortran and Python use the same 36-bin legacy grid (Mo=1.5625e-23 kg,
mass-doubling), eliminating grid mismatch as a variable.

The Fortran uses tmcond (top-hat redistribution) per VBS species.
The JAX uses PPM (Piecewise Parabolic Method) redistribution.

Scenarios:
    A. Pure condensation (288K, 1atm, high gas, low C*)
    B. Mixed condensation/evaporation (270K, 800hPa, uniform gas)
    C. Warm evaporative (310K, 1atm, low gas, high C*)

All scenarios seed particles with 10% sulfate mass as organic (VBS bin 1)
to enable Raoult-law partitioning in Fortran.

Figures (8):
    1. VBS gas-phase evolution (6 panels per scenario)
    2. VBS particle-phase distribution (bar chart at 4 times)
    3. Size distribution evolution (dN/dlogDp at 4 snapshots)
    4. Size-resolved organic vs inorganic mass
    5. Total mass/number timeseries
    6. Mass conservation check
    7. Relative error summary (Python vs Fortran)
    8. Condensation sink and Kelvin diagnostics

Usage::

    python -m benchmarks.python.benchmark_soa --run
    python -m benchmarks.python.benchmark_soa --plot-only
"""
import os
import re
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTORG1, SRTORGLAST, SRTNH4, SRTH2O,
    MW_H2SO4, KB, AVOGADRO, PI, R_GAS,
    NBINS_LEGACY,
)
from tomas_jax.physics.soa_condensation import soa_condensation_step
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics.vbs_config import DEFAULT_VBS_CONFIG

# =========================================================================
# Constants
# =========================================================================
NBINS = NBINS_LEGACY  # 36 bins — match Fortran exactly
MO = 1.0e-21 * 2.0 ** (-6)  # = 1.5625e-23 kg, exact Fortran initbounds.f
DENS_INIT = 1770.0   # kg/m3 (pure sulfate)
BOXVOL = 1.0e6       # cm3
DT = 10.0            # seconds per step — match Fortran dt=10s
ALPHA = 1.0
N_VBS = 6            # 6 VBS bins
NHOURS = 24
NSTEPS = NHOURS * 3600 // 10  # 8640 steps at dt=10s = 24h

FORTRAN_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                           'tomas_fortran', 'output', 'soa')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results',
                           'soa_benchmark')


def _make_xk():
    """36-bin grid matching Fortran initbounds.f exactly."""
    return jnp.array([MO * 2.0 ** k for k in range(NBINS + 1)])


def _make_xk_np():
    """36-bin grid as numpy array."""
    return np.array([MO * 2.0 ** k for k in range(NBINS + 1)])


# =========================================================================
# Scenario definitions (must match Fortran harness exactly)
# =========================================================================
SCENARIOS = {
    'sA': {
        'label': 'A: Pure condensation (288K)',
        'temp': 288.0, 'pres': 101325.0, 'rh': 0.5,
        'n_total': 1e4, 'gmd': 50e-9, 'gsd': 1.6,
        # VBS gas concentrations [kg/cell] — match Fortran harness
        # µg/m³: [0.1, 0.1, 1, 3, 10, 20] (more mass in volatile bins)
        'Gc_org': [1e-10, 1e-10, 1e-9, 3e-9, 1e-8, 2e-8],
    },
    'sB': {
        'label': 'B: Mixed cond/evap (270K)',
        'temp': 270.0, 'pres': 80000.0, 'rh': 0.3,
        'n_total': 5e3, 'gmd': 80e-9, 'gsd': 1.5,
        # µg/m³: [0.1, 0.1, 1, 3, 10, 20]
        'Gc_org': [1e-10, 1e-10, 1e-9, 3e-9, 1e-8, 2e-8],
    },
    'sC': {
        'label': 'C: Warm evaporative (310K)',
        'temp': 310.0, 'pres': 101325.0, 'rh': 0.6,
        'n_total': 2e4, 'gmd': 30e-9, 'gsd': 1.8,
        'Gc_org': [2e-9, 2e-9, 2e-9, 2e-9, 2e-9, 2e-9],
    },
}
SCENARIO_ORDER = ['sA', 'sB', 'sC']


# =========================================================================
# Fortran CSV loader (with missing-E fix)
# =========================================================================
_FORTRAN_RE = re.compile(r'(\d)([-+])(\d)')

def _fix_fortran_float(s):
    return _FORTRAN_RE.sub(r'\1E\2\3', s.strip())

def _load_fortran_csv(fname, delimiter=None):
    with open(fname) as f:
        text = _fix_fortran_float(f.read())
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if delimiter:
        return np.array([[float(x) for x in l.split(delimiter)] for l in lines])
    else:
        return np.array([float(l) for l in lines])


def load_fortran_scenario(label):
    """Load all hourly Fortran data for one scenario."""
    Nk_hourly = []
    Mk_hourly = []
    Gc_hourly = []
    for hr in range(NHOURS + 1):
        nk_f = os.path.join(FORTRAN_DIR, f'{label}_soa_hour{hr:02d}_Nk.csv')
        mk_f = os.path.join(FORTRAN_DIR, f'{label}_soa_hour{hr:02d}_Mk.csv')
        gc_f = os.path.join(FORTRAN_DIR, f'{label}_soa_hour{hr:02d}_Gc.csv')
        if not os.path.exists(nk_f):
            raise FileNotFoundError(
                f"Missing Fortran output: {nk_f}. "
                "Run: cd tomas_fortran && make benchmark_soa && ./benchmark_soa.exe")
        Nk_hourly.append(_load_fortran_csv(nk_f))
        Mk_hourly.append(_load_fortran_csv(mk_f, delimiter=','))
        Gc_hourly.append(_load_fortran_csv(gc_f))
    return {
        'Nk': np.array(Nk_hourly),   # (25, 36)
        'Mk': np.array(Mk_hourly),   # (25, 36, 44)
        'Gc': np.array(Gc_hourly),    # (25, 43)
    }


# =========================================================================
# Grid utilities
# =========================================================================

def _dp_midpoints(xk_np, rho=DENS_INIT):
    """Diameter midpoints [m] from bin boundaries [kg]."""
    m_mid = np.sqrt(xk_np[:-1] * xk_np[1:])
    return np.cbrt(m_mid / rho * (6.0 / PI))

def _dlogDp(xk_np, rho=DENS_INIT):
    """dlog10(Dp) width per bin."""
    dp_edges = np.cbrt(xk_np / rho * (6.0 / PI))
    return np.log10(dp_edges[1:] / dp_edges[:-1])


# =========================================================================
# Python simulation (SOA-only, 36-bin, PPM)
# =========================================================================

def make_initial_state(sc):
    """Create initial Nk, Mk, Gc matching Fortran harness exactly."""
    xk_np = _make_xk_np()

    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))

    n_total = sc['n_total']
    gmd_um = sc['gmd'] * 1e6
    gsd = sc['gsd']

    # Lognormal initialization (same formula as Fortran)
    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dk = np.sqrt(Dl * Dh)
        np_init = ((n_total * BOXVOL)
                   / (np.sqrt(2.0 * PI) * Dk * np.log(gsd))
                   * np.exp(-(np.log(Dk / gmd_um) ** 2
                              / (2.0 * np.log(gsd) ** 2)))
                   * (Dh - Dl))
        Nk[k] = np_init
        Mk[k, SRTSO4] = np_init * np.sqrt(xk_np[k] * xk_np[k + 1])

    # Seed 10% organic mass (VBS bin 1) for Raoult partitioning
    Mk[:, SRTORG1] = 0.1 * Mk[:, SRTSO4]

    # Neps preprocessing (match Fortran)
    for k in range(NBINS):
        if Nk[k] < 1e-3:
            Nk[k] = 1e-3
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = 1e-3 * 1.4 * xk_np[k]

    # Gas phase — NO H2SO4 gas
    Gc = np.zeros(N_GAS_SPECIES)
    # Gc[SRTSO4] = 0  (no H2SO4)
    for j in range(N_VBS):
        Gc[SRTORG1 + j] = sc['Gc_org'][j]

    return jnp.array(Nk), jnp.array(Mk), jnp.array(Gc)


def run_scenario(label, verbose=True):
    """Run one SOA-only benchmark scenario (36-bin, PPM)."""
    sc = SCENARIOS[label]
    xk = _make_xk()
    temp = jnp.float64(sc['temp'])
    pres = jnp.float64(sc['pres'])
    rh = jnp.float64(sc['rh'])
    alpha = jnp.float64(ALPHA)
    boxvol = jnp.float64(BOXVOL)
    dt = jnp.float64(DT)

    Nk, Mk, Gc = make_initial_state(sc)

    # JIT-compile the SOA step + MNFIX
    @jax.jit
    def soa_step(Nk, Mk, Gc):
        # SOA condensation with PPM redistribution
        Nk, Mk, Gc = soa_condensation_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
            redistribution='tfl',
        )
        # MNFIX after SOA step
        Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)
        return Nk, Mk, Gc

    # Warmup JIT
    _ = soa_step(Nk, Mk, Gc)

    # Storage (hourly: 25 snapshots including t=0)
    Nk_hourly = np.zeros((NHOURS + 1, NBINS))
    Mk_hourly = np.zeros((NHOURS + 1, NBINS, ICOMP))
    Gc_hourly = np.zeros((NHOURS + 1, N_GAS_SPECIES))

    # Record initial state
    Nk_hourly[0] = np.array(Nk)
    Mk_hourly[0] = np.array(Mk)
    Gc_hourly[0] = np.array(Gc)

    t0 = time.time()

    for i in range(NSTEPS):
        Nk, Mk, Gc = soa_step(Nk, Mk, Gc)

        steps_per_hour = 3600 // int(DT)  # 360 at dt=10s
        if (i + 1) % steps_per_hour == 0:
            hr = (i + 1) // steps_per_hour
            Nk_hourly[hr] = np.array(Nk)
            Mk_hourly[hr] = np.array(Mk)
            Gc_hourly[hr] = np.array(Gc)

    wall_time = time.time() - t0

    if verbose:
        print(f"  {label}: {wall_time:.2f}s, N_total={float(jnp.sum(Nk))/BOXVOL:.2e}/cm3")

    return {
        'Nk': Nk_hourly,
        'Mk': Mk_hourly,
        'Gc': Gc_hourly,
        'wall_time_s': wall_time,
        'nbins': NBINS,
    }


def run_all(verbose=True):
    """Run all scenarios and save NPZ."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print(f"SOA/VBS Benchmark — {NHOURS}h, {NBINS} bins (Fortran-matching grid)")
    print("  SOA-only: soa_condensation_step(PPM) + mnfix_jax")
    print("  No H2SO4 condensation")
    print("=" * 72)

    for label in SCENARIO_ORDER:
        res = run_scenario(label, verbose)
        fname = f"soa_{label}_{NBINS}bin.npz"
        np.savez_compressed(
            os.path.join(RESULTS_DIR, fname),
            **{k: v for k, v in res.items()
               if isinstance(v, (np.ndarray, np.floating, float, int))},
        )

    print("-" * 72)
    print(f"Results saved to {RESULTS_DIR}")


# =========================================================================
# Plotting helpers
# =========================================================================

def _load_python(label):
    fname = os.path.join(RESULTS_DIR, f"soa_{label}_{NBINS}bin.npz")
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Missing: {fname}. Run: python -m benchmarks.python.benchmark_soa --run")
    return dict(np.load(fname, allow_pickle=True))


# =========================================================================
# Figures
# =========================================================================

def plot_fig1_gas_evolution():
    """Fig 1: VBS gas-phase evolution (6 panels x 3 scenarios)."""
    import matplotlib.pyplot as plt

    cstar_labels = ['0.01', '0.1', '1', '10', '100', '1000']
    hours = np.arange(NHOURS + 1)

    fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True)
    fig.suptitle('Fig 1: VBS Gas-Phase Evolution — Fortran (top-hat) vs Python (PPM)',
                 fontsize=14, fontweight='bold')

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            fort = None

        try:
            py = _load_python(label)
        except FileNotFoundError:
            py = None

        for col in range(N_VBS):
            ax = axes[row, col]
            # Fortran: Gc has 43 elements (icomp-1), species srtorg1=2 -> 0-indexed = 1
            f_idx = 1 + col
            p_idx = SRTORG1 + col

            if fort is not None:
                ax.semilogy(hours, fort['Gc'][:, f_idx],
                            'k-', lw=2, label='Fortran (top-hat)')
            if py is not None:
                ax.semilogy(hours, py['Gc'][:, p_idx],
                            'C0--', lw=1.5, label='Python (PPM)')

            if row == 0:
                ax.set_title(f"C*={cstar_labels[col]} µg/m³", fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nGc [kg/cell]', fontsize=9)
            if row == 2:
                ax.set_xlabel('Hours')
            ax.grid(True, alpha=0.3, which='both')
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_fig2_particle_vbs():
    """Fig 2: VBS particle-phase distribution at 0, 6, 12, 24h."""
    import matplotlib.pyplot as plt

    times = [0, 6, 12, 24]
    cstar_labels = ['0.01', '0.1', '1', '10', '100', '1000']
    x = np.arange(N_VBS)
    width = 0.25

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Fig 2: VBS Particle-Phase Mass — Fortran (top-hat) vs Python (PPM)',
                 fontsize=14, fontweight='bold')

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            fort = None

        try:
            py = _load_python(label)
        except FileNotFoundError:
            py = None

        for col, hr in enumerate(times):
            ax = axes[row, col]

            if fort is not None:
                fort_org = np.array([np.sum(fort['Mk'][hr, :, 1 + j])
                                     for j in range(N_VBS)])
                ax.bar(x - width/2, fort_org, width, color='gray',
                       edgecolor='black', label='Fortran')

            if py is not None:
                py_org = np.array([np.sum(py['Mk'][hr, :, SRTORG1 + j])
                                   for j in range(N_VBS)])
                ax.bar(x + width/2, py_org, width, color='C0',
                       edgecolor='navy', label='Python (PPM)')

            ax.set_xticks(x)
            ax.set_xticklabels(cstar_labels, fontsize=8)
            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nMk_org [kg/cell]', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_fig3_sizedist():
    """Fig 3: Size distribution evolution at 4 snapshots."""
    import matplotlib.pyplot as plt

    times = [0, 6, 12, 24]
    xk = _make_xk_np()

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Fig 3: Size Distribution (dN/dlogDp) — Fortran vs Python (same 36-bin grid)',
                 fontsize=14, fontweight='bold')

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            fort = None

        try:
            py = _load_python(label)
        except FileNotFoundError:
            py = None

        dp = _dp_midpoints(xk) * 1e9
        dlogDp = _dlogDp(xk)

        for col, hr in enumerate(times):
            ax = axes[row, col]

            if fort is not None:
                dNdlogDp_f = fort['Nk'][hr] / BOXVOL / dlogDp
                ax.loglog(dp, dNdlogDp_f, 'k-', lw=2, label='Fortran (top-hat)')

            if py is not None:
                dNdlogDp_p = py['Nk'][hr] / BOXVOL / dlogDp
                ax.loglog(dp, dNdlogDp_p, 'C0--', lw=1.5, label='Python (PPM)')

            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\ndN/dlogDp [#/cm³]', fontsize=9)
            if row == 2:
                ax.set_xlabel('Dp [nm]')
            ax.set_xlim(1, 2e4)
            ax.grid(True, alpha=0.3, which='both')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_fig3b_sizedist_relerror():
    """Fig 3b: Per-bin relative error of dN/dlogDp (Python vs Fortran)."""
    import matplotlib.pyplot as plt

    times = [0, 6, 12, 24]
    xk = _make_xk_np()

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Fig 3b: Size Distribution Relative Error — (Python - Fortran) / Fortran',
                 fontsize=14, fontweight='bold')

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            fort = None
        try:
            py = _load_python(label)
        except FileNotFoundError:
            py = None

        dp = _dp_midpoints(xk) * 1e9
        dlogDp = _dlogDp(xk)

        for col, hr in enumerate(times):
            ax = axes[row, col]

            if fort is not None and py is not None:
                dNdlogDp_f = fort['Nk'][hr] / BOXVOL / dlogDp
                dNdlogDp_p = py['Nk'][hr] / BOXVOL / dlogDp

                # Only compute relative error where Fortran has significant particles
                threshold = np.max(dNdlogDp_f) * 1e-4
                active = dNdlogDp_f > threshold
                rel_err = np.full(NBINS, np.nan)
                rel_err[active] = (dNdlogDp_p[active] - dNdlogDp_f[active]) / dNdlogDp_f[active]

                ax.semilogx(dp, rel_err * 100, 'C0o-', lw=1.5, ms=3)
                ax.axhline(0, color='k', ls='--', lw=0.5)

            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nRelative error [%]', fontsize=9)
            if row == 2:
                ax.set_xlabel('Dp [nm]')
            ax.set_xlim(1, 2e4)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fig3c_sizedist_linear():
    """Fig 3c: Size distribution with linear y-axis."""
    import matplotlib.pyplot as plt

    times = [0, 6, 12, 24]
    xk = _make_xk_np()

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Fig 3c: Size Distribution (dN/dlogDp, linear y) — Fortran vs Python (same 36-bin grid)',
                 fontsize=14, fontweight='bold')

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            fort = None
        try:
            py = _load_python(label)
        except FileNotFoundError:
            py = None

        dp = _dp_midpoints(xk) * 1e9
        dlogDp = _dlogDp(xk)

        for col, hr in enumerate(times):
            ax = axes[row, col]

            if fort is not None:
                dNdlogDp_f = fort['Nk'][hr] / BOXVOL / dlogDp
                ax.plot(dp, dNdlogDp_f, 'k-', lw=2, label='Fortran (top-hat)')

            if py is not None:
                dNdlogDp_p = py['Nk'][hr] / BOXVOL / dlogDp
                ax.plot(dp, dNdlogDp_p, 'C0--', lw=1.5, label='Python (PPM)')

            ax.set_xscale('log')
            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\ndN/dlogDp [#/cm³]', fontsize=9)
            if row == 2:
                ax.set_xlabel('Dp [nm]')
            ax.set_xlim(1, 2e4)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_fig4_composition():
    """Fig 4: Size-resolved organic vs inorganic mass at 12h and 24h."""
    import matplotlib.pyplot as plt

    times = [12, 24]
    xk = _make_xk_np()
    dp = _dp_midpoints(xk) * 1e9

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Fig 4: Size-Resolved Composition (SO4 + Organic) — 12h & 24h',
                 fontsize=14, fontweight='bold')

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            fort = None

        try:
            py = _load_python(label)
        except FileNotFoundError:
            py = None

        for col, hr in enumerate(times):
            ax = axes[row, col]

            if fort is not None:
                Mk_so4_f = fort['Mk'][hr, :, 0]
                Mk_org_f = np.sum(fort['Mk'][hr, :, 1:7], axis=1)
                ax.loglog(dp, Mk_so4_f / BOXVOL, 'k-', lw=2, label='SO4 (Fortran)')
                ax.loglog(dp, np.maximum(Mk_org_f, 1e-30) / BOXVOL,
                          'k--', lw=2, label='Org (Fortran)')

            if py is not None:
                Mk_so4_p = py['Mk'][hr, :, SRTSO4]
                Mk_org_p = np.sum(py['Mk'][hr, :, SRTORG1:SRTORG1+N_VBS], axis=1)
                ax.loglog(dp, Mk_so4_p / BOXVOL, 'C0-', lw=1.5,
                          label='SO4 (Python)')
                ax.loglog(dp, np.maximum(Mk_org_p, 1e-30) / BOXVOL,
                          'C0--', lw=1.5, label='Org (Python)')

            if row == 0:
                ax.set_title(f't = {hr}h', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nMk [kg/cm³]', fontsize=9)
            if row == 2:
                ax.set_xlabel('Dp [nm]')
            ax.set_xlim(1, 2e4)
            ax.grid(True, alpha=0.3, which='both')
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_fig5_totals():
    """Fig 5: Total N, M_dry, M_organic, M_sulfate timeseries."""
    import matplotlib.pyplot as plt

    hours = np.arange(NHOURS + 1)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('Fig 5: Total Mass & Number — Fortran vs Python',
                 fontsize=14, fontweight='bold')

    metric_labels = ['N_total [#/cm³]', 'M_dry [kg/cell]',
                     'M_organic [kg/cell]', 'M_sulfate [kg/cell]']

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            fort = None

        try:
            py = _load_python(label)
        except FileNotFoundError:
            py = None

        metrics_f = {}
        metrics_p = {}

        if fort is not None:
            metrics_f['N_total'] = np.sum(fort['Nk'], axis=1) / BOXVOL
            metrics_f['M_dry'] = np.sum(fort['Mk'][:, :, :42], axis=(1, 2))
            metrics_f['M_org'] = np.sum(fort['Mk'][:, :, 1:7], axis=(1, 2))
            metrics_f['M_so4'] = np.sum(fort['Mk'][:, :, 0], axis=1)

        if py is not None:
            metrics_p['N_total'] = np.sum(py['Nk'], axis=1) / BOXVOL
            metrics_p['M_dry'] = np.sum(py['Mk'][:, :, :SRTH2O], axis=(1, 2))
            metrics_p['M_org'] = np.sum(
                py['Mk'][:, :, SRTORG1:SRTORG1+N_VBS], axis=(1, 2))
            metrics_p['M_so4'] = np.sum(py['Mk'][:, :, SRTSO4], axis=1)

        keys = ['N_total', 'M_dry', 'M_org', 'M_so4']
        for col, key in enumerate(keys):
            ax = axes[row, col]
            if fort is not None and key in metrics_f:
                ax.semilogy(hours, metrics_f[key], 'k-', lw=2, label='Fortran')
            if py is not None and key in metrics_p:
                ax.semilogy(hours, metrics_p[key], 'C0--', lw=1.5, label='Python')

            if row == 0:
                ax.set_title(metric_labels[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(sc['label'], fontsize=9)
            if row == 2:
                ax.set_xlabel('Hours')
            ax.grid(True, alpha=0.3, which='both')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_fig6_conservation():
    """Fig 6: Mass conservation (gas + particle total per VBS bin)."""
    import matplotlib.pyplot as plt

    hours = np.arange(NHOURS + 1)
    cstar_labels = ['0.01', '0.1', '1', '10', '100', '1000']

    fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True)
    fig.suptitle('Fig 6: Mass Conservation (gas+particle) — Fractional Error vs Time',
                 fontsize=14, fontweight='bold')

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            fort = None

        try:
            py = _load_python(label)
        except FileNotFoundError:
            py = None

        for col in range(N_VBS):
            ax = axes[row, col]
            spec_f = 1 + col
            spec_p = SRTORG1 + col

            if fort is not None:
                gas_f = fort['Gc'][:, spec_f]
                part_f = np.sum(fort['Mk'][:, :, spec_f], axis=1)
                total_f = gas_f + part_f
                err_f = (total_f - total_f[0]) / np.maximum(np.abs(total_f[0]), 1e-30)
                ax.plot(hours, err_f, 'k-', lw=2, label='Fortran')

            if py is not None:
                gas_p = py['Gc'][:, spec_p]
                part_p = np.sum(py['Mk'][:, :, spec_p], axis=1)
                total_p = gas_p + part_p
                err_p = (total_p - total_p[0]) / np.maximum(np.abs(total_p[0]), 1e-30)
                ax.plot(hours, err_p, 'C0--', lw=1.5, label='Python')

            ax.axhline(0, color='gray', ls=':', lw=0.5)
            if row == 0:
                ax.set_title(f"C*={cstar_labels[col]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nFrac. error', fontsize=9)
            if row == 2:
                ax.set_xlabel('Hours')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_fig7_error():
    """Fig 7: Relative error summary (Python vs Fortran)."""
    import matplotlib.pyplot as plt

    hours = np.arange(NHOURS + 1)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Fig 7: Relative Error (Python PPM vs Fortran top-hat, same 36-bin grid)',
                 fontsize=14, fontweight='bold')
    metric_labels = ['N_total error', 'M_dry error', 'M_organic error']

    for row, label in enumerate(SCENARIO_ORDER):
        sc = SCENARIOS[label]
        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            continue

        try:
            py = _load_python(label)
        except FileNotFoundError:
            continue

        # N_total (same grid — direct comparison)
        N_f = np.sum(fort['Nk'], axis=1) / BOXVOL
        N_p = np.sum(py['Nk'], axis=1) / BOXVOL
        err_N = (N_p - N_f) / np.maximum(np.abs(N_f), 1e-30)

        # M_dry
        M_f = np.sum(fort['Mk'][:, :, :42], axis=(1, 2))
        M_p = np.sum(py['Mk'][:, :, :SRTH2O], axis=(1, 2))
        err_M = (M_p - M_f) / np.maximum(np.abs(M_f), 1e-30)

        # M_organic
        Morg_f = np.sum(fort['Mk'][:, :, 1:7], axis=(1, 2))
        Morg_p = np.sum(py['Mk'][:, :, SRTORG1:SRTORG1+N_VBS], axis=(1, 2))
        err_org = (Morg_p - Morg_f) / np.maximum(np.abs(Morg_f), 1e-30)

        errors = [err_N, err_M, err_org]

        for col, (err, mlabel) in enumerate(zip(errors, metric_labels)):
            ax = axes[row, col]
            ax.plot(hours, err * 100, 'C0-', lw=2)
            ax.axhline(0, color='gray', ls=':', lw=0.5)
            if row == 0:
                ax.set_title(mlabel, fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{sc["label"]}\nError [%]', fontsize=9)
            if row == 2:
                ax.set_xlabel('Hours')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fig8_diagnostics():
    """Fig 8: Condensation sink and Kelvin effect diagnostics."""
    import matplotlib.pyplot as plt
    from tomas_jax.physics.vbs_config import calc_Cstar_T, cstar_ug_to_Pa
    from tomas_jax.physics.kelvin_effect import calc_kelvin_factor

    cstar_ref = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    dHvap = np.array([150.0, 136.0, 122.0, 108.0, 94.0, 80.0])
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Fig 8: VBS Diagnostics — C*(T), Kelvin Effect, Gas Depletion',
                 fontsize=14, fontweight='bold')

    # Panel 1: Temperature-dependent C*
    temps = np.linspace(250, 320, 100)
    ax = axes[0, 0]
    for j in range(N_VBS):
        cstar_T = np.array([float(calc_Cstar_T(
            jnp.float64(cstar_ref[j]), jnp.float64(dHvap[j]), jnp.float64(T)))
            for T in temps])
        ax.semilogy(temps, cstar_T, color=colors[j], lw=2,
                    label=f"C*_ref={cstar_ref[j]}")
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('C*(T) [µg/m³]')
    ax.set_title('Temperature-corrected C*')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: Kelvin factor vs diameter
    ax = axes[0, 1]
    Dp_nm = np.logspace(0, 4, 200)
    Dp_m = Dp_nm * 1e-9
    for j, T in enumerate([270, 288, 310]):
        Ke = np.array([float(calc_kelvin_factor(
            jnp.array([d]), jnp.float64(0.025), jnp.float64(200.0),
            jnp.float64(1200.0), jnp.float64(T))[0]) for d in Dp_m])
        ax.semilogx(Dp_nm, Ke, lw=2, label=f'T={T}K')
    ax.set_xlabel('Dp [nm]')
    ax.set_ylabel('Kelvin factor')
    ax.set_title('Kelvin Effect (σ=0.025, MW=200)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 5)

    # Panel 3: C*(T) as Pa at 3 temperatures
    ax = axes[0, 2]
    for j, T in enumerate([270, 288, 310]):
        cstar_Pa = np.array([float(cstar_ug_to_Pa(
            calc_Cstar_T(jnp.float64(c), jnp.float64(dHvap[i]),
                         jnp.float64(T)),
            jnp.float64(200.0), jnp.float64(T)))
            for i, c in enumerate(cstar_ref)])
        ax.semilogy(np.arange(N_VBS), cstar_Pa, 'o-', lw=2,
                    label=f'T={T}K')
    ax.set_xticks(np.arange(N_VBS))
    ax.set_xticklabels(['0.01', '0.1', '1', '10', '100', '1000'], fontsize=8)
    ax.set_xlabel('C* ref [µg/m³]')
    ax.set_ylabel('C*(T) [Pa]')
    ax.set_title('Saturation pressure at 3 temperatures')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 4-6: Gas depletion ratios per scenario
    for col, label in enumerate(SCENARIO_ORDER):
        ax = axes[1, col]
        sc = SCENARIOS[label]
        hours = np.arange(NHOURS + 1)

        try:
            fort = load_fortran_scenario(label)
        except FileNotFoundError:
            fort = None

        try:
            py = _load_python(label)
        except FileNotFoundError:
            py = None

        for j in range(N_VBS):
            f_idx = 1 + j
            p_idx = SRTORG1 + j

            if fort is not None:
                gc_init_f = fort['Gc'][0, f_idx]
                if gc_init_f > 0:
                    ax.plot(hours, fort['Gc'][:, f_idx] / gc_init_f,
                            color=colors[j], ls='-', lw=1.5)

            if py is not None:
                gc_init_p = py['Gc'][0, p_idx]
                if gc_init_p > 0:
                    ax.plot(hours, py['Gc'][:, p_idx] / gc_init_p,
                            color=colors[j], ls='--', lw=1.5,
                            label=f'C*={cstar_ref[j]}' if fort is None else None)

        # Legend: solid=Fortran, dashed=Python
        if col == 0:
            from matplotlib.lines import Line2D
            leg_elems = [
                Line2D([0], [0], color='gray', ls='-', lw=2, label='Fortran'),
                Line2D([0], [0], color='gray', ls='--', lw=2, label='Python (PPM)'),
            ]
            for j in range(N_VBS):
                leg_elems.append(
                    Line2D([0], [0], color=colors[j], lw=2,
                           label=f'C*={cstar_ref[j]}'))
            ax.legend(handles=leg_elems, fontsize=6, loc='best')

        ax.set_xlabel('Hours')
        ax.set_ylabel('Gc(t)/Gc(0)')
        ax.set_title(f'{sc["label"]}')
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =========================================================================
# Main
# =========================================================================

def plot_all():
    """Generate all 8 figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    figures = [
        ('fig1_gas_evolution.png', plot_fig1_gas_evolution),
        ('fig2_particle_vbs.png', plot_fig2_particle_vbs),
        ('fig3_sizedist.png', plot_fig3_sizedist),
        ('fig3b_sizedist_relerror.png', plot_fig3b_sizedist_relerror),
        ('fig3c_sizedist_linear.png', plot_fig3c_sizedist_linear),
        ('fig4_composition.png', plot_fig4_composition),
        ('fig5_totals.png', plot_fig5_totals),
        ('fig6_conservation.png', plot_fig6_conservation),
        ('fig7_error.png', plot_fig7_error),
        ('fig8_diagnostics.png', plot_fig8_diagnostics),
    ]

    for fname, plot_fn in figures:
        print(f"  Plotting {fname}...")
        fig = plot_fn()
        fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"All figures saved to {RESULTS_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SOA/VBS Benchmark')
    parser.add_argument('--run', action='store_true',
                        help='Run Python simulations')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only regenerate plots from existing data')
    args = parser.parse_args()

    if args.run:
        run_all()

    if args.plot_only or args.run:
        plot_all()

    if not args.run and not args.plot_only:
        parser.print_help()
