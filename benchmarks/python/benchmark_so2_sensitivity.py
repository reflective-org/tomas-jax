"""SO2 sensitivity benchmark: SO2 decay, H2SO4 production, aerosol response.

Runs 5 SO2 levels × 4 process modes × 3 altitudes × 2 grid resolutions = 120
scenarios over 48 hours, producing 12 figures + timing CSV.

Usage::

    python -m benchmarks.python.benchmark_so2_sensitivity           # run all + plot
    python -m benchmarks.python.benchmark_so2_sensitivity --plot-only  # replot from NPZ
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
    make_grid, make_grid_80bin,
)
from tomas_jax.solvers.condensation import make_step

# =========================================================================
# Constants
# =========================================================================
DENS_INIT = 1770.0  # kg/m3
BOXVOL = 1.0e6      # cm3

DT = 60.0           # seconds per step
NHOURS = 48
NSTEPS = NHOURS * 60  # 2880

SO2_LEVELS = [1e8, 1e9, 1e10, 5e10, 1e11]  # molec/cm3
SO2_LABELS = ['1e8', '1e9', '1e10', '5e10', '1e11']

MODES = ['cond_only', 'nucl_cond', 'coag_cond', 'full']
MODE_PROCESSES = {
    'cond_only': ['so2_chemistry', 'condensation'],
    'nucl_cond': ['so2_chemistry', 'nucleation', 'condensation'],
    'coag_cond': ['so2_chemistry', 'coagulation', 'condensation'],
    'full':      ['so2_chemistry', 'nucleation', 'coagulation', 'condensation'],
}

ALTITUDES = {
    'sfc': {'temp': 288.0, 'pres': 101325.0, 'rh': 0.50, 'label': 'Surface (0 km)'},
    'ft':  {'temp': 256.0, 'pres': 54000.0,  'rh': 0.30, 'label': 'Free tropo (5 km)'},
    'ut':  {'temp': 223.0, 'pres': 26500.0,  'rh': 0.10, 'label': 'Upper tropo (10 km)'},
}

GRIDS = {
    40: {'nbins': 40, 'xk_fn': lambda: make_grid(40, XK0, 2.0)},
    80: {'nbins': 80, 'xk_fn': make_grid_80bin},
}

# Nucleation parameters
NUC_ORG = jnp.float64(1e7)
NUC_NH3 = jnp.float64(1e9)
NUC_FION = jnp.float64(3.0)

OH_CONC = jnp.float64(1e6)  # molec/cm3 (constant)

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'so2_sensitivity')

# =========================================================================
# Unit conversion helpers
# =========================================================================

def so2_molec_to_ppb(molec_cm3, temp, pres):
    """Convert molec/cm3 to ppbv."""
    M_total = pres / (KB * temp) * 1e-6  # molec/cm3
    return molec_cm3 / M_total * 1e9

def so2_molec_to_ugm3(molec_cm3):
    """Convert molec/cm3 to μg/m3."""
    return molec_cm3 * MW_SO2 * 1e12 / AVOGADRO


# =========================================================================
# Initialization
# =========================================================================

def make_initial_state(so2_init, nbins, xk, temp, pres):
    """Create initial Nk, Mk, Gc for a scenario.

    Args:
        so2_init: Initial SO2 [molec/cm3]
        nbins: Number of bins
        xk: Bin boundaries [kg], shape (nbins+1,)
        temp, pres: Thermodynamic conditions

    Returns:
        Nk, Mk, Gc (JAX arrays)
    """
    n_total = 1e4    # #/cm3
    gmd = 50e-9      # 50 nm
    gsd = 1.6
    h2so4_init = 1e7  # molec/cm3

    xk_np = np.array(xk)

    # Lognormal initialization (Fortran-style)
    Nk = np.zeros(nbins)
    Mk = np.zeros((nbins, ICOMP))

    for k in range(nbins):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI)) ** (1.0/3.0)
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI)) ** (1.0/3.0)
        Dk = np.sqrt(Dl * Dh)
        gmd_um = gmd * 1e6

        np_init = ((n_total * BOXVOL)
                   / (np.sqrt(2.0 * PI) * Dk * np.log(gsd))
                   * np.exp(-(np.log(Dk / gmd_um) ** 2
                              / (2.0 * np.log(gsd) ** 2)))
                   * (Dh - Dl))

        Nk[k] = max(np_init, 1e-3)
        Mk[k, SRTSO4] = Nk[k] * np.sqrt(xk_np[k] * xk_np[k + 1])

    # Gas phase
    Gc = np.zeros(N_GAS_SPECIES)
    Gc[SRTSO4] = h2so4_init * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO
    Gc[SRTSO2] = so2_init * BOXVOL * (MW_SO2 / 1000.0) / AVOGADRO

    return jnp.array(Nk), jnp.array(Mk), jnp.array(Gc)


# =========================================================================
# Single scenario runner
# =========================================================================

def run_scenario(so2_init, mode, alt_key, nbins, xk, verbose=True):
    """Run one 48h SO2 sensitivity scenario.

    Returns dict with timeseries and final state.
    """
    alt = ALTITUDES[alt_key]
    temp = jnp.float64(alt['temp'])
    pres = jnp.float64(alt['pres'])
    rh = jnp.float64(alt['rh'])
    alpha = jnp.float64(1.0)
    boxvol = jnp.float64(BOXVOL)
    dt = jnp.float64(DT)

    Nk, Mk, Gc = make_initial_state(so2_init, nbins, xk, float(temp), float(pres))

    # Build step function
    processes = MODE_PROCESSES[mode]
    step_fn_jit = make_step(processes, cond_method='ppm_jit')

    # Build kwargs
    kw = dict(oh_conc=OH_CONC)
    if 'nucleation' in processes:
        kw.update(org_conc=NUC_ORG, nh3_conc=NUC_NH3, fion=NUC_FION)

    # Warmup JIT
    _ = step_fn_jit(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, **kw)

    # Storage — every-minute Nk for banana, hourly for timeseries
    Nk_every = np.zeros((NSTEPS, nbins))
    Gc_SO2_every = np.zeros(NSTEPS)
    Gc_SO4_every = np.zeros(NSTEPS)
    N_tot_hourly = np.zeros(NHOURS)
    M_dry_hourly = np.zeros(NHOURS)
    Nk_hourly = np.zeros((NHOURS, nbins))

    t0 = time.time()

    for i in range(NSTEPS):
        # Record every-minute state BEFORE stepping
        Nk_every[i] = np.array(Nk)
        gc_np = np.array(Gc)
        Gc_SO2_every[i] = gc_np[SRTSO2]
        Gc_SO4_every[i] = gc_np[SRTSO4]

        # Hourly snapshot
        if i % 60 == 0:
            hr = i // 60
            N_tot_hourly[hr] = float(jnp.sum(Nk))
            M_dry_hourly[hr] = float(jnp.sum(Mk[:, :SRTH2O]))
            Nk_hourly[hr] = np.array(Nk)

        # Step
        Nk, Mk, Gc = step_fn_jit(Nk, Mk, Gc, xk, temp, pres, boxvol,
                                   rh, alpha, dt, **kw)

    wall_time = time.time() - t0

    # Convert SO2 gas from kg/gridcell to molec/cm3 for output
    so2_to_molec = AVOGADRO / (MW_SO2 / 1000.0) / BOXVOL
    so4_to_molec = AVOGADRO / (MW_H2SO4 / 1000.0) / BOXVOL

    result = {
        'Nk_every': Nk_every,
        'Gc_SO2_molec': Gc_SO2_every * so2_to_molec,
        'Gc_SO4_molec': Gc_SO4_every * so4_to_molec,
        'N_tot_hourly': N_tot_hourly,
        'M_dry_hourly': M_dry_hourly,
        'Nk_hourly': Nk_hourly,
        'Nk_final': np.array(Nk),
        'Mk_final': np.array(Mk),
        'wall_time_s': wall_time,
        'so2_init': so2_init,
        'mode': mode,
        'alt': alt_key,
        'nbins': nbins,
    }

    if verbose:
        so2_label = f"{so2_init:.0e}"
        print(f"  {so2_label:>6s} | {mode:<10s} | {alt_key:<3s} | {nbins:>2d}bin | "
              f"{wall_time:6.2f}s | N={N_tot_hourly[0]:.2e}")

    return result


# =========================================================================
# Run all scenarios
# =========================================================================

def run_all(grid_sizes=None, alt_keys=None, verbose=True):
    """Run all SO2 sensitivity scenarios and save results."""
    if grid_sizes is None:
        grid_sizes = [40, 80]
    if alt_keys is None:
        alt_keys = list(ALTITUDES.keys())

    os.makedirs(OUTDIR, exist_ok=True)
    results = []
    timing_rows = []

    print("=" * 72)
    print("SO2 Sensitivity Benchmark — 48h simulations")
    print(f"  SO2 levels: {SO2_LABELS}")
    print(f"  Modes:      {MODES}")
    print(f"  Altitudes:  {alt_keys}")
    print(f"  Grids:      {grid_sizes}")
    print("=" * 72)
    print(f"  {'SO2':>6s} | {'Mode':<10s} | {'Alt':<3s} | {'Grid':>5s} | "
          f"{'Time':>6s} | {'N_init':>10s}")
    print("-" * 72)

    for gsize in grid_sizes:
        xk = GRIDS[gsize]['xk_fn']()
        nbins = GRIDS[gsize]['nbins']

        for alt_key in alt_keys:
            for mode in MODES:
                for so2_init, so2_label in zip(SO2_LEVELS, SO2_LABELS):
                    res = run_scenario(so2_init, mode, alt_key, nbins, xk, verbose)
                    results.append(res)

                    # Save individual NPZ
                    fname = f"so2_{so2_label}_{mode}_{alt_key}_{nbins}bin.npz"
                    np.savez_compressed(
                        os.path.join(OUTDIR, fname),
                        **{k: v for k, v in res.items()
                           if isinstance(v, np.ndarray)},
                        so2_init=so2_init, wall_time_s=res['wall_time_s'],
                        mode=mode, alt=alt_key, nbins=nbins,
                    )

                    timing_rows.append({
                        'so2_init': so2_init,
                        'so2_label': so2_label,
                        'mode': mode,
                        'alt': alt_key,
                        'nbins': nbins,
                        'wall_time_s': res['wall_time_s'],
                    })

    # Write timing CSV
    csv_path = os.path.join(OUTDIR, 'timing.csv')
    with open(csv_path, 'w') as f:
        f.write('so2_init,so2_label,mode,alt,nbins,wall_time_s\n')
        for row in timing_rows:
            f.write(f"{row['so2_init']:.0e},{row['so2_label']},"
                    f"{row['mode']},{row['alt']},{row['nbins']},"
                    f"{row['wall_time_s']:.3f}\n")

    print("-" * 72)
    total_time = sum(r['wall_time_s'] for r in results)
    print(f"Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to {OUTDIR}")

    return results


# =========================================================================
# Plotting helpers
# =========================================================================

def _load_result(so2_label, mode, alt_key, nbins):
    """Load a single NPZ result. Raises FileNotFoundError if missing."""
    fname = f"so2_{so2_label}_{mode}_{alt_key}_{nbins}bin.npz"
    path = os.path.join(OUTDIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing NPZ: {fname}. Run simulations first "
            f"(python -m benchmarks.python.benchmark_so2_sensitivity).")
    return dict(np.load(path, allow_pickle=True))


def _dp_midpoints(xk_np):
    """Geometric mean diameter [m] from bin boundaries [kg]."""
    m_mid = np.sqrt(xk_np[:-1] * xk_np[1:])
    return np.cbrt(m_mid / DENS_INIT * (6.0 / PI))


def _dlogDp(xk_np):
    """Log-width of bins."""
    dp_edges = np.cbrt(xk_np)
    return np.log10(dp_edges[1:] / dp_edges[:-1])


def _time_minutes():
    """Time array in minutes for every-step data."""
    return np.arange(NSTEPS)  # dt=60s → 1 point per minute


def _time_hours():
    """Time array in hours for hourly data."""
    return np.arange(NHOURS)


MODE_LABELS = {
    'cond_only': 'SO2 + Cond',
    'nucl_cond': 'SO2 + Nucl + Cond',
    'coag_cond': 'SO2 + Coag + Cond',
    'full':      'SO2 + Nucl + Coag + Cond',
}

SO2_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']


# =========================================================================
# Figures
# =========================================================================

def _fig_so2_decay(unit, ylabel, convert_fn, fig_num, alt_key='sfc', nbins=40):
    """SO2 decay: 2×2 panels (modes), 5 lines (SO2 levels)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Fig {fig_num}: SO2 Decay ({ylabel}) — {ALTITUDES[alt_key]["label"]}, {nbins}-bin',
                 fontsize=14, fontweight='bold')

    t_hr = _time_minutes() / 60.0

    for idx, mode in enumerate(MODES):
        ax = axes.flat[idx]
        for j, (so2_init, so2_label) in enumerate(zip(SO2_LEVELS, SO2_LABELS)):
            res = _load_result(so2_label, mode, alt_key, nbins)
            so2_molec = res['Gc_SO2_molec']
            y = convert_fn(so2_molec, ALTITUDES[alt_key]['temp'], ALTITUDES[alt_key]['pres'])
            ax.semilogy(t_hr, y, color=SO2_COLORS[j], lw=1.5,
                        label=f'SO2={so2_label}')

        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which='both')
        if idx >= 2:
            ax.set_xlabel('Time [hours]')
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    return fig


def plot_so2_decay_molec(alt_key='sfc', nbins=40):
    """Fig 1: SO2 in molec/cm3."""
    return _fig_so2_decay(
        'molec', 'SO2 [molec/cm³]',
        lambda y, t, p: y,
        1, alt_key, nbins)


def plot_so2_decay_ppb(alt_key='sfc', nbins=40):
    """Fig 2: SO2 in ppb."""
    return _fig_so2_decay(
        'ppb', 'SO2 [ppb]',
        lambda y, t, p: so2_molec_to_ppb(y, t, p),
        2, alt_key, nbins)


def plot_so2_decay_ugm3(alt_key='sfc', nbins=40):
    """Fig 3: SO2 in μg/m3."""
    return _fig_so2_decay(
        'ugm3', 'SO2 [μg/m³]',
        lambda y, t, p: so2_molec_to_ugm3(y),
        3, alt_key, nbins)


def plot_h2so4_gas(alt_key='sfc', nbins=40):
    """Fig 4: H2SO4 gas timeseries — 2×2 panels, 5 lines."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Fig 4: H2SO4 Gas Phase — {ALTITUDES[alt_key]["label"]}, {nbins}-bin',
                 fontsize=14, fontweight='bold')

    t_hr = _time_minutes() / 60.0

    for idx, mode in enumerate(MODES):
        ax = axes.flat[idx]
        for j, (so2_init, so2_label) in enumerate(zip(SO2_LEVELS, SO2_LABELS)):
            res = _load_result(so2_label, mode, alt_key, nbins)
            ax.semilogy(t_hr, res['Gc_SO4_molec'], color=SO2_COLORS[j],
                        lw=1.5, label=f'SO2={so2_label}')

        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_ylabel('H2SO4 [molec/cm³]')
        ax.grid(True, alpha=0.3, which='both')
        if idx >= 2:
            ax.set_xlabel('Time [hours]')
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    return fig


def plot_n_total(alt_key='sfc', nbins=40):
    """Fig 5: Total N timeseries — 2×2 panels, 5 lines."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Fig 5: Total Particle Number — {ALTITUDES[alt_key]["label"]}, {nbins}-bin',
                 fontsize=14, fontweight='bold')

    t_hr = _time_hours()

    for idx, mode in enumerate(MODES):
        ax = axes.flat[idx]
        for j, (so2_init, so2_label) in enumerate(zip(SO2_LEVELS, SO2_LABELS)):
            res = _load_result(so2_label, mode, alt_key, nbins)
            N_cm3 = res['N_tot_hourly'] / BOXVOL
            ax.semilogy(t_hr, N_cm3, color=SO2_COLORS[j], lw=1.5,
                        label=f'SO2={so2_label}')

        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_ylabel('N_tot [#/cm³]')
        ax.grid(True, alpha=0.3, which='both')
        if idx >= 2:
            ax.set_xlabel('Time [hours]')
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_m_dry(alt_key='sfc', nbins=40):
    """Fig 6: Dry mass timeseries — 2×2 panels, 5 lines."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Fig 6: Dry Aerosol Mass — {ALTITUDES[alt_key]["label"]}, {nbins}-bin',
                 fontsize=14, fontweight='bold')

    t_hr = _time_hours()

    for idx, mode in enumerate(MODES):
        ax = axes.flat[idx]
        for j, (so2_init, so2_label) in enumerate(zip(SO2_LEVELS, SO2_LABELS)):
            res = _load_result(so2_label, mode, alt_key, nbins)
            M_ugm3 = res['M_dry_hourly'] / BOXVOL * 1e9  # kg/cm3 → μg/m3
            ax.plot(t_hr, M_ugm3, color=SO2_COLORS[j], lw=1.5,
                    label=f'SO2={so2_label}')

        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_ylabel('M_dry [μg/m³]')
        ax.grid(True, alpha=0.3)
        if idx >= 2:
            ax.set_xlabel('Time [hours]')
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_banana(alt_key='sfc', nbins=40, fig_num=7):
    """Fig 7/8: Banana plot — 2×2 panels (modes), highest SO2, surface."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    so2_label = SO2_LABELS[-1]  # highest SO2
    xk_np = np.array(GRIDS[nbins]['xk_fn']())
    dp_mid = _dp_midpoints(xk_np) * 1e9  # nm
    dlog = _dlogDp(xk_np)
    t_hr = _time_minutes() / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Fig {fig_num}: Banana Plot (SO2={so2_label}) — '
                 f'{ALTITUDES[alt_key]["label"]}, {nbins}-bin',
                 fontsize=14, fontweight='bold')

    for idx, mode in enumerate(MODES):
        ax = axes.flat[idx]
        res = _load_result(so2_label, mode, alt_key, nbins)

        # dN/dlogDp in #/cm3
        Nk_all = res['Nk_every']  # (NSTEPS, nbins)
        Z = (Nk_all / BOXVOL) / dlog[None, :]
        Z = Z.T  # (nbins, NSTEPS)
        Z = np.maximum(Z, 1e-2)

        vmax = Z.max()
        vmin = max(vmax * 1e-4, 1e-2)

        mesh = ax.pcolormesh(t_hr, dp_mid, Z,
                             norm=LogNorm(vmin=vmin, vmax=vmax),
                             cmap='jet', shading='auto')
        ax.set_yscale('log')
        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_ylabel('Dp [nm]')
        if idx >= 2:
            ax.set_xlabel('Time [hours]')
        plt.colorbar(mesh, ax=ax, label='dN/dlogDp [#/cm³]')

    plt.tight_layout()
    return fig


def plot_size_dist_final(alt_key='sfc', nbins=40, fig_num=9):
    """Fig 9/10: Final dN/dlogDp — 2×2 panels (modes), 5 lines (SO2 levels)."""
    import matplotlib.pyplot as plt

    xk_np = np.array(GRIDS[nbins]['xk_fn']())
    dp_mid = _dp_midpoints(xk_np) * 1e9  # nm
    dlog = _dlogDp(xk_np)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Fig {fig_num}: Final Size Distribution (t=48h) — '
                 f'{ALTITUDES[alt_key]["label"]}, {nbins}-bin',
                 fontsize=14, fontweight='bold')

    for idx, mode in enumerate(MODES):
        ax = axes.flat[idx]
        for j, (so2_init, so2_label) in enumerate(zip(SO2_LEVELS, SO2_LABELS)):
            res = _load_result(so2_label, mode, alt_key, nbins)
            Nk_f = res['Nk_final']
            dNdlogDp = (Nk_f / BOXVOL) / dlog
            ax.semilogx(dp_mid, dNdlogDp, color=SO2_COLORS[j], lw=1.5,
                        label=f'SO2={so2_label}')

        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_ylabel('dN/dlogDp [#/cm³]')
        ax.set_xlabel('Dp [nm]')
        ax.grid(True, alpha=0.3, which='both')
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_altitude_so2(nbins=40):
    """Fig 11: SO2 decay at 3 altitudes — highest SO2, full mode."""
    import matplotlib.pyplot as plt

    so2_label = SO2_LABELS[-1]
    mode = 'full'
    t_hr = _time_minutes() / 60.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Fig 11: SO2 Decay at 3 Altitudes (SO2={so2_label}, {mode}) — {nbins}-bin',
                 fontsize=14, fontweight='bold')

    alt_colors = {'sfc': '#E91E63', 'ft': '#FF9800', 'ut': '#2196F3'}
    units = [
        ('molec/cm³', lambda y, t, p: y),
        ('ppb', lambda y, t, p: so2_molec_to_ppb(y, t, p)),
        ('μg/m³', lambda y, t, p: so2_molec_to_ugm3(y)),
    ]

    for ax_idx, (unit_label, conv) in enumerate(units):
        ax = axes[ax_idx]
        for alt_key in ['sfc', 'ft', 'ut']:
            res = _load_result(so2_label, mode, alt_key, nbins)
            alt = ALTITUDES[alt_key]
            y = conv(res['Gc_SO2_molec'], alt['temp'], alt['pres'])
            ax.semilogy(t_hr, y, color=alt_colors[alt_key], lw=2,
                        label=alt['label'])
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel(f'SO2 [{unit_label}]')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_altitude_aerosol(nbins=40):
    """Fig 12: N_tot and M_dry at 3 altitudes — highest SO2, full mode."""
    import matplotlib.pyplot as plt

    so2_label = SO2_LABELS[-1]
    mode = 'full'
    t_hr = _time_hours()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fig 12: Aerosol Response at 3 Altitudes (SO2={so2_label}, {mode}) — {nbins}-bin',
                 fontsize=14, fontweight='bold')

    alt_colors = {'sfc': '#E91E63', 'ft': '#FF9800', 'ut': '#2196F3'}

    # N_tot
    ax = axes[0]
    for alt_key in ['sfc', 'ft', 'ut']:
        res = _load_result(so2_label, mode, alt_key, nbins)
        N_cm3 = res['N_tot_hourly'] / BOXVOL
        ax.semilogy(t_hr, N_cm3, color=alt_colors[alt_key], lw=2,
                    label=ALTITUDES[alt_key]['label'])
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('N_tot [#/cm³]')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)

    # M_dry
    ax = axes[1]
    for alt_key in ['sfc', 'ft', 'ut']:
        res = _load_result(so2_label, mode, alt_key, nbins)
        M_ugm3 = res['M_dry_hourly'] / BOXVOL * 1e9
        ax.plot(t_hr, M_ugm3, color=alt_colors[alt_key], lw=2,
                label=ALTITUDES[alt_key]['label'])
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('M_dry [μg/m³]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


# =========================================================================
# Master plot function
# =========================================================================

def plot_results():
    """Generate all 12 figures from saved NPZ files.

    Raises FileNotFoundError if any required NPZ file is missing —
    run simulations first.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(OUTDIR, exist_ok=True)

    figures = [
        ('fig01_so2_decay_molec.png', plot_so2_decay_molec),
        ('fig02_so2_decay_ppb.png', plot_so2_decay_ppb),
        ('fig03_so2_decay_ugm3.png', plot_so2_decay_ugm3),
        ('fig04_h2so4_gas.png', plot_h2so4_gas),
        ('fig05_n_total.png', plot_n_total),
        ('fig06_m_dry.png', plot_m_dry),
        ('fig07_banana_40bin.png', lambda: plot_banana(nbins=40, fig_num=7)),
        ('fig08_banana_80bin.png', lambda: plot_banana(nbins=80, fig_num=8)),
        ('fig09_sizedist_40bin.png', lambda: plot_size_dist_final(nbins=40, fig_num=9)),
        ('fig10_sizedist_80bin.png', lambda: plot_size_dist_final(nbins=80, fig_num=10)),
        ('fig11_altitude_so2.png', plot_altitude_so2),
        ('fig12_altitude_aerosol.png', plot_altitude_aerosol),
    ]

    for fname, plot_fn in figures:
        print(f"  Generating {fname}...")
        fig = plot_fn()
        fig.savefig(os.path.join(OUTDIR, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"All 12 figures saved to {OUTDIR}")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SO2 sensitivity benchmark (48h, 5 SO2 levels × 4 modes × 3 altitudes × 2 grids)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip simulation, replot from saved NPZ files')
    parser.add_argument('--grids', type=int, nargs='+', default=[40, 80],
                        choices=[40, 80], help='Grid resolutions (default: 40 80)')
    parser.add_argument('--altitudes', nargs='+', default=None,
                        choices=['sfc', 'ft', 'ut'],
                        help='Altitude conditions (default: all)')
    args = parser.parse_args()

    if not args.plot_only:
        run_all(grid_sizes=args.grids, alt_keys=args.altitudes)

    plot_results()


if __name__ == '__main__':
    main()
