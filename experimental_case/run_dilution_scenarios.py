"""SAI box model — 30-scenario dilution ensemble with multi-resolution stepping.

Time stepping (adaptive resolution to capture fast early dynamics):
    Phase 1:  t = 0 – 20 min    dt = 1 s       (1200 steps)
    Phase 2:  t = 20 min – 4 h  dt = 10 s      (1320 steps)
    Phase 3:  t = 4 h – data end dt = 60 s      (up to 9840 steps for 168 h)

Scenarios: 5 subgroups × 3 normalizations × 2 variants (raw, fit) = 30.

Species settings:
    OH = 1e6 molec/cm³ constant.  NH3 = 0, organics = 0, ion-pairs = 0.
    Background aerosol: pure SO4, 5 #/cm³, GMD = 80 nm, GSD = 1.6.
    SO2: from plume_concentration.py (~2e13 molec/cm³).

Output per scenario: NPZ with full time series + time array.

Plots (generated after all runs):
    1. Banana plot  (dN/dlogDp vs time)
    2. Size dist    (dN/dlogDp at h0, d1, d2, d7)  log-y + linear-y
    3. Size dist    (dA/dlogDp at same times)       log-y + linear-y
    4. Size dist    (dM/dlogDp at same times)       log-y + linear-y

Usage::

    python -m experimental_case.run_dilution_scenarios
    python -m experimental_case.run_dilution_scenarios --dilution-dir /path/to/data
    python -m experimental_case.run_dilution_scenarios --scenario sg02_first_value_raw
    python -m experimental_case.run_dilution_scenarios --plot-only
"""

import os
import sys
import time
import argparse
import numpy as np
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, N_GAS_SPECIES,
    SRTSO4, SRTSO2, SRTH2O,
    MW_H2SO4, MW_SO2, AVOGADRO, PI,
    make_grid, XK0,
)
from tomas_jax.solvers.condensation import make_step

from .plume_concentration import plume_so2_concentration

# =========================================================================
# Physical / grid constants
# =========================================================================
DENS_INIT = 1770.0      # kg/m³  (sulfate)
BOXVOL    = 1.0e6       # cm³   (1 m³)
ALPHA     = 1.0

TEMP = 216.0            # K
PRES = 5500.0           # Pa  (55 hPa)
RH   = 0.01

N_BG   = 5.0            # background #/cm³
GMD_BG = 80e-9          # m
GSD_BG = 1.6

H2SO4_INIT = 1e5        # molec/cm³
OH_CONC    = 1e6        # molec/cm³

NUC_ORG  = jnp.float64(0.0)
NUC_NH3  = jnp.float64(0.0)
NUC_FION = jnp.float64(0.0)

# =========================================================================
# Multi-resolution time schedule
# =========================================================================
# Phase boundaries in seconds from t=0
PHASE1_END  = 20 * 60           # 1200 s  (20 min)
PHASE2_END  = 4  * 3600         # 14400 s (4 h)
PHASE3_END  = 168 * 3600        # 604800 s (7 days, capped by data)

DT1 = 1.0    # s  — first 20 min
DT2 = 10.0   # s  — 20 min – 4 h
DT3 = 60.0   # s  — 4 h onward

# Pre-compute phase step counts
N1 = int(round(PHASE1_END / DT1))        # 1200
N2 = int(round((PHASE2_END - PHASE1_END) / DT2))  # 1320
N3_MAX = int(round((PHASE3_END - PHASE2_END) / DT3))  # 9840

# Total maximum steps
NSTEPS_MAX = N1 + N2 + N3_MAX

# =========================================================================
# Snapshot times for size-distribution plots
# =========================================================================
SNAPSHOT_TIMES_S = {
    'h0':  0.0,
    'd1':  24  * 3600.0,
    'd2':  48  * 3600.0,
    'd6':  144 * 3600.0,
    'd7':  168 * 3600.0,
}

# =========================================================================
# Scenario enumeration
# =========================================================================
SUBGROUPS = ['2', '4', '10', '10b', '12']
NORMS     = ['first_hour_mean', 'first_value', 'smoothed']
VARIANTS  = ['raw', 'fit']

_DEFAULT_DILUTION_DIR = os.path.expanduser(
    '~/Documents/GitHub/experiment_sai/'
    'dilution_determination/output/box_model'
)
_OUTDIR = os.path.join(os.path.dirname(__file__), 'results', 'dilution_scenarios')


def _sg_prefix(sg):
    """'2' → 'sg02',  '10b' → 'sg10b'."""
    return f'sg{sg}' if 'b' in sg else f'sg{int(sg):02d}'


def scenario_name(sg, norm, variant):
    return f'{_sg_prefix(sg)}_{norm}_{variant}'


def all_scenarios():
    return [(sg, norm, var)
            for sg in SUBGROUPS
            for norm in NORMS
            for var in VARIANTS]


# =========================================================================
# Time schedule builder
# =========================================================================

def build_time_schedule(max_hours):
    """Build arrays of (t_start, dt) for each step.

    Returns
    -------
    t_starts : ndarray  — time at START of each step [s]
    dts      : ndarray  — dt for each step [s]
    """
    max_s = max_hours * 3600.0
    t = 0.0
    t_starts = []
    dts = []

    # Phase 1: dt = 1 s
    while t < PHASE1_END - 1e-9 and t < max_s - 1e-9:
        t_starts.append(t)
        dts.append(DT1)
        t += DT1

    # Phase 2: dt = 10 s
    while t < PHASE2_END - 1e-9 and t < max_s - 1e-9:
        t_starts.append(t)
        dts.append(DT2)
        t += DT2

    # Phase 3: dt = 60 s
    while t < max_s - 1e-9:
        t_starts.append(t)
        dts.append(DT3)
        t += DT3

    return np.array(t_starts), np.array(dts)


# =========================================================================
# Dilution: V(t) → kdil per step
# =========================================================================

def _get_V_at(t_seconds, sg, norm, variant, dilution_dir,
              _cache={}):
    """Return V(t) for an array of times [s], using cache for CSV data."""
    key = (sg, norm, variant)
    if key not in _cache:
        if variant == 'fit':
            if dilution_dir not in sys.path:
                sys.path.insert(0, dilution_dir)
            from volume_ratios import V_ratio, fit_info
            info = fit_info(str(sg), norm)
            max_hours = info['max_hours']
            _cache[key] = ('fit', None, None, float(max_hours))
        else:
            prefix = _sg_prefix(sg)
            csv_path = os.path.join(dilution_dir, f'{prefix}_{norm}.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f'Dilution CSV not found: {csv_path}\n'
                    f'Check --dilution-dir.')
            # Use genfromtxt to tolerate empty/NaN cells in the CSV
            data = np.genfromtxt(csv_path, delimiter=',', skip_header=1,
                                 filling_values=np.nan)
            hours_csv = data[:, 0]
            raw_csv   = data[:, 1]
            # Drop rows where hours or raw_ratio is NaN
            valid = np.isfinite(hours_csv) & np.isfinite(raw_csv)
            hours_csv = hours_csv[valid]
            raw_csv   = raw_csv[valid]
            max_hours = float(hours_csv[-1])
            _cache[key] = ('raw', hours_csv, raw_csv, max_hours)

    mode, hours_csv, raw_csv, max_hours = _cache[key]
    t_hours = np.clip(
        np.asarray(t_seconds, dtype=np.float64) / 3600.0,
        0.0, max_hours,
    )

    if mode == 'fit':
        from volume_ratios import V_ratio
        return np.asarray(V_ratio(t_hours, str(sg), norm), dtype=np.float64), max_hours
    else:
        return np.interp(t_hours, hours_csv, raw_csv).astype(np.float64), max_hours


def build_kdil_array(t_starts, dts, sg, norm, variant, dilution_dir):
    """Compute per-step kdil [s⁻¹].

    kdil[i] = max(0, ln(V(t+dt) / V(t)) / dt)
    For 'fit' variant: clamped to 0 (fits are monotonically increasing anyway).
    For 'raw' variant: negative values allowed — shrinking volume re-concentrates
    particles, consistent with the observed V(t) data.
    """
    t_ends = t_starts + dts
    V_start, max_hours = _get_V_at(t_starts, sg, norm, variant, dilution_dir)
    V_end,   _         = _get_V_at(t_ends,   sg, norm, variant, dilution_dir)
    safe = np.clip(V_end / np.maximum(V_start, 1e-300), 1e-300, None)
    kdil = np.log(safe) / dts
    if variant == 'fit':
        kdil = np.maximum(0.0, kdil)
    return kdil, V_start, max_hours


# =========================================================================
# Initial state
# =========================================================================

def _make_initial_state(nbins, xk):
    xk_np = np.array(xk)
    Nk = np.zeros(nbins)
    Mk = np.zeros((nbins, ICOMP))

    for k in range(nbins):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dk = np.sqrt(Dl * Dh)
        gmd_um = GMD_BG * 1e6
        np_init = (
            (N_BG * BOXVOL)
            / (np.sqrt(2.0 * PI) * Dk * np.log(GSD_BG))
            * np.exp(-(np.log(Dk / gmd_um) ** 2 / (2.0 * np.log(GSD_BG) ** 2)))
            * (Dh - Dl)
        )
        Nk[k] = max(np_init, 1e-3)
        Mk[k, SRTSO4] = Nk[k] * np.sqrt(xk_np[k] * xk_np[k + 1])

    so2_init = plume_so2_concentration(
        release_rate_ton_per_min=1.0,
        aircraft_speed_m_per_s=250.0,
        plume_radius_m=100.0,
        temp_K=TEMP,
        pres_Pa=PRES,
    )['molec_per_cm3']

    Gc = np.zeros(N_GAS_SPECIES)
    Gc[SRTSO4] = H2SO4_INIT * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO
    Gc[SRTSO2] = so2_init   * BOXVOL * (MW_SO2   / 1000.0) / AVOGADRO

    return jnp.array(Nk), jnp.array(Mk), jnp.array(Gc), float(so2_init)


# =========================================================================
# Single scenario runner
# =========================================================================

def run_scenario(sg, norm, variant, step_fn, nbins, xk, dilution_dir,
                 verbose=True):
    """Run one scenario with multi-resolution time stepping.

    Returns dict with time series arrays.
    """
    name = scenario_name(sg, norm, variant)

    # Build time schedule (capped at data max_hours)
    # First get max_hours from data
    _, _, max_hours = build_kdil_array(
        np.array([0.0]), np.array([DT3]), sg, norm, variant, dilution_dir)
    max_hours = min(max_hours, 168.0)

    t_starts, dts = build_time_schedule(max_hours)
    nsteps = len(t_starts)

    kdil_arr, V_arr, _ = build_kdil_array(
        t_starts, dts, sg, norm, variant, dilution_dir)

    if verbose:
        print(f'    {name}: {nsteps} steps, max={max_hours:.1f}h, '
              f'kdil ∈ [{kdil_arr.min():.2e}, {kdil_arr.max():.2e}] s⁻¹')

    Nk, Mk, Gc, so2_init = _make_initial_state(nbins, xk)

    temp   = jnp.float64(TEMP)
    pres   = jnp.float64(PRES)
    rh     = jnp.float64(RH)
    alpha  = jnp.float64(ALPHA)
    boxvol = jnp.float64(BOXVOL)
    oh_conc = jnp.float64(OH_CONC)

    Nk_bg = jnp.zeros_like(Nk)
    Mk_bg = jnp.zeros_like(Mk)
    Gc_bg = jnp.zeros_like(Gc)

    # Storage (one record per step)
    Nk_every     = np.zeros((nsteps, nbins))
    Mk_dry_every = np.zeros((nsteps, nbins))
    N_tot_every  = np.zeros(nsteps)
    M_dry_every  = np.zeros(nsteps)
    Gc_SO2_every = np.zeros(nsteps)
    Gc_SO4_every = np.zeros(nsteps)
    tracer_every = np.zeros(nsteps)

    so2_to_molec = AVOGADRO / (MW_SO2   / 1000.0) / BOXVOL
    so4_to_molec = AVOGADRO / (MW_H2SO4 / 1000.0) / BOXVOL

    tracer = jnp.float64(1.0)
    t0 = time.time()

    for i in range(nsteps):
        # Record state before stepping
        Nk_np = np.array(Nk)
        Mk_np = np.array(Mk)
        Gc_np = np.array(Gc)

        Nk_every[i]     = Nk_np
        Mk_dry_every[i] = Mk_np[:, :SRTH2O].sum(axis=1)
        N_tot_every[i]  = float(jnp.sum(Nk))
        M_dry_every[i]  = float(jnp.sum(Mk[:, :SRTH2O]))
        Gc_SO2_every[i] = Gc_np[SRTSO2] * so2_to_molec
        Gc_SO4_every[i] = Gc_np[SRTSO4] * so4_to_molec
        tracer_every[i] = float(tracer)

        # Progress: report each day boundary
        t_s = float(t_starts[i])
        if verbose and i > 0 and abs(t_s % 86400) < float(dts[i]):
            elapsed = time.time() - t0
            print(f'      d{t_s/86400:.0f}  N={N_tot_every[i]/BOXVOL:.2e}/cm³  '
                  f'elapsed={elapsed:.0f}s')

        kdil_i = jnp.float64(kdil_arr[i])
        dt_i   = jnp.float64(dts[i])

        Nk, Mk, Gc = step_fn(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt_i,
            oh_conc=oh_conc,
            org_conc=NUC_ORG, nh3_conc=NUC_NH3, fion=NUC_FION,
            kdil=kdil_i, Nk_bg=Nk_bg, Mk_bg=Mk_bg, Gc_bg=Gc_bg,
        )
        tracer = tracer * jnp.exp(-kdil_i * dt_i)

    wall_time = time.time() - t0
    if verbose:
        print(f'      → {wall_time:.1f}s wall, '
              f'N_final={N_tot_every[-1]/BOXVOL:.2e}/cm³')

    return {
        't_seconds':     t_starts,
        'dts':           dts,
        'Nk_every':      Nk_every,
        'Mk_dry_every':  Mk_dry_every,
        'N_tot_every':   N_tot_every,
        'M_dry_every':   M_dry_every,
        'Gc_SO2_molec':  Gc_SO2_every,
        'Gc_SO4_molec':  Gc_SO4_every,
        'tracer_every':  tracer_every,
        'kdil_every':    kdil_arr,
        'V_array':       V_arr,
        'wall_time_s':   wall_time,
        'so2_init_molec_cm3': so2_init,
        'nbins':         nbins,
        'max_hours':     max_hours,
        'nsteps':        nsteps,
    }


# =========================================================================
# Run all 30 scenarios
# =========================================================================

def run_all(dilution_dir=_DEFAULT_DILUTION_DIR, outdir=_OUTDIR,
            skip_existing=True, verbose=True):
    nbins = NBINS
    xk = make_grid(nbins, XK0, 2.0)
    os.makedirs(outdir, exist_ok=True)

    so2_init = plume_so2_concentration(
        release_rate_ton_per_min=1.0, aircraft_speed_m_per_s=250.0,
        plume_radius_m=100.0, temp_K=TEMP, pres_Pa=PRES,
    )['molec_per_cm3']

    print('=' * 72)
    print(f'SAI Dilution Ensemble — up to 168h, 30 scenarios')
    print(f'  Conditions : T={TEMP}K, P={PRES/100:.0f}hPa, RH={RH}')
    print(f'  Initial SO2: {so2_init:.2e} molec/cm³')
    print(f'  OH={OH_CONC:.0e} molec/cm³, NH3=0, organics=0, ions=0')
    print(f'  Background : pure SO4, {N_BG} #/cm³, GMD={GMD_BG*1e9:.0f}nm')
    print(f'  Time steps : dt=1s (0–20min), 10s (20min–4h), 60s (4h–end)')
    print(f'  Grid       : {nbins} bins')
    print(f'  Dilution   : {dilution_dir}')
    print(f'  Output     : {outdir}')
    print('=' * 72)

    # Single step function shared across all 30 scenarios
    step_fn = make_step(
        ['so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'],
        cond_method='ppm_jit',
    )

    # JIT warmup (once)
    print('  JIT warmup ... ', end='', flush=True)
    Nk0, Mk0, Gc0, _ = _make_initial_state(nbins, xk)
    Nk_bg = jnp.zeros_like(Nk0)
    Mk_bg = jnp.zeros_like(Mk0)
    Gc_bg = jnp.zeros_like(Gc0)
    for _dt_w in [DT1, DT2, DT3]:   # warm up all three dt values
        _ = step_fn(
            Nk0, Mk0, Gc0, xk,
            jnp.float64(TEMP), jnp.float64(PRES),
            jnp.float64(BOXVOL), jnp.float64(RH),
            jnp.float64(ALPHA), jnp.float64(_dt_w),
            oh_conc=jnp.float64(OH_CONC),
            org_conc=NUC_ORG, nh3_conc=NUC_NH3, fion=NUC_FION,
            kdil=jnp.float64(1e-5),
            Nk_bg=Nk_bg, Mk_bg=Mk_bg, Gc_bg=Gc_bg,
        )
    print('done.', flush=True)

    scenarios = all_scenarios()
    t_total = time.time()
    n_done, n_skip, n_err = 0, 0, 0

    for idx, (sg, norm, variant) in enumerate(scenarios):
        name = scenario_name(sg, norm, variant)
        npz_path = os.path.join(outdir, f'{name}.npz')

        if skip_existing and os.path.exists(npz_path):
            print(f'  [{idx+1:2d}/30] {name:<44s} SKIP')
            n_skip += 1
            continue

        print(f'  [{idx+1:2d}/30] {name}')
        try:
            res = run_scenario(sg, norm, variant, step_fn, nbins, xk,
                               dilution_dir, verbose=verbose)
        except Exception as exc:
            print(f'    ERROR: {exc}')
            n_err += 1
            continue

        save_dict = {k: v for k, v in res.items()
                     if isinstance(v, (np.ndarray, float, int))}
        save_dict['subgroup']      = np.bytes_(sg)
        save_dict['normalization'] = np.bytes_(norm)
        save_dict['variant']       = np.bytes_(variant)
        np.savez_compressed(npz_path, **save_dict)
        n_done += 1

    print('-' * 72)
    print(f'Done: {n_done} ran, {n_skip} skipped, {n_err} errors  '
          f'({time.time()-t_total:.0f}s total)')
    print(f'Results in: {outdir}')


# =========================================================================
# Data loading
# =========================================================================

def load_scenario(sg, norm, variant, outdir=_OUTDIR):
    name = scenario_name(sg, norm, variant)
    path = os.path.join(outdir, f'{name}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Missing: {path}\n'
            f'Run: python -m experimental_case.run_dilution_scenarios')
    return dict(np.load(path, allow_pickle=True))


def load_all(outdir=_OUTDIR):
    results = {}
    for sg, norm, variant in all_scenarios():
        try:
            results[scenario_name(sg, norm, variant)] = \
                load_scenario(sg, norm, variant, outdir)
        except FileNotFoundError:
            pass
    return results


# =========================================================================
# Plot helpers
# =========================================================================

def _grid_geometry(nbins=NBINS):
    """Return (dp_nm, dlogDp, xk_np) for the default grid."""
    xk_np = np.array(make_grid(nbins, XK0, 2.0))
    m_mid = np.sqrt(xk_np[:-1] * xk_np[1:])
    dp_m  = (6.0 * m_mid / (DENS_INIT * PI)) ** (1.0 / 3.0)
    dp_nm = dp_m * 1e9
    dp_edges = (6.0 * xk_np / (DENS_INIT * PI)) ** (1.0 / 3.0)
    dlogDp   = np.log10(dp_edges[1:] / dp_edges[:-1])
    return dp_nm, dlogDp, xk_np, dp_m


def _snap_idx(t_seconds, snap_t):
    """Index of the recorded step closest to snap_t [s]."""
    return int(np.argmin(np.abs(t_seconds - snap_t)))


def _surface_conc(Nk_cm3, dp_m):
    """Particle surface area concentration dA/dlogDp basis [m²/cm³]."""
    A_per_bin = np.pi * dp_m ** 2 * Nk_cm3  # [m² per cm³] per bin
    return A_per_bin


# =========================================================================
# Plots
# =========================================================================

def plot_banana(d, sg, norm, variant, outdir):
    """Banana plot: dN/dlogDp vs time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    dp_nm, dlogDp, _, _ = _grid_geometry(int(d['nbins']))
    t_h = d['t_seconds'] / 3600.0
    Nk_all = d['Nk_every']
    dNdlogDp = (Nk_all / BOXVOL) / dlogDp[np.newaxis, :]

    # Sub-sample for pcolormesh (max ~2000 time points)
    nsub = max(1, len(t_h) // 2000)
    t_sub  = t_h[::nsub]
    dat_sub = dNdlogDp[::nsub]

    fig, ax = plt.subplots(figsize=(12, 5))
    pcm = ax.pcolormesh(
        t_sub, dp_nm, dat_sub.T,
        norm=mcolors.LogNorm(vmin=1.0, vmax=1e8),
        cmap='inferno', shading='nearest',
    )
    ax.set_yscale('log')
    ax.set_ylim(1, 2e4)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Dp [nm]')
    name = scenario_name(sg, norm, variant)
    ax.set_title(f'Banana — {name}  (T={TEMP}K, P={PRES/100:.0f}hPa)')
    fig.colorbar(pcm, ax=ax, label='dN/dlogDp [#/cm³]')
    fig.tight_layout()

    out = os.path.join(outdir, 'figs', f'banana_{name}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_size_distributions(d, sg, norm, variant, outdir):
    """dN/dA/dM dlogDp at h0, d1, d2, d7 — log-y and linear-y."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dp_nm, dlogDp, _, dp_m = _grid_geometry(int(d['nbins']))
    t_s    = d['t_seconds']
    kg_to_ug_m3 = 1e9 / (BOXVOL * 1e-6)   # [kg/cell → µg/m³]

    # Build snapshot records; skip if time exceeds data range
    snaps = []
    for label, t_snap in SNAPSHOT_TIMES_S.items():
        if t_snap > t_s[-1] + float(d.get('dts', [DT3])[-1]):
            continue  # beyond this scenario's data
        idx = _snap_idx(t_s, t_snap)
        Nk_cm3  = d['Nk_every'][idx] / BOXVOL
        Mk_dry  = d['Mk_dry_every'][idx]  # kg/cell per bin
        snaps.append({
            'label': label,
            't_h':   t_s[idx] / 3600.0,
            'dN':    Nk_cm3 / dlogDp,
            'dA':    _surface_conc(Nk_cm3, dp_m) / dlogDp,
            'dM':    (Mk_dry * kg_to_ug_m3) / dlogDp,
        })

    if not snaps:
        return []

    snap_colors = ['#333333', '#1E88E5', '#E53935', '#F57C00', '#7B1FA2']
    name = scenario_name(sg, norm, variant)

    saved = []
    for qty, ylabel_log, ylabel_lin, ylabel_unit in [
        ('dN', 'dN/dlogDp [#/cm³]',  'dN/dlogDp [#/cm³]',  ''),
        ('dA', 'dA/dlogDp [m²/cm³]', 'dA/dlogDp [m²/cm³]', ''),
        ('dM', 'dM/dlogDp [µg/m³]',  'dM/dlogDp [µg/m³]',  ''),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'{qty} size distribution — {name}\n'
            f'(T={TEMP}K, P={PRES/100:.0f}hPa)',
            fontsize=11, fontweight='bold'
        )

        for s, c in zip(snaps, snap_colors[:len(snaps)]):
            vals = s[qty]
            lbl  = f"{s['label']} (t={s['t_h']:.1f}h)"
            axes[0].loglog(dp_nm, np.maximum(vals, 1e-30), color=c, lw=1.8, label=lbl)
            axes[1].semilogx(dp_nm, vals, color=c, lw=1.8, label=lbl)

        for ax, title in zip(axes, ['log-y', 'linear-y']):
            ax.set_xlabel('Dp [nm]')
            ax.set_ylabel(ylabel_log if title == 'log-y' else ylabel_lin)
            ax.set_title(title)
            ax.set_xlim(1, 2e4)
            ax.grid(True, alpha=0.3, which='both')
            ax.legend(fontsize=8)
            if title == 'log-y' and qty == 'dN':
                ax.set_ylim(bottom=1.0)
            elif title == 'linear-y' and qty == 'dN':
                ax.set_ylim(0, 1e4)
            elif title == 'linear-y':
                ax.set_ylim(bottom=0)

        fig.tight_layout()
        out = os.path.join(outdir, 'figs', f'{qty}_{name}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(out)

    return saved


def plot_all_for_scenario(sg, norm, variant, outdir=_OUTDIR):
    """Generate banana + 3×2 size-dist plots for one scenario."""
    d = load_scenario(sg, norm, variant, outdir)
    figs = []
    figs.append(plot_banana(d, sg, norm, variant, outdir))
    figs.extend(plot_size_distributions(d, sg, norm, variant, outdir))
    return figs


def plot_all_scenarios(outdir=_OUTDIR):
    """Generate all plots for all available scenarios."""
    import matplotlib
    matplotlib.use('Agg')
    print(f'  Generating plots → {os.path.join(outdir, "figs")}/...')
    for sg, norm, variant in all_scenarios():
        name = scenario_name(sg, norm, variant)
        try:
            figs = plot_all_for_scenario(sg, norm, variant, outdir)
            print(f'    {name}: {len(figs)} figures')
        except FileNotFoundError:
            print(f'    {name}: no data, skipped')


def plot_ensemble_summary(outdir=_OUTDIR):
    """N_total and M_dry envelope plot (all 30 scenarios)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey='row')
    fig.suptitle(
        f'SAI Dilution Ensemble — N_tot and M_dry  (T={TEMP}K, {PRES/100:.0f}hPa)',
        fontsize=12, fontweight='bold')

    colors  = {'first_hour_mean': '#E53935', 'first_value': '#1E88E5',
               'smoothed': '#43A047'}
    ls_map  = {'raw': '--', 'fit': '-'}

    for col, sg in enumerate(SUBGROUPS):
        ax_n = axes[0, col]
        ax_m = axes[1, col]
        for norm in NORMS:
            for variant in VARIANTS:
                try:
                    d = load_scenario(sg, norm, variant, outdir)
                except FileNotFoundError:
                    continue
                t_h  = d['t_seconds'] / 3600.0
                N_cm3 = d['N_tot_every'] / BOXVOL
                M_ug  = d['M_dry_every'] * 1e9 / (BOXVOL * 1e-6)
                c  = colors[norm]
                ls = ls_map[variant]
                lw = 1.4 if variant == 'fit' else 0.7
                ax_n.semilogy(t_h, N_cm3, color=c, ls=ls, lw=lw, alpha=0.85)
                ax_m.semilogy(t_h, M_ug,  color=c, ls=ls, lw=lw, alpha=0.85)

        ax_n.set_title(f'sg{sg}', fontsize=10, fontweight='bold')
        for ax, yl in zip([ax_n, ax_m],
                          ['N_tot [#/cm³]', r'M$_{dry}$ [µg/m³]']):
            ax.set_xlabel('Time [h]')
            if col == 0:
                ax.set_ylabel(yl)
            ax.grid(True, alpha=0.3, which='both')

    # Legend
    from matplotlib.lines import Line2D
    handles = (
        [Line2D([0], [0], color=colors[n], lw=2, label=n) for n in NORMS]
        + [Line2D([0], [0], color='k', ls=ls_map[v], lw=1.5, label=v)
           for v in VARIANTS]
    )
    axes[0, -1].legend(handles=handles, fontsize=7)

    fig.tight_layout()
    out = os.path.join(outdir, 'ensemble_summary.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SAI dilution ensemble — 30 scenarios, multi-resolution stepping')
    parser.add_argument('--dilution-dir', default=_DEFAULT_DILUTION_DIR,
                        help='Path to CSV files and volume_ratios.py')
    parser.add_argument('--outdir', default=_OUTDIR,
                        help='Output directory for NPZ files and figures')
    parser.add_argument('--scenario',
                        help='Run a single scenario (e.g. sg02_first_value_raw)')
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-run even if NPZ exists')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip simulations, regenerate plots only')
    args = parser.parse_args()

    if not args.plot_only:
        if args.scenario:
            # Parse scenario name
            for var in VARIANTS:
                if args.scenario.endswith('_' + var):
                    name_base = args.scenario[: -(len(var) + 1)]
                    variant = var
                    break
            else:
                parser.error(f'Scenario must end with _raw or _fit: {args.scenario}')

            matched = None
            for sg in SUBGROUPS:
                prefix = _sg_prefix(sg) + '_'
                if name_base.startswith(prefix):
                    norm_cand = name_base[len(prefix):]
                    if norm_cand in NORMS:
                        matched = (sg, norm_cand, variant)
                        break
            if matched is None:
                parser.error(f'Cannot parse scenario: {args.scenario}')

            sg, norm, variant = matched
            nbins = NBINS
            xk = make_grid(nbins, XK0, 2.0)
            step_fn = make_step(
                ['so2_chemistry', 'nucleation', 'coagulation',
                 'condensation', 'dilution'],
                cond_method='ppm_jit',
            )
            res = run_scenario(sg, norm, variant, step_fn, nbins, xk,
                               args.dilution_dir, verbose=True)
            os.makedirs(args.outdir, exist_ok=True)
            name = scenario_name(sg, norm, variant)
            npz_path = os.path.join(args.outdir, f'{name}.npz')
            save_dict = {k: v for k, v in res.items()
                         if isinstance(v, (np.ndarray, float, int))}
            np.savez_compressed(npz_path, **save_dict)
            print(f'Saved: {npz_path}')
        else:
            run_all(
                dilution_dir=args.dilution_dir,
                outdir=args.outdir,
                skip_existing=not args.no_skip,
            )

    plot_all_scenarios(args.outdir)
    plot_ensemble_summary(args.outdir)
