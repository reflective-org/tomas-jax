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
    make_grid, make_grid_80bin, XK0,
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
    't0': 0.0,
    'd1': 24  * 3600.0,
    'd2': 48  * 3600.0,
    'd3': 72  * 3600.0,
    'd4': 96  * 3600.0,
    'd5': 120 * 3600.0,
    'd6': 144 * 3600.0,
    'd7': 168 * 3600.0,
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

def _make_initial_state(nbins, xk, bg_dist='lognormal', h2so4_init=H2SO4_INIT):
    """
    bg_dist : 'lognormal'  — sparse 5 #/cm³ lognormal at 80 nm (default)
              'tabulated'  — mapped from the measured stratospheric distribution
    """
    xk_np = np.array(xk)
    Nk = np.zeros(nbins)
    Mk = np.zeros((nbins, ICOMP))

    if bg_dist in ('tabulated', 'tabulated2'):
        from .background_aerosol_distribution import get_initial_state
        Nk, Mk = get_initial_state(nbins=nbins, boxvol=BOXVOL, dist=bg_dist)
        Nk = np.array(Nk)
        Mk = np.array(Mk)
    else:
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
    Gc[SRTSO4] = h2so4_init * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO
    Gc[SRTSO2] = so2_init   * BOXVOL * (MW_SO2   / 1000.0) / AVOGADRO

    return jnp.array(Nk), jnp.array(Mk), jnp.array(Gc), float(so2_init)


# =========================================================================
# Single scenario runner
# =========================================================================

def run_scenario(sg, norm, variant, step_fn, nbins, xk, dilution_dir,
                 bg_dist='lognormal', h2so4_init=H2SO4_INIT, verbose=True):
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

    Nk, Mk, Gc, so2_init = _make_initial_state(nbins, xk, bg_dist, h2so4_init)

    temp   = jnp.float64(TEMP)
    pres   = jnp.float64(PRES)
    rh     = jnp.float64(RH)
    alpha  = jnp.float64(ALPHA)
    boxvol = jnp.float64(BOXVOL)
    oh_conc = jnp.float64(OH_CONC)

    # Zero background arrays (used when box shrinks, regardless of bg_dist)
    _zeros_Nk = jnp.zeros_like(Nk)
    _zeros_Mk = jnp.zeros_like(Mk)
    _zeros_Gc = jnp.zeros_like(Gc)

    if bg_dist in ('tabulated', 'tabulated2'):
        # Entrain ambient stratospheric aerosol when box expands
        from .background_aerosol_distribution import get_initial_state
        Nk_bg_np, Mk_bg_np = get_initial_state(nbins=nbins, boxvol=BOXVOL,
                                               dist=bg_dist)
        Nk_bg = jnp.array(Nk_bg_np)
        Mk_bg = jnp.array(Mk_bg_np)
        Gc_bg = jnp.zeros(N_GAS_SPECIES, dtype=jnp.float64)
        Gc_bg = Gc_bg.at[SRTSO4].set(
            h2so4_init * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO)
    else:
        Nk_bg = _zeros_Nk
        Mk_bg = _zeros_Mk
        Gc_bg = _zeros_Gc

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

        # When box shrinks (kdil < 0): re-concentrate only, no entrainment.
        # When box expands (kdil >= 0): entrain ambient background air.
        if float(kdil_arr[i]) >= 0.0:
            _Nk_bg, _Mk_bg, _Gc_bg = Nk_bg, Mk_bg, Gc_bg
        else:
            _Nk_bg, _Mk_bg, _Gc_bg = _zeros_Nk, _zeros_Mk, _zeros_Gc

        Nk, Mk, Gc = step_fn(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt_i,
            oh_conc=oh_conc,
            org_conc=NUC_ORG, nh3_conc=NUC_NH3, fion=NUC_FION,
            kdil=kdil_i, Nk_bg=_Nk_bg, Mk_bg=_Mk_bg, Gc_bg=_Gc_bg,
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
            skip_existing=True, bg_dist='lognormal', h2so4_init=H2SO4_INIT,
            nbins=40, verbose=True):
    if nbins == 80:
        xk = make_grid_80bin()
    else:
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
    bg_label = {'tabulated':  'tabulated dist1 (peak 45nm)',
                'tabulated2': 'tabulated dist2 (peak 150nm)'}.get(
                bg_dist, f'lognormal {N_BG} #/cm³ GMD={GMD_BG*1e9:.0f}nm')
    print(f'  Background : {bg_label}')
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
    h2so4_label = f'{h2so4_init:.0e}' if h2so4_init > 0 else '0'
    print(f'  H2SO4 init : {h2so4_label} molec/cm³')
    print('  JIT warmup ... ', end='', flush=True)
    Nk0, Mk0, Gc0, _ = _make_initial_state(nbins, xk, bg_dist, h2so4_init)
    if bg_dist in ('tabulated', 'tabulated2'):
        from .background_aerosol_distribution import get_initial_state
        _Nk_bg_np, _Mk_bg_np = get_initial_state(nbins=nbins, boxvol=BOXVOL,
                                                  dist=bg_dist)
        Nk_bg = jnp.array(_Nk_bg_np)
        Mk_bg = jnp.array(_Mk_bg_np)
        Gc_bg = jnp.zeros(N_GAS_SPECIES, dtype=jnp.float64)
        Gc_bg = Gc_bg.at[SRTSO4].set(
            H2SO4_INIT * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO)
    else:
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
                               dilution_dir, bg_dist=bg_dist,
                               h2so4_init=h2so4_init, verbose=verbose)
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
    """Return (dp_nm, dlogDp, xk_np) for the correct grid."""
    if nbins == 80:
        xk_np = np.array(make_grid_80bin())
    else:
        xk_np = np.array(make_grid(nbins, XK0, 2.0))
    m_mid = np.sqrt(xk_np[:-1] * xk_np[1:])
    dp_m  = (6.0 * m_mid / (DENS_INIT * PI)) ** (1.0 / 3.0)
    dp_nm = dp_m * 1e9
    dp_edges = (6.0 * xk_np / (DENS_INIT * PI)) ** (1.0 / 3.0)
    dlogDp   = np.log10(dp_edges[1:] / dp_edges[:-1])
    return dp_nm, dlogDp, xk_np, dp_m


def _snap_avg(t_seconds, Nk_every, Mk_dry_every, snap_t, window_s=300.0):
    """Average Nk and Mk_dry over steps within ±window_s of snap_t [s].

    Falls back to the single nearest step if no steps fall in the window.
    """
    mask = np.abs(t_seconds - snap_t) <= window_s
    if mask.sum() == 0:
        idx = int(np.argmin(np.abs(t_seconds - snap_t)))
        mask = np.zeros(len(t_seconds), dtype=bool)
        mask[idx] = True
    t_actual = float(t_seconds[mask].mean())
    Nk_mean  = Nk_every[mask].mean(axis=0)
    Mk_mean  = Mk_dry_every[mask].mean(axis=0)
    return Nk_mean, Mk_mean, t_actual


def _surface_conc(Nk_cm3, dp_m):
    """Particle surface area concentration dA/dlogDp basis [m²/cm³]."""
    A_per_bin = np.pi * dp_m ** 2 * Nk_cm3  # [m² per cm³] per bin
    return A_per_bin


# =========================================================================
# Plots
# =========================================================================

_RF_DP_LO = 400.0   # nm — optimal RF diameter lower bound
_RF_DP_HI = 440.0   # nm — optimal RF diameter upper bound


def _style_banana_ax(ax):
    """Keep bottom and left spines only, remove top and right."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', direction='out', length=5, width=1.0,
                   labelsize=9)


def _add_rf_band_banana(ax):
    """Horizontal shaded band + dashed lines at the optimal RF size range."""
    ax.axhspan(_RF_DP_LO, _RF_DP_HI, color='white', alpha=0.12, zorder=2)
    for dp in (_RF_DP_LO, _RF_DP_HI):
        ax.axhline(dp, color='white', lw=0.9, ls='--', alpha=0.7, zorder=3)


def _add_rf_band_sizedist(ax):
    """Vertical shaded band + dashed lines at the optimal RF size range."""
    ax.axvspan(_RF_DP_LO, _RF_DP_HI, color='#1565C0', alpha=0.10, zorder=1)
    for dp in (_RF_DP_LO, _RF_DP_HI):
        ax.axvline(dp, color='#1565C0', lw=1.0, ls='--', alpha=0.7, zorder=2)


def plot_banana(d, sg, norm, variant, outdir):
    """Banana plot: dN/dlogDp vs time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    dp_nm, dlogDp, _, _ = _grid_geometry(int(d['nbins']))
    t_h = d['t_seconds'] / 3600.0
    dNdlogDp = (d['Nk_every'] / BOXVOL) / dlogDp[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(12, 5))

    pcm = ax.pcolormesh(
        t_h, dp_nm, dNdlogDp.T,
        norm=mcolors.LogNorm(vmin=1.0, vmax=1e8),
        cmap='inferno', shading='nearest',
    )
    ax.set_yscale('log')
    ax.set_ylim(1, 2e4)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Dp [nm]')
    name = scenario_name(sg, norm, variant)
    ax.set_title(f'Banana — {name}  (T={TEMP}K, P={PRES/100:.0f}hPa)')
    _style_banana_ax(ax)
    _add_rf_band_banana(ax)

    fig.colorbar(pcm, ax=ax, label='dN/dlogDp [#/cm³]', pad=0.01)
    fig.tight_layout()

    out = os.path.join(outdir, 'figs', f'banana_{name}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_size_distributions(d, sg, norm, variant, outdir):
    """dN/dA/dM dlogDp at snapshot times — log-y and linear-y."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='ticks', font_scale=1.0)

    dp_nm, dlogDp, _, dp_m = _grid_geometry(int(d['nbins']))
    t_s    = d['t_seconds']
    kg_to_ug_m3 = 1e9 / (BOXVOL * 1e-6)

    snaps = []
    for label, t_snap in SNAPSHOT_TIMES_S.items():
        if t_snap > t_s[-1] + float(d.get('dts', [DT3])[-1]):
            continue
        Nk_avg, Mk_avg, t_actual = _snap_avg(
            t_s, d['Nk_every'], d['Mk_dry_every'], t_snap, window_s=300.0)
        Nk_cm3 = Nk_avg / BOXVOL
        snaps.append({
            'label': label,
            't_h':   t_actual / 3600.0,
            'dN':    Nk_cm3 / dlogDp,
            'dA':    _surface_conc(Nk_cm3, dp_m) / dlogDp,
            'dM':    (Mk_avg * kg_to_ug_m3) / dlogDp,
        })

    if not snaps:
        sns.reset_defaults()
        return []

    snap_colors = ['#333333', '#1E88E5', '#43A047', '#E53935',
                   '#FB8C00', '#8E24AA', '#00ACC1', '#795548']
    name = scenario_name(sg, norm, variant)

    saved = []
    for qty, ylabel_log, ylabel_lin in [
        ('dN', 'dN/dlogDp [#/cm³]',  'dN/dlogDp [#/cm³]'),
        ('dA', 'dA/dlogDp [m²/cm³]', 'dA/dlogDp [m²/cm³]'),
        ('dM', 'dM/dlogDp [µg/m³]',  'dM/dlogDp [µg/m³]'),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'{qty} size distribution — {name}\n'
            f'(T={TEMP}K, P={PRES/100:.0f}hPa)',
            fontsize=11, fontweight='bold'
        )

        for s, c in zip(snaps, snap_colors[:len(snaps)]):
            lbl = f"{s['label']} (t={s['t_h']:.1f}h)"
            axes[0].loglog(dp_nm, np.maximum(s[qty], 1e-30),
                           color=c, lw=1.8, label=lbl)
            axes[1].semilogx(dp_nm, s[qty], color=c, lw=1.8, label=lbl)

        for ax, title, ylabel in zip(axes,
                                     ['log-y', 'linear-y'],
                                     [ylabel_log, ylabel_lin]):
            _add_rf_band_sizedist(ax)
            sns.despine(ax=ax)
            ax.set_xlabel('Dp [nm]')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlim(1, 2e4)
            ax.grid(True, alpha=0.25, which='both', axis='both')
            ax.legend(fontsize=8, frameon=False)
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

    sns.reset_defaults()
    return saved


def plot_subgroup_banana(sg, outdir):
    """3-row × 2-col banana grid for one subgroup."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f'Banana plots — sg{sg}  (T={TEMP}K, P={PRES/100:.0f}hPa)',
                 fontsize=13, fontweight='bold')

    norm_labels = {'first_hour_mean': 'First hour mean',
                   'first_value': 'First value',
                   'smoothed': 'Smoothed'}
    log_norm = mcolors.LogNorm(vmin=1.0, vmax=1e8)
    pcm = None

    for row, norm in enumerate(NORMS):
        for col, variant in enumerate(VARIANTS):
            ax = axes[row, col]
            try:
                d = load_scenario(sg, norm, variant, outdir)
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                continue

            nbins = int(d['nbins'])
            dp_nm, dlogDp, _, _ = _grid_geometry(nbins)
            t_h = d['t_seconds'] / 3600.0
            dNdlogDp = (d['Nk_every'] / BOXVOL) / dlogDp[np.newaxis, :]

            pcm = ax.pcolormesh(t_h, dp_nm, dNdlogDp.T,
                                norm=log_norm, cmap='inferno', shading='nearest')
            ax.set_yscale('log')
            ax.set_ylim(1, 2e4)
            _style_banana_ax(ax)
            _add_rf_band_banana(ax)

            if col == 0:
                ax.set_ylabel(f'{norm_labels[norm]}\nDp [nm]', fontsize=9)
            if row == 2:
                ax.set_xlabel('Time [h]')
            ax.set_title(variant, fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    if pcm is not None:
        cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
        fig.colorbar(pcm, cax=cax, label='dN/dlogDp [#/cm³]')

    out = os.path.join(outdir, 'figs', f'subgroup_banana_sg{sg}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_subgroup_sizedist(sg, qty, outdir):
    """3-row × 2-col size distribution grid for one subgroup and quantity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='ticks', font_scale=1.0)

    qty_labels = {'dN': 'dN/dlogDp [#/cm³]',
                  'dA': 'dA/dlogDp [m²/cm³]',
                  'dM': 'dM/dlogDp [µg/m³]'}
    snap_colors = ['#333333', '#1E88E5', '#43A047', '#E53935',
                   '#FB8C00', '#8E24AA', '#00ACC1', '#795548']
    norm_labels = {'first_hour_mean': 'First hour mean',
                   'first_value': 'First value',
                   'smoothed': 'Smoothed'}

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f'{qty} size distributions — sg{sg}  '
                 f'(T={TEMP}K, P={PRES/100:.0f}hPa)',
                 fontsize=13, fontweight='bold')

    kg_to_ug_m3 = 1e9 / (BOXVOL * 1e-6)
    all_handles, all_labels = [], []

    for row, norm in enumerate(NORMS):
        for col, variant in enumerate(VARIANTS):
            ax = axes[row, col]
            try:
                d = load_scenario(sg, norm, variant, outdir)
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                sns.despine(ax=ax)
                continue

            nbins = int(d['nbins'])
            dp_nm, dlogDp, _, dp_m = _grid_geometry(nbins)
            t_s = d['t_seconds']

            for (label, t_snap), c in zip(SNAPSHOT_TIMES_S.items(),
                                          snap_colors[:len(SNAPSHOT_TIMES_S)]):
                if t_snap > t_s[-1] + float(d.get('dts', [DT3])[-1]):
                    continue
                Nk_avg, Mk_avg, t_actual = _snap_avg(
                    t_s, d['Nk_every'], d['Mk_dry_every'], t_snap)
                Nk_cm3 = Nk_avg / BOXVOL
                if qty == 'dN':
                    vals = Nk_cm3 / dlogDp
                elif qty == 'dA':
                    vals = _surface_conc(Nk_cm3, dp_m) / dlogDp
                else:
                    vals = (Mk_avg * kg_to_ug_m3) / dlogDp

                ln, = ax.loglog(dp_nm, np.maximum(vals, 1e-30), color=c,
                                lw=1.5, label=f'{label} ({t_actual/3600:.1f}h)')
                if row == 0 and col == 0:
                    all_handles.append(ln)
                    all_labels.append(f'{label} ({t_actual/3600:.1f}h)')

            _add_rf_band_sizedist(ax)
            sns.despine(ax=ax)
            ax.grid(True, alpha=0.25, which='both')
            ax.set_xlim(1, 2e4)
            if qty == 'dN':
                ax.set_ylim(bottom=1.0)

            if col == 0:
                ax.set_ylabel(f'{norm_labels[norm]}\n{qty_labels[qty]}',
                              fontsize=9)
            if row == 2:
                ax.set_xlabel('Dp [nm]')
            ax.set_title(variant, fontsize=10)

    if all_handles:
        fig.legend(all_handles, all_labels, loc='lower center',
                   ncol=len(all_handles), fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(outdir, 'figs', f'subgroup_{qty}_sg{sg}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    sns.reset_defaults()
    return out


def plot_gas_timeseries(d, sg, norm, variant, outdir):
    """SO2 and H2SO4 gas-phase concentration timeseries."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='ticks', font_scale=1.0)

    t_h  = d['t_seconds'] / 3600.0
    so2  = d['Gc_SO2_molec']
    h2so4 = d['Gc_SO4_molec']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    name = scenario_name(sg, norm, variant)
    fig.suptitle(f'Gas-phase concentrations — {name}\n'
                 f'(T={TEMP}K, P={PRES/100:.0f}hPa)',
                 fontsize=11, fontweight='bold')

    axes[0].semilogy(t_h, np.maximum(so2,   1e-10), color='#E53935', lw=1.8)
    axes[1].semilogy(t_h, np.maximum(h2so4, 1e-10), color='#1E88E5', lw=1.8)

    axes[0].set_xlabel('Time [h]')
    axes[0].set_ylabel('SO₂ [molec/cm³]')
    axes[0].set_title('SO₂')

    axes[1].set_xlabel('Time [h]')
    axes[1].set_ylabel('H₂SO₄ [molec/cm³]')
    axes[1].set_title('H₂SO₄')

    for ax in axes:
        sns.despine(ax=ax)
        ax.grid(True, alpha=0.25, which='both')

    fig.tight_layout()
    out = os.path.join(outdir, 'figs', f'gas_{name}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    sns.reset_defaults()
    return out


def plot_subgroup_gas(sg, outdir):
    """3-row × 2-col SO2 and H2SO4 timeseries grid for one subgroup."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='ticks', font_scale=1.0)

    norm_labels = {'first_hour_mean': 'First hour mean',
                   'first_value':     'First value',
                   'smoothed':        'Smoothed'}

    # 3 rows (norms) × 2 cols (raw/fit) × 2 panels (SO2, H2SO4) — use a 3×4 grid
    fig, axes = plt.subplots(3, 4, figsize=(22, 12), sharex=True)
    fig.suptitle(f'Gas-phase concentrations — sg{sg}  '
                 f'(T={TEMP}K, P={PRES/100:.0f}hPa)',
                 fontsize=13, fontweight='bold')

    for row, norm in enumerate(NORMS):
        for col, variant in enumerate(VARIANTS):
            ax_so2  = axes[row, col * 2]
            ax_h2so4 = axes[row, col * 2 + 1]
            try:
                d = load_scenario(sg, norm, variant, outdir)
            except FileNotFoundError:
                for ax in (ax_so2, ax_h2so4):
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                            transform=ax.transAxes, color='gray')
                    sns.despine(ax=ax)
                continue

            t_h   = d['t_seconds'] / 3600.0
            so2   = np.maximum(d['Gc_SO2_molec'],  1e-10)
            h2so4 = np.maximum(d['Gc_SO4_molec'],  1e-10)

            ax_so2.semilogy(t_h, so2,   color='#E53935', lw=1.5)
            ax_h2so4.semilogy(t_h, h2so4, color='#1E88E5', lw=1.5)

            for ax in (ax_so2, ax_h2so4):
                sns.despine(ax=ax)
                ax.grid(True, alpha=0.25, which='both')

            if row == 2:
                ax_so2.set_xlabel('Time [h]')
                ax_h2so4.set_xlabel('Time [h]')
            if col == 0:
                ax_so2.set_ylabel(f'{norm_labels[norm]}\nSO₂ [molec/cm³]',
                                  fontsize=9)
            if col == 1:
                ax_h2so4.set_ylabel('H₂SO₄ [molec/cm³]', fontsize=9)

            ax_so2.set_title(f'{variant} — SO₂', fontsize=9)
            ax_h2so4.set_title(f'{variant} — H₂SO₄', fontsize=9)

    fig.tight_layout()
    out = os.path.join(outdir, 'figs', f'subgroup_gas_sg{sg}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    sns.reset_defaults()
    return out


def plot_all_for_scenario(sg, norm, variant, outdir=_OUTDIR):
    """Generate banana + size-dist + gas plots for one scenario."""
    d = load_scenario(sg, norm, variant, outdir)
    figs = []
    figs.append(plot_banana(d, sg, norm, variant, outdir))
    figs.extend(plot_size_distributions(d, sg, norm, variant, outdir))
    figs.append(plot_gas_timeseries(d, sg, norm, variant, outdir))
    return figs


def plot_all_scenarios(outdir=_OUTDIR):
    """Generate all per-scenario and per-subgroup plots."""
    import matplotlib
    matplotlib.use('Agg')
    print(f'  Generating plots → {os.path.join(outdir, "figs")}/...')

    # Per-scenario individual plots
    for sg, norm, variant in all_scenarios():
        name = scenario_name(sg, norm, variant)
        try:
            figs = plot_all_for_scenario(sg, norm, variant, outdir)
            print(f'    {name}: {len(figs)} figures')
        except FileNotFoundError:
            print(f'    {name}: no data, skipped')

    # Per-subgroup composite plots (raw vs fit grid)
    for sg in SUBGROUPS:
        out = plot_subgroup_banana(sg, outdir)
        print(f'    subgroup_banana_sg{sg}: {out}')
        out = plot_subgroup_gas(sg, outdir)
        print(f'    subgroup_gas_sg{sg}: {out}')
        for qty in ['dN', 'dA', 'dM']:
            out = plot_subgroup_sizedist(sg, qty, outdir)
            print(f'    subgroup_{qty}_sg{sg}: {out}')


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
    parser.add_argument('--outdir', default=None,
                        help='Output directory (default: auto from --bg-dist)')
    parser.add_argument('--bg-dist', default='lognormal',
                        choices=['lognormal', 'tabulated', 'tabulated2'],
                        help='Background aerosol: lognormal, tabulated (dist1), or tabulated2 (dist2)')
    parser.add_argument('--zero-h2so4', action='store_true',
                        help='Start with zero H2SO4 gas and zero in entrained air')
    parser.add_argument('--nbins', type=int, default=40, choices=[40, 80],
                        help='Number of size bins (default: 40)')
    parser.add_argument('--scenario',
                        help='Run a single scenario (e.g. sg02_first_value_raw)')
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-run even if NPZ exists')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip simulations, regenerate plots only')
    args = parser.parse_args()

    # Auto-select output directory based on settings
    h2so4_init = 0.0 if args.zero_h2so4 else H2SO4_INIT
    if args.outdir is None:
        if args.bg_dist == 'tabulated':
            suffix = '_tabulated'
        elif args.bg_dist == 'tabulated2':
            suffix = '_tabulated2'
        else:
            suffix = ''
        if args.zero_h2so4:
            suffix += '_zero_h2so4'
        if args.nbins == 80:
            suffix += '_80bin'
        args.outdir = os.path.join(
            os.path.dirname(__file__), 'results',
            f'dilution_scenarios{suffix}'
        )

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
                               args.dilution_dir, bg_dist=args.bg_dist,
                               h2so4_init=h2so4_init, verbose=True)
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
                bg_dist=args.bg_dist,
                h2so4_init=h2so4_init,
                nbins=args.nbins,
            )

    plot_all_scenarios(args.outdir)
    plot_ensemble_summary(args.outdir)
