"""Marianna SAI dilution case — 10-day box-model run with explicit V(t)/V0.

See docs/marianna_dilution.md for the full specification. Summary:

  Conditions : T=210 K, P=55 hPa, [H2O]=4 ppm (via rh), OH=5e5 molec/cm³ const,
               NH3=0, organics=0, ion-pair production fion=30 pairs/cm³/s.
  Initial    : SO2 = 2.9e9 ppt (=2900 ppm = 5.50e15 molec/cm³),
               H2SO4 = 1e5 molec/cm³,
               aerosol = red-circles obs (220-230 ppbv), STP->ambient (x0.0706).
  Dilution   : explicit analytic V(t)/V0 (piecewise), entrains background aerosol
               (red-circles ambient) + background SO2 = 0.01 ppb, background
               H2SO4 = 0.  Inert tracer integrated alongside (Cbg=0) to isolate
               the pure dilution signal.
  Processes  : so2_chemistry, nucleation (ricco_dunne), coagulation, condensation,
               dilution — via make_step(cond_method='ppm_jit').
  Duration   : 10 days (240 h) with multi-resolution stepping.

Usage::

    python -m experimental_case.run_marianna_dilution            # run + plots
    python -m experimental_case.run_marianna_dilution --plot-only
    python -m experimental_case.run_marianna_dilution --hours 12  # short test
"""
import os
import time
import math
import argparse
import numpy as np
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, N_GAS_SPECIES,
    SRTSO4, SRTSO2, SRTH2O,
    MW_H2SO4, MW_SO2, AVOGADRO, PI,
    make_grid, XK0, KB,
)
from tomas_jax.solvers.condensation import make_step
from tomas_jax.physics.so2_chemistry import (
    _BUCK_A, _BUCK_B, _BUCK_C, _BUCK_D,
)
from .background_aerosol_distribution import (
    get_initial_state, stp_to_ambient_factor,
)

# =========================================================================
# Conditions / constants
# =========================================================================
TEMP   = 210.0          # K
PRES   = 5500.0         # Pa  (55 hPa)
H2O_PPM = 4.0           # ppm (volume mixing ratio) -> rh below
ALPHA  = 1.0
BOXVOL = 1.0e6          # cm³ (1 m³ reference cell)
DENS_INIT = 1770.0      # kg/m³ sulfate

OH_CONC    = 5e5        # molec/cm³ constant
H2SO4_INIT = 1e5        # molec/cm³ initial plume H2SO4 (future runs: 0)
FION       = 30.0       # pairs/cm³/s ion-pair production
NUC_ORG    = 0.0
NUC_NH3    = 0.0

# Initial SO2: 2.9e9 ppt = 2900 ppm (literal, as specified). NB: only 1.76 t in
# V0=3e6 m³ — see docs/marianna_dilution.md §6 (mass-equivalence caveat).
SO2_INIT_PPT = 2.9e9

# Background (entrained air)
BG_SO2_PPB   = 0.01     # ppb  -> molec/cm³ below
BG_H2SO4     = 0.0      # molec/cm³ (no background H2SO4)

# Initial plume geometry (informational only — box model is in concentration
# space; only V(t)/V0 enters via kdil)
V0_M3 = 10.0 * 10.0 * 30000.0   # 3.0e6 m³

# =========================================================================
# Time schedule (multi-resolution) and snapshots
# =========================================================================
PHASE1_END = 20 * 60        # 20 min
PHASE2_END = 4 * 3600       # 4 h
DT1, DT2, DT3 = 1.0, 10.0, 60.0

SNAPSHOT_HOURS = [12, 24, 48, 72, 168, 240]

_OUTDIR = os.path.join(os.path.dirname(__file__), 'results', 'marianna')
_NPZ = os.path.join(_OUTDIR, 'marianna_dilution.npz')


# =========================================================================
# Unit helpers
# =========================================================================

def n_air_cm3(temp=TEMP, pres=PRES):
    """Air number density [molec/cm³]."""
    return pres / (KB * temp) * 1.0e-6


def rh_for_h2o_ppm(h2o_ppm, temp=TEMP, pres=PRES):
    """RH [fraction] that yields a given H2O volume mixing ratio, using the same
    Buck formula the SO2 chemistry uses internally:  x_h2o = rh * e_sat / P."""
    T_C = temp - 273.15
    e_sat = _BUCK_A * math.exp((_BUCK_B - T_C / _BUCK_D) * T_C / (T_C + _BUCK_C))
    return (h2o_ppm * 1e-6) * pres / e_sat


def conc_to_gc(conc_cm3, mw):
    """molec/cm³ -> kg per box cell."""
    return conc_cm3 * BOXVOL * (mw / 1000.0) / AVOGADRO


def gc_to_conc(gc_kg, mw):
    """kg per box cell -> molec/cm³."""
    return gc_kg * AVOGADRO / (mw / 1000.0) / BOXVOL


# =========================================================================
# Explicit volume-dilution parameterization V(t)/V0
# =========================================================================

def V_ratio(t_seconds):
    """V(t)/V0 (vectorized). Piecewise, continuous at t=1e4 s:

        t^0.8                                  0 < t <= 1e4
        1585 * exp{8.89e-9 * (t-1e4)^(3/2)}    t  > 1e4

    The plume starts at V0 (V/V0 = 1) at t=0 and only expands, so the early
    branch is clamped to >= 1 (t^0.8 < 1 only for t < 1 s). This removes the
    t->0 singularity (t^0.8 -> 0) that would otherwise give an infinite kdil
    on the first step. Net effect is confined to the first second.
    """
    t = np.asarray(t_seconds, dtype=np.float64)
    t_safe = np.maximum(t, 0.0)
    early = np.maximum(1.0, np.power(np.maximum(t_safe, 1e-30), 0.8))
    dt_late = np.maximum(t_safe - 1e4, 0.0)
    late = 1585.0 * np.exp(8.89e-9 * np.power(dt_late, 1.5))
    return np.where(t_safe <= 1e4, early, late)


def build_time_schedule(max_hours):
    """Return (t_starts, dts) arrays for multi-resolution stepping."""
    max_s = max_hours * 3600.0
    t, t_starts, dts = 0.0, [], []
    while t < PHASE1_END - 1e-9 and t < max_s - 1e-9:
        t_starts.append(t); dts.append(DT1); t += DT1
    while t < PHASE2_END - 1e-9 and t < max_s - 1e-9:
        t_starts.append(t); dts.append(DT2); t += DT2
    while t < max_s - 1e-9:
        t_starts.append(t); dts.append(DT3); t += DT3
    return np.array(t_starts), np.array(dts)


def build_kdil_array(t_starts, dts):
    """Per-step kdil = ln(V(t+dt)/V(t)) / dt  [s^-1]  (>=0; volume only grows)."""
    V0 = V_ratio(t_starts)
    V1 = V_ratio(t_starts + dts)
    ratio = np.clip(V1 / np.maximum(V0, 1e-300), 1e-300, None)
    return np.log(ratio) / dts


# =========================================================================
# Initial state
# =========================================================================

def make_initial_state(nbins, xk):
    """Red-circles ambient aerosol + SO2/H2SO4 gas seed. Returns (Nk, Mk, Gc)."""
    Nk_np, Mk_np = get_initial_state(nbins=nbins, boxvol=BOXVOL, dist='redcircles',
                                     to_ambient=True, temp=TEMP, pres=PRES)
    so2_init_cm3 = SO2_INIT_PPT * 1e-12 * n_air_cm3()
    Gc = np.zeros(N_GAS_SPECIES)
    Gc[SRTSO4] = conc_to_gc(H2SO4_INIT, MW_H2SO4)
    Gc[SRTSO2] = conc_to_gc(so2_init_cm3, MW_SO2)
    return jnp.array(Nk_np), jnp.array(Mk_np), jnp.array(Gc), so2_init_cm3


def make_background(nbins):
    """Entrained-air background: red-circles ambient aerosol, SO2=0.01 ppb, H2SO4=0."""
    Nk_bg_np, Mk_bg_np = get_initial_state(nbins=nbins, boxvol=BOXVOL, dist='redcircles',
                                           to_ambient=True, temp=TEMP, pres=PRES)
    bg_so2_cm3 = BG_SO2_PPB * 1e-9 * n_air_cm3()
    Gc_bg = np.zeros(N_GAS_SPECIES)
    Gc_bg[SRTSO4] = BG_H2SO4
    Gc_bg[SRTSO2] = conc_to_gc(bg_so2_cm3, MW_SO2)
    return jnp.array(Nk_bg_np), jnp.array(Mk_bg_np), jnp.array(Gc_bg), bg_so2_cm3


# =========================================================================
# Run
# =========================================================================

def run(max_hours=240.0, nbins=NBINS, verbose=True):
    xk = make_grid(nbins, XK0, 2.0)
    rh = rh_for_h2o_ppm(H2O_PPM)

    t_starts, dts = build_time_schedule(max_hours)
    kdil_arr = build_kdil_array(t_starts, dts)
    V_arr = V_ratio(t_starts)
    nsteps = len(t_starts)

    Nk, Mk, Gc, so2_init = make_initial_state(nbins, xk)
    Nk_bg, Mk_bg, Gc_bg, bg_so2 = make_background(nbins)

    if verbose:
        print('=' * 72)
        print('Marianna SAI dilution — 10-day box model')
        print(f'  T={TEMP}K  P={PRES/100:.0f}hPa  rh={rh:.4f} (=> {H2O_PPM} ppm H2O)')
        print(f'  n_air={n_air_cm3():.3e} molec/cm³')
        print(f'  SO2_init = {SO2_INIT_PPT:.2e} ppt = {so2_init:.3e} molec/cm³')
        print(f'  H2SO4_init = {H2SO4_INIT:.0e}  OH = {OH_CONC:.0e}  fion = {FION}')
        print(f'  bg: SO2={BG_SO2_PPB} ppb ({bg_so2:.2e} cm⁻³), H2SO4={BG_H2SO4:.0e}, '
              f'aerosol=red-circles ambient (N={float(jnp.sum(Nk_bg))/BOXVOL:.2f}/cm³)')
        print(f'  V0={V0_M3:.2e} m³  V(t)/V0: t^0.8 then 1585·exp(...)  '
              f'-> V({max_hours:.0f}h)/V0={V_ratio(max_hours*3600.0):.3e}')
        print(f'  steps={nsteps}  (dt=1s/10s/60s)  '
              f'kdil∈[{kdil_arr.min():.2e},{kdil_arr.max():.2e}] s⁻¹')
        print('=' * 72)

    step_fn = make_step(
        ['so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'],
        cond_method='ppm_jit', nucl_scheme='ricco_dunne')

    temp = jnp.float64(TEMP); pres = jnp.float64(PRES)
    rhj = jnp.float64(rh); alpha = jnp.float64(ALPHA); boxvol = jnp.float64(BOXVOL)
    oh = jnp.float64(OH_CONC); fion = jnp.float64(FION)
    org = jnp.float64(NUC_ORG); nh3 = jnp.float64(NUC_NH3)

    # Storage
    Nk_every     = np.zeros((nsteps, nbins))
    Mk_dry_every = np.zeros((nsteps, nbins))
    N_tot_every  = np.zeros(nsteps)
    M_dry_every  = np.zeros(nsteps)
    SO2_every    = np.zeros(nsteps)
    SO4_every    = np.zeros(nsteps)
    tracer_every = np.zeros(nsteps)

    tracer = 1.0
    t0 = time.time()
    for i in range(nsteps):
        Nk_np = np.array(Nk); Mk_np = np.array(Mk); Gc_np = np.array(Gc)
        Nk_every[i]     = Nk_np
        Mk_dry_every[i] = Mk_np[:, :SRTH2O].sum(axis=1)
        N_tot_every[i]  = float(jnp.sum(Nk))
        M_dry_every[i]  = float(jnp.sum(Mk[:, :SRTH2O]))
        SO2_every[i]    = gc_to_conc(Gc_np[SRTSO2], MW_SO2)
        SO4_every[i]    = gc_to_conc(Gc_np[SRTSO4], MW_H2SO4)
        tracer_every[i] = tracer

        t_s = float(t_starts[i])
        if verbose and i > 0 and abs(t_s % 86400) < float(dts[i]):
            print(f'  d{t_s/86400:.0f}  N={N_tot_every[i]/BOXVOL:.3e}/cm³  '
                  f'SO2={SO2_every[i]:.2e}  H2SO4={SO4_every[i]:.2e}  '
                  f'tracer={tracer:.3e}  elapsed={time.time()-t0:.0f}s')

        kdil_i = jnp.float64(kdil_arr[i]); dt_i = jnp.float64(dts[i])
        Nk, Mk, Gc = step_fn(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rhj, alpha, dt_i,
            oh_conc=oh, org_conc=org, nh3_conc=nh3, fion=fion,
            kdil=kdil_i, Nk_bg=Nk_bg, Mk_bg=Mk_bg, Gc_bg=Gc_bg,
        )
        tracer = tracer * float(jnp.exp(-kdil_i * dt_i))

    wall = time.time() - t0
    if verbose:
        print(f'  -> {wall:.1f}s wall, N_final={N_tot_every[-1]/BOXVOL:.3e}/cm³, '
              f'tracer_final={tracer_every[-1]:.3e}')

    os.makedirs(_OUTDIR, exist_ok=True)
    np.savez_compressed(
        _NPZ,
        t_seconds=t_starts, dts=dts,
        Nk_every=Nk_every, Mk_dry_every=Mk_dry_every,
        N_tot_every=N_tot_every, M_dry_every=M_dry_every,
        SO2_molec_cm3=SO2_every, SO4_molec_cm3=SO4_every,
        tracer_every=tracer_every, kdil_every=kdil_arr, V_ratio_every=V_arr,
        nbins=nbins, max_hours=max_hours, so2_init_molec_cm3=so2_init,
        temp=TEMP, pres=PRES, rh=rh, boxvol=BOXVOL,
    )
    if verbose:
        print(f'  saved: {_NPZ}')
    return _NPZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=240.0,
                    help='Simulation duration in hours (default 240 = 10 days)')
    ap.add_argument('--nbins', type=int, default=NBINS)
    ap.add_argument('--plot-only', action='store_true',
                    help='Skip the run; regenerate plots from the existing NPZ')
    args = ap.parse_args()

    if not args.plot_only:
        run(max_hours=args.hours, nbins=args.nbins)

    # Plotting added in plot_marianna_dilution (next commit)
    try:
        from .plot_marianna_dilution import plot_all
        plot_all(_NPZ)
    except ImportError:
        print('(plotting module not available yet)')


if __name__ == '__main__':
    main()
