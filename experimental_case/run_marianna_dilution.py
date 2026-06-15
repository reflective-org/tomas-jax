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
from dataclasses import dataclass
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
# Fixed numerics (shared across scenarios)
# =========================================================================
ALPHA  = 1.0
BOXVOL = 1.0e6          # cm³ (1 m³ reference cell)
DENS_INIT = 1770.0      # kg/m³ sulfate

PHASE1_END = 20 * 60        # 20 min
PHASE2_END = 4 * 3600       # 4 h
DT1, DT2, DT3 = 1.0, 10.0, 60.0
SNAPSHOT_HOURS = [12, 24, 48, 72, 168, 240]

_RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results', 'marianna')


# =========================================================================
# Scenario configuration
# =========================================================================

@dataclass
class ScenarioConfig:
    """All inputs that define one Marianna dilution simulation."""
    id: str
    name: str
    # Ambient conditions
    temp: float = 210.0          # K
    pres: float = 5500.0         # Pa
    h2o_ppm: float = 4.0         # ppm volume mixing ratio (-> rh internally)
    # Chemistry / nucleation
    oh_conc: float = 5e5         # molec/cm³ constant
    fion: float = 30.0           # pairs/cm³/s ion-pair production
    nuc_org: float = 0.0
    nuc_nh3: float = 0.0
    # Initial plume
    so2_init_ppt: float = 2.9e9  # ppt volume mixing ratio
    h2so4_init: float = 1e5      # molec/cm³
    init_dist: str = 'redcircles'
    init_to_ambient: bool = True
    # Entrained-air background
    bg_dist: str = 'redcircles'  # aerosol dist entrained as plume expands ('' = clean)
    bg_to_ambient: bool = True
    bg_so2_ppb: float = 0.01     # ppb
    bg_h2so4: float = 0.0        # molec/cm³
    # Explicit volume dilution V(t)/V0 (piecewise)
    v0_m3: float = 10.0 * 10.0 * 30000.0   # informational only
    v_early_exp: float = 0.8     # V/V0 = t^v_early_exp for t <= v_t_break
    v_t_break: float = 1e4       # s
    v_prefactor: float = 1585.0  # continuity: v_t_break ** v_early_exp
    v_k: float = 8.89e-9         # exp prefactor for late branch
    v_late_exp: float = 1.5      # (t - v_t_break) ** v_late_exp
    # Run length
    max_hours: float = 240.0

    @property
    def slug(self):
        s = self.name.lower()
        for ch in ' ,/:();':
            s = s.replace(ch, '_')
        while '__' in s:
            s = s.replace('__', '_')
        return f"{self.id}_{s.strip('_')}"

    @property
    def outdir(self):
        return os.path.join(_RESULTS_ROOT, self.slug)

    @property
    def npz(self):
        return os.path.join(self.outdir, 'data.npz')


# Scenario registry. Add new scenarios here.
SCENARIOS = {
    '1': ScenarioConfig(
        id='1',
        name='Low Latitude, High Altitude, Clean Stratosphere',
        # all other fields use the defaults above (the run already executed)
    ),
}


# =========================================================================
# Unit helpers
# =========================================================================

def n_air_cm3(temp, pres):
    """Air number density [molec/cm³]."""
    return pres / (KB * temp) * 1.0e-6


def rh_for_h2o_ppm(h2o_ppm, temp, pres):
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

def V_ratio(t_seconds, cfg):
    """V(t)/V0 (vectorized). Piecewise, continuous at t=v_t_break:

        t^v_early_exp                              0 < t <= v_t_break
        v_prefactor * exp{v_k * (t-v_t_break)^v_late_exp}   t > v_t_break

    The plume starts at V0 (V/V0 = 1) at t=0 and only expands, so the early
    branch is clamped to >= 1 (removes the t->0 singularity giving infinite
    kdil on the first step; effect confined to t < 1 s for exp 0.8).
    """
    t = np.asarray(t_seconds, dtype=np.float64)
    t_safe = np.maximum(t, 0.0)
    early = np.maximum(1.0, np.power(np.maximum(t_safe, 1e-30), cfg.v_early_exp))
    dt_late = np.maximum(t_safe - cfg.v_t_break, 0.0)
    late = cfg.v_prefactor * np.exp(cfg.v_k * np.power(dt_late, cfg.v_late_exp))
    return np.where(t_safe <= cfg.v_t_break, early, late)


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


def build_kdil_array(t_starts, dts, cfg):
    """Per-step kdil = ln(V(t+dt)/V(t)) / dt  [s^-1]  (>=0; volume only grows)."""
    V0 = V_ratio(t_starts, cfg)
    V1 = V_ratio(t_starts + dts, cfg)
    ratio = np.clip(V1 / np.maximum(V0, 1e-300), 1e-300, None)
    return np.log(ratio) / dts


# =========================================================================
# Initial state & background
# =========================================================================

def _aerosol(cfg, dist, to_ambient, nbins):
    if not dist:
        return (np.zeros(nbins), np.zeros((nbins, ICOMP)))
    return get_initial_state(nbins=nbins, boxvol=BOXVOL, dist=dist,
                             to_ambient=to_ambient, temp=cfg.temp, pres=cfg.pres)


def make_initial_state(cfg, nbins):
    """Plume aerosol + SO2/H2SO4 gas seed. Returns (Nk, Mk, Gc, so2_init_cm3)."""
    Nk_np, Mk_np = _aerosol(cfg, cfg.init_dist, cfg.init_to_ambient, nbins)
    so2_init_cm3 = cfg.so2_init_ppt * 1e-12 * n_air_cm3(cfg.temp, cfg.pres)
    Gc = np.zeros(N_GAS_SPECIES)
    Gc[SRTSO4] = conc_to_gc(cfg.h2so4_init, MW_H2SO4)
    Gc[SRTSO2] = conc_to_gc(so2_init_cm3, MW_SO2)
    return jnp.array(Nk_np), jnp.array(Mk_np), jnp.array(Gc), so2_init_cm3


def make_background(cfg, nbins):
    """Entrained-air background aerosol + gases. Returns (Nk_bg, Mk_bg, Gc_bg, bg_so2_cm3)."""
    Nk_bg_np, Mk_bg_np = _aerosol(cfg, cfg.bg_dist, cfg.bg_to_ambient, nbins)
    bg_so2_cm3 = cfg.bg_so2_ppb * 1e-9 * n_air_cm3(cfg.temp, cfg.pres)
    Gc_bg = np.zeros(N_GAS_SPECIES)
    Gc_bg[SRTSO4] = cfg.bg_h2so4
    Gc_bg[SRTSO2] = conc_to_gc(bg_so2_cm3, MW_SO2)
    return jnp.array(Nk_bg_np), jnp.array(Mk_bg_np), jnp.array(Gc_bg), bg_so2_cm3


# =========================================================================
# Run
# =========================================================================

def run(scenario='1', nbins=NBINS, max_hours=None, verbose=True):
    cfg = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    if max_hours is None:
        max_hours = cfg.max_hours
    xk = make_grid(nbins, XK0, 2.0)
    rh = rh_for_h2o_ppm(cfg.h2o_ppm, cfg.temp, cfg.pres)

    t_starts, dts = build_time_schedule(max_hours)
    kdil_arr = build_kdil_array(t_starts, dts, cfg)
    V_arr = V_ratio(t_starts, cfg)
    nsteps = len(t_starts)

    Nk, Mk, Gc, so2_init = make_initial_state(cfg, nbins)
    Nk_bg, Mk_bg, Gc_bg, bg_so2 = make_background(cfg, nbins)
    N_init = float(jnp.sum(Nk)) / BOXVOL
    N_bg   = float(jnp.sum(Nk_bg)) / BOXVOL

    if verbose:
        print('=' * 72)
        print(f'Marianna scenario {cfg.id}: {cfg.name}')
        print(f'  T={cfg.temp}K  P={cfg.pres/100:.0f}hPa  rh={rh:.4f} (=> {cfg.h2o_ppm} ppm H2O)')
        print(f'  n_air={n_air_cm3(cfg.temp, cfg.pres):.3e} molec/cm³')
        print(f'  SO2_init = {cfg.so2_init_ppt:.2e} ppt = {so2_init:.3e} molec/cm³')
        print(f'  H2SO4_init = {cfg.h2so4_init:.0e}  OH = {cfg.oh_conc:.0e}  fion = {cfg.fion}')
        print(f'  init aerosol: {cfg.init_dist} (N={N_init:.2f}/cm³)')
        print(f'  bg: SO2={cfg.bg_so2_ppb} ppb ({bg_so2:.2e} cm⁻³), H2SO4={cfg.bg_h2so4:.0e}, '
              f'aerosol={cfg.bg_dist or "clean"} (N={N_bg:.2f}/cm³)')
        print(f'  V0={cfg.v0_m3:.2e} m³  V({max_hours:.0f}h)/V0={V_ratio(max_hours*3600.0, cfg):.3e}')
        print(f'  steps={nsteps}  kdil∈[{kdil_arr.min():.2e},{kdil_arr.max():.2e}] s⁻¹')
        print('=' * 72)

    step_fn = make_step(
        ['so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'],
        cond_method='ppm_jit', nucl_scheme='ricco_dunne')

    temp = jnp.float64(cfg.temp); pres = jnp.float64(cfg.pres)
    rhj = jnp.float64(rh); alpha = jnp.float64(ALPHA); boxvol = jnp.float64(BOXVOL)
    oh = jnp.float64(cfg.oh_conc); fion = jnp.float64(cfg.fion)
    org = jnp.float64(cfg.nuc_org); nh3 = jnp.float64(cfg.nuc_nh3)

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

    os.makedirs(cfg.outdir, exist_ok=True)
    np.savez_compressed(
        cfg.npz,
        t_seconds=t_starts, dts=dts,
        Nk_every=Nk_every, Mk_dry_every=Mk_dry_every,
        N_tot_every=N_tot_every, M_dry_every=M_dry_every,
        SO2_molec_cm3=SO2_every, SO4_molec_cm3=SO4_every,
        tracer_every=tracer_every, kdil_every=kdil_arr, V_ratio_every=V_arr,
        nbins=nbins, max_hours=max_hours, so2_init_molec_cm3=so2_init,
        # Scenario identity + full input parameter set (for the param-summary plot)
        scenario_id=cfg.id, scenario_name=cfg.name,
        temp=cfg.temp, pres=cfg.pres, h2o_ppm=cfg.h2o_ppm, rh=rh, boxvol=BOXVOL,
        oh_conc=cfg.oh_conc, fion=cfg.fion, nuc_org=cfg.nuc_org, nuc_nh3=cfg.nuc_nh3,
        so2_init_ppt=cfg.so2_init_ppt, h2so4_init=cfg.h2so4_init,
        init_dist=cfg.init_dist, init_to_ambient=cfg.init_to_ambient, N_init=N_init,
        bg_dist=(cfg.bg_dist or 'clean'), bg_to_ambient=cfg.bg_to_ambient,
        bg_so2_ppb=cfg.bg_so2_ppb, bg_so2_cm3=bg_so2, bg_h2so4=cfg.bg_h2so4, N_bg=N_bg,
        v0_m3=cfg.v0_m3, v_early_exp=cfg.v_early_exp, v_t_break=cfg.v_t_break,
        v_prefactor=cfg.v_prefactor, v_k=cfg.v_k, v_late_exp=cfg.v_late_exp,
        V_final=float(V_ratio(max_hours*3600.0, cfg)),
        n_air_cm3=n_air_cm3(cfg.temp, cfg.pres),
    )
    if verbose:
        print(f'  saved: {cfg.npz}')
    return cfg.npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='1', choices=sorted(SCENARIOS),
                    help='Scenario id to run (default 1)')
    ap.add_argument('--hours', type=float, default=None,
                    help='Override duration in hours (default: scenario max_hours)')
    ap.add_argument('--nbins', type=int, default=NBINS)
    ap.add_argument('--plot-only', action='store_true',
                    help='Skip the run; regenerate plots from the existing NPZ')
    args = ap.parse_args()
    cfg = SCENARIOS[args.scenario]

    if not args.plot_only:
        run(scenario=cfg, nbins=args.nbins, max_hours=args.hours)

    from .plot_marianna_dilution import plot_all
    plot_all(cfg.npz)


if __name__ == '__main__':
    main()
