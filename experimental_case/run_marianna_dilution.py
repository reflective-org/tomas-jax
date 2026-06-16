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

SNAPSHOT_HOURS = [12, 24, 48, 72, 168, 240]

# Default multi-resolution timestep schedule: list of (phase_end_seconds, dt).
# Fine head resolves the operator-split stress while H2SO4 production is highest
# (sub-second H2SO4 turnover in the first ~2 min); coarsens as dilution drops SO2.
DEFAULT_DT_SCHEDULE = (
    (120.0,    0.01),   # 0–2 min     : 0.01 s  (resolves H2SO4 transient)
    (1200.0,   0.1),    # 2–20 min    : 0.1  s
    (14400.0,  10.0),   # 20 min–4 h  : 10   s
    (1.30e6,   60.0),   # 4 h–>15 d   : 60   s  (covers up to 336 h runs)
)

_RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results', 'marianna')


# =========================================================================
# Dilution regimes — piecewise V(t)/V0
# =========================================================================
# A regime is an ordered tuple of (t_end_seconds, segment). Segment kinds:
#   ('power', p)          -> V/V0 = max(1, t^p)              (early Schumann t^0.8)
#   ('exp', A, k, t0, q)  -> V/V0 = A * exp{ k * (t-t0)^q }  (late turbulent growth)
# Shared early branch: t^0.8 clamped >=1 for 1<t<1e4 s. Late branches differ by Kz.
_T_BREAK = 1e4
_BIG_T = 1e12   # sentinel end for the final segment


def _D_two_piece(k):
    """Standard two-piece regime: t^0.8 then 1585*exp{k (t-1e4)^1.5}."""
    return (
        (_T_BREAK, ('power', 0.8)),
        (_BIG_T,   ('exp', 1585.0, k, _T_BREAK, 1.5)),
    )


# D1-D5 (see plan / Schumann et al. 1998 scaling). D2 == prior scenario 1.
DILUTIONS = {
    'D1': dict(name='Low Kz',      segments=_D_two_piece(2.811e-9)),
    'D2': dict(name='Med Kz',      segments=_D_two_piece(8.89e-9)),
    'D3': dict(name='High Kz',     segments=_D_two_piece(2.811e-8)),
    'D4': dict(name='Burst', segments=(
        (_T_BREAK,  ('power', 0.8)),
        (1.728e5,   ('exp', 1585.0,  2.811e-9, _T_BREAK, 1.5)),
        (2.238e5,   ('exp', 1906.0,  2.811e-7, 1.728e5,  1.5)),   # 14.2 h burst
        (_BIG_T,    ('exp', 4.83e4,  2.811e-9, 2.238e5,  1.5)),
    )),
    'D5': dict(name='Very High',   segments=_D_two_piece(1.33e-7)),
}


def dilution_str(segments):
    """Compact human-readable description of a dilution regime."""
    parts, t0 = [], 0.0
    for t_end, seg in segments:
        rng = f"{t0:g}-{'inf' if t_end >= _BIG_T else f'{t_end:g}'}s"
        if seg[0] == 'power':
            parts.append(f"{rng}: t^{seg[1]:g}")
        else:
            _, A, k, t_ref, q = seg
            parts.append(f"{rng}: {A:g}*exp[{k:g}(t-{t_ref:g})^{q:g}]")
        t0 = t_end
    return " | ".join(parts)


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
    # Explicit volume dilution V(t)/V0 (piecewise segments; see DILUTIONS)
    v0_m3: float = 10.0 * 10.0 * 30000.0   # informational only
    dilution: tuple = DILUTIONS['D2']['segments']   # default = Med Kz (scenario 1)
    baseline: str = ''           # e.g. 'B1'
    dilution_id: str = ''        # e.g. 'D2'
    # Numerics
    dt_schedule: tuple = DEFAULT_DT_SCHEDULE   # ((phase_end_s, dt), ...)
    # Run length
    max_hours: float = 240.0

    @property
    def dt_schedule_str(self):
        parts, t0 = [], 0.0
        for t_end, dt in self.dt_schedule:
            parts.append(f"{t0:g}-{t_end:g}s@{dt:g}s")
            t0 = t_end
        return ", ".join(parts)

    @property
    def dilution_desc(self):
        return dilution_str(self.dilution)

    @property
    def slug(self):
        return self.id   # e.g. 'B1-D2' -> results/marianna/B1-D2/

    @property
    def outdir(self):
        return os.path.join(_RESULTS_ROOT, self.slug)

    @property
    def npz(self):
        return os.path.join(self.outdir, 'data.npz')


# =========================================================================
# Baselines + run matrix (3 baselines × 5 dilution regimes = 15 runs)
# =========================================================================
# Only these atmospheric/chemistry params vary per baseline. The aerosol size
# distribution (initial AND entrained background) is the digitized red-circles
# for ALL baselines (per-baseline background-aerosol specs in the brief ignored).
# B1 reproduces the prior "scenario 1"; B1-D2 == that exact run.
BASELINES = {
    'B1': dict(name='Low Lat/High Alt, Clean',
               temp=210.0, pres=5500.0,  so2_init_ppt=2.9e9, oh_conc=5.0e5,
               h2o_ppm=4.0, h2so4_init=1e5, fion=30.0),
    'B2': dict(name='High Lat/Low Alt, Clean',
               temp=210.0, pres=12000.0, so2_init_ppt=1.3e9, oh_conc=2.0e5,
               h2o_ppm=4.0, h2so4_init=5e4, fion=40.0),
    'B3': dict(name='Low Lat/High Alt, Geoengineered',
               temp=213.0, pres=5500.0,  so2_init_ppt=2.9e9, oh_conc=3.5e5,
               h2o_ppm=5.5, h2so4_init=4e5, fion=30.0),
}

MATRIX_MAX_HOURS = 336.0   # 14 days


def _make_matrix():
    out = {}
    for bid, b in BASELINES.items():
        for did, dd in DILUTIONS.items():
            rid = f"{bid}-{did}"
            out[rid] = ScenarioConfig(
                id=rid,
                name=f"{bid} {b['name']} — {did} {dd['name']}",
                baseline=bid, dilution_id=did,
                temp=b['temp'], pres=b['pres'], h2o_ppm=b['h2o_ppm'],
                oh_conc=b['oh_conc'], fion=b['fion'],
                so2_init_ppt=b['so2_init_ppt'], h2so4_init=b['h2so4_init'],
                dilution=dd['segments'], max_hours=MATRIX_MAX_HOURS,
            )
    return out


SCENARIOS = _make_matrix()   # keys: 'B1-D1' ... 'B3-D5'


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

def _eval_segment(t_safe, seg):
    """Evaluate one dilution segment over the whole time array (vectorized)."""
    if seg[0] == 'power':
        p = seg[1]
        # clamp >=1: plume starts at V0 and only expands (removes t->0 singularity)
        return np.maximum(1.0, np.power(np.maximum(t_safe, 1e-30), p))
    _, A, k, t0, q = seg
    dt = np.maximum(t_safe - t0, 0.0)
    return A * np.exp(k * np.power(dt, q))


def V_ratio(t_seconds, cfg):
    """V(t)/V0 (vectorized), evaluated from cfg.dilution piecewise segments.

    Segments are (t_end, kind, params): 'power' -> max(1, t^p); 'exp' ->
    A*exp{k (t-t0)^q}. Continuous by construction (prefactors chosen to match
    at the breakpoints). The early t^0.8 branch is clamped >=1.
    """
    t_safe = np.maximum(np.asarray(t_seconds, dtype=np.float64), 0.0)
    conds, vals, t0 = [], [], 0.0
    for t_end, seg in cfg.dilution:
        conds.append((t_safe >= t0) & (t_safe < t_end))
        vals.append(_eval_segment(t_safe, seg))
        t0 = t_end
    return np.select(conds, vals, default=vals[-1])


def build_time_schedule(dt_schedule, max_hours):
    """Return (t_starts, dts) arrays for multi-resolution stepping.

    dt_schedule: iterable of (phase_end_seconds, dt). Phases run in order;
    the final phase end is capped at max_hours.
    """
    max_s = max_hours * 3600.0
    t, t_starts, dts = 0.0, [], []
    for t_end, dt in dt_schedule:
        end = min(t_end, max_s)
        while t < end - 1e-9:
            t_starts.append(t); dts.append(dt); t += dt
        if t >= max_s - 1e-9:
            break
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

def run(scenario='B1-D2', nbins=NBINS, max_hours=None, verbose=True):
    cfg = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    if max_hours is None:
        max_hours = cfg.max_hours
    xk = make_grid(nbins, XK0, 2.0)
    rh = rh_for_h2o_ppm(cfg.h2o_ppm, cfg.temp, cfg.pres)

    t_starts, dts = build_time_schedule(cfg.dt_schedule, max_hours)
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
        print(f'  dilution {cfg.dilution_id}: {cfg.dilution_desc}')
        print(f'  dt schedule: {cfg.dt_schedule_str}')
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
        v0_m3=cfg.v0_m3, baseline=cfg.baseline, dilution_id=cfg.dilution_id,
        dilution_desc=cfg.dilution_desc,
        V_final=float(V_ratio(max_hours*3600.0, cfg)),
        n_air_cm3=n_air_cm3(cfg.temp, cfg.pres),
        dt_schedule_str=cfg.dt_schedule_str,
    )
    if verbose:
        print(f'  saved: {cfg.npz}')
    return cfg.npz


def select_scenarios(scenario=None, baseline=None, dilution=None, run_all=False):
    """Resolve which scenario ids to run from the CLI selectors."""
    if scenario:
        return [scenario]
    ids = list(SCENARIOS)
    if run_all and not (baseline or dilution):
        return ids
    sel = [rid for rid in ids
           if (baseline is None or SCENARIOS[rid].baseline == baseline)
           and (dilution is None or SCENARIOS[rid].dilution_id == dilution)]
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default=None, choices=sorted(SCENARIOS),
                    help='Single run id, e.g. B1-D2')
    ap.add_argument('--baseline', default=None, choices=sorted(BASELINES),
                    help='Run all dilutions for this baseline (B1/B2/B3)')
    ap.add_argument('--dilution', default=None, choices=sorted(DILUTIONS),
                    help='Run this dilution across all baselines (D1..D5)')
    ap.add_argument('--all', action='store_true', help='Run the full 15-run matrix')
    ap.add_argument('--hours', type=float, default=None,
                    help='Override duration in hours (default: scenario max_hours)')
    ap.add_argument('--nbins', type=int, default=NBINS)
    ap.add_argument('--plot-only', action='store_true',
                    help='Skip the runs; regenerate per-run plots from existing NPZs')
    args = ap.parse_args()

    ids = select_scenarios(args.scenario, args.baseline, args.dilution, args.all)
    if not ids:
        ids = ['B1-D2']
    from .plot_marianna_dilution import plot_all
    for i, rid in enumerate(ids, 1):
        cfg = SCENARIOS[rid]
        if len(ids) > 1:
            print(f'\n##### [{i}/{len(ids)}] {rid} #####')
        if not args.plot_only:
            run(scenario=cfg, nbins=args.nbins, max_hours=args.hours)
        plot_all(cfg.npz)


if __name__ == '__main__':
    main()
