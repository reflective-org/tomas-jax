"""TFL vs PPM convergence test at multiple bin resolutions.

Runs condensation-only, coag+cond combined, or coag-only for a single
scenario at 36/72/144 bins to test whether distributions converge at
higher resolution. TFL and PPM are compared for condensation modes;
coag-only uses Euler + MNFIX (single method).

Usage::

    source .venv/bin/activate
    python -m benchmarks.python.convergence_test
    python -m benchmarks.python.convergence_test --scenario 5
    python -m benchmarks.python.convergence_test --no-plots
    python -m benchmarks.python.convergence_test --mode combined --constant-gc
    python -m benchmarks.python.convergence_test --mode coag_only --dt 60
"""
import os
import sys
import time
import argparse
import numpy as np

import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO,
    make_grid,
)
from tomas_jax.solvers.condensation import (
    run_condensation_scan_tfl,
    run_condensation_scan,
    run_combined_scan_tfl,
    run_combined_scan_ppm,
    condensation_step_tfl_jax,
    condensation_step_jax,
    combined_step_tfl_jax,
    combined_step_ppm_jax,
)
from tomas_jax.solvers.diffrax import coag_euler_step
from benchmarks.python.scenarios import get_scenarios

from functools import partial


def _run_constant_gc_scan_tfl(Nk, Mk, Gc, xk, temp, pres, boxvol, rh,
                               alpha, dt, nsteps):
    """TFL scan with Gc reset to initial value after each step."""
    @partial(jax.jit, static_argnums=(10,))
    def _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps):
        Gc0 = Gc.copy()
        def step(carry, _):
            Nk_c, Mk_c, Gc_c = carry
            Nk_c, Mk_c, _ = condensation_step_tfl_jax(
                Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt)
            diag = jnp.array([jnp.sum(Nk_c),
                               jnp.sum(Mk_c[:, :SRTH2O]),
                               Gc0[SRTSO4]])
            return (Nk_c, Mk_c, Gc0), diag
        (Nk_f, Mk_f, Gc_f), hist = jax.lax.scan(
            step, (Nk, Mk, Gc), None, length=nsteps)
        return Nk_f, Mk_f, Gc_f, hist
    return _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps)


def _run_constant_gc_scan_ppm(Nk, Mk, Gc, xk, temp, pres, boxvol, rh,
                               alpha, dt, nsteps):
    """PPM scan with Gc reset to initial value after each step."""
    @partial(jax.jit, static_argnums=(10,))
    def _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps):
        Gc0 = Gc.copy()
        def step(carry, _):
            Nk_c, Mk_c, Gc_c = carry
            Nk_c, Mk_c, _ = condensation_step_jax(
                Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt)
            diag = jnp.array([jnp.sum(Nk_c),
                               jnp.sum(Mk_c[:, :SRTH2O]),
                               Gc0[SRTSO4]])
            return (Nk_c, Mk_c, Gc0), diag
        (Nk_f, Mk_f, Gc_f), hist = jax.lax.scan(
            step, (Nk, Mk, Gc), None, length=nsteps)
        return Nk_f, Mk_f, Gc_f, hist
    return _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps)


def _run_constant_gc_combined_scan_tfl(Nk, Mk, Gc, xk, temp, pres, boxvol,
                                        rh, alpha, dt, nsteps):
    """Coag + TFL condensation scan with Gc reset to initial value each step."""
    @partial(jax.jit, static_argnums=(10,))
    def _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps):
        Gc0 = Gc.copy()
        def step(carry, _):
            Nk_c, Mk_c, Gc_c = carry
            Nk_c, Mk_c, _ = combined_step_tfl_jax(
                Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt)
            diag = jnp.array([jnp.sum(Nk_c),
                               jnp.sum(Mk_c[:, :SRTH2O]),
                               Gc0[SRTSO4]])
            return (Nk_c, Mk_c, Gc0), diag
        (Nk_f, Mk_f, Gc_f), hist = jax.lax.scan(
            step, (Nk, Mk, Gc), None, length=nsteps)
        return Nk_f, Mk_f, Gc_f, hist
    return _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps)


def _run_constant_gc_combined_scan_ppm(Nk, Mk, Gc, xk, temp, pres, boxvol,
                                        rh, alpha, dt, nsteps):
    """Coag + PPM condensation scan with Gc reset to initial value each step."""
    @partial(jax.jit, static_argnums=(10,))
    def _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps):
        Gc0 = Gc.copy()
        def step(carry, _):
            Nk_c, Mk_c, Gc_c = carry
            Nk_c, Mk_c, _ = combined_step_ppm_jax(
                Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt)
            diag = jnp.array([jnp.sum(Nk_c),
                               jnp.sum(Mk_c[:, :SRTH2O]),
                               Gc0[SRTSO4]])
            return (Nk_c, Mk_c, Gc0), diag
        (Nk_f, Mk_f, Gc_f), hist = jax.lax.scan(
            step, (Nk, Mk, Gc), None, length=nsteps)
        return Nk_f, Mk_f, Gc_f, hist
    return _scan(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, nsteps)


def _run_coag_only_scan(Nk, Mk, xk, temp, pres, boxvol, dt, nsteps,
                        n_substeps=3):
    """Coag-only scan: Euler + MNFIX per timestep."""
    @partial(jax.jit, static_argnums=(7, 8))
    def _scan(Nk, Mk, xk, temp, pres, boxvol, dt, nsteps, n_substeps):
        def step(carry, _):
            Nk_c, Mk_c = carry
            Nk_c, Mk_c = coag_euler_step(Nk_c, Mk_c, xk, temp, pres, boxvol,
                                          dt=dt, n_substeps=n_substeps)
            diag = jnp.array([jnp.sum(Nk_c),
                              jnp.sum(Mk_c[:, :SRTH2O]),
                              0.0])
            return (Nk_c, Mk_c), diag
        (Nk_f, Mk_f), hist = jax.lax.scan(step, (Nk, Mk), None, length=nsteps)
        return Nk_f, Mk_f, hist
    return _scan(Nk, Mk, xk, temp, pres, boxvol, dt, nsteps, n_substeps)


# Physical constants matching Fortran harness
PI = 3.141592654
DENS_INIT = 1770.0  # kg/m3 (sulfate density)
NEPS = 1.0e-3
BOXVOL = 1.0e6  # cm3
DT = 60.0
NSTEPS = 1440  # 24 hours

# Grid starting points
XK0_17NM = (np.pi / 6.0) * (1.7e-9) ** 3 * DENS_INIT  # 1.7nm start
XK0_STANDARD = 1.0e-21 * 2.0 ** (-6)  # standard TOMAS (Mo ≈ 1.5625e-23, ~3.2nm)


def dp_nm_to_xk0(dp_nm):
    """Convert a diameter in nm to the corresponding xk0 mass in kg."""
    return (np.pi / 6.0) * (dp_nm * 1e-9) ** 3 * DENS_INIT

# Grid configurations: (label, nbins, doubling_factor)
# Extra bins offset with --extra-bins N (e.g., +4 → 40/80/160 for 1nm start)
BASE_GRID_CONFIGS = [
    (36, 2.0,       "×2"),
    (72, 2.0**0.5,  "×√2"),
    (144, 2.0**0.25, "×2^¼"),
]


def _make_grid_configs(extra_bins=0):
    """Build GRID_CONFIGS with optional extra bins per resolution."""
    configs = []
    for base_n, factor, factor_label in BASE_GRID_CONFIGS:
        n = base_n + extra_bins * int(round(base_n / 36))
        configs.append((f"{n} bins ({factor_label})", n, factor))
    return configs


GRID_CONFIGS = _make_grid_configs(0)

# Fortran output directory (relative to tomas_fortran/)
FORTRAN_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                           'tomas_fortran', 'output', 'constgc')


def init_lognormal(N_total, Dp_gmd_um, sigma_gsd, xk):
    """Initialize lognormal Nk, Mk on an arbitrary grid.

    Args:
        N_total: Total number concentration [#/cm3]
        Dp_gmd_um: Geometric mean diameter [um]
        sigma_gsd: Geometric standard deviation
        xk: Bin boundaries [kg], shape (nbins+1,)

    Returns:
        Nk: [#/grid cell], shape (nbins,)
        Mk: [kg/grid cell], shape (nbins, ICOMP)
    """
    xk_np = np.array(xk)
    nbins = len(xk_np) - 1
    Nk = np.zeros(nbins)
    Mk = np.zeros((nbins, ICOMP))

    for k in range(nbins):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
        Dk = np.sqrt(Dl * Dh)

        dN = ((N_total * BOXVOL)
              / (np.sqrt(2.0 * PI) * Dk * np.log(sigma_gsd))
              * np.exp(-(np.log(Dk / Dp_gmd_um) ** 2
                         / (2.0 * np.log(sigma_gsd) ** 2)))
              * (Dh - Dl))

        Nk[k] = dN
        Mk[k, SRTSO4] = dN * np.sqrt(xk_np[k] * xk_np[k + 1])

    # Neps preprocessing
    for k in range(nbins):
        if Nk[k] < NEPS:
            Nk[k] = NEPS
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = NEPS * 1.4 * xk_np[k]

    return Nk, Mk


def _fix_fortran_float(s):
    """Fix Fortran exponents missing 'E' (e.g., '0.1234+05' -> '0.1234E+05')."""
    import re
    return re.sub(r'(\d)([\+\-])(\d)', r'\1E\2\3', s.strip())


def load_fortran_results(dt_val, fortran_dir=None, mode='cond_only'):
    """Load Fortran Nk/Mk output for a given dt value.

    Returns dict with keys: Nk, Mk, xk (standard 36-bin grid), or None if not found.
    """
    if fortran_dir is None:
        fortran_dir = FORTRAN_DIR
    dt_int = int(dt_val)

    prefix = 'constgc_combined' if mode == 'combined' else 'constgc'
    nk_path = os.path.join(fortran_dir, f'{prefix}_dt{dt_int:02d}_final_Nk.csv')
    mk_path = os.path.join(fortran_dir, f'{prefix}_dt{dt_int:02d}_final_Mk.csv')
    xk_path = os.path.join(fortran_dir, 'constgc_xk.csv')

    if not os.path.exists(nk_path):
        print(f"  [Fortran] Not found: {nk_path}")
        return None

    # Load Nk (one value per line)
    Nk = np.array([float(_fix_fortran_float(line))
                    for line in open(nk_path).readlines()])

    # Load Mk (ibins rows × icomp columns, comma-separated)
    Mk = []
    for line in open(mk_path).readlines():
        row = [float(_fix_fortran_float(v)) for v in line.split(',')]
        Mk.append(row)
    Mk = np.array(Mk)

    # Load xk if available, otherwise reconstruct standard grid
    if os.path.exists(xk_path):
        xk = np.array([float(_fix_fortran_float(line))
                        for line in open(xk_path).readlines()])
    else:
        xk = np.array([XK0_STANDARD * 2.0 ** k for k in range(37)])

    return {'Nk': Nk, 'Mk': Mk, 'xk': xk}


def _bin_diameters(xk):
    """Compute bin-edge and midpoint diameters [nm] from mass boundaries [kg]."""
    xk_np = np.array(xk)
    Dp_lo = 1e9 * ((6.0 * xk_np[:-1]) / (DENS_INIT * np.pi)) ** (1.0 / 3.0)
    Dp_hi = 1e9 * ((6.0 * xk_np[1:]) / (DENS_INIT * np.pi)) ** (1.0 / 3.0)
    Dp_mid = np.sqrt(Dp_lo * Dp_hi)
    dlogDp = np.log10(Dp_hi / Dp_lo)
    return Dp_mid, dlogDp


def compute_dNdlogDp(Nk, xk):
    """Compute dN/dlogDp [#/cm3] for plotting size distributions."""
    Dp_mid, dlogDp = _bin_diameters(xk)
    dNdlogDp = (np.array(Nk) / BOXVOL) / np.maximum(dlogDp, 1e-30)
    return Dp_mid, dNdlogDp


def compute_dMdlogDp(Mk, xk):
    """Compute dM/dlogDp [ug/m3] for plotting mass distributions.

    Uses dry mass (species 0 to SRTH2O-1). Since boxvol=1e6 cm3 = 1 m3,
    kg/cell = kg/m3, and ×1e9 = ug/m3.
    """
    Dp_mid, dlogDp = _bin_diameters(xk)
    Mk_np = np.array(Mk)
    M_dry_per_bin = np.sum(Mk_np[:, :SRTH2O], axis=1)  # kg/cell per bin
    dMdlogDp = (M_dry_per_bin * 1e9) / np.maximum(dlogDp, 1e-30)  # ug/m3
    return Dp_mid, dMdlogDp


def _fmt_exp(val):
    """Format a number in compact scientific notation (e.g., 1e4, 3.5e-12)."""
    if val == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(val))))
    coeff = val / 10 ** exp
    if abs(coeff - round(coeff)) < 0.01:
        return f"{round(coeff):g}e{exp}"
    return f"{coeff:.1f}e{exp}"


def _build_file_tag(meta, n_total_override, gmd_override, gsd_override,
                    dt_override, temp_override=None, pres_override=None):
    """Build a unique, descriptive filename tag for this simulation.

    Examples:
        S01                              — default scenario 1
        S01_N1e4                         — scenario 1 with N override
        S01_N1e4_GMD0.05_GSD1.8_dt30    — scenario 1 with multiple overrides
        depletion_H2SO4-1e9_N1e4        — depletion mode
    """
    parts = []

    if meta.get('constant_gc_mode'):
        parts.append(f"constGc_H2SO4-{_fmt_exp(meta['h2so4_molec_cm3'])}")
    elif meta['depletion_mode']:
        parts.append(f"depletion_H2SO4-{_fmt_exp(meta['h2so4_molec_cm3'])}")
    else:
        parts.append(f"S{meta['scenario_id']:02d}")

    if n_total_override is not None:
        parts.append(f"N{_fmt_exp(n_total_override)}")
    if gmd_override is not None:
        parts.append(f"GMD{gmd_override:g}")
    if gsd_override is not None:
        parts.append(f"GSD{gsd_override:g}")
    if temp_override is not None:
        parts.append(f"T{int(temp_override)}K")
    if pres_override is not None:
        parts.append(f"P{int(pres_override)}Pa")
    if dt_override is not None:
        parts.append(f"dt{int(dt_override)}")

    return "_".join(parts)


def run_convergence(scenario_idx=0, make_plots=True, dt_override=None,
                    depletion_mode=False, constant_gc_mode=False,
                    h2so4_molec_cm3=1e7,
                    n_total_override=None, gmd_override=None,
                    gsd_override=None, temp_override=None,
                    pres_override=None, standard_grid=False,
                    include_fortran=False, mode='cond_only',
                    xk0_dp_nm=None, extra_bins=0):
    """Run TFL vs PPM convergence test at multiple resolutions.

    Args:
        scenario_idx: 0-based index into LHC scenarios
        make_plots: Generate matplotlib plots
        dt_override: Override timestep [s] (default: DT=60)
        depletion_mode: If True, use fixed initial H2SO4 with zero production
        constant_gc_mode: If True, hold gas at h2so4_molec_cm3 (no depletion)
        h2so4_molec_cm3: H2SO4 concentration [molec/cm3]
        n_total_override: Override N_total [#/cm3] (default: use scenario value)
        gmd_override: Override GMD [um] (default: use scenario value)
        gsd_override: Override GSD (default: use scenario value)
        temp_override: Override temperature [K] (default: use scenario value)
        pres_override: Override pressure [Pa] (default: use scenario value)
        standard_grid: Use standard TOMAS grid (xk0=1.5625e-23, ~3.2nm)
        include_fortran: Load and overlay Fortran TFL results at 36 bins
        mode: 'cond_only' (default) or 'combined' (coag+cond)
        xk0_dp_nm: First bin edge diameter [nm] (overrides --standard-grid)

    Returns:
        results: dict of results per grid config
    """
    scenarios = get_scenarios(n=50, seed=42)
    sc = scenarios[scenario_idx]
    sid = sc['scenario_id']

    # Override parameters if requested
    if n_total_override is not None:
        sc['N_total'] = n_total_override
    if gmd_override is not None:
        sc['GMD_um'] = gmd_override
    if gsd_override is not None:
        sc['GSD'] = gsd_override
    if temp_override is not None:
        sc['temp'] = temp_override
    if pres_override is not None:
        sc['pres'] = pres_override

    # Convert H2SO4 molec/cm3 to kg/grid cell
    h2so4_kg = h2so4_molec_cm3 * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO

    if mode == 'combined':
        mode_label = 'Coag+Cond'
    elif mode == 'coag_only':
        mode_label = 'Coag-only'
    else:
        mode_label = 'Cond-only'

    if constant_gc_mode:
        prod_val = 0.0
        gc_so4 = h2so4_kg
        print(f"=== Convergence Test: Constant Gas Mode ({mode_label}) ===")
        print(f"  H2SO4 = {h2so4_molec_cm3:.1e} molec/cm3 "
              f"= {h2so4_kg:.2e} kg/cell (held constant)")
    elif depletion_mode:
        prod_val = 0.0
        gc_so4 = h2so4_kg
        print(f"=== Convergence Test: Depletion Mode ({mode_label}) ===")
        print(f"  Initial H2SO4 = {h2so4_molec_cm3:.1e} molec/cm3 "
              f"= {h2so4_kg:.2e} kg/cell")
        print(f"  Production = 0 (pure depletion)")
    else:
        prod_val = sc['h2so4_prod_kg_per_s']
        gc_so4 = sc['Gc_SO4_kg']
        print(f"=== Convergence Test: Scenario S{sid:02d} ({mode_label}) ===")
        print(f"  Gc_SO4={gc_so4:.2e} kg/cell, "
              f"prod={sc['h2so4_prod']:.1e} molec/cm3/s")

    print(f"  N={sc['N_total']:.1e} #/cm3, GMD={sc['GMD_um']:.3f} um, "
          f"GSD={sc['GSD']:.2f}")
    print(f"  T={sc['temp']:.0f} K, P={sc['pres']:.0f} Pa, RH={sc['RH']:.2f}")

    # Grid starting point
    if xk0_dp_nm is not None:
        xk0 = dp_nm_to_xk0(xk0_dp_nm)
        grid_label = f"{xk0_dp_nm:.1f}nm"
    elif standard_grid:
        xk0 = XK0_STANDARD
        grid_label = "standard TOMAS (~3.2nm)"
    else:
        xk0 = XK0_17NM
        grid_label = "1.7nm"
    print(f"  Grid: {grid_label}, xk0={xk0:.4e} kg")

    # Timestep configuration
    use_dt = dt_override if dt_override is not None else DT
    nsteps = int(24 * 3600 / use_dt)
    print(f"  dt={use_dt:.1f}s, nsteps={nsteps}")
    print()

    grid_configs = _make_grid_configs(extra_bins)
    results = {}

    for label, nbins, factor in grid_configs:
        print(f"--- {label} ---")
        xk = make_grid(nbins=nbins, xk0=xk0, doubling_factor=factor)

        # Initialize
        Nk, Mk = init_lognormal(sc['N_total'], sc['GMD_um'], sc['GSD'], xk)
        Gc = np.zeros(N_GAS_SPECIES)
        Gc[SRTSO4] = gc_so4

        Nk_jax = jnp.array(Nk)
        Mk_jax = jnp.array(Mk)
        Gc_jax = jnp.array(Gc)
        xk_jax = jnp.array(xk)

        prod_rate = jnp.array(prod_val)
        temp = jnp.array(sc['temp'])
        pres = jnp.array(sc['pres'])
        boxvol = jnp.array(float(BOXVOL))
        rh = jnp.array(sc['RH'])
        alpha = jnp.array(1.0)
        dt = jnp.array(use_dt)

        # Initial total mass (gas + aerosol dry)
        M_aer_init = float(np.sum(Mk[:, :SRTH2O]))
        M_gas_init = float(Gc[SRTSO4])
        M_total_init = M_aer_init + M_gas_init

        entry = {
            'label': label, 'nbins': nbins, 'factor': factor,
            'xk': np.array(xk),
            'Nk_init': Nk.copy(), 'Mk_init': Mk.copy(),
            'Gc_init': Gc.copy(),
            'M_total_init': M_total_init,
        }

        # --- Coag-only shortcut ---
        if mode == 'coag_only':
            # No gas involvement — total init mass is aerosol only
            entry['M_total_init'] = M_aer_init

            # Scale substeps with bin refinement: narrower bins need more
            # substeps to prevent Euler overshoot across bin boundaries
            n_sub = max(3, int(round(3 * nbins / 36)))

            print(f"  Coag ({n_sub} substeps): compiling...",
                  end="", flush=True)
            _ = _run_coag_only_scan(Nk_jax, Mk_jax, xk_jax,
                                    temp, pres, boxvol, dt, nsteps,
                                    n_substeps=n_sub)
            jax.block_until_ready(_)

            print(f" running...", end="", flush=True)
            t0 = time.perf_counter()
            Nk_f, Mk_f, hist_coag = _run_coag_only_scan(
                Nk_jax, Mk_jax, xk_jax, temp, pres, boxvol, dt, nsteps,
                n_substeps=n_sub)
            jax.block_until_ready(Nk_f)
            t_coag = time.perf_counter() - t0

            M_aer_coag = float(jnp.sum(Mk_f[:, :SRTH2O]))

            entry['tfl_Nk'] = np.array(Nk_f)
            entry['tfl_Mk'] = np.array(Mk_f)
            entry['tfl_Gc'] = np.zeros(N_GAS_SPECIES)
            entry['tfl_hist'] = np.array(hist_coag)
            entry['tfl_time'] = t_coag
            entry['tfl_N_tot'] = float(jnp.sum(Nk_f))
            entry['tfl_M_dry'] = M_aer_coag
            entry['tfl_M_gas'] = 0.0
            entry['tfl_M_total'] = M_aer_coag
            entry['tfl_mass_err'] = abs(M_aer_coag - M_aer_init) / max(M_aer_init, 1e-30)
            print(f" {t_coag:.2f}s  N={entry['tfl_N_tot']:.4e}  "
                  f"M_cons={entry['tfl_mass_err']:.2e}")

            results[nbins] = entry
            continue

        # --- TFL ---
        def _run_tfl():
            if mode == 'combined':
                if constant_gc_mode:
                    return _run_constant_gc_combined_scan_tfl(
                        Nk_jax, Mk_jax, Gc_jax, xk_jax,
                        temp, pres, boxvol, rh, alpha, dt, nsteps)
                # run_combined_scan_tfl returns N_history (1D);
                # repack as 3-col for compatibility
                Nk_f, Mk_f, Gc_f, N_hist = run_combined_scan_tfl(
                    Nk_jax, Mk_jax, Gc_jax, xk_jax,
                    temp, pres, boxvol, rh, alpha, dt, nsteps, prod_rate)
                # N_hist is (nsteps,), wrap as (nsteps, 3) [N_tot, M_dry=0, Gc=0]
                # (time-series plots disabled for this path — use dummy)
                hist = jnp.stack([N_hist,
                                  jnp.zeros_like(N_hist),
                                  jnp.zeros_like(N_hist)], axis=1)
                return Nk_f, Mk_f, Gc_f, hist
            if constant_gc_mode:
                return _run_constant_gc_scan_tfl(
                    Nk_jax, Mk_jax, Gc_jax, xk_jax,
                    temp, pres, boxvol, rh, alpha, dt, nsteps)
            return run_condensation_scan_tfl(
                Nk_jax, Mk_jax, Gc_jax, xk_jax,
                temp, pres, boxvol, rh, alpha, dt, nsteps, prod_rate)

        print(f"  TFL: compiling...", end="", flush=True)
        _ = _run_tfl()
        jax.block_until_ready(_)

        print(f" running...", end="", flush=True)
        t0 = time.perf_counter()
        Nk_f, Mk_f, Gc_f, hist_tfl = _run_tfl()
        jax.block_until_ready(Nk_f)
        t_tfl = time.perf_counter() - t0

        M_aer_tfl = float(jnp.sum(Mk_f[:, :SRTH2O]))
        M_gas_tfl = float(Gc_f[SRTSO4])
        M_total_tfl = M_aer_tfl + M_gas_tfl

        entry['tfl_Nk'] = np.array(Nk_f)
        entry['tfl_Mk'] = np.array(Mk_f)
        entry['tfl_Gc'] = np.array(Gc_f)
        entry['tfl_hist'] = np.array(hist_tfl)  # (nsteps, 3): N_tot, M_dry, Gc_SO4
        entry['tfl_time'] = t_tfl
        entry['tfl_N_tot'] = float(jnp.sum(Nk_f))
        entry['tfl_M_dry'] = M_aer_tfl
        entry['tfl_M_gas'] = M_gas_tfl
        entry['tfl_M_total'] = M_total_tfl
        entry['tfl_mass_err'] = abs(M_total_tfl - M_total_init) / max(M_total_init, 1e-30)
        print(f" {t_tfl:.2f}s  N={entry['tfl_N_tot']:.4e}  "
              f"M_cons={entry['tfl_mass_err']:.2e}")

        # --- PPM ---
        def _run_ppm():
            if mode == 'combined':
                if constant_gc_mode:
                    return _run_constant_gc_combined_scan_ppm(
                        Nk_jax, Mk_jax, Gc_jax, xk_jax,
                        temp, pres, boxvol, rh, alpha, dt, nsteps)
                Nk_f, Mk_f, Gc_f, N_hist = run_combined_scan_ppm(
                    Nk_jax, Mk_jax, Gc_jax, xk_jax,
                    temp, pres, boxvol, rh, alpha, dt, nsteps, prod_rate)
                hist = jnp.stack([N_hist,
                                  jnp.zeros_like(N_hist),
                                  jnp.zeros_like(N_hist)], axis=1)
                return Nk_f, Mk_f, Gc_f, hist
            if constant_gc_mode:
                return _run_constant_gc_scan_ppm(
                    Nk_jax, Mk_jax, Gc_jax, xk_jax,
                    temp, pres, boxvol, rh, alpha, dt, nsteps)
            return run_condensation_scan(
                Nk_jax, Mk_jax, Gc_jax, xk_jax,
                temp, pres, boxvol, rh, alpha, dt, nsteps, prod_rate)

        print(f"  PPM: compiling...", end="", flush=True)
        _ = _run_ppm()
        jax.block_until_ready(_)

        print(f" running...", end="", flush=True)
        t0 = time.perf_counter()
        Nk_f_ppm, Mk_f_ppm, Gc_f_ppm, hist_ppm = _run_ppm()
        jax.block_until_ready(Nk_f_ppm)
        t_ppm = time.perf_counter() - t0

        M_aer_ppm = float(jnp.sum(Mk_f_ppm[:, :SRTH2O]))
        M_gas_ppm = float(Gc_f_ppm[SRTSO4])
        M_total_ppm = M_aer_ppm + M_gas_ppm

        entry['ppm_Nk'] = np.array(Nk_f_ppm)
        entry['ppm_Mk'] = np.array(Mk_f_ppm)
        entry['ppm_Gc'] = np.array(Gc_f_ppm)
        entry['ppm_hist'] = np.array(hist_ppm)  # (nsteps, 3): N_tot, M_dry, Gc_SO4
        entry['ppm_time'] = t_ppm
        entry['ppm_N_tot'] = float(jnp.sum(Nk_f_ppm))
        entry['ppm_M_dry'] = M_aer_ppm
        entry['ppm_M_gas'] = M_gas_ppm
        entry['ppm_M_total'] = M_total_ppm
        entry['ppm_mass_err'] = abs(M_total_ppm - M_total_init) / max(M_total_init, 1e-30)
        print(f" {t_ppm:.2f}s  N={entry['ppm_N_tot']:.4e}  "
              f"M_cons={entry['ppm_mass_err']:.2e}")

        results[nbins] = entry

    # --- Summary table ---
    print("\n=== Summary ===")
    has_ppm = 'ppm_N_tot' in results[sorted(results.keys())[0]]
    if has_ppm:
        hdr = (f"{'Config':<22s} {'TFL time':>9s} {'TFL N_tot':>12s} {'TFL M_aer':>12s}"
               f" {'TFL M_cons':>10s} {'PPM N_tot':>12s} {'PPM M_aer':>12s}"
               f" {'PPM M_cons':>10s}")
    else:
        method_lbl = 'Coag' if mode == 'coag_only' else 'TFL'
        hdr = (f"{'Config':<22s} {'Time':>9s} {method_lbl+' N_tot':>12s} "
               f"{method_lbl+' M_aer':>12s} {method_lbl+' M_cons':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for nbins_key in sorted(results.keys()):
        r = results[nbins_key]
        if has_ppm:
            print(f"{r['label']:<22s} {r['tfl_time']:>8.2f}s {r['tfl_N_tot']:>12.4e} "
                  f"{r['tfl_M_dry']:>12.4e} {r['tfl_mass_err']:>10.2e} "
                  f"{r['ppm_N_tot']:>12.4e} {r['ppm_M_dry']:>12.4e} "
                  f"{r['ppm_mass_err']:>10.2e}")
        else:
            print(f"{r['label']:<22s} {r['tfl_time']:>8.2f}s {r['tfl_N_tot']:>12.4e} "
                  f"{r['tfl_M_dry']:>12.4e} {r['tfl_mass_err']:>10.2e}")

    # Mass in μg/m³ (boxvol = 1e6 cm³ = 1 m³, so kg/cell = kg/m³; × 1e9 = μg/m³)
    KG_TO_UG = 1.0e9
    print(f"\n=== Mass Budget (μg/m³) ===")
    r0 = results[sorted(results.keys())[0]]
    M_init_ug = r0['M_total_init'] * KG_TO_UG
    M_aer_init_ug = (r0['M_total_init'] - float(r0['Gc_init'][SRTSO4])) * KG_TO_UG
    M_gas_init_ug = float(r0['Gc_init'][SRTSO4]) * KG_TO_UG
    print(f"  Initial: aerosol={M_aer_init_ug:.4f}, gas={M_gas_init_ug:.4f}, "
          f"total={M_init_ug:.4f} μg/m³")
    for nbins_key in sorted(results.keys()):
        r = results[nbins_key]
        print(f"  {r['label']}:")
        print(f"    TFL final: aerosol={r['tfl_M_dry']*KG_TO_UG:.4f}, "
              f"gas={r['tfl_M_gas']*KG_TO_UG:.6f}, "
              f"total={r['tfl_M_total']*KG_TO_UG:.4f} μg/m³")
        if has_ppm:
            print(f"    PPM final: aerosol={r['ppm_M_dry']*KG_TO_UG:.4f}, "
                  f"gas={r['ppm_M_gas']*KG_TO_UG:.6f}, "
                  f"total={r['ppm_M_total']*KG_TO_UG:.4f} μg/m³")

    if depletion_mode:
        print(f"\n  Initial total mass (gas+aerosol): {results[sorted(results.keys())[0]]['M_total_init']:.6e} kg")
        for nbins_key in sorted(results.keys()):
            r = results[nbins_key]
            if has_ppm:
                print(f"  {r['label']}: TFL gas_left={r['tfl_M_gas']:.4e}  "
                      f"PPM gas_left={r['ppm_M_gas']:.4e}")
            else:
                print(f"  {r['label']}: gas_left={r['tfl_M_gas']:.4e}")

    # Store metadata for plots
    meta = {
        'scenario_id': sid,
        'N_total': sc['N_total'],
        'GMD_um': sc['GMD_um'],
        'GSD': sc['GSD'],
        'temp': sc['temp'],
        'pres': sc['pres'],
        'RH': sc['RH'],
        'h2so4_molec_cm3': h2so4_molec_cm3 if (depletion_mode or constant_gc_mode) else None,
        'depletion_mode': depletion_mode,
        'constant_gc_mode': constant_gc_mode,
        'mode': mode,
        'dt': use_dt,
        'nsteps': nsteps,
        'M_aer_init_ug': M_aer_init_ug,
        'M_gas_init_ug': M_gas_init_ug,
        'M_total_init_ug': M_init_ug,
        '_prod_rate_kg': float(prod_val),
    }

    # Load Fortran results if requested
    fortran_data = None
    if include_fortran:
        fortran_label = "Fortran Coag+Cond" if mode == 'combined' else "Fortran TFL"
        print(f"\n--- Loading {fortran_label} results (36 bins, standard grid) ---")
        fortran_data = load_fortran_results(use_dt, mode=mode)
        if fortran_data is not None:
            f_Nk = fortran_data['Nk']
            f_Mk = fortran_data['Mk']
            f_M_dry = float(np.sum(f_Mk[:, :SRTH2O]))
            print(f"  {fortran_label} N_tot={np.sum(f_Nk):.4e}  "
                  f"M_dry={f_M_dry:.4e}  ({f_M_dry*KG_TO_UG:.4f} μg/m³)")

    # Build unique filename tag from simulation parameters
    file_tag = _build_file_tag(meta, n_total_override, gmd_override,
                               gsd_override, dt_override,
                               temp_override, pres_override)
    if xk0_dp_nm is not None:
        file_tag = file_tag + f'_xk0-{xk0_dp_nm:g}nm'
    if mode == 'combined':
        file_tag = file_tag + '_combined'
    elif mode == 'coag_only':
        file_tag = file_tag + '_coag'

    # --- Plots ---
    if make_plots:
        plot_convergence(results, file_tag, meta=meta,
                         fortran_data=fortran_data)

    return results


def plot_convergence(results, scenario_id_or_label, meta=None,
                     fortran_data=None):
    """Generate convergence comparison plots.

    Args:
        results: dict of results per grid config
        scenario_id_or_label: int scenario ID or string label for filenames
        meta: dict with scenario metadata (N_total, GMD_um, GSD, etc.)
        fortran_data: dict with Fortran Nk/Mk/xk (36 bins) or None
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'convergence')
    os.makedirs(outdir, exist_ok=True)

    tag = scenario_id_or_label              # for filenames (e.g., S01_N1e4)
    display_tag = tag.replace("_", " ")     # for plot titles (e.g., S01 N1e4)
    sorted_keys = sorted(results.keys())
    is_combined = meta is not None and meta.get('mode') == 'combined'
    is_coag_only = meta is not None and meta.get('mode') == 'coag_only'
    if is_combined:
        process_label = 'Coag+Cond'
    elif is_coag_only:
        process_label = 'Coagulation'
    else:
        process_label = 'Condensation'
    fortran_legend = 'Fortran Coag+Cond' if is_combined else 'Fortran TFL'

    # Build info string for plot annotations
    KG_TO_UG = 1.0e9
    if meta is not None:
        # Line 1: Aerosol initial conditions
        line1 = (f"Aerosol: N₀ = {meta['N_total']:.1e} #/cm³, "
                 f"GMD = {meta['GMD_um']:.3f} μm, GSD = {meta['GSD']:.2f}")

        # Line 2: Environment + simulation setup
        mode_info = f"  |  Mode: {process_label}"
        line2 = (f"Environment: T = {meta['temp']:.0f} K, "
                 f"P = {meta['pres']:.0f} Pa, RH = {meta['RH']:.0%}"
                 f"{mode_info}")

        # Line 3: Gas + simulation timing
        if meta.get('constant_gc_mode'):
            line3 = (f"Gas: H₂SO₄ = {meta['h2so4_molec_cm3']:.0e} molec/cm³ "
                     f"(held constant)  |  dt = {meta['dt']:.0f}s × {meta['nsteps']} steps")
        elif meta.get('depletion_mode'):
            line3 = (f"Gas: H₂SO₄₀ = {meta['h2so4_molec_cm3']:.0e} molec/cm³ "
                     f"(no production)  |  dt = {meta['dt']:.0f}s × {meta['nsteps']} steps")
        else:
            line3 = f"Simulation: dt = {meta['dt']:.0f}s × {meta['nsteps']} steps (24h)"

        # Line 4: Mass budget table header + initial
        line4 = (f"Initial mass [μg/m³]: "
                 f"aerosol = {meta['M_aer_init_ug']:.4f}, "
                 f"gas = {meta['M_gas_init_ug']:.4f}, "
                 f"total = {meta['M_total_init_ug']:.4f}")

        # Lines 5+: Final aerosol dry mass per resolution
        mass_lines = ["Final aerosol dry mass [μg/m³]:"]
        for nbins_key in sorted_keys:
            r = results[nbins_key]
            tfl_m = r['tfl_M_dry'] * KG_TO_UG
            if 'ppm_M_dry' in r:
                ppm_m = r['ppm_M_dry'] * KG_TO_UG
                mass_lines.append(
                    f"  {r['label']:<20s}  TFL = {tfl_m:.4f},  PPM = {ppm_m:.4f}"
                )
            else:
                method_lbl = 'Coag' if is_coag_only else 'TFL'
                mass_lines.append(
                    f"  {r['label']:<20s}  {method_lbl} = {tfl_m:.4f}"
                )

        info_text = "\n".join([line1, line2, line3, line4] + mass_lines)
    else:
        info_text = None

    # -----------------------------------------------------------------
    # NYT-inspired style: muted palette, no chart junk, clean geometry
    # -----------------------------------------------------------------
    _BG = '#FFFFFF'
    _TEXT = '#333333'
    _GRID = '#E0E0E0'
    _ANNOT = '#666666'

    # Fortran style
    _FORTRAN_C = '#2CA02C'   # green (distinct from TFL/PPM palette)
    _FORTRAN_LW = 2.0
    _FORTRAN_LS = ':'        # dotted

    # Muted, high-contrast palette — distinguishable in print & on screen
    _PALETTE = {
        36:  '#1A1A2E',   # near-black (anchor)
        40:  '#1A1A2E',   # near-black (anchor, +4 variant)
        72:  '#C44E52',   # muted red
        80:  '#C44E52',   # muted red (+4 variant)
        144: '#4C72B0',   # steel blue
        160: '#4C72B0',   # steel blue (+4 variant)
        108: '#8172B2',   # muted purple
    }
    _LW = {36: 2.2, 40: 2.2, 72: 1.8, 80: 1.8, 144: 1.5, 160: 1.5, 108: 1.5}

    def _color(nbins_key):
        return _PALETTE.get(nbins_key, '#555555')

    def _lw(nbins_key):
        return _LW.get(nbins_key, 1.5)

    def _get_xlim_from_ppm(results, sorted_keys):
        all_Dp = []
        for nbins_key in sorted_keys:
            r = results[nbins_key]
            key = 'ppm_Nk' if 'ppm_Nk' in r else 'tfl_Nk'
            Dp, dN = compute_dNdlogDp(r[key], r['xk'])
            mask = dN > np.max(dN) * 0.01
            if np.any(mask):
                all_Dp.extend(Dp[mask])
        if all_Dp:
            return min(all_Dp) * 0.3, max(all_Dp) * 3.0
        return 1, 10000

    def _nyt_ax(ax, xlabel='', ylabel='', title='', xlim_log=True):
        """Apply NYT-style formatting to an axis."""
        if xlim_log:
            ax.set_xscale('log')
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', which='major', length=4, width=0.6,
                       labelsize=9, colors=_TEXT)
        ax.tick_params(axis='both', which='minor', length=2, width=0.4,
                       colors=_TEXT)
        ax.yaxis.grid(True, which='major', color=_GRID, linewidth=0.6)
        ax.yaxis.grid(True, which='minor', color=_GRID, linewidth=0.3, alpha=0.35)
        ax.xaxis.grid(True, which='major', color=_GRID, linewidth=0.4, alpha=0.5)
        ax.xaxis.grid(True, which='minor', color=_GRID, linewidth=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, color=_TEXT, labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, color=_TEXT, labelpad=8)
        if title:
            ax.set_title(title, fontsize=12, color=_TEXT, fontweight='bold',
                         loc='left', pad=10)

    def _nyt_legend(ax, **kwargs):
        """Minimal legend — no frame, small text."""
        leg = ax.legend(fontsize=8.5, frameon=False, labelcolor=_TEXT, **kwargs)
        return leg

    def _nyt_fig(nrows, ncols, figsize, suptitle=''):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 facecolor=_BG)
        if suptitle:
            fig.suptitle(suptitle, fontsize=13, color=_TEXT,
                         fontweight='bold', x=0.02, ha='left', y=0.98)
        return fig, axes

    def _add_info_box(fig, info_text):
        if info_text:
            fig.text(0.5, -0.01, info_text, ha='center', va='top',
                     fontsize=7, color=_ANNOT, family='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7F7F7',
                               edgecolor='#E0E0E0', linewidth=0.5))

    # =====================================================================
    # Figure 1: Size distributions
    # =====================================================================
    if is_coag_only:
        fig1_panels = [('Initial', 'Nk_init'), ('After 24 h', 'tfl_Nk')]
    else:
        fig1_panels = [('Initial', 'Nk_init'),
                       ('After 24 h — TFL', 'tfl_Nk'),
                       ('After 24 h — PPM', 'ppm_Nk')]
    n_fig1 = len(fig1_panels)
    fig, axes = plt.subplots(1, n_fig1, figsize=(6 * n_fig1, 5), facecolor=_BG,
                             sharey=True)
    if n_fig1 == 1:
        axes = [axes]
    fig.suptitle(f'Size Distribution Convergence ({process_label})  /  {display_tag}',
                 fontsize=13, color=_TEXT, fontweight='bold',
                 x=0.02, ha='left', y=0.98)

    for panel_idx, (ax, (panel_title, nk_key)) in enumerate(zip(axes, fig1_panels)):
        for nbins_key in sorted_keys:
            r = results[nbins_key]
            if nk_key not in r:
                continue
            Dp, dN = compute_dNdlogDp(r[nk_key], r['xk'])
            ax.plot(Dp, dN, color=_color(nbins_key), lw=_lw(nbins_key),
                    label=r['label'])
        # Overlay Fortran on final panels (TFL comparison)
        if fortran_data is not None and nk_key in ('tfl_Nk', 'ppm_Nk'):
            Dp_f, dN_f = compute_dNdlogDp(fortran_data['Nk'],
                                           fortran_data['xk'])
            ax.plot(Dp_f, dN_f, color=_FORTRAN_C, lw=_FORTRAN_LW,
                    ls=_FORTRAN_LS, label=fortran_legend)
        _nyt_ax(ax, xlabel='Diameter [nm]',
                ylabel='dN/dlog Dp  [cm$^{-3}$]' if panel_idx == 0 else '',
                title=panel_title)
        _nyt_legend(ax)
        ax.set_xlim(1, 10000)
        if panel_idx > 0:
            ax.tick_params(labelleft=False)

    _add_info_box(fig, info_text)
    fig.subplots_adjust(wspace=0.08)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path1 = os.path.join(outdir, f'convergence_sizedist_{tag}.png')
    fig.savefig(path1, dpi=180, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f"\nSaved: {path1}")

    # =====================================================================
    # Figure 2: Zoomed comparison
    # =====================================================================
    xlim = _get_xlim_from_ppm(results, sorted_keys)
    if is_coag_only:
        fig2_panels = [('After 24 h', 'tfl_Nk')]
    else:
        fig2_panels = [('TFL', 'tfl_Nk'), ('PPM', 'ppm_Nk')]
    n_fig2 = len(fig2_panels)
    fig, axes = plt.subplots(1, n_fig2, figsize=(7 * n_fig2, 5.5), facecolor=_BG,
                             sharey=True)
    if n_fig2 == 1:
        axes = [axes]
    fig.suptitle(f'Zoomed Comparison ({process_label})  /  {display_tag}',
                 fontsize=13, color=_TEXT, fontweight='bold',
                 x=0.02, ha='left', y=0.98)

    for panel_idx, (ax, (panel_title, nk_key)) in enumerate(zip(axes, fig2_panels)):
        for nbins_key in sorted_keys:
            r = results[nbins_key]
            if nk_key not in r:
                continue
            Dp, dN = compute_dNdlogDp(r[nk_key], r['xk'])
            ax.plot(Dp, dN, color=_color(nbins_key), lw=_lw(nbins_key),
                    label=r['label'])
        if fortran_data is not None:
            Dp_f, dN_f = compute_dNdlogDp(fortran_data['Nk'],
                                           fortran_data['xk'])
            ax.plot(Dp_f, dN_f, color=_FORTRAN_C, lw=_FORTRAN_LW,
                    ls=_FORTRAN_LS, label=fortran_legend)
        _nyt_ax(ax, xlabel='Diameter [nm]',
                ylabel='dN/dlog Dp  [cm$^{-3}$]' if panel_idx == 0 else '',
                title=panel_title)
        _nyt_legend(ax)
        ax.set_xlim(xlim)
        if panel_idx > 0:
            ax.tick_params(labelleft=False)

    _add_info_box(fig, info_text)
    fig.subplots_adjust(wspace=0.08)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path2 = os.path.join(outdir, f'convergence_zoomed_{tag}.png')
    fig.savefig(path2, dpi=180, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f"Saved: {path2}")

    # =====================================================================
    # Figure 3: Convergence of total N, M_dry, and mass conservation
    # =====================================================================
    nbins_list = sorted(results.keys())
    tfl_N = [results[n]['tfl_N_tot'] for n in nbins_list]
    tfl_M = [results[n]['tfl_M_dry'] for n in nbins_list]
    ppm_nbins = [n for n in nbins_list if 'ppm_N_tot' in results[n]]
    ppm_N = [results[n]['ppm_N_tot'] for n in ppm_nbins]
    ppm_M = [results[n]['ppm_M_dry'] for n in ppm_nbins]

    _TFL_C = '#1A1A2E'
    _PPM_C = '#C44E52'

    has_mass_cons = 'tfl_mass_err' in results[nbins_list[0]]
    ncols = 3 if has_mass_cons else 2
    fig, axes = _nyt_fig(1, ncols, (5 * ncols, 4.5),
                         f'Integral Convergence  /  {display_tag}')

    ax = axes[0]
    ax.plot(nbins_list, tfl_N, '-', color=_TFL_C, lw=1.8, marker='o',
            markersize=5, label='TFL')
    if ppm_N:
        ax.plot(ppm_nbins, ppm_N, '-', color=_PPM_C, lw=1.8, marker='o',
                markersize=5, label='PPM')
    _nyt_ax(ax, xlabel='Number of bins', ylabel='Total N  [# / cell]',
            title='Total Number', xlim_log=False)
    _nyt_legend(ax)

    ax = axes[1]
    ax.plot(nbins_list, tfl_M, '-', color=_TFL_C, lw=1.8, marker='o',
            markersize=5, label='TFL')
    if ppm_M:
        ax.plot(ppm_nbins, ppm_M, '-', color=_PPM_C, lw=1.8, marker='o',
                markersize=5, label='PPM')
    _nyt_ax(ax, xlabel='Number of bins', ylabel='Dry mass  [kg / cell]',
            title='Dry Aerosol Mass', xlim_log=False)
    _nyt_legend(ax)

    if has_mass_cons:
        ax = axes[2]
        tfl_err = [results[n]['tfl_mass_err'] for n in nbins_list]
        ppm_err = [results[n]['ppm_mass_err'] for n in ppm_nbins]
        ax.semilogy(nbins_list, tfl_err, '-', color=_TFL_C, lw=1.8,
                    marker='o', markersize=5, label='TFL')
        if ppm_err:
            ax.semilogy(ppm_nbins, ppm_err, '-', color=_PPM_C, lw=1.8,
                        marker='o', markersize=5, label='PPM')
        _nyt_ax(ax, xlabel='Number of bins',
                ylabel='Relative mass error', title='Mass Conservation',
                xlim_log=False)
        _nyt_legend(ax)

    _add_info_box(fig, info_text)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path3 = os.path.join(outdir, f'convergence_totals_{tag}.png')
    fig.savefig(path3, dpi=180, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f"Saved: {path3}")

    # =====================================================================
    # Figure 4: Timing comparison (horizontal bars — cleaner)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=_BG)

    labels_list = [results[n]['label'] for n in nbins_list]
    tfl_times = [results[n]['tfl_time'] for n in nbins_list]
    ppm_times = [results[n].get('ppm_time', 0) for n in nbins_list]

    y = np.arange(len(labels_list))
    if is_coag_only:
        h = 0.5
        bar_label = 'Coag (Euler)'
        ax.barh(y, tfl_times, h, color=_TFL_C, alpha=0.85, label=bar_label)
        for i, t in enumerate(tfl_times):
            ax.text(t + 0.08, y[i], f'{t:.2f}s',
                    va='center', ha='left', fontsize=8.5, color=_TEXT)
    else:
        h = 0.35
        ax.barh(y + h / 2, tfl_times, h, color=_TFL_C, alpha=0.85, label='TFL')
        ax.barh(y - h / 2, ppm_times, h, color=_PPM_C, alpha=0.85, label='PPM')
        for i, t in enumerate(tfl_times):
            ax.text(t + 0.08, y[i] + h / 2, f'{t:.2f}s',
                    va='center', ha='left', fontsize=8.5, color=_TEXT)
        for i, t in enumerate(ppm_times):
            if t > 0:
                ax.text(t + 0.08, y[i] - h / 2, f'{t:.2f}s',
                        va='center', ha='left', fontsize=8.5, color=_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_list, fontsize=10)
    ax.invert_yaxis()
    _nyt_ax(ax, xlabel='Wall time [s]', title=f'Runtime  /  {display_tag}',
            xlim_log=False)
    ax.xaxis.grid(True, color=_GRID, linewidth=0.6)
    _nyt_legend(ax, loc='upper right')

    _add_info_box(fig, info_text)
    plt.tight_layout()
    path4 = os.path.join(outdir, f'convergence_timing_{tag}.png')
    fig.savefig(path4, dpi=180, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f"Saved: {path4}")

    # =====================================================================
    # Figure 5: Mass distribution
    # =====================================================================
    if is_coag_only:
        fig5_panels = [('Initial', 'Mk_init'), ('After 24 h', 'tfl_Mk')]
    else:
        fig5_panels = [('Initial', 'Mk_init'),
                       ('After 24 h — TFL', 'tfl_Mk'),
                       ('After 24 h — PPM', 'ppm_Mk')]
    n_fig5 = len(fig5_panels)
    fig, axes = plt.subplots(1, n_fig5, figsize=(6 * n_fig5, 5), facecolor=_BG,
                             sharey=True)
    if n_fig5 == 1:
        axes = [axes]
    fig.suptitle(f'Mass Distribution Convergence ({process_label})  /  {display_tag}',
                 fontsize=13, color=_TEXT, fontweight='bold',
                 x=0.02, ha='left', y=0.98)

    for panel_idx, (ax, (panel_title, mk_key)) in enumerate(zip(axes, fig5_panels)):
        for nbins_key in sorted_keys:
            r = results[nbins_key]
            if mk_key not in r:
                continue
            Dp, dM = compute_dMdlogDp(r[mk_key], r['xk'])
            ax.plot(Dp, dM, color=_color(nbins_key), lw=_lw(nbins_key),
                    label=r['label'])
        if fortran_data is not None and mk_key in ('tfl_Mk', 'ppm_Mk'):
            Dp_f, dM_f = compute_dMdlogDp(fortran_data['Mk'],
                                           fortran_data['xk'])
            ax.plot(Dp_f, dM_f, color=_FORTRAN_C, lw=_FORTRAN_LW,
                    ls=_FORTRAN_LS, label=fortran_legend)
        _nyt_ax(ax, xlabel='Diameter [nm]',
                ylabel='dM/dlog Dp  [$\\mu$g m$^{-3}$]' if panel_idx == 0 else '',
                title=panel_title)
        _nyt_legend(ax)
        ax.set_xlim(1, 10000)
        if panel_idx > 0:
            ax.tick_params(labelleft=False)

    _add_info_box(fig, info_text)
    fig.subplots_adjust(wspace=0.08)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path5 = os.path.join(outdir, f'convergence_massdist_{tag}.png')
    fig.savefig(path5, dpi=180, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f"Saved: {path5}")

    # -----------------------------------------------------------------
    # Shared legend helper for time-series figures (Figs 6 & 7)
    # Convention: solid = TFL, dashed = PPM; color = resolution
    # -----------------------------------------------------------------
    def _add_shared_legend(fig, results, sorted_keys):
        """Add a compact legend: color lines for resolution + style note."""
        from matplotlib.lines import Line2D
        handles = []
        for nbins_key in sorted_keys:
            r = results[nbins_key]
            handles.append(Line2D([0], [0], color=_color(nbins_key),
                                  lw=1.8, label=r['label']))
        # Style indicators — only show TFL/PPM distinction when both present
        if not is_coag_only:
            handles.append(Line2D([0], [0], color='#888888', lw=1.5,
                                  ls='-', label='TFL (solid)'))
            handles.append(Line2D([0], [0], color='#888888', lw=1.5,
                                  ls='--', label='PPM (dashed)'))
        fig.legend(handles=handles, loc='upper right',
                   bbox_to_anchor=(0.98, 0.97),
                   ncol=len(sorted_keys) + (0 if is_coag_only else 2),
                   fontsize=8, frameon=False, labelcolor=_TEXT)

    def _plot_timeseries_panel(ax, results, sorted_keys, t_hours,
                               col_idx=None, combine_cols=None):
        """Plot TFL (solid) + PPM (dashed) for each resolution on one axis."""
        for nbins_key in sorted_keys:
            r = results[nbins_key]
            c = _color(nbins_key)
            for method, ls_val in [('tfl', '-'), ('ppm', '--')]:
                if f'{method}_hist' not in r:
                    continue
                h = r[f'{method}_hist']
                if col_idx is not None:
                    y = h[:, col_idx]
                elif combine_cols is not None:
                    y = sum(h[:, c] for c in combine_cols)
                ax.plot(t_hours, y, color=c, lw=1.5, ls=ls_val)

    # =====================================================================
    # Figure 6: Time series — 1x3 (Gas SO4, Aerosol M_dry, Total M)
    # =====================================================================
    has_hist = 'tfl_hist' in results[sorted_keys[0]]
    if has_hist and meta is not None:
        use_dt = meta['dt']
        use_nsteps = meta['nsteps']
        t_hours = np.arange(1, use_nsteps + 1) * use_dt / 3600.0

        fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=_BG)
        fig.suptitle(f'Time Evolution  /  {display_tag}',
                     fontsize=13, color=_TEXT, fontweight='bold',
                     x=0.02, ha='left', y=0.98)

        # Gas SO4
        _plot_timeseries_panel(axes[0], results, sorted_keys, t_hours,
                               col_idx=2)
        _nyt_ax(axes[0], xlabel='Time [hours]', ylabel='kg / cell',
                title='Gas H$_2$SO$_4$', xlim_log=False)

        # Aerosol dry mass
        _plot_timeseries_panel(axes[1], results, sorted_keys, t_hours,
                               col_idx=1)
        _nyt_ax(axes[1], xlabel='Time [hours]', ylabel='kg / cell',
                title='Aerosol dry mass', xlim_log=False)

        # Total (gas + aerosol)
        _plot_timeseries_panel(axes[2], results, sorted_keys, t_hours,
                               combine_cols=[1, 2])
        _nyt_ax(axes[2], xlabel='Time [hours]', ylabel='kg / cell',
                title='Total (gas + aerosol)', xlim_log=False)

        _add_shared_legend(fig, results, sorted_keys)
        _add_info_box(fig, info_text)
        fig.subplots_adjust(wspace=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        path6 = os.path.join(outdir, f'convergence_timeseries_{tag}.png')
        fig.savefig(path6, dpi=180, bbox_inches='tight', facecolor=_BG)
        plt.close(fig)
        print(f"Saved: {path6}")

        # =================================================================
        # Figure 7: Diagnostics — 1x3
        #   (a) N_tot over time
        #   (b) Mean particle diameter over time
        #   (c) Mass conservation error over time
        # =================================================================
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=_BG)
        fig.suptitle(f'Diagnostics  /  {display_tag}',
                     fontsize=13, color=_TEXT, fontweight='bold',
                     x=0.02, ha='left', y=0.98)

        # (a) N_tot over time
        _plot_timeseries_panel(axes[0], results, sorted_keys, t_hours,
                               col_idx=0)
        _nyt_ax(axes[0], xlabel='Time [hours]',
                ylabel='N$_{tot}$  [# / cell]',
                title='Total number', xlim_log=False)

        # (b) Mean diameter over time
        ax = axes[1]
        for nbins_key in sorted_keys:
            r = results[nbins_key]
            c = _color(nbins_key)
            for method, ls_val in [('tfl', '-'), ('ppm', '--')]:
                if f'{method}_hist' not in r:
                    continue
                h = r[f'{method}_hist']
                safe_N = np.maximum(h[:, 0], 1e-30)
                Dp_mean = 1e9 * ((6.0 * h[:, 1])
                                 / (DENS_INIT * np.pi * safe_N)) ** (1.0/3.0)
                ax.plot(t_hours, Dp_mean, color=c, lw=1.5, ls=ls_val)
        _nyt_ax(ax, xlabel='Time [hours]', ylabel='Mean Dp  [nm]',
                title='Mean particle diameter', xlim_log=False)

        # (c) Mass conservation error over time
        ax = axes[2]
        for nbins_key in sorted_keys:
            r = results[nbins_key]
            c = _color(nbins_key)
            M_init_k = r['M_total_init']
            for method, ls_val in [('tfl', '-'), ('ppm', '--')]:
                if f'{method}_hist' not in r:
                    continue
                h = r[f'{method}_hist']
                M_total_t = h[:, 1] + h[:, 2]
                if meta.get('depletion_mode'):
                    M_expected = M_init_k
                else:
                    prod_per_step = meta['_prod_rate_kg'] * use_dt
                    M_expected = M_init_k + np.cumsum(
                        np.full(use_nsteps, prod_per_step))
                rel_err = np.abs(M_total_t - M_expected) / max(M_init_k, 1e-30)
                ax.semilogy(t_hours, np.maximum(rel_err, 1e-16),
                            color=c, lw=1.5, ls=ls_val)
        _nyt_ax(ax, xlabel='Time [hours]', ylabel='Relative mass error',
                title='Mass conservation', xlim_log=False)

        _add_shared_legend(fig, results, sorted_keys)
        _add_info_box(fig, info_text)
        fig.subplots_adjust(wspace=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        path7 = os.path.join(outdir, f'convergence_diagnostics_{tag}.png')
        fig.savefig(path7, dpi=180, bbox_inches='tight', facecolor=_BG)
        plt.close(fig)
        print(f"Saved: {path7}")


def main():
    parser = argparse.ArgumentParser(
        description='TFL vs PPM convergence test at multiple bin resolutions')
    parser.add_argument('--scenario', type=int, default=1,
                        help='Scenario ID (1-50, default: 1)')
    parser.add_argument('--dt', type=float, default=None,
                        help='Timestep in seconds (default: 60)')
    parser.add_argument('--depletion', action='store_true',
                        help='Depletion mode: fixed initial H2SO4, no production')
    parser.add_argument('--constant-gc', action='store_true',
                        help='Constant gas mode: hold H2SO4 at --h2so4 value')
    parser.add_argument('--h2so4', type=float, default=1e7,
                        help='H2SO4 [molec/cm3] for depletion/constant-gc mode (default: 1e7)')
    parser.add_argument('--n-total', type=float, default=None,
                        help='Override N_total [#/cm3] (default: use scenario value)')
    parser.add_argument('--gmd', type=float, default=None,
                        help='Override GMD [um] (default: use scenario value)')
    parser.add_argument('--gsd', type=float, default=None,
                        help='Override GSD (default: use scenario value)')
    parser.add_argument('--temp', type=float, default=None,
                        help='Override temperature [K] (default: use scenario value)')
    parser.add_argument('--pres', type=float, default=None,
                        help='Override pressure [Pa] (default: use scenario value)')
    parser.add_argument('--standard-grid', action='store_true',
                        help='Use standard TOMAS grid (~3.2nm) instead of 1.7nm')
    parser.add_argument('--xk0-dp', type=float, default=None,
                        help='First bin edge diameter [nm] (overrides --standard-grid)')
    parser.add_argument('--fortran', action='store_true',
                        help='Load and overlay Fortran TFL results (36 bins)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--mode', choices=['cond_only', 'combined', 'coag_only'],
                        default='cond_only',
                        help='Mode: cond_only (default), combined (coag+cond), or coag_only')
    parser.add_argument('--extra-bins', type=int, default=0,
                        help='Extra bins per resolution (e.g., 4 → 40/80/160 for 1nm start)')
    args = parser.parse_args()

    run_convergence(
        scenario_idx=args.scenario - 1,
        make_plots=not args.no_plots,
        dt_override=args.dt,
        depletion_mode=args.depletion,
        constant_gc_mode=args.constant_gc,
        h2so4_molec_cm3=args.h2so4,
        n_total_override=args.n_total,
        gmd_override=args.gmd,
        gsd_override=args.gsd,
        temp_override=args.temp,
        pres_override=args.pres,
        standard_grid=args.standard_grid,
        include_fortran=args.fortran,
        mode=args.mode,
        xk0_dp_nm=args.xk0_dp,
        extra_bins=args.extra_bins,
    )


if __name__ == '__main__':
    main()
