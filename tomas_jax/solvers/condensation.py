"""Top-level condensation operator-split step for TOMAS-JAX.

Orchestrates the full condensation pipeline in sequence:

    1. H2SO4 condensation — exponential gas depletion + ezcond
    2. NH3 equilibrium   — stoichiometric NH4:SO4 partitioning
    3. Water equilibrium  — hygroscopic uptake (ISORROPIA fits)
    4. MNFIX             — mass-number consistency cleanup

This is called once per model timestep, after the coagulation step.

Condensation methods:
    - method='ppm_jit': Fully JIT-compiled PPM pipeline (default, fastest)
    - method='tfl_jit': Fully JIT-compiled TFL (Fortran-matching)
    - method='tfl': Sequential (non-JIT), Fortran-faithful TFL algorithm
    - method='ppm': Sequential wrapper, PPM advection (JIT internally)

Scan-fused time loops (single XLA program, zero Python dispatch):

    Function                          | Coag      | Cond | Nucl
    ----------------------------------|-----------|------|-----
    run_condensation_scan             | —         | PPM  | —
    run_condensation_scan_tfl         | —         | TFL  | —
    run_combined_scan_ppm             | Euler(3)  | PPM  | —
    run_combined_scan_tfl             | Euler(3)  | TFL  | —
    run_nucleation_condensation_scan  | —         | both | Yes
    run_full_scan                     | Euler(10) | TFL  | Yes

Composable process API:

    step_fn = make_step(['nucleation', 'coagulation', 'condensation'],
                        cond_method='ppm_jit')
    Nk, Mk, Gc = step_fn(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                          org_conc=..., nh3_conc=..., fion=...)

Usage::

    from tomas_jax.solvers.condensation import condensation_step
    Nk_new, Mk_new, Gc_new = condensation_step(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
    )

    # JIT-compiled version with scan-fused time loop:
    from tomas_jax.solvers.condensation import run_condensation_scan_tfl
    Nk_f, Mk_f, Gc_f, history = run_condensation_scan_tfl(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
        dt=60.0, nsteps=1440, prod_rate=h2so4_prod_kg_s
    )
    # history shape (nsteps, 3): columns [N_tot, M_dry, Gc_SO4]
"""
import warnings
import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple
from functools import partial

from ..core.config import (
    SRTSO4, SRTSO2, SRTNH4, SRTH2O, ICOMP_NODIAG,
    MW_H2SO4, SV_H2SO4, CS_EPS,
)
from ..physics.so2_chemistry import (
    so2_oxidation_step, calc_k1_so2_oh,
    calc_solar_zenith_angle, calc_oh_concentration,
)
from ..physics.dilution import dilution_step
from ..physics.nucleation import (
    nucleation_step, ricco_dunne_nucleation_rate, compute_nucleation_substeps,
    zhao2024_nucleation_step, ZHAO2024_ALL_ENABLED,
)
from ..physics.condensation_sink import calc_condensation_sink
from ..physics.ezcond import ezcond
from ..physics.ezcond_ppm_jax import ezcond_ppm_jax
from ..physics.condensation_tfl_jax import ezcond_tfl_jax
from ..physics.nh3_equilibrium import eznh3eqm
from ..physics.water_equilibrium import (calc_equilibrium_water,
                                          calc_equilibrium_water_h2so4)
from ..core.mnfix_jax import mnfix_jax
from ..solvers.diffrax import coag_euler_step

# Valid kwargs for make_step() step function — used for typo detection
_VALID_MAKE_STEP_KWARGS = {
    # so2_chemistry
    'oh_conc',
    # nucleation (common)
    'org_conc', 'nh3_conc', 'fion', 'fn_scale',
    # nucleation (ricco_dunne)
    'enable_organic', 'enable_inorganic',
    # nucleation (zhao2024)
    'hno3', 'ulvoc', 'dma', 'hio3', 'enable_masks',
    # coagulation
    'icomp_nodiag', 'coag_kernel_scale',
    # dilution
    'kdil', 'Nk_bg', 'Mk_bg', 'Gc_bg',
}

# Canonical physical process ordering
_CANONICAL_PROCESS_ORDER = [
    'so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution',
]


# =========================================================================
# Layer 0: Non-JIT dispatcher (sequential numpy paths)
# =========================================================================

# =========================================================================
# Layer 0: Non-JIT dispatcher (sequential numpy paths)
# =========================================================================

def condensation_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    Gc: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    rh: float,
    alpha: float,
    dt: float,
    method: str = 'ppm_jit'
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Execute one condensation operator-split step.

    Steps:
        1. H2SO4 condensation via ezcond (TFL) or ezcond_ppm (PPM)
        2. NH3 equilibrium
        3. Water equilibrium
        4. MNFIX cleanup

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        Gc: Gas-phase concentrations [kg/grid cell], shape (N_GAS_SPECIES,)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        rh: Relative humidity [fraction 0-1]
        alpha: Accommodation coefficient
        dt: Timestep [s]
        method: Condensation method - 'ppm_jit' (default), 'tfl', 'ppm', or 'tfl_jit'

    Returns:
        Nk_new: Updated number concentration
        Mk_new: Updated mass concentration
        Gc_new: Updated gas concentrations
    """
    # Dispatch to pure-JAX path if requested
    if method == 'ppm_jit':
        return condensation_step_jax(
            Nk, Mk, Gc, xk,
            jnp.asarray(temp), jnp.asarray(pres),
            jnp.asarray(boxvol), jnp.asarray(rh),
            jnp.asarray(alpha), jnp.asarray(dt)
        )

    if method == 'tfl_jit':
        return condensation_step_tfl_jax(
            Nk, Mk, Gc, xk,
            jnp.asarray(temp), jnp.asarray(pres),
            jnp.asarray(boxvol), jnp.asarray(rh),
            jnp.asarray(alpha), jnp.asarray(dt)
        )

    # Legacy numpy paths — not JIT-compilable, not GPU-compatible
    warnings.warn(
        f"method='{method}' uses sequential numpy and is NOT JIT-compilable. "
        "Use 'ppm_jit' or 'tfl_jit' for GPU deployment.",
        DeprecationWarning, stacklevel=2,
    )

    # Convert to numpy for sequential operations
    Nk_np = np.array(Nk)
    Mk_np = np.array(Mk)
    Gc_np = np.array(Gc)
    xk_np = np.array(xk)

    # =========================================================
    # 1. H2SO4 Condensation
    # =========================================================
    # Calculate condensation sink for H2SO4
    CS, sinkfrac = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        MW_H2SO4, SV_H2SO4, alpha, xk=xk
    )
    CS_val = float(CS)

    if CS_val > CS_EPS and Gc_np[SRTSO4] > 0.0:
        # Exponential gas depletion: mcond = Gc * (1 - exp(-CS*dt))
        mcond_so4 = Gc_np[SRTSO4] * (1.0 - np.exp(-CS_val * dt))

        # Deplete gas phase
        Gc_np[SRTSO4] -= mcond_so4

        # Condense onto particles
        if method == 'ppm':
            from ..physics.ezcond_ppm import ezcond_ppm
            Nk_np, Mk_np = ezcond_ppm(
                Nk_np, Mk_np, mcond_so4, SRTSO4,
                xk_np, temp, pres, boxvol, alpha
            )
        else:
            Nk_np, Mk_np = ezcond(
                Nk_np, Mk_np, mcond_so4, SRTSO4,
                xk_np, temp, pres, boxvol, alpha
            )
    elif Gc_np[SRTSO4] > 0.0:
        # CS too small — dump all gas into first bin
        Mk_np[0, SRTSO4] += Gc_np[SRTSO4]
        Nk_np[0] += Gc_np[SRTSO4] / np.sqrt(xk_np[0] * xk_np[1])
        Gc_np[SRTSO4] = 0.0

    # =========================================================
    # 2. NH3 Equilibrium
    # =========================================================
    Gc_jax = jnp.array(Gc_np)
    Mk_jax = jnp.array(Mk_np)
    Gc_jax, Mk_jax = eznh3eqm(Gc_jax, Mk_jax)
    Gc_np = np.array(Gc_jax)
    Mk_np = np.array(Mk_jax)

    # =========================================================
    # 3. Water Equilibrium
    # =========================================================
    Mk_jax = jnp.array(Mk_np)
    Mk_jax = calc_equilibrium_water(Mk_jax, rh)
    Mk_np = np.array(Mk_jax)

    # =========================================================
    # 4. MNFIX
    # =========================================================
    Nk_jax, Mk_jax = mnfix_jax(
        jnp.array(Nk_np), jnp.array(Mk_np),
        jnp.array(xk_np), ICOMP_NODIAG
    )

    return Nk_jax, Mk_jax, jnp.array(Gc_np)


# =========================================================================
# Layer 1: _condensation_step_core — unified JIT condensation
# =========================================================================

def _condensation_step_core(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                            ezcond_fn, water_fn=None):
    """Core condensation: MNFIX -> CS -> gas depletion -> ezcond_fn -> NH3 -> water -> MNFIX.

    This is the single implementation that both PPM and TFL JIT paths share.
    The only difference is which ezcond function is passed in.

    water_fn selects the equilibrium-water scheme. None (default) uses the
    ISORROPIA ammonium-bisulfate fit `calc_equilibrium_water(Mk, rh)`. Otherwise
    it is called as `water_fn(Mk, rh, temp)` (e.g. the Tabazadeh-1997 pure
    H2SO4/H2O scheme for clean, NH3-free stratospheric aerosol).
    """
    # 0. MNFIX input
    Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)

    # 1. Condensation sink
    CS, sinkfrac = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        MW_H2SO4, SV_H2SO4, alpha, xk=xk
    )

    # 2. H2SO4 condensation
    gc_so4 = Gc[SRTSO4]
    mcond_so4 = gc_so4 * (1.0 - jnp.exp(-CS * dt))
    mcond_so4 = jnp.where((CS > CS_EPS) & (gc_so4 > 0.0), mcond_so4, 0.0)

    Gc_new = Gc.at[SRTSO4].add(-mcond_so4)

    # Dump path (CS too small but gas available)
    dump_N = gc_so4 / jnp.sqrt(xk[0] * xk[1])
    Nk_dump = Nk.at[0].add(dump_N)
    Mk_dump = Mk.at[0, SRTSO4].add(gc_so4)
    Gc_dump = Gc.at[SRTSO4].set(0.0)

    should_dump = (CS <= CS_EPS) & (gc_so4 > 0.0)

    # Condense via ezcond_fn (PPM or TFL)
    Nk_cond, Mk_cond = ezcond_fn(
        Nk, Mk, mcond_so4, SRTSO4,
        xk, temp, pres, boxvol, alpha
    )

    # Select: dump path vs condensation path vs no-op
    has_gas = gc_so4 > 0.0
    Nk = jnp.where(should_dump, Nk_dump,
                    jnp.where(has_gas & (CS > CS_EPS), Nk_cond, Nk))
    Mk = jnp.where(should_dump, Mk_dump,
                    jnp.where(has_gas & (CS > CS_EPS), Mk_cond, Mk))
    Gc = jnp.where(should_dump, Gc_dump, Gc_new)

    # 3. NH3 equilibrium
    Gc, Mk = eznh3eqm(Gc, Mk)

    # 4. Water equilibrium (ISORROPIA NH4HSO4 by default; water_fn overrides)
    if water_fn is None:
        Mk = calc_equilibrium_water(Mk, rh)
    else:
        Mk = water_fn(Mk, rh, temp)

    # 5. MNFIX cleanup
    Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)

    return Nk, Mk, Gc


# =========================================================================
# Layer 1 wrappers: backward-compatible public API
# =========================================================================

def condensation_step_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    Gc: jnp.ndarray,
    xk: jnp.ndarray,
    temp: jnp.ndarray,
    pres: jnp.ndarray,
    boxvol: jnp.ndarray,
    rh: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure-JAX condensation step using PPM — fully JIT-compilable."""
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_ppm_jax,
    )


condensation_step_jit = jax.jit(condensation_step_jax)


def condensation_step_tfl_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    Gc: jnp.ndarray,
    xk: jnp.ndarray,
    temp: jnp.ndarray,
    pres: jnp.ndarray,
    boxvol: jnp.ndarray,
    rh: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure-JAX condensation step using TFL — Fortran-matching, JIT-compilable."""
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_tfl_jax,
    )


condensation_step_tfl_jit = jax.jit(condensation_step_tfl_jax)


# =========================================================================
# Layer 2: _combined_step_core / _full_step_core — multi-process steps
# =========================================================================

def _combined_step_core(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                        ezcond_fn, icomp_nodiag=ICOMP_NODIAG, n_coag_substeps=3):
    """Coagulation + condensation in one step, parameterized by ezcond_fn."""
    # 1. Coagulation (forward Euler + MNFIX)
    Nk, Mk = coag_euler_step(
        Nk, Mk, xk, temp, pres, boxvol,
        dt=dt, icomp_nodiag=icomp_nodiag,
        n_substeps=n_coag_substeps,
    )

    # 2. Condensation
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_fn,
    )


def _full_step_core(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                    ezcond_fn, org_conc, nh3_conc, fion,
                    enable_organic=1.0, enable_inorganic=1.0, fn_scale=1.0,
                    icomp_nodiag=ICOMP_NODIAG, n_coag_substeps=10,
                    max_nucleation_frac=0.5, max_nuc_substeps=20):
    """Nucleation + coagulation + condensation in one step.

    Nucleation uses adaptive sub-stepping: when dN would exceed
    max_nucleation_frac * N_total, the nucleation timestep is subdivided
    (up to max_nuc_substeps) with MNFIX between substeps.
    """
    # 1. Adaptive nucleation sub-stepping
    fn = ricco_dunne_nucleation_rate(
        Gc, temp, pres, boxvol,
        org_conc, nh3_conc, fion,
        enable_organic, enable_inorganic, fn_scale,
    )
    N_total = jnp.sum(Nk)
    n_nuc = compute_nucleation_substeps(
        fn, boxvol, dt, N_total,
        max_nucleation_frac, max_nuc_substeps,
    )
    dt_nuc = dt / n_nuc

    def nuc_body(i, carry):
        Nk_s, Mk_s, Gc_s = carry
        Nk_s, Mk_s, Gc_s = nucleation_step(
            Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
            org_conc, nh3_conc, fion,
            enable_organic, enable_inorganic, fn_scale,
        )
        Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, icomp_nodiag)
        return (Nk_s, Mk_s, Gc_s)

    Nk, Mk, Gc = jax.lax.fori_loop(0, n_nuc, nuc_body, (Nk, Mk, Gc))

    # 2. Coagulation (forward Euler + MNFIX)
    Nk, Mk = coag_euler_step(
        Nk, Mk, xk, temp, pres, boxvol,
        dt=dt, icomp_nodiag=icomp_nodiag,
        n_substeps=n_coag_substeps,
    )

    # 3. Condensation
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_fn,
    )


# =========================================================================
# Layer 2 wrappers: backward-compatible public API
# =========================================================================

def combined_step_ppm_jax(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    icomp_nodiag=ICOMP_NODIAG, n_coag_substeps=3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Combined coagulation + PPM condensation step (JIT-compilable)."""
    return _combined_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_ppm_jax,
        icomp_nodiag=icomp_nodiag, n_coag_substeps=n_coag_substeps,
    )


def combined_step_tfl_jax(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    icomp_nodiag=ICOMP_NODIAG, n_coag_substeps=3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Combined coagulation + TFL condensation step (JIT-compilable)."""
    return _combined_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_tfl_jax,
        icomp_nodiag=icomp_nodiag, n_coag_substeps=n_coag_substeps,
    )


def condensation_step_with_nucleation_jax(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    org_conc, nh3_conc, fion,
    enable_organic=1.0, enable_inorganic=1.0, fn_scale=1.0,
    use_tfl=1.0,
    max_nucleation_frac=0.5, max_nuc_substeps=20,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Nucleation then condensation in one JIT-compilable step.

    Args:
        use_tfl: 1.0 for TFL condensation, 0.0 for PPM (Python-level dispatch)
        max_nucleation_frac: Max dN/N_total per nucleation substep (0.5 = 50%)
        max_nuc_substeps: Hard cap on nucleation substeps
    """
    # Python-level dispatch — no double compute
    ezcond_fn = ezcond_tfl_jax if float(use_tfl) > 0.5 else ezcond_ppm_jax

    # 1. Adaptive nucleation sub-stepping
    fn = ricco_dunne_nucleation_rate(
        Gc, temp, pres, boxvol,
        org_conc, nh3_conc, fion,
        enable_organic, enable_inorganic, fn_scale,
    )
    N_total = jnp.sum(Nk)
    n_nuc = compute_nucleation_substeps(
        fn, boxvol, dt, N_total,
        max_nucleation_frac, max_nuc_substeps,
    )
    dt_nuc = dt / n_nuc

    def nuc_body(i, carry):
        Nk_s, Mk_s, Gc_s = carry
        Nk_s, Mk_s, Gc_s = nucleation_step(
            Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
            org_conc, nh3_conc, fion,
            enable_organic, enable_inorganic, fn_scale,
        )
        Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG)
        return (Nk_s, Mk_s, Gc_s)

    Nk, Mk, Gc = jax.lax.fori_loop(0, n_nuc, nuc_body, (Nk, Mk, Gc))

    # 2. Condensation
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_fn,
    )


condensation_step_with_nucleation_jit = jax.jit(
    condensation_step_with_nucleation_jax,
    static_argnums=(16,),  # use_tfl must be static for Python-level dispatch
    # max_nucleation_frac (17) and max_nuc_substeps (18) are captured as tracers
)


def full_step_jax(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    org_conc, nh3_conc, fion,
    enable_organic=1.0, enable_inorganic=1.0, fn_scale=1.0,
    use_tfl=1.0, icomp_nodiag=ICOMP_NODIAG,
    max_nucleation_frac=0.5, max_nuc_substeps=20,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full step: nucleation + coagulation + condensation (JIT-compilable).

    Uses fixed-step forward Euler coagulation (no diffrax) for minimal XLA overhead.
    Nucleation uses adaptive sub-stepping when dN would exceed max_nucleation_frac * N_total.
    """
    ezcond_fn = ezcond_tfl_jax if float(use_tfl) > 0.5 else ezcond_ppm_jax
    return _full_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_fn,
        org_conc=org_conc, nh3_conc=nh3_conc, fion=fion,
        enable_organic=enable_organic, enable_inorganic=enable_inorganic,
        fn_scale=fn_scale, icomp_nodiag=icomp_nodiag, n_coag_substeps=10,
        max_nucleation_frac=max_nucleation_frac,
        max_nuc_substeps=max_nuc_substeps,
    )


# =========================================================================
# Layer 3: _run_scan — single scan implementation
# =========================================================================

def _run_scan(Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, diag_mode='rich',
              so2_prod_rate=0.0):
    """Internal scan: adds H2SO4/SO2 prod, calls step_fn, collects diagnostics.

    Args:
        step_fn: Callable(Nk, Mk, Gc) -> (Nk, Mk, Gc)
        nsteps: Number of scan steps
        dt: Timestep [s]
        prod_rate: H2SO4 production rate [kg/s]
        diag_mode: 'rich' returns (N_tot, M_dry, Gc_SO4) per step,
                   'light' returns N_tot only
        so2_prod_rate: SO2 emission rate [kg/s] (added to Gc[SRTSO2] each step)

    Returns:
        (Nk_f, Mk_f, Gc_f), diagnostics
    """
    if diag_mode == 'rich':
        def body(carry, _):
            Nk_c, Mk_c, Gc_c = carry
            Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
            Gc_c = Gc_c.at[SRTSO2].add(so2_prod_rate * dt)
            Nk_c, Mk_c, Gc_c = step_fn(Nk_c, Mk_c, Gc_c)
            diag = jnp.array([jnp.sum(Nk_c),
                               jnp.sum(Mk_c[:, :SRTH2O]),
                               Gc_c[SRTSO4]])
            return (Nk_c, Mk_c, Gc_c), diag
    else:
        def body(carry, _):
            Nk_c, Mk_c, Gc_c = carry
            Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
            Gc_c = Gc_c.at[SRTSO2].add(so2_prod_rate * dt)
            Nk_c, Mk_c, Gc_c = step_fn(Nk_c, Mk_c, Gc_c)
            return (Nk_c, Mk_c, Gc_c), jnp.sum(Nk_c)

    (Nk_f, Mk_f, Gc_f), history = jax.lax.scan(
        body, (Nk, Mk, Gc), None, length=nsteps
    )
    return (Nk_f, Mk_f, Gc_f), history


# =========================================================================
# Layer 3 wrappers: backward-compatible scan-fused loops
# =========================================================================

@partial(jax.jit, static_argnums=(10,))
def run_condensation_scan(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run nsteps PPM condensation steps fused into a single XLA program.

    Returns:
        Nk_f, Mk_f, Gc_f: Final state arrays
        history: shape (nsteps, 3) — columns [N_tot, M_dry, Gc_SO4]
    """
    def step_fn(Nk_c, Mk_c, Gc_c):
        return _condensation_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_ppm_jax,
        )
    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'rich'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10,))
def run_condensation_scan_tfl(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run nsteps TFL condensation steps fused into a single XLA program.

    Returns:
        Nk_f, Mk_f, Gc_f: Final state arrays
        history: shape (nsteps, 3) — columns [N_tot, M_dry, Gc_SO4]
    """
    def step_fn(Nk_c, Mk_c, Gc_c):
        return _condensation_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_tfl_jax,
        )
    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'rich'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10,))
def run_combined_scan_ppm(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused Euler coagulation + PPM condensation loop."""
    def step_fn(Nk_c, Mk_c, Gc_c):
        return _combined_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_ppm_jax,
        )
    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'light'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10,))
def run_combined_scan_tfl(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused Euler coagulation + TFL condensation loop."""
    def step_fn(Nk_c, Mk_c, Gc_c):
        return _combined_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_tfl_jax,
        )
    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'light'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10, 18))
def run_nucleation_condensation_scan(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
    org_conc, nh3_conc, fion,
    enable_organic, enable_inorganic, fn_scale,
    use_tfl,
    max_nucleation_frac=0.5, max_nuc_substeps=20,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused nucleation + condensation loop with adaptive sub-stepping."""
    ezcond_fn = ezcond_tfl_jax if float(use_tfl) > 0.5 else ezcond_ppm_jax

    def step_fn(Nk_c, Mk_c, Gc_c):
        # Adaptive nucleation sub-stepping
        fn = ricco_dunne_nucleation_rate(
            Gc_c, temp, pres, boxvol,
            org_conc, nh3_conc, fion,
            enable_organic, enable_inorganic, fn_scale,
        )
        N_total = jnp.sum(Nk_c)
        n_nuc = compute_nucleation_substeps(
            fn, boxvol, dt, N_total,
            max_nucleation_frac, max_nuc_substeps,
        )
        dt_nuc = dt / n_nuc

        def nuc_body(i, carry):
            Nk_s, Mk_s, Gc_s = carry
            Nk_s, Mk_s, Gc_s = nucleation_step(
                Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
                org_conc, nh3_conc, fion,
                enable_organic, enable_inorganic, fn_scale,
            )
            Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG)
            return (Nk_s, Mk_s, Gc_s)

        Nk_c, Mk_c, Gc_c = jax.lax.fori_loop(
            0, n_nuc, nuc_body, (Nk_c, Mk_c, Gc_c))

        return _condensation_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_fn,
        )

    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'light'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10, 18))
def run_full_scan(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
    org_conc, nh3_conc, fion,
    enable_organic, enable_inorganic, fn_scale,
    use_tfl,
    max_nucleation_frac=0.5, max_nuc_substeps=20,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused nucleation + coagulation + condensation loop with adaptive sub-stepping."""
    ezcond_fn = ezcond_tfl_jax if float(use_tfl) > 0.5 else ezcond_ppm_jax

    def step_fn(Nk_c, Mk_c, Gc_c):
        return _full_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_fn,
            org_conc=org_conc, nh3_conc=nh3_conc, fion=fion,
            enable_organic=enable_organic, enable_inorganic=enable_inorganic,
            fn_scale=fn_scale,
            max_nucleation_frac=max_nucleation_frac,
            max_nuc_substeps=max_nuc_substeps,
        )

    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'light'
    )
    return Nk_f, Mk_f, Gc_f, history


# =========================================================================
# Layer 4: make_step() — public composable API
# =========================================================================

def make_step(processes, cond_method='ppm_jit', nucl_scheme='ricco_dunne',
              n_coag_substeps=10,
              max_nucleation_frac=0.5, max_nuc_substeps=20,
              water_scheme='isorropia', jit=True):
    """Build a step function from an ordered list of process names.

    The returned function has signature:
        step_fn(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, **kwargs)
    and returns (Nk, Mk, Gc).

    The ``for process in processes`` loop is Python-level — unrolled at JAX
    trace time. This makes it trivial to reorder, skip, or add processes.

    Nucleation uses adaptive sub-stepping: when dN would exceed
    max_nucleation_frac * N_total, the nucleation timestep is subdivided.

    Args:
        processes: Ordered list of process names, e.g.
            ['so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution']
        cond_method: 'ppm_jit' or 'tfl_jit'
        nucl_scheme: 'ricco_dunne' (Riccobono 2014 + Dunne 2016) or
            'zhao2024' (11-mechanism Zhao et al. 2024). For zhao2024,
            pass extra kwargs: hno3, ulvoc, dma, hio3, enable_masks.
        n_coag_substeps: Number of forward-Euler substeps for coagulation
        max_nucleation_frac: Max dN/N_total per nucleation substep (0.5 = 50%)
        max_nuc_substeps: Hard cap on nucleation substeps
        water_scheme: Equilibrium-water uptake scheme. 'isorropia' (default)
            uses the ammonium-bisulfate ISORROPIA fit; 'h2so4_tabazadeh' uses
            the Tabazadeh-1997 pure H2SO4/H2O binary scheme (for clean, NH3-free
            stratospheric aerosol — T-dependent). Default preserves prior behavior.
        jit: If True (default), wrap the returned function in ``jax.jit``.
            Callers no longer need to wrap manually. Double-JIT is a no-op.

    Returns:
        A callable step function (JIT-compiled by default).

    For JIT stability, always pass the same set of kwargs across calls.

    Example::

        step = make_step(['nucleation', 'coagulation', 'condensation'],
                         cond_method='ppm_jit')
        Nk, Mk, Gc = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                           org_conc=org_conc, nh3_conc=nh3_conc, fion=fion)

        # With SO2 chemistry:
        step = make_step(['so2_chemistry', 'nucleation', 'coagulation', 'condensation'])
        Nk, Mk, Gc = step(..., oh_conc=1e6)

        # Zhao 2024 11-mechanism scheme:
        step = make_step(['nucleation', 'coagulation', 'condensation'],
                         nucl_scheme='zhao2024')
        Nk, Mk, Gc = step(..., org_conc=..., nh3_conc=..., fion=...,
                           hno3=..., ulvoc=..., dma=..., hio3=...)
    """
    ezcond_fn = ezcond_tfl_jax if 'tfl' in cond_method else ezcond_ppm_jax

    valid_water = {'isorropia', 'h2so4_tabazadeh'}
    if water_scheme not in valid_water:
        raise ValueError(f"Unknown water_scheme '{water_scheme}'. "
                         f"Valid: {sorted(valid_water)}")
    # None => ISORROPIA (default, behavior-preserving); else the Tabazadeh fn.
    water_fn = (calc_equilibrium_water_h2so4
                if water_scheme == 'h2so4_tabazadeh' else None)

    valid = {'so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'}
    for p in processes:
        if p not in valid:
            raise ValueError(f"Unknown process '{p}'. Valid: {sorted(valid)}")

    # Warn if process order deviates from canonical physical ordering
    canonical_indices = [_CANONICAL_PROCESS_ORDER.index(p) for p in processes]
    if canonical_indices != sorted(canonical_indices):
        canonical_subset = [p for p in _CANONICAL_PROCESS_ORDER if p in processes]
        warnings.warn(
            f"Non-standard process order: {list(processes)}. "
            f"Canonical physical order is: {canonical_subset}. "
            f"Non-standard ordering may produce physically incorrect results.",
            UserWarning, stacklevel=2,
        )

    valid_schemes = {'ricco_dunne', 'zhao2024'}
    if nucl_scheme not in valid_schemes:
        raise ValueError(f"Unknown nucl_scheme '{nucl_scheme}'. Valid: {sorted(valid_schemes)}")

    use_zhao = nucl_scheme == 'zhao2024'

    def step_fn(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, **kwargs):
        unknown = set(kwargs) - _VALID_MAKE_STEP_KWARGS
        if unknown:
            warnings.warn(
                f"Unknown kwargs passed to make_step step function: {unknown}. "
                f"These will be silently ignored. "
                f"Valid kwargs: {sorted(_VALID_MAKE_STEP_KWARGS)}",
                UserWarning, stacklevel=2,
            )
        for process in processes:
            if process == 'so2_chemistry':
                oh_conc = kwargs.get('oh_conc', 0.0)
                Gc = so2_oxidation_step(Gc, temp, pres, boxvol, dt, oh_conc, rh)

            elif process == 'nucleation':
                # Adaptive nucleation sub-stepping
                # Rate estimate always uses Ricco+Dunne (fast, conservative)
                fn = ricco_dunne_nucleation_rate(
                    Gc, temp, pres, boxvol,
                    kwargs.get('org_conc', 0.0), kwargs.get('nh3_conc', 0.0),
                    kwargs.get('fion', 0.0),
                    kwargs.get('enable_organic', 1.0),
                    kwargs.get('enable_inorganic', 1.0),
                    kwargs.get('fn_scale', 1.0),
                )
                N_total = jnp.sum(Nk)
                n_nuc = compute_nucleation_substeps(
                    fn, boxvol, dt, N_total,
                    max_nucleation_frac, max_nuc_substeps,
                )
                dt_nuc = dt / n_nuc

                if use_zhao:
                    enable_masks = kwargs.get('enable_masks', ZHAO2024_ALL_ENABLED)

                    def nuc_body(i, carry):
                        Nk_s, Mk_s, Gc_s = carry
                        Nk_s, Mk_s, Gc_s = zhao2024_nucleation_step(
                            Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
                            kwargs.get('org_conc', 0.0),
                            kwargs.get('nh3_conc', 0.0),
                            kwargs.get('fion', 0.0),
                            hno3=kwargs.get('hno3', 0.0),
                            ulvoc=kwargs.get('ulvoc', 0.0),
                            dma=kwargs.get('dma', 0.0),
                            hio3=kwargs.get('hio3', 0.0),
                            enable_masks=enable_masks,
                            fn_scale=kwargs.get('fn_scale', 1.0),
                        )
                        Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG)
                        return (Nk_s, Mk_s, Gc_s)
                else:
                    def nuc_body(i, carry):
                        Nk_s, Mk_s, Gc_s = carry
                        Nk_s, Mk_s, Gc_s = nucleation_step(
                            Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
                            kwargs.get('org_conc', 0.0),
                            kwargs.get('nh3_conc', 0.0),
                            kwargs.get('fion', 0.0),
                            kwargs.get('enable_organic', 1.0),
                            kwargs.get('enable_inorganic', 1.0),
                            kwargs.get('fn_scale', 1.0),
                        )
                        Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG)
                        return (Nk_s, Mk_s, Gc_s)

                Nk, Mk, Gc = jax.lax.fori_loop(
                    0, n_nuc, nuc_body, (Nk, Mk, Gc))

            elif process == 'coagulation':
                Nk, Mk = coag_euler_step(
                    Nk, Mk, xk, temp, pres, boxvol,
                    dt=dt,
                    icomp_nodiag=kwargs.get('icomp_nodiag', ICOMP_NODIAG),
                    n_substeps=n_coag_substeps,
                    coag_kernel_scale=kwargs.get('coag_kernel_scale', 1.0),
                )
            elif process == 'condensation':
                Nk, Mk, Gc = _condensation_step_core(
                    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                    ezcond_fn=ezcond_fn, water_fn=water_fn,
                )
            elif process == 'dilution':
                kdil = kwargs.get('kdil', 0.0)
                Nk, Mk, Gc = dilution_step(
                    Nk, Mk, Gc, dt, kdil,
                    kwargs.get('Nk_bg', jnp.zeros_like(Nk)),
                    kwargs.get('Mk_bg', jnp.zeros_like(Mk)),
                    kwargs.get('Gc_bg', jnp.zeros_like(Gc)),
                )
        return Nk, Mk, Gc

    return jax.jit(step_fn) if jit else step_fn
