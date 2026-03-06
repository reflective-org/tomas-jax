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
import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple
from functools import partial

from ..core.config import (
    SRTSO4, SRTNH4, SRTH2O, ICOMP_NODIAG,
    MW_H2SO4, SV_H2SO4, CS_EPS,
)
from ..physics.nucleation import nucleation_step
from ..physics.condensation_sink import calc_condensation_sink
from ..physics.ezcond import ezcond
from ..physics.ezcond_ppm_jax import ezcond_ppm_jax
from ..physics.condensation_tfl_jax import ezcond_tfl_jax
from ..physics.nh3_equilibrium import eznh3eqm
from ..physics.water_equilibrium import calc_equilibrium_water
from ..core.mnfix_jax import mnfix_jax


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
# Pure-JAX condensation step (JIT-compilable)
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
    """Pure-JAX condensation step — fully JIT-compilable.

    Same algorithm as condensation_step() with method='ppm' but
    uses ezcond_ppm_jax instead of ezcond_ppm, and keeps everything
    as JAX arrays throughout. No numpy conversions.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        Gc: Gas-phase concentrations [kg/grid cell], shape (N_GAS_SPECIES,)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K] (scalar JAX array)
        pres: Pressure [Pa] (scalar JAX array)
        boxvol: Grid cell volume [cm^3] (scalar JAX array)
        rh: Relative humidity [fraction 0-1] (scalar JAX array)
        alpha: Accommodation coefficient (scalar JAX array)
        dt: Timestep [s] (scalar JAX array)

    Returns:
        Nk_new, Mk_new, Gc_new
    """
    # 0. MNFIX input (matching ezcond_ppm.py line 80)
    Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)

    # 1. Condensation sink
    CS, sinkfrac = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        MW_H2SO4, SV_H2SO4, alpha, xk=xk
    )

    # 2. H2SO4 condensation
    gc_so4 = Gc[SRTSO4]
    mcond_so4 = gc_so4 * (1.0 - jnp.exp(-CS * dt))
    # Only condense if CS is significant and gas is available
    mcond_so4 = jnp.where((CS > CS_EPS) & (gc_so4 > 0.0), mcond_so4, 0.0)

    # Deplete gas
    Gc_new = Gc.at[SRTSO4].add(-mcond_so4)

    # Dump-to-bin-0 path when CS too small but gas available
    dump_N = gc_so4 / jnp.sqrt(xk[0] * xk[1])
    Nk_dump = Nk.at[0].add(dump_N)
    Mk_dump = Mk.at[0, SRTSO4].add(gc_so4)
    Gc_dump = Gc.at[SRTSO4].set(0.0)

    should_dump = (CS <= CS_EPS) & (gc_so4 > 0.0)

    # Condense via PPM
    Nk_cond, Mk_cond = ezcond_ppm_jax(
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

    # 4. Water equilibrium
    Mk = calc_equilibrium_water(Mk, rh)

    # 5. MNFIX cleanup
    Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)

    return Nk, Mk, Gc


# Pre-compiled JIT versions
condensation_step_jit = jax.jit(condensation_step_jax)


# =========================================================================
# Pure-JAX TFL condensation step (JIT-compilable, Fortran-matching)
# =========================================================================

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
    """Pure-JAX condensation step using TFL algorithm — Fortran-matching.

    Same pipeline as condensation_step_jax but uses ezcond_tfl_jax
    (TFL tmcond) instead of ezcond_ppm_jax (PPM advection).

    Args/Returns: Same as condensation_step_jax.
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

    # Dump path
    dump_N = gc_so4 / jnp.sqrt(xk[0] * xk[1])
    Nk_dump = Nk.at[0].add(dump_N)
    Mk_dump = Mk.at[0, SRTSO4].add(gc_so4)
    Gc_dump = Gc.at[SRTSO4].set(0.0)

    should_dump = (CS <= CS_EPS) & (gc_so4 > 0.0)

    # Condense via TFL
    Nk_cond, Mk_cond = ezcond_tfl_jax(
        Nk, Mk, mcond_so4, SRTSO4,
        xk, temp, pres, boxvol, alpha
    )

    # Select path
    has_gas = gc_so4 > 0.0
    Nk = jnp.where(should_dump, Nk_dump,
                    jnp.where(has_gas & (CS > CS_EPS), Nk_cond, Nk))
    Mk = jnp.where(should_dump, Mk_dump,
                    jnp.where(has_gas & (CS > CS_EPS), Mk_cond, Mk))
    Gc = jnp.where(should_dump, Gc_dump, Gc_new)

    # 3. NH3 equilibrium
    Gc, Mk = eznh3eqm(Gc, Mk)

    # 4. Water equilibrium
    Mk = calc_equilibrium_water(Mk, rh)

    # 5. MNFIX cleanup
    Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)

    return Nk, Mk, Gc


condensation_step_tfl_jit = jax.jit(condensation_step_tfl_jax)


# =========================================================================
# Scan-fused time loop (eliminates 1440 Python dispatch calls)
# =========================================================================

@partial(jax.jit, static_argnums=(10,))
def run_condensation_scan(
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
    nsteps: int,
    prod_rate: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run nsteps PPM condensation steps fused into a single XLA program.

    Returns:
        Nk_f, Mk_f, Gc_f: Final state arrays
        history: shape (nsteps, 3) — columns [N_tot, M_dry, Gc_SO4]
    """
    def step_fn(carry, _):
        Nk_c, Mk_c, Gc_c = carry
        Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
        Nk_c, Mk_c, Gc_c = condensation_step_jax(
            Nk_c, Mk_c, Gc_c, xk,
            temp, pres, boxvol, rh, alpha, dt
        )
        diag = jnp.array([jnp.sum(Nk_c),
                           jnp.sum(Mk_c[:, :SRTH2O]),
                           Gc_c[SRTSO4]])
        return (Nk_c, Mk_c, Gc_c), diag

    (Nk_f, Mk_f, Gc_f), history = jax.lax.scan(
        step_fn, (Nk, Mk, Gc), None, length=nsteps
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10,))
def run_condensation_scan_tfl(
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
    nsteps: int,
    prod_rate: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run nsteps TFL condensation steps fused into a single XLA program.

    Returns:
        Nk_f, Mk_f, Gc_f: Final state arrays
        history: shape (nsteps, 3) — columns [N_tot, M_dry, Gc_SO4]
    """
    def step_fn_tfl(carry, _):
        Nk_c, Mk_c, Gc_c = carry
        Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
        Nk_c, Mk_c, Gc_c = condensation_step_tfl_jax(
            Nk_c, Mk_c, Gc_c, xk,
            temp, pres, boxvol, rh, alpha, dt
        )
        diag = jnp.array([jnp.sum(Nk_c),
                           jnp.sum(Mk_c[:, :SRTH2O]),
                           Gc_c[SRTSO4]])
        return (Nk_c, Mk_c, Gc_c), diag

    (Nk_f, Mk_f, Gc_f), history = jax.lax.scan(
        step_fn_tfl, (Nk, Mk, Gc), None, length=nsteps
    )
    return Nk_f, Mk_f, Gc_f, history


# =========================================================================
# Nucleation + Condensation combined step (JIT-compilable)
# =========================================================================

def condensation_step_with_nucleation_jax(
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
    org_conc: jnp.ndarray,
    nh3_conc: jnp.ndarray,
    fion: jnp.ndarray,
    enable_organic: float = 1.0,
    enable_inorganic: float = 1.0,
    fn_scale: float = 1.0,
    use_tfl: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Nucleation then condensation in one JIT-compilable step.

    Args:
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt: standard state
        org_conc: Organic vapor [molec/cm3]
        nh3_conc: NH3 [molec/cm3]
        fion: Ion-pair production rate [pairs/cm3/s]
        enable_organic: 1.0/0.0 mask for Riccobono
        enable_inorganic: 1.0/0.0 mask for Dunne
        fn_scale: nucleation rate scaling factor
        use_tfl: 1.0 for TFL condensation, 0.0 for PPM

    Returns:
        (Nk_new, Mk_new, Gc_new)
    """
    # 1. Nucleation
    Nk, Mk, Gc = nucleation_step(
        Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
        org_conc, nh3_conc, fion,
        enable_organic, enable_inorganic, fn_scale,
    )

    # 2. Condensation (TFL or PPM based on use_tfl flag)
    Nk_tfl, Mk_tfl, Gc_tfl = condensation_step_tfl_jax(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
    )
    Nk_ppm, Mk_ppm, Gc_ppm = condensation_step_jax(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
    )
    Nk = jnp.where(use_tfl > 0.5, Nk_tfl, Nk_ppm)
    Mk = jnp.where(use_tfl > 0.5, Mk_tfl, Mk_ppm)
    Gc = jnp.where(use_tfl > 0.5, Gc_tfl, Gc_ppm)

    return Nk, Mk, Gc


condensation_step_with_nucleation_jit = jax.jit(
    condensation_step_with_nucleation_jax
)


@partial(jax.jit, static_argnums=(10,))
def run_nucleation_condensation_scan(
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
    nsteps: int,
    prod_rate: jnp.ndarray,
    org_conc: jnp.ndarray,
    nh3_conc: jnp.ndarray,
    fion: jnp.ndarray,
    enable_organic: jnp.ndarray,
    enable_inorganic: jnp.ndarray,
    fn_scale: jnp.ndarray,
    use_tfl: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused nucleation + condensation loop."""
    def step_fn(carry, _):
        Nk_c, Mk_c, Gc_c = carry
        Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
        Nk_c, Mk_c, Gc_c = condensation_step_with_nucleation_jax(
            Nk_c, Mk_c, Gc_c, xk,
            temp, pres, boxvol, rh, alpha, dt,
            org_conc, nh3_conc, fion,
            enable_organic, enable_inorganic, fn_scale, use_tfl,
        )
        return (Nk_c, Mk_c, Gc_c), jnp.sum(Nk_c)

    (Nk_f, Mk_f, Gc_f), N_history = jax.lax.scan(
        step_fn, (Nk, Mk, Gc), None, length=nsteps
    )
    return Nk_f, Mk_f, Gc_f, N_history


# =========================================================================
# Combined mode: Coagulation + Condensation (scan-fused, no nucleation)
# =========================================================================

def combined_step_ppm_jax(
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
    icomp_nodiag: int = 42,
    n_coag_substeps: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Combined coagulation + PPM condensation step (JIT-compilable).

    Uses forward Euler coagulation (scan-fusable) + PPM condensation.
    """
    from .diffrax import coag_euler_step

    # 1. Coagulation (forward Euler + MNFIX)
    Nk, Mk = coag_euler_step(
        Nk, Mk, xk, temp, pres, boxvol,
        dt=dt, icomp_nodiag=icomp_nodiag,
        n_substeps=n_coag_substeps,
    )

    # 2. Condensation (PPM only)
    Nk, Mk, Gc = condensation_step_jax(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
    )

    return Nk, Mk, Gc


def combined_step_tfl_jax(
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
    icomp_nodiag: int = 42,
    n_coag_substeps: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Combined coagulation + TFL condensation step (JIT-compilable).

    Uses forward Euler coagulation (scan-fusable) + TFL condensation.
    """
    from .diffrax import coag_euler_step

    # 1. Coagulation (forward Euler + MNFIX)
    Nk, Mk = coag_euler_step(
        Nk, Mk, xk, temp, pres, boxvol,
        dt=dt, icomp_nodiag=icomp_nodiag,
        n_substeps=n_coag_substeps,
    )

    # 2. Condensation (TFL only)
    Nk, Mk, Gc = condensation_step_tfl_jax(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
    )

    return Nk, Mk, Gc


@partial(jax.jit, static_argnums=(10,))
def run_combined_scan_ppm(
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
    nsteps: int,
    prod_rate: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused Euler coagulation + PPM condensation loop."""
    def step_fn(carry, _):
        Nk_c, Mk_c, Gc_c = carry
        Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
        Nk_c, Mk_c, Gc_c = combined_step_ppm_jax(
            Nk_c, Mk_c, Gc_c, xk,
            temp, pres, boxvol, rh, alpha, dt,
        )
        return (Nk_c, Mk_c, Gc_c), jnp.sum(Nk_c)

    (Nk_f, Mk_f, Gc_f), N_history = jax.lax.scan(
        step_fn, (Nk, Mk, Gc), None, length=nsteps
    )
    return Nk_f, Mk_f, Gc_f, N_history


@partial(jax.jit, static_argnums=(10,))
def run_combined_scan_tfl(
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
    nsteps: int,
    prod_rate: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused Euler coagulation + TFL condensation loop."""
    def step_fn(carry, _):
        Nk_c, Mk_c, Gc_c = carry
        Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
        Nk_c, Mk_c, Gc_c = combined_step_tfl_jax(
            Nk_c, Mk_c, Gc_c, xk,
            temp, pres, boxvol, rh, alpha, dt,
        )
        return (Nk_c, Mk_c, Gc_c), jnp.sum(Nk_c)

    (Nk_f, Mk_f, Gc_f), N_history = jax.lax.scan(
        step_fn, (Nk, Mk, Gc), None, length=nsteps
    )
    return Nk_f, Mk_f, Gc_f, N_history


# =========================================================================
# Full mode: Nucleation + Coagulation + Condensation (scan-fused)
# =========================================================================

def full_step_jax(
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
    org_conc: jnp.ndarray,
    nh3_conc: jnp.ndarray,
    fion: jnp.ndarray,
    enable_organic: float = 1.0,
    enable_inorganic: float = 1.0,
    fn_scale: float = 1.0,
    use_tfl: float = 1.0,
    icomp_nodiag: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full step: nucleation + coagulation + condensation (JIT-compilable).

    Uses fixed-step forward Euler coagulation (no diffrax) for minimal XLA overhead.
    """
    from .diffrax import coag_euler_step

    # 1. Nucleation
    Nk, Mk, Gc = nucleation_step(
        Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
        org_conc, nh3_conc, fion,
        enable_organic, enable_inorganic, fn_scale,
    )

    # 2. Coagulation (forward Euler + MNFIX, scan-fusable)
    Nk, Mk = coag_euler_step(
        Nk, Mk, xk, temp, pres, boxvol,
        dt=dt, icomp_nodiag=icomp_nodiag,
        n_substeps=10,
    )

    # 3. Condensation (TFL only — PPM path removed to halve XLA graph)
    Nk, Mk, Gc = condensation_step_tfl_jax(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
    )

    return Nk, Mk, Gc


@partial(jax.jit, static_argnums=(10,))
def run_full_scan(
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
    nsteps: int,
    prod_rate: jnp.ndarray,
    org_conc: jnp.ndarray,
    nh3_conc: jnp.ndarray,
    fion: jnp.ndarray,
    enable_organic: jnp.ndarray,
    enable_inorganic: jnp.ndarray,
    fn_scale: jnp.ndarray,
    use_tfl: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused nucleation + coagulation + condensation loop."""
    def step_fn(carry, _):
        Nk_c, Mk_c, Gc_c = carry
        Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
        Nk_c, Mk_c, Gc_c = full_step_jax(
            Nk_c, Mk_c, Gc_c, xk,
            temp, pres, boxvol, rh, alpha, dt,
            org_conc, nh3_conc, fion,
            enable_organic, enable_inorganic, fn_scale, use_tfl,
        )
        return (Nk_c, Mk_c, Gc_c), jnp.sum(Nk_c)

    (Nk_f, Mk_f, Gc_f), N_history = jax.lax.scan(
        step_fn, (Nk, Mk, Gc), None, length=nsteps
    )
    return Nk_f, Mk_f, Gc_f, N_history
