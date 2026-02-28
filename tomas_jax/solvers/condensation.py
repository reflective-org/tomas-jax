"""Top-level condensation operator-split step for TOMAS-JAX.

Orchestrates the full condensation pipeline in sequence:

    1. H2SO4 condensation — exponential gas depletion + ezcond
    2. NH3 equilibrium   — stoichiometric NH4:SO4 partitioning
    3. Water equilibrium  — hygroscopic uptake (ISORROPIA fits)
    4. MNFIX             — mass-number consistency cleanup

This is called once per model timestep, after the coagulation step.

Three implementations:
    - method='tfl': Sequential (non-JIT), Fortran-faithful TFL algorithm
    - method='ppm': Sequential wrapper, PPM advection (JIT internally)
    - method='ppm_jit': Fully JIT-compiled PPM pipeline (fastest)

Usage::

    from tomas_jax.solvers.condensation import condensation_step
    Nk_new, Mk_new, Gc_new = condensation_step(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
    )

    # JIT-compiled version with scan-fused time loop:
    from tomas_jax.solvers.condensation import run_condensation_scan
    Nk_f, Mk_f, Gc_f, N_hist = run_condensation_scan(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
        dt=60.0, nsteps=1440, prod_rate=h2so4_prod_kg_s
    )
"""
import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple
from functools import partial

from ..core.config import (
    SRTSO4, SRTNH4, ICOMP_NODIAG,
    MW_H2SO4, SV_H2SO4, CS_EPS
)
from ..physics.condensation_sink import calc_condensation_sink
from ..physics.ezcond import ezcond
from ..physics.ezcond_ppm_jax import ezcond_ppm_jax
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
    method: str = 'tfl'
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
        method: Condensation method - 'tfl' (default), 'ppm', or 'ppm_jit'

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
        MW_H2SO4, SV_H2SO4, alpha
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
        MW_H2SO4, SV_H2SO4, alpha
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


# Pre-compiled JIT version
condensation_step_jit = jax.jit(condensation_step_jax)


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
    """Run nsteps condensation steps fused into a single XLA program.

    Eliminates Python dispatch overhead by compiling the entire time loop.

    Args:
        Nk, Mk, Gc, xk: Initial state arrays
        temp, pres, boxvol, rh, alpha: Environment parameters
        dt: Timestep [s]
        nsteps: Number of steps (static, must be known at compile time)
        prod_rate: H2SO4 production rate [kg/grid cell/s]

    Returns:
        Nk_final, Mk_final, Gc_final: Final state
        N_history: Total number at each step, shape (nsteps,)
    """
    def step_fn(carry, _):
        Nk_c, Mk_c, Gc_c = carry

        # Add H2SO4 production
        Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)

        # Condensation step
        Nk_c, Mk_c, Gc_c = condensation_step_jax(
            Nk_c, Mk_c, Gc_c, xk,
            temp, pres, boxvol, rh, alpha, dt
        )

        return (Nk_c, Mk_c, Gc_c), jnp.sum(Nk_c)

    (Nk_f, Mk_f, Gc_f), N_history = jax.lax.scan(
        step_fn, (Nk, Mk, Gc), None, length=nsteps
    )

    return Nk_f, Mk_f, Gc_f, N_history
