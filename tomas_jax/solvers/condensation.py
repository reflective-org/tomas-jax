"""Top-level condensation operator-split step for TOMAS-JAX.

Orchestrates the full condensation pipeline in sequence:

    1. H2SO4 condensation — exponential gas depletion + ezcond
    2. NH3 equilibrium   — stoichiometric NH4:SO4 partitioning
    3. Water equilibrium  — hygroscopic uptake (ISORROPIA fits)
    4. MNFIX             — mass-number consistency cleanup

This is called once per model timestep, after the coagulation step.

Phase 1: Sequential (non-JIT) version. Internally converts between
JAX and numpy arrays at the condensation boundary. The equilibrium
routines (NH3, water) use JAX arrays directly.

Usage::

    from tomas_jax.solvers.condensation import condensation_step
    Nk_new, Mk_new, Gc_new = condensation_step(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
    )
"""
import numpy as np
import jax.numpy as jnp
from typing import Tuple

from ..core.config import (
    SRTSO4, SRTNH4, ICOMP_NODIAG,
    MW_H2SO4, SV_H2SO4, CS_EPS
)
from ..physics.condensation_sink import calc_condensation_sink
from ..physics.ezcond import ezcond
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
        method: Condensation method - 'tfl' (default) or 'ppm'

    Returns:
        Nk_new: Updated number concentration
        Mk_new: Updated mass concentration
        Gc_new: Updated gas concentrations
    """
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
