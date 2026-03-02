"""Pure-JAX coagulation solver — forward Euler, JIT-compilable.

Replaces the diffrax Tsit5 adaptive solver with simple forward Euler
matching Fortran's integration scheme. All components are already pure JAX:
  - calc_particle_properties (vectorized)
  - calc_coagulation_kernel (vectorized, O(N^2) broadcasting)
  - calc_coagulation_rates (vectorized TFL algorithm)
  - mnfix_jax (vectorized)

The outer time loop can be fused with jax.lax.scan.

Usage::

    from tomas_jax.solvers.coagulation_jax import coagulation_step_jax
    Nk_new, Mk_new = coagulation_step_jax(Nk, Mk, xk, temp, pres, boxvol, dt)

    # Pre-compiled:
    from tomas_jax.solvers.coagulation_jax import coagulation_step_jit
    Nk_new, Mk_new = coagulation_step_jit(Nk, Mk, xk, temp, pres, boxvol, dt)
"""
import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial

from ..core.config import ICOMP_NODIAG
from ..physics.properties import calc_particle_properties
from ..physics.coagulation_kernel import calc_coagulation_kernel
from ..physics.coagulation_rates import calc_coagulation_rates
from ..core.mnfix_jax import mnfix_jax


def coagulation_step_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    temp: jnp.ndarray,
    pres: jnp.ndarray,
    boxvol: jnp.ndarray,
    dt: jnp.ndarray,
    icomp_nodiag: int = ICOMP_NODIAG,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single coagulation step — forward Euler, matching Fortran.

    Pipeline:
        1. Compute particle properties (Dpk, Dk, ck)
        2. Compute coagulation kernel kij (36x36)
        3. Compute rates dNdt, dMdt (vectorized TFL)
        4. Forward Euler update: Nk += dNdt * dt, Mk += dMdt * dt
        5. Clamp negatives to zero
        6. MNFIX cleanup

    Args:
        Nk: Number [#/grid cell], shape (ibins,)
        Mk: Mass [kg/grid cell], shape (ibins, icomp)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp, pres, boxvol: Environment (scalar JAX arrays)
        dt: Timestep [s] (scalar JAX array)
        icomp_nodiag: Number of non-diagnostic species (static)

    Returns:
        Nk_new, Mk_new
    """
    # 1. Particle properties
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres)

    # 2. Coagulation kernel
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    # 3. Rates
    dNdt, dMdt = calc_coagulation_rates(Nk, Mk, kij, xk, icomp_nodiag)

    # 4. Forward Euler
    Nk_new = Nk + dNdt * dt
    Mk_new = Mk + dMdt * dt

    # 5. Clamp negatives
    Nk_new = jnp.maximum(Nk_new, 0.0)
    Mk_new = jnp.maximum(Mk_new, 0.0)

    # 6. MNFIX
    Nk_new, Mk_new = mnfix_jax(Nk_new, Mk_new, xk, icomp_nodiag)

    return Nk_new, Mk_new


# Pre-compiled JIT version
coagulation_step_jit = jax.jit(
    coagulation_step_jax,
    static_argnums=(7,),  # icomp_nodiag
)
