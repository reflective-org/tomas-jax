"""Forward Euler integrator — exact port of multicoag.f time-stepping.

This module provides a FORTRAN-equivalent Forward Euler solver with
adaptive sub-stepping for validation against the original FORTRAN code.

The FORTRAN multicoag.f uses:
  - Forward Euler: y(t+dt) = y(t) + dt * f(y(t))
  - Adaptive dt: dtlimit = 0.25 * Nk / |dNdt|, itlimit = 10 * Mk / |dMdt|
  - Sub-stepping: dt = min(dtlimit, itlimit, remaining_time)
  - MNFIX after each sub-step

For production use, prefer diffrax.py (Tsit5 adaptive RK) which is:
  - Higher-order accurate (5th order vs 1st order)
  - Better error control (embedded error estimator)
  - More efficient (larger stable timesteps)
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Tuple, Optional

from ..physics.properties import calc_particle_properties
from ..physics.coagulation_kernel import calc_coagulation_kernel
from ..physics.coagulation_rates import calc_coagulation_rates
from ..core.mnfix_jax import mnfix_jax
from ..core.mnfix_fortran import mnfix_fortran


def _compute_adaptive_dt(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    dNdt: jnp.ndarray,
    dMdt: jnp.ndarray,
    dt_remaining: float,
    icomp_nodiag: int = 42
) -> jnp.ndarray:
    """Compute adaptive timestep matching FORTRAN multicoag.f logic.

    FORTRAN:
        dtlimit = 0.25 * Nk(k) / max(abs(dNdt(k)), 1e-30)
        itlimit = 10.0 * Mk(k,j) / max(abs(dMdt(k,j)), 1e-30)
        dt = min(dtlimit, itlimit) over all k,j

    Args:
        Nk: Number concentrations, shape (ibins,)
        Mk: Mass concentrations, shape (ibins, icomp)
        dNdt: Number rates, shape (ibins,)
        dMdt: Mass rates, shape (ibins, icomp)
        dt_remaining: Time remaining in the outer step [s]
        icomp_nodiag: Number of non-diagnostic species

    Returns:
        dt: Adaptive sub-step size [s]
    """
    # Number-based limit: dt < 0.25 * N / |dN/dt|
    abs_dNdt = jnp.maximum(jnp.abs(dNdt), 1e-30)
    dtlimit = 0.25 * jnp.abs(Nk) / abs_dNdt
    dt_N = jnp.min(dtlimit)

    # Mass-based limit: dt < 10.0 * M / |dM/dt|
    Mk_prog = Mk[:, :icomp_nodiag]
    dMdt_prog = dMdt[:, :icomp_nodiag]
    abs_dMdt = jnp.maximum(jnp.abs(dMdt_prog), 1e-30)
    itlimit = 10.0 * jnp.abs(Mk_prog) / abs_dMdt
    dt_M = jnp.min(itlimit)

    # Take minimum of all constraints
    dt = jnp.minimum(dt_N, dt_M)
    dt = jnp.minimum(dt, dt_remaining)

    # Safety floor to prevent infinitely small steps
    dt = jnp.maximum(dt, 1e-10)

    return dt


def euler_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    dt: float,
    icomp_nodiag: int = 42,
    use_fortran_mnfix: bool = False,
    recompute_kernel: bool = False,
    max_substeps: int = 10000
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward Euler integration of coagulation over timestep dt.

    Matches the FORTRAN multicoag.f approach:
    1. Compute particle properties and coagulation kernel
    2. Compute rates (dNdt, dMdt)
    3. Determine adaptive sub-step dt_sub
    4. Apply Euler update: N += dt_sub * dNdt, M += dt_sub * dMdt
    5. Apply MNFIX
    6. Repeat until dt is consumed

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        dt: Total integration time [s]
        icomp_nodiag: Number of non-diagnostic species
        use_fortran_mnfix: If True, use FORTRAN-equivalent mnfix
        recompute_kernel: If True, recompute kernel each sub-step (slower but more accurate)
        max_substeps: Maximum number of sub-steps

    Returns:
        Nk_final: Updated number concentration
        Mk_final: Updated mass concentration
    """
    # Select MNFIX function
    mnfix_fn = mnfix_fortran if use_fortran_mnfix else mnfix_jax

    # Initial kernel computation
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    # State for while_loop: (Nk, Mk, kij, time_elapsed, substep_count)
    def cond_fn(state):
        _, _, _, t_elapsed, n_steps = state
        return (t_elapsed < dt - 1e-15) & (n_steps < max_substeps)

    def body_fn(state):
        Nk_s, Mk_s, kij_s, t_elapsed, n_steps = state

        # Optionally recompute kernel (FORTRAN recomputes each sub-step)
        def recompute():
            Dpk_new, Dk_new, ck_new = calc_particle_properties(Nk_s, Mk_s, temp, pres)
            return calc_coagulation_kernel(Dpk_new, Dk_new, ck_new, boxvol)

        kij_use = jax.lax.cond(
            recompute_kernel & (n_steps > 0),
            recompute,
            lambda: kij_s
        )

        # Compute rates
        dNdt, dMdt = calc_coagulation_rates(Nk_s, Mk_s, kij_use, xk, icomp_nodiag)

        # Adaptive timestep
        dt_remaining = dt - t_elapsed
        dt_sub = _compute_adaptive_dt(
            Nk_s, Mk_s, dNdt, dMdt, dt_remaining, icomp_nodiag
        )

        # Forward Euler update
        Nk_new = Nk_s + dt_sub * dNdt
        Mk_new = Mk_s + dt_sub * dMdt

        # Enforce positivity
        Nk_new = jnp.maximum(Nk_new, 0.0)
        Mk_new = jnp.maximum(Mk_new, 0.0)

        # Apply MNFIX
        Nk_new, Mk_new = mnfix_fn(Nk_new, Mk_new, xk, icomp_nodiag)

        return Nk_new, Mk_new, kij_use, t_elapsed + dt_sub, n_steps + 1

    init_state = (Nk, Mk, kij, jnp.float64(0.0), jnp.int32(0))
    Nk_final, Mk_final, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    return Nk_final, Mk_final


def euler_single_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    kij: jnp.ndarray,
    xk: jnp.ndarray,
    dt_sub: float,
    icomp_nodiag: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Forward Euler step without MNFIX (for benchmarking level 9).

    Args:
        Nk: Number concentration, shape (ibins,)
        Mk: Mass concentration, shape (ibins, icomp)
        kij: Pre-computed kernel, shape (ibins, ibins)
        xk: Bin boundaries, shape (ibins+1,)
        dt_sub: Sub-step size [s]
        icomp_nodiag: Number of non-diagnostic species

    Returns:
        Nk_new, Mk_new after one Euler step
    """
    dNdt, dMdt = calc_coagulation_rates(Nk, Mk, kij, xk, icomp_nodiag)

    Nk_new = Nk + dt_sub * dNdt
    Mk_new = Mk + dt_sub * dMdt

    Nk_new = jnp.maximum(Nk_new, 0.0)
    Mk_new = jnp.maximum(Mk_new, 0.0)

    return Nk_new, Mk_new
