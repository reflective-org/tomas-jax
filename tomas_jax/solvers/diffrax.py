"""Production ODE solver using diffrax.

Optimized for High Performance Computing (HPC) via JAX JIT compilation.
Implements 'Operator Splitting' for the Coagulation Kernel.

Uses jax.lax.scan instead of fori_loop to support Automatic Differentiation.
"""
import warnings

import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
import jax.tree_util
import diffrax
from typing import Tuple, NamedTuple

from ..physics.properties import calc_particle_properties
from ..physics.coagulation_kernel import calc_coagulation_kernel
from ..physics.coagulation_rates import calc_coagulation_rates
from ..core.mnfix_jax import mnfix_jax
from ..core.config import ICOMP_NODIAG

# Optimized Argument Container
class CoagArgs(NamedTuple):
    kij: jnp.ndarray
    xk: jnp.ndarray
    icomp_nodiag: int

def coagulation_rhs(t: float, state: Tuple[jnp.ndarray, jnp.ndarray], args: CoagArgs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Right-hand side function for coagulation ODE.

    State is strictly (Nk, Mk) — static variables live in CoagArgs
    to avoid polluting the Diffrax PID error norm.
    """
    Nk, Mk = state
    kij, xk, icomp_nodiag = args

    dNdt, dMdt, _overflow = calc_coagulation_rates(Nk, Mk, kij, xk, icomp_nodiag)

    return (dNdt, dMdt)

def diffrax_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    dt: float,
    icomp_nodiag: int = ICOMP_NODIAG,
    n_substeps: int = 10,
    rtol: float = 1e-4,
    atol: float = 1e-10,
    return_status: bool = False,
    return_overflow: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate coagulation over timestep dt using Tsit5 adaptive solver.

    Uses jax.lax.scan for the inner splitting loop to ensure the function
    is fully differentiable (compatible with jax.grad).

    Args:
        n_substeps: Number of MNFIX splitting intervals (static integer).
        return_status: If True, returns (Nk, Mk, all_ok) where all_ok is
            a boolean JAX scalar indicating whether all substeps converged.
        return_overflow: If True, returns overflow mass [kg/cell] that
            overflowed the top bin, shape (icomp,). Approximated using
            the overflow rate at the start of each chunk.
    """

    # 1. Physics Setup
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    dt_chunk = dt / n_substeps

    # Pack static arguments
    args = CoagArgs(kij=kij, xk=xk, icomp_nodiag=icomp_nodiag)

    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
        dtmin=1e-13,
        dtmax=dt_chunk
    )
    term = diffrax.ODETerm(coagulation_rhs)

    # 2. Integration Loop (Splitting for MNFIX)
    def scan_body(current_state, _):
        Nk_prev, Mk_prev, overflow_acc = current_state

        # Estimate overflow rate at start of chunk (first-order approximation)
        _dNdt, _dMdt, dM_overflow = calc_coagulation_rates(
            Nk_prev, Mk_prev, kij, xk, icomp_nodiag)
        overflow_acc = overflow_acc + dt_chunk * dM_overflow

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=dt_chunk,
            dt0=dt_chunk / 10.0,
            y0=(Nk_prev, Mk_prev),
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=5000,
            throw=False
        )

        # Extract result
        Nk_sol, Mk_sol = jax.tree_util.tree_map(lambda x: x[0], solution.ys)

        # Check solver status: revert to previous state on failure
        ok = diffrax.is_successful(solution.result)
        Nk_out = jnp.where(ok, Nk_sol, Nk_prev)
        Mk_out = jnp.where(ok, Mk_sol, Mk_prev)

        # Apply MNFIX
        Nk_fixed, Mk_fixed = mnfix_jax(Nk_out, Mk_out, xk, icomp_nodiag)

        return (Nk_fixed, Mk_fixed, overflow_acc), ok

    overflow_init = jnp.zeros(Mk.shape[1])
    (final_Nk, final_Mk, overflow_total), substep_ok = jax.lax.scan(
        scan_body, (Nk, Mk, overflow_init), None, length=n_substeps)

    if return_status and return_overflow:
        return final_Nk, final_Mk, jnp.all(substep_ok), overflow_total
    if return_status:
        return final_Nk, final_Mk, jnp.all(substep_ok)
    if return_overflow:
        return final_Nk, final_Mk, overflow_total
    return final_Nk, final_Mk


def coag_euler_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    dt: float = 60.0,
    icomp_nodiag: int = ICOMP_NODIAG,
    n_substeps: int = 3,
    return_overflow: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fixed-step forward Euler coagulation solver (scan-fusable).

    Matches Fortran approach: explicit forward step + MNFIX after each substep.
    Forward Euler avoids the intermediate-stage amplification that makes
    higher-order methods unstable for high-N coagulation (N^2 rates create
    positive feedback in intermediate stage evaluations).

    Args:
        return_overflow: If True, returns (Nk, Mk, overflow) where overflow
            is the total mass [kg/cell] that overflowed the top bin during
            this timestep, shape (icomp,).
    """
    # Pre-compute coagulation kernel (once per timestep)
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    dt_sub = dt / n_substeps

    def substep(carry, _):
        Nk_c, Mk_c, overflow_acc = carry

        # Forward Euler
        dNdt, dMdt, dM_overflow = calc_coagulation_rates(Nk_c, Mk_c, kij, xk, icomp_nodiag)
        Nk_new = Nk_c + dt_sub * dNdt
        Mk_new = Mk_c + dt_sub * dMdt

        # Accumulate overflow
        overflow_acc = overflow_acc + dt_sub * dM_overflow

        # Positivity enforcement
        Nk_new = jnp.maximum(Nk_new, 0.0)
        Mk_new = jnp.maximum(Mk_new, 0.0)

        # MNFIX after each substep
        Nk_new, Mk_new = mnfix_jax(Nk_new, Mk_new, xk, icomp_nodiag)

        return (Nk_new, Mk_new, overflow_acc), None

    overflow_init = jnp.zeros(Mk.shape[1])
    (Nk_f, Mk_f, overflow_total), _ = jax.lax.scan(
        substep, (Nk, Mk, overflow_init), None, length=n_substeps)

    if return_overflow:
        return Nk_f, Mk_f, overflow_total
    return Nk_f, Mk_f


def coag_rk4_step(*args, **kwargs):
    """Deprecated alias for coag_euler_step."""
    warnings.warn(
        "coag_rk4_step is deprecated, use coag_euler_step instead. "
        "The solver was always forward Euler, not RK4.",
        DeprecationWarning, stacklevel=2,
    )
    return coag_euler_step(*args, **kwargs)