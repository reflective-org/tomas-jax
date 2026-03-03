"""Production ODE solver using diffrax.

Optimized for High Performance Computing (HPC) via JAX JIT compilation.
Implements 'Operator Splitting' for the Coagulation Kernel.

UPDATED: Uses jax.lax.scan instead of fori_loop to support Automatic Differentiation.
"""
import math
import jax
# Enforce float64 for mass conservation precision
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.tree_util
import diffrax
from typing import Tuple, NamedTuple, Optional

# Assumes these modules exist in your project structure
from ..core.state import TomasState
from ..core.config import N_GAS_SPECIES
from ..physics.properties import calc_particle_properties
from ..physics.coagulation_kernel import calc_coagulation_kernel
from ..physics.coagulation_rates import calc_coagulation_rates
from ..core.mnfix_jax import mnfix_jax

# Optimized Argument Container
class CoagArgs(NamedTuple):
    kij: jnp.ndarray
    xk: jnp.ndarray
    icomp_nodiag: int

def coagulation_rhs(t: float, state: TomasState, args: CoagArgs) -> TomasState:
    """Right-hand side function for coagulation ODE.
    
    PERFORMANCE CRITICAL: 
    This function is called thousands of times by the solver.
    It deliberately avoids re-calculating the Coagulation Kernel (O(N^2) heavy math),
    using the pre-computed 'kij' from args instead.
    """
    # Unpack pre-computed arguments
    kij, xk, icomp_nodiag = args

    # Calculate Rates
    dNdt, dMdt = calc_coagulation_rates(state.Nk, state.Mk, kij, xk, icomp_nodiag)

    return TomasState(
        Nk=dNdt,
        Mk=dMdt,
        xk=jnp.zeros_like(xk),
        temp=0.0,
        pres=0.0,
        boxvol=0.0,
        Gc=jnp.zeros(N_GAS_SPECIES),
        rh=0.0,
        alpha=0.0
    )

def diffrax_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    dt: float,
    icomp_nodiag: int = 42,
    mnfix_interval: Optional[float] = None,
    rtol: float = 1e-4,
    atol: float = 1e-10
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate coagulation over timestep dt.
    
    Uses jax.lax.scan for the inner splitting loop to ensure the function
    is fully differentiable (compatible with jax.grad).
    """
    
    # 1. Physics Setup (The Heavy Lifting)
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    # 2. Solver Configuration
    if mnfix_interval is None:
        mnfix_interval = dt / 10.0

    # CRITICAL CHANGE: Calculate num_intervals as a Python integer.
    # jax.lax.scan requires a static length argument.
    # This works because 'dt' is usually passed as a static float/constant in scripts.
    try:
        n_substeps = int(max(1, math.ceil(dt / mnfix_interval)))
    except TypeError:
        # If dt is a Tracer (dynamic), we default to 10 steps to keep it compilable
        # This is a safe fallback for AD
        n_substeps = 10
        
    dt_chunk = dt / n_substeps

    # Pack static arguments
    args = CoagArgs(kij=kij, xk=xk, icomp_nodiag=icomp_nodiag)

    # Solver Definition — Tsit5 (explicit RK5(4), relaxed tolerances for physical units)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
        dtmin=1e-13,
        dtmax=dt_chunk
    )
    term = diffrax.ODETerm(coagulation_rhs)

    # Initial State Wrapper (Gc, rh, alpha are preserved unchanged through coagulation)
    Gc_zero = jnp.zeros(N_GAS_SPECIES)
    state = TomasState(
        Nk=Nk, Mk=Mk, xk=xk, temp=temp, pres=pres, boxvol=boxvol,
        Gc=Gc_zero, rh=0.0, alpha=0.0
    )

    # 3. Integration Loop (Splitting for MNFIX)
    # REPLACED fori_loop with SCAN
    
    def scan_body(current_state, _):
        # We don't use the loop index '_', just the carried state
        
        # Solve ODE for this chunk (from t=0 to t=dt_chunk)
        # We perform a "local integration" relative to the chunk start
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=dt_chunk,
            dt0=dt_chunk / 10.0, 
            y0=current_state,
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=5000,
            throw=False
        )
        
        # Extract result
        state_sol = jax.tree_util.tree_map(lambda x: x[0], solution.ys)
        
        # Apply MNFIX (Mass-Number Fix)
        Nk_fixed, Mk_fixed = mnfix_jax(
            state_sol.Nk, state_sol.Mk, xk, icomp_nodiag
        )
        
        # Repack state for next iteration (preserve Gc, rh, alpha)
        next_state = TomasState(
            Nk=Nk_fixed,
            Mk=Mk_fixed,
            xk=xk,
            temp=temp,
            pres=pres,
            boxvol=boxvol,
            Gc=Gc_zero,
            rh=0.0,
            alpha=0.0
        )
        
        # Carry, Output
        return next_state, None

    # Execute the loop using scan (Differentiable!)
    final_state, _ = jax.lax.scan(scan_body, state, None, length=n_substeps)

    return final_state.Nk, final_state.Mk


def coag_euler_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    dt: float = 60.0,
    icomp_nodiag: int = 42,
    n_substeps: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fixed-step forward Euler coagulation solver (scan-fusable).

    Matches Fortran approach: explicit forward step + MNFIX after each substep.
    Forward Euler avoids the intermediate-stage amplification that makes
    higher-order methods unstable for high-N coagulation (N^2 rates create
    positive feedback in intermediate stage evaluations).
    """
    # Pre-compute coagulation kernel (once per timestep)
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    dt_sub = dt / n_substeps

    def substep(carry, _):
        Nk_c, Mk_c = carry

        # Forward Euler
        dNdt, dMdt = calc_coagulation_rates(Nk_c, Mk_c, kij, xk, icomp_nodiag)
        Nk_new = Nk_c + dt_sub * dNdt
        Mk_new = Mk_c + dt_sub * dMdt

        # Positivity enforcement
        Nk_new = jnp.maximum(Nk_new, 0.0)
        Mk_new = jnp.maximum(Mk_new, 0.0)

        # MNFIX after each substep
        Nk_new, Mk_new = mnfix_jax(Nk_new, Mk_new, xk, icomp_nodiag)

        return (Nk_new, Mk_new), None

    (Nk_f, Mk_f), _ = jax.lax.scan(substep, (Nk, Mk), None, length=n_substeps)
    return Nk_f, Mk_f


# Deprecated alias (was misnamed as RK4 when it's actually forward Euler)
coag_rk4_step = coag_euler_step