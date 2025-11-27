"""Production ODE solver using diffrax.

Optimized for High Performance Computing (HPC) via JAX JIT compilation.
Implements 'Operator Splitting' for the Coagulation Kernel.
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
from ..physics.properties import calc_particle_properties
from ..physics.coagulation_kernel import calc_coagulation_kernel
from ..physics.coagulation_rates import calc_coagulation_rates
from ..core.mnfix_jax import mnfix_jax

# Optimized Argument Container
# Passing a NamedTuple is more JIT-friendly than a raw tuple
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

    # Calculate Rates (Matrix Multiplications - Fast on AVX/GPU)
    dNdt, dMdt = calc_coagulation_rates(state.Nk, state.Mk, kij, xk, icomp_nodiag)

    # Return derivatives
    # Note: Returning zeros for T, P, etc. allows XLA to optimize them out completely.
    return TomasState(
        Nk=dNdt,
        Mk=dMdt,
        temp=0.0,
        pres=0.0,
        xk=jnp.zeros_like(xk),
        boxvol=0.0
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
    rtol: float = 1e-4,  # Tuned for aerosol physics (1e-6 is often overkill)
    atol: float = 1e-10
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate coagulation over timestep dt.
    
    This entire function is designed to be JIT-compiled.
    It fuses the Physics Setup (Kernel) + ODE Solver + MNFIX Correction.
    """
    
    # 1. Physics Setup (The Heavy Lifting)
    # Calculated ONCE per global timestep.
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    # 2. Solver Configuration
    if mnfix_interval is None:
        mnfix_interval = dt / 10.0

    # Determine loop bounds (must be static for JIT, or use lax.scan)
    # Using python control flow here works if this function is JIT-compiled 
    # with 'dt' and 'mnfix_interval' as static, OR if we strictly divide time.
    # For robustness in JAX, we determine steps roughly:
    num_intervals = jnp.maximum(1, jnp.ceil(dt / mnfix_interval).astype(int))
    dt_chunk = dt / num_intervals

    # Pack static arguments
    args = CoagArgs(kij=kij, xk=xk, icomp_nodiag=icomp_nodiag)

    # Solver Definition
    # Tsit5 is a 5th order Runge-Kutta (standard for non-stiff problems)
    # If coagulation is extremely fast (stiff), consider diffrax.Kvaerno5()
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(
        rtol=rtol, 
        atol=atol, 
        dtmin=1e-13, 
        dtmax=dt_chunk
    )
    term = diffrax.ODETerm(coagulation_rhs)

    # Initial State Wrapper
    state = TomasState(Nk=Nk, Mk=Mk, temp=temp, pres=pres, xk=xk, boxvol=boxvol)

    # 3. Integration Loop (Splitting for MNFIX)
    # We use jax.lax.fori_loop to keep the loop inside the compiled XLA kernel.
    # This is much faster than a Python loop for many iterations.
    
    def loop_body(i, current_state):
        t0 = i * dt_chunk
        t1 = (i + 1) * dt_chunk
        
        # Solve ODE for this chunk
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt_chunk / 10.0, # Guess for first step
            y0=current_state,
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=5000
        )
        
        # Extract result (remove time dimension)
        # diffrax returns shape (1, ...), we take index 0
        state_sol = jax.tree_util.tree_map(lambda x: x[0], solution.ys)
        
        # Apply MNFIX (Mass-Number Fix)
        # This MUST be a pure JAX implementation, not NumPy!
        Nk_fixed, Mk_fixed = mnfix_jax(
            state_sol.Nk, state_sol.Mk, xk, icomp_nodiag
        )
        
        # Repack state
        return TomasState(
            Nk=Nk_fixed,
            Mk=Mk_fixed,
            temp=temp,
            pres=pres,
            xk=xk,
            boxvol=boxvol
        )

    # Execute the loop
    final_state = jax.lax.fori_loop(0, num_intervals, loop_body, state)

    return final_state.Nk, final_state.Mk