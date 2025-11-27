"""
Production Driver for Long Simulations.

Wraps the entire simulation time loop in a JAX scan to maximize speed.
"""
import jax
import jax.numpy as jnp
from typing import Tuple
from tomas_jax.integration.diffrax_solver import diffrax_step

def run_production_simulation(
    Nk_init: jnp.ndarray,
    Mk_init: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    total_hours: int,
    output_interval: float,
    internal_timestep: float,
    icomp_nodiag: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run a long simulation entirely within a JIT-compiled XLA kernel.
    
    Args:
        Nk_init, Mk_init: Initial state.
        total_hours: Total simulation duration in hours (e.g., 72).
        output_interval: How often to save output in seconds (e.g., 3600.0).
        internal_timestep: The physics/mnfix splitting time step (e.g., 10.0).
        
    Returns:
        times: Array of time points [s].
        Nk_history: History of Number concentration.
        Mk_history: History of Mass concentration.
    """
    
    # 1. Setup Time Grid
    # We want output every hour, so we scan over this array
    # Since output_interval is now static, we can compute this shape concretely
    num_outputs = int((total_hours * 3600.0) / output_interval)
    times = jnp.arange(0, (total_hours * 3600.0) + output_interval, output_interval)
    
    # 2. Define the Loop Body (The "Scan" function)
    def scan_body(carrier, t):
        Nk_curr, Mk_curr = carrier
        
        # Run one output interval (e.g., 1 hour)
        # Inside this call, diffrax breaks it down into chunks based on internal_timestep
        Nk_next, Mk_next = diffrax_step(
            Nk_curr, Mk_curr, xk, temp, pres, boxvol, 
            dt=output_interval,          # Run for 1 hour
            mnfix_interval=internal_timestep, # Apply mnfix every 10s
            icomp_nodiag=icomp_nodiag
        )
        
        # New state becomes carrier for next step, and output for history
        return (Nk_next, Mk_next), (Nk_next, Mk_next)

    # 3. Initial Carry State
    init_carrier = (Nk_init, Mk_init)
    
    # 4. Run the loop (This happens entirely in XLA)
    # We scan over times[:-1] because we start at t=0 and step forward
    final_carrier, (Nk_stacked, Mk_stacked) = jax.lax.scan(
        scan_body, init_carrier, times[:-1]
    )
    
    # 5. Prepend Initial Condition to history
    Nk_history = jnp.vstack([Nk_init[None, :], Nk_stacked])
    Mk_history = jnp.vstack([Mk_init[None, :, :], Mk_stacked])
    
    return times, Nk_history, Mk_history

# Create the JIT-compiled version
# FIXED: Added 'output_interval' and 'internal_timestep' to static_argnames
# This ensures loop bounds inside this function AND inside diffrax_step are static.
jit_run_simulation = jax.jit(
    run_production_simulation, 
    static_argnames=['total_hours', 'output_interval', 'internal_timestep', 'icomp_nodiag']
)