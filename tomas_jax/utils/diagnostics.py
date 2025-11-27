"""Diagnostic utilities for TOMAS-JAX.

Functions here calculate instantaneous rates or properties 
without advancing the simulation state.
"""
import jax
from functools import partial
import tomas_jax.physics.properties as properties
import tomas_jax.physics.coagulation_kernel as kernel
import tomas_jax.physics.coagulation_rates as rates

# JIT-compiled diagnostic function
@partial(jax.jit, static_argnames=['icomp_nodiag'])
def get_coagulation_rates(
    Nk, Mk, xk, temp, pres, boxvol, icomp_nodiag
):
    """
    Calculate instantaneous dN/dt and dM/dt for the current state.
    Useful for debugging and plotting rates.
    """
    # 1. Calculate Properties (O(N))
    Dpk, Dk, ck = properties.calc_particle_properties(Nk, Mk, temp, pres)
    
    # 2. Calculate Kernel (O(N^2))
    kij = kernel.calc_coagulation_kernel(Dpk, Dk, ck, boxvol)
    
    # 3. Calculate Rates (O(N^2))
    dNdt, dMdt = rates.calc_coagulation_rates(Nk, Mk, kij, xk, icomp_nodiag)
    
    return dNdt, dMdt