"""Pure JAX implementation of MNFIX - Vectorized for High Performance.

This module restores consistency between Number (Nk) and Mass (Mk) concentrations.
It replaces the sequential Fortran loops with parallel array operations (shifts),
making it suitable for GPU/TPU execution and JIT compilation.

Algorithm:
    1. Calculate average mass per particle (xbar) for each bin.
    2. Identify bins where xbar has drifted outside bin boundaries.
    3. Shift the entire population of drifting bins to the appropriate neighbor.
    
Assumption:
    Since this is used inside an adaptive ODE solver (Diffrax), time steps are
    small enough that particles typically drift only one bin at a time.
"""
import jax
# Enforce 64-bit precision for mass conservation
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Tuple

def mnfix_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Correct mass/number drift in bins using vectorized operations.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        xk: Bin boundaries [kg], shape (ibins+1,)
        icomp_nodiag: Number of non-diagnostic species.

    Returns:
        Nk_new: Corrected number concentration
        Mk_new: Corrected mass concentration
    """
    # =========================================================================
    # 1. Constants & Preprocessing
    # =========================================================================
    nbins = Nk.shape[0]
    NEPS = 1.0e-20  # Threshold to avoid division by zero
    TINY_N = 1.0e-10 # Minimum number concentration for empty bins
    
    # Enforce physical positivity immediately
    Nk = jnp.maximum(Nk, 0.0)
    Mk = jnp.maximum(Mk, 0.0)

    # =========================================================================
    # 2. Diagnostics (Calculate Average Mass)
    # =========================================================================
    # Sum dry mass components
    drymass = jnp.sum(Mk[:, :icomp_nodiag], axis=1)
    
    # Calculate average mass (xbar), handling empty bins safely
    # If Nk < NEPS, xbar is irrelevant (set to 0.0)
    xbar = jnp.where(Nk > NEPS, drymass / Nk, 0.0)
    
    # Bin boundaries
    xk_lo = xk[:-1]
    xk_hi = xk[1:]

    # =========================================================================
    # 3. Identify Drifting Bins
    # =========================================================================
    # Particles that have grown too large for their current bin
    mask_up = (xbar > xk_hi) & (Nk > NEPS)
    
    # Particles that have shrunk too small for their current bin
    mask_down = (xbar < xk_lo) & (Nk > NEPS)

    # =========================================================================
    # 4. Calculate Fluxes (Vectorized)
    # =========================================================================
    # Instead of looping, we determine the flux for the whole grid at once.
    
    # Prepare Flux Arrays (Amount to move out of current bin)
    N_flux_up = jnp.where(mask_up, Nk, 0.0)
    M_flux_up = jnp.where(mask_up[:, None], Mk, 0.0)
    
    N_flux_down = jnp.where(mask_down, Nk, 0.0)
    M_flux_down = jnp.where(mask_down[:, None], Mk, 0.0)

    # Apply Boundary Conditions (Cannot shift out of the domain)
    # 1. Last bin (nbins-1) cannot shift UP. 
    #    (Standard TFL choice: keep them there, or let them rain out. We keep them.)
    N_flux_up = N_flux_up.at[nbins-1].set(0.0)
    M_flux_up = M_flux_up.at[nbins-1].set(0.0)

    # 2. First bin (0) cannot shift DOWN.
    #    (They cannot shrink smaller than the smallest bin.)
    N_flux_down = N_flux_down.at[0].set(0.0)
    M_flux_down = M_flux_down.at[0].set(0.0)

    # =========================================================================
    # 5. Apply Shifts (Vectorized Roll)
    # =========================================================================
    # Step A: Remove drifting particles from their source bins
    Nk_new = Nk - N_flux_up - N_flux_down
    Mk_new = Mk - M_flux_up - M_flux_down

    # Step B: Add drifting particles to their destination bins
    # Shift Up -> Moves to index k+1. We use roll with shift +1.
    # Note: jnp.roll shifts indices: index 0 moves to 1.
    Nk_new = Nk_new + jnp.roll(N_flux_up, 1)
    Mk_new = Mk_new + jnp.roll(M_flux_up, 1, axis=0)

    # Shift Down -> Moves to index k-1. We use roll with shift -1.
    Nk_new = Nk_new + jnp.roll(N_flux_down, -1)
    Mk_new = Mk_new + jnp.roll(M_flux_down, -1, axis=0)

    # Step C: Clean up "wrap-around" artifacts from roll
    # jnp.roll is circular. 
    # - Anything shifting UP from the last bin would wrap to 0. (Handled by boundary condition above).
    # - Anything shifting DOWN from bin 0 would wrap to last bin. (Handled by boundary condition above).
    # However, strictly enforcing zero at the "entry" points of the roll is good practice.
    # (Implicitly handled by BCs, but floating point noise can exist).

    # =========================================================================
    # 6. Final Cleanup (Empty Bins)
    # =========================================================================
    # Ensure empty bins have consistent (tiny) values to prevent NaNs in next step
    
    mask_empty = Nk_new < TINY_N
    
    # Reset N to tiny epsilon
    Nk_final = jnp.where(mask_empty, TINY_N, Nk_new)
    
    # Reset Mass: Zero out, then add tiny diagnostic mass (e.g. Sulfate)
    Mk_temp = jnp.where(mask_empty[:, None], 0.0, Mk_new)
    
    # Calculate geometric mean mass for the bin
    mean_mass_bin = jnp.sqrt(xk_lo * xk_hi)
    
    # Add tiny mass to first component (usually SO4, index 0)
    Mk_final = Mk_temp.at[:, 0].set(
        jnp.where(mask_empty, TINY_N * mean_mass_bin, Mk_temp[:, 0])
    )

    return Nk_final, Mk_final