"""Visualization utilities for TOMAS-JAX results.

Handles:
1. Size Distributions (Number & Mass)
2. Time Series (Conservation checks)
3. Banana Plots (Evolution of size distribution)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax.numpy as jnp

def calculate_dlogDp(xk: jnp.ndarray) -> np.ndarray:
    """Calculate log-width of bins for normalization (dN/dlogDp)."""
    # Convert to numpy for plotting
    xk_np = np.array(xk)
    
    # Calculate diameters from mass (assuming spherical, density approx)
    # Note: For bin width calculation, absolute density doesn't matter much
    # as long as mass ~ D^3.
    # m = rho * pi/6 * D^3  -> D ~ m^(1/3)
    dp_edges = np.cbrt(xk_np)
    
    # dlogDp = log10(D_upper) - log10(D_lower)
    dlog_dp = np.log10(dp_edges[1:] / dp_edges[:-1])
    return dlog_dp

def calculate_dp_midpoints(xk: jnp.ndarray, density: float = 1770.0) -> np.ndarray:
    """Calculate geometric mean diameter of bins [m]."""
    xk_np = np.array(xk)
    # Mass midpoints
    m_mid = np.sqrt(xk_np[:-1] * xk_np[1:])
    # Diameter midpoints
    dp_mid = np.cbrt(m_mid / density * (6.0 / np.pi))
    return dp_mid

def plot_size_distribution(
    Nk_init: jnp.ndarray,
    Nk_final: jnp.ndarray,
    Mk_init: jnp.ndarray,
    Mk_final: jnp.ndarray,
    xk: jnp.ndarray,
    boxvol: float = 1.0e6
):
    """Plot Initial vs Final Number and Mass Size Distributions."""
    
    # Convert to Numpy
    Nk_i = np.array(Nk_init) / boxvol # Convert to #/cm3
    Nk_f = np.array(Nk_final) / boxvol
    
    # Sum mass across species for distribution plot
    if Mk_init.ndim > 1:
        Mk_i = np.sum(np.array(Mk_init), axis=1) / boxvol
        Mk_f = np.sum(np.array(Mk_final), axis=1) / boxvol
    else:
        Mk_i = np.array(Mk_init) / boxvol
        Mk_f = np.array(Mk_final) / boxvol

    dp_mid = calculate_dp_midpoints(xk)
    dlog_dp = calculate_dlogDp(xk)
    
    # Normalize (dN/dlogDp)
    dNdlogDp_i = Nk_i / dlog_dp
    dNdlogDp_f = Nk_f / dlog_dp
    dMdlogDp_i = Mk_i / dlog_dp
    dMdlogDp_f = Mk_f / dlog_dp

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Number Distribution
    ax = axes[0]
    ax.semilogx(dp_mid, dNdlogDp_i, '--', label='Initial', color='gray')
    ax.semilogx(dp_mid, dNdlogDp_f, '-', label='Final', color='blue', lw=2)
    ax.set_xlabel('Particle Diameter [m]')
    ax.set_ylabel('dN/dlogDp [#/cm³]')
    ax.set_title('Number Size Distribution')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    
    # 2. Mass Distribution
    ax = axes[1]
    ax.semilogx(dp_mid, dMdlogDp_i, '--', label='Initial', color='gray')
    ax.semilogx(dp_mid, dMdlogDp_f, '-', label='Final', color='red', lw=2)
    ax.set_xlabel('Particle Diameter [m]')
    ax.set_ylabel('dM/dlogDp [kg/cm³]')
    ax.set_title('Mass Size Distribution')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_time_series(
    time_arr: list,
    total_N_arr: list,
    total_M_arr: list
):
    """Plot evolution of Total Number and Total Mass over time."""
    t = np.array(time_arr)
    N = np.array(total_N_arr)
    M = np.array(total_M_arr)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Left Axis: Number
    color = 'tab:blue'
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Total Number [#]', color=color)
    ax1.plot(t, N, color=color, lw=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Right Axis: Mass
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Total Mass [kg]', color=color)
    # Plot mass as a dashed line (should be flat if conserved)
    ax2.plot(t, M, color=color, linestyle='--', lw=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Zoom in mass axis to show tiny errors if they exist
    # ax2.set_ylim(M[0]*0.9999, M[0]*1.0001) 
    
    plt.title("Time Evolution (Mass Conservation Check)")
    plt.tight_layout()
    plt.show()

def plot_banana(
    time_arr: list,
    Nk_history: list,
    xk: jnp.ndarray,
    boxvol: float = 1.0e6
):
    """Create a 'Banana Plot' (Contour of Size Dist vs Time)."""
    
    # Prepare Data
    times = np.array(time_arr)  # X-axis
    dp_mid = calculate_dp_midpoints(xk) # Y-axis
    dlog_dp = calculate_dlogDp(xk)
    
    # Matrix: (Time, Bins)
    # Normalize raw Nk to dN/dlogDp
    matrix = np.array(Nk_history) / boxvol # Shape (T, Bins)
    matrix = matrix / dlog_dp[None, :] # Broadcast divide
    
    # Transpose for plotting: (Bins, Time)
    Z = matrix.T 
    
    # Handle zeros for log-scale plotting
    Z = np.maximum(Z, 1e-5) 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pcolormesh
    # Note: we use times and dp_mid. 
    # Technically pcolormesh wants edges, but using centers with 'nearest' or 'gouraud' works usually.
    # For accuracy, we assume X=Time, Y=Diameter
    
    mesh = ax.pcolormesh(
        times, 
        dp_mid, 
        Z, 
        norm=LogNorm(vmin=Z.max()*1e-4, vmax=Z.max()),
        cmap='jet',
        shading='auto'
    )
    
    ax.set_yscale('log')
    ax.set_ylabel('Particle Diameter [m]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Aerosol Size Distribution Evolution (Banana Plot)')
    
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('dN/dlogDp [#/cm³]')
    
    plt.tight_layout()
    plt.show()

def plot_rates(xk: jnp.ndarray, dNdt: jnp.ndarray, boxvol: float = 1.0e6):
    """Plot the instantaneous rate of change dN/dt vs Diameter."""
    dp_mid = calculate_dp_midpoints(xk)
    
    # Normalize per cm3 per second
    rate_plot = np.array(dNdt) / boxvol
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Separate gains (positive) and losses (negative) for log plotting
    gains = np.maximum(rate_plot, 0)
    losses = np.abs(np.minimum(rate_plot, 0))
    
    ax.loglog(dp_mid, gains, 'b-', label='Gain (Coagulation In)')
    ax.loglog(dp_mid, losses, 'r--', label='Loss (Coagulation Out)')
    
    ax.set_xlabel('Particle Diameter [m]')
    ax.set_ylabel('Rate dN/dt [#/cm³/s]')
    ax.set_title('Instantaneous Coagulation Rates')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
