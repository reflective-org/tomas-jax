import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # Needed for LogNorm in banana plot

def plot_size_distributions(
    Nk_init, Mk_init, 
    Nk_final, Mk_final, 
    Dp_centers, 
    boxvol, 
    density, 
    title_suffix=""
):
    """
    Plot Number, Mass, and Volume distributions (Initial vs Final).
    
    Args:
        Nk_init, Mk_init: Initial state arrays.
        Nk_final, Mk_final: Final state arrays.
        Dp_centers: Particle diameter for bin centers [m].
        boxvol: Grid cell volume [cm3].
        density: Particle density [kg/m3].
        title_suffix: String to append to plot titles (e.g., "Euler" or "Diffrax").
    """
    # Constants
    # dlogDp for mass doubling bins: log10(2^(1/3))
    dlogDp = np.log10(2**(1.0/3.0))
    Dp_nm = Dp_centers * 1e9  # Convert to nm
    
    # --- 1. Calculate Distributions ---
    
    # Number Distribution: dN/dlogDp [#/cm3]
    dNdlogDp_init = Nk_init / dlogDp / boxvol
    dNdlogDp_final = Nk_final / dlogDp / boxvol
    
    # Mass Distribution: dM/dlogDp [ug/m3]
    # Mk is [kg/box]. 
    # To get [ug/m3]:
    # 1. / boxvol -> [kg/cm3]
    # 2. * 1e6 -> [kg/m3] (since 1e6 cm3 = 1 m3)
    # 3. * 1e9 -> [ug/m3]
    # Total factor: 1e15 / boxvol
    scale_mass = 1e15 / boxvol
    dMdlogDp_init = np.sum(Mk_init, axis=1) / dlogDp * scale_mass
    dMdlogDp_final = np.sum(Mk_final, axis=1) / dlogDp * scale_mass
    
    # Volume Distribution: dV/dlogDp [um3/cm3]
    # Volume = Mass / Density [m3]
    # To get [um3/cm3]:
    # 1. Mass [kg] / Density [kg/m3] -> Volume [m3]
    # 2. / boxvol -> [m3/cm3]
    # 3. * 1e18 -> [um3/cm3] (since 1 m3 = 1e18 um3)
    scale_vol = 1e18 / density / boxvol
    dVdlogDp_init = np.sum(Mk_init, axis=1) / dlogDp * scale_vol
    dVdlogDp_final = np.sum(Mk_final, axis=1) / dlogDp * scale_vol

    # --- 2. Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot A: Number
    axes[0].plot(Dp_nm, dNdlogDp_init, 'k--', label='Initial')
    axes[0].plot(Dp_nm, dNdlogDp_final, 'r-', label='Final', linewidth=2, alpha=0.8)
    axes[0].set_ylabel('dN/dlogDp [#/cm$^3$]')
    axes[0].set_title(f'Number Distribution {title_suffix}')
    
    # Plot B: Mass
    axes[1].plot(Dp_nm, dMdlogDp_init, 'k--', label='Initial')
    axes[1].plot(Dp_nm, dMdlogDp_final, 'r-', label='Final', linewidth=2, alpha=0.8)
    axes[1].set_ylabel('dM/dlogDp [$\mu$g/m$^3$]')
    axes[1].set_title(f'Mass Distribution {title_suffix}')

    # Plot C: Volume
    axes[2].plot(Dp_nm, dVdlogDp_init, 'k--', label='Initial')
    axes[2].plot(Dp_nm, dVdlogDp_final, 'r-', label='Final', linewidth=2, alpha=0.8)
    axes[2].set_ylabel('dV/dlogDp [$\mu$m$^3$/cm$^3$]')
    axes[2].set_title(f'Volume Distribution {title_suffix}')

    # Formatting
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('Diameter [nm]')
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    
    # Save to figures/ directory
    os.makedirs('figures', exist_ok=True)
    filename = f'dist_{title_suffix.strip().lower().replace("(", "").replace(")", "")}.png'
    filepath = os.path.join('figures', filename)
    
    print(f"Saving distribution plot to {filepath}")
    plt.savefig(filepath, dpi=300)
    plt.show()


def plot_time_series(
    times, 
    Nk_history, 
    Mk_history, 
    boxvol, 
    density, 
    title_suffix=""
):
    """
    Plot Total Number, Mass, and Volume concentration over time.
    
    Args:
        times: Array of time points [s].
        Nk_history: Array of Nk states [steps, bins].
        Mk_history: Array of Mk states [steps, bins, species].
        boxvol: Grid cell volume [cm3].
        density: Particle density [kg/m3].
        title_suffix: String to append to titles.
    """
    # Convert times to hours
    t_hrs = np.array(times) / 3600.0
    
    # Calculate Totals
    # Total Number [#/cm3]
    N_total = np.sum(Nk_history, axis=1) / boxvol
    
    # Total Mass [ug/m3]
    # Sum over bins AND species
    M_total_kg = np.sum(np.sum(Mk_history, axis=2), axis=1)
    M_total = M_total_kg / boxvol * 1e15 # Convert to ug/m3
    
    # Total Volume [um3/cm3]
    # V = M / rho
    V_total = (M_total_kg / density) / boxvol * 1e18 # Convert to um3/cm3

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot A: Total Number
    axes[0].plot(t_hrs, N_total, 'b-', linewidth=2)
    axes[0].set_ylabel('Total N [#/cm$^3$]')
    axes[0].set_title(f'Total Number Conc. {title_suffix}')
    
    # Plot B: Total Mass
    # Use scientific notation offset if changes are tiny (conservation check)
    axes[1].plot(t_hrs, M_total, 'g-', linewidth=2)
    axes[1].set_ylabel('Total M [$\mu$g/m$^3$]')
    axes[1].set_title(f'Total Mass Conc. {title_suffix}')
    axes[1].ticklabel_format(useOffset=False) # Show actual values for conservation check
    
    # Plot C: Total Volume
    axes[2].plot(t_hrs, V_total, 'm-', linewidth=2)
    axes[2].set_ylabel('Total V [$\mu$m$^3$/cm$^3$]')
    axes[2].set_title(f'Total Volume Conc. {title_suffix}')
    axes[2].ticklabel_format(useOffset=False)

    # Formatting
    for ax in axes:
        ax.set_xlabel('Time [h]')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    # Save to figures/ directory
    os.makedirs('figures', exist_ok=True)
    filename = f'series_{title_suffix.strip().lower().replace("(", "").replace(")", "")}.png'
    filepath = os.path.join('figures', filename)
    
    print(f"Saving time series plot to {filepath}")
    plt.savefig(filepath, dpi=300)
    plt.show()

def plot_banana_dist(
    times,
    Nk_history,
    Dp_centers,
    boxvol,
    title_suffix=""
):
    """
    Generate a 'Banana Plot' (Contour of Size Distribution over Time).
    
    Args:
        times: Array of time points [s].
        Nk_history: Array of Nk states [steps, bins].
        Dp_centers: Particle diameter for bin centers [m].
        boxvol: Grid cell volume [cm3].
        title_suffix: String to append to titles.
    """
    # 1. Prepare Grid
    # Convert inputs
    t_hrs = np.array(times) / 3600.0
    Dp_nm = Dp_centers * 1e9
    
    # Calculate dNdlogDp matrix
    dlogDp = np.log10(2**(1.0/3.0))
    # Shape: [steps, bins] -> We need Transpose for pcolormesh [bins, steps]
    dNdlogDp = (Nk_history / dlogDp / boxvol).T
    
    # Create Meshgrid for plotting
    # Note: pcolormesh expects bin edges. For simplicity, we use centers 
    # and let matplotlib infer shading.
    T, D = np.meshgrid(t_hrs, Dp_nm)

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use LogNorm for color scale because concentrations span orders of magnitude
    # Clip vmin to avoid log(0)
    vmin = max(1e-1, np.min(dNdlogDp[dNdlogDp > 0]))
    vmax = np.max(dNdlogDp)
    
    mesh = ax.pcolormesh(
        T, D, dNdlogDp,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap='jet',
        shading='auto' # Interpolates appropriately
    )
    
    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('dN/dlogDp [#/cm$^3$]')
    
    # Labels
    ax.set_yscale('log')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Particle Diameter [nm]')
    ax.set_title(f'Aerosol Size Distribution Evolution (Banana Plot) {title_suffix}')
    
    # Save
    os.makedirs('figures', exist_ok=True)
    filename = f'banana_{title_suffix.strip().lower().replace("(", "").replace(")", "")}.png'
    filepath = os.path.join('figures', filename)
    print(f"Saving banana plot to {filepath}")
    plt.savefig(filepath, dpi=300)
    plt.show()