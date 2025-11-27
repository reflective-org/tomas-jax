"""
TOMAS Coagulation Box Model Driver.

This script acts as the entry point for the simulation.
It demonstrates:
1. Initialization of state (Log-normal distribution).
2. JIT-compilation of the solver.
3. Time-stepping loop with dynamic environment (Operator Splitting).
4. Validation of Mass Conservation.
"""
import jax
import jax.numpy as jnp

# Import the solver from the new location
from tomas_jax.solvers.diffrax import diffrax_step

import time
import numpy as np # Used for printing/logging (CPU side)

# 1. CRITICAL: Import config before JAX to enforce float64
import tomas_jax.core.config as config
from tomas_jax.core.config import NBINS, ICOMP, ICOMP_NODIAG
from tomas_jax.core.config import SRTSO4, SRTH2O

from tomas_jax.core.state import TomasState
from tomas_jax.solvers.diffrax import diffrax_step
from tomas_jax.utils import plotting
from tomas_jax.utils.diagnostics import get_coagulation_rates

# =========================================================================
# 1. Helper Functions (Initialization)
# =========================================================================

def create_lognormal_dist(
    nbins: int, 
    n_total: float, 
    gmd: float, 
    gsd: float, 
    mass_min: float, 
    mass_max: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Initialize a log-normal size distribution.
    
    Args:
        nbins: Number of bins
        n_total: Total number concentration [#/cm3] -> converted to #/cell
        gmd: Geometric Mean Diameter [m]
        gsd: Geometric Standard Deviation
        mass_min: Smallest particle mass [kg]
        mass_max: Largest particle mass [kg]
    """
    # 1. Create Bin Boundaries (Log-spaced mass)
    xk = jnp.logspace(jnp.log10(mass_min), jnp.log10(mass_max), nbins + 1)
    
    # 2. Calculate Bin Midpoints (Mass and Diameter)
    # Use geometric mean for mass midpoints
    m_mid = jnp.sqrt(xk[:-1] * xk[1:])
    
    # Approx density for initialization (assume 1770 kg/m3 for sulfate)
    rho_init = 1770.0
    dp_mid = jnp.cbrt(m_mid / rho_init * (6.0 / np.pi))

    # 3. Calculate Lognormal Distribution dN/dlogDp
    # n(Dp) = (N / (sqrt(2pi) * log(gsd))) * exp(-(log(Dp) - log(gmd))^2 / (2 * log(gsd)^2))
    log_dp = jnp.log(dp_mid)
    log_gmd = jnp.log(gmd)
    log_gsd = jnp.log(gsd)
    
    dNdlogDp = (n_total / (jnp.sqrt(2 * np.pi) * log_gsd)) * \
               jnp.exp(-(log_dp - log_gmd)**2 / (2 * log_gsd**2))
    
    # 4. Integrate to get Number per Bin (Nk)
    # Simple approximation: Nk = dN/dlogDp * dlogDp
    # Calculate bin width in log space
    dlogDp = jnp.log(xk[1:] / xk[:-1]) / 3.0 # mass ~ D^3, so dlogm = 3*dlogD
    
    Nk = dNdlogDp * dlogDp
    
    # 5. Calculate Mass per Bin (Mk)
    # Initialize Mass array (Bins, Species)
    Mk = jnp.zeros((nbins, ICOMP))
    
    # Assign all mass to Sulfate for this test
    # Mass = Number * Average Mass per particle
    Mk = Mk.at[:, SRTSO4].set(Nk * m_mid)
    
    return Nk, Mk, xk

# =========================================================================
# 2. Main Simulation Loop
# =========================================================================

def run_box_model():
    print("="*60)
    print(f"🚀 TOMAS Coagulation Model - Initializing")
    print(f"   Bins: {NBINS}, Components: {ICOMP}")
    print(f"   Precision: {'float64' if jax.config.jax_enable_x64 else 'float32'}")
    print("="*60)

    # --- A. Setup Environment ---
    boxvol = 1.0e6  # 1 m^3 in cm^3 (standard box size)
    temp = 298.0    # Kelvin
    pres = 101325.0 # Pascals
    
    # --- B. Initialize Particles ---
    # Create a distribution centered at 100nm
    Nk_init, Mk_init, xk = create_lognormal_dist(
        nbins=NBINS,
        n_total=100000.0,       # 1000 particles/cm3
        gmd=1.0e-7,           # 100 nm geometric mean diameter
        gsd=1.6,              # Standard width
        mass_min=1e-23,       # ~2 nm
        mass_max=1e-12        # ~10 um
    )
    
    # Scale inputs by box volume
    Nk = Nk_init * boxvol
    Mk = Mk_init * boxvol
    
    print(f"Nk: {Nk}")

    # Calculate initial totals for conservation check
    total_N_init = jnp.sum(Nk)
    total_M_init = jnp.sum(Mk)
    
    print(f"Initial Total Particles: {total_N_init:.4e}")
    print(f"Initial Total Mass:      {total_M_init:.4e} kg")

    # --- C. Prepare Solver ---
    # JIT Compile the solver step function.
    # We pass 'icomp_nodiag' as static because it determines loop/array sizes.
    print("\n[System] JIT Compiling Solver... (this happens once)")
    
    solver_jit = jax.jit(
        diffrax_step, 
        static_argnames=['icomp_nodiag']
    )
    
    # Trigger compilation with a dummy call (optional, but good for timing accuracy)
    # Using small dt=0.01 just to compile
    _ = solver_jit(Nk, Mk, xk, temp, pres, boxvol, 0.01, ICOMP_NODIAG)
    print("[System] Compilation Complete.")

    # --- D. Time Loop ---
    total_time = 3600.0*24   # 1 Hour simulation
    dt_model = 600.0       # Update environment/output every 60 seconds
    
    current_time = 0.0
    step_count = 0
    
    start_time_wall = time.time()
    
    print(f"\n[Simulation] Starting {int(total_time)}s run (dt={dt_model}s)...")
    print(f"{'Time [s]':<10} | {'Particles [#]':<15} | {'Mass [kg]':<15} | {'Status'}")
    print("-" * 60)

    # Data Collection for Plotting
    history_time = []
    history_Nk = []
    history_N_tot = []
    history_M_tot = []
    history_dNdt = []
    history_dMdt = []

    while current_time < total_time:
        # Store History
        history_time.append(current_time)
        history_Nk.append(Nk) # JAX array, but list append is fine
        history_N_tot.append(jnp.sum(Nk))
        history_M_tot.append(jnp.sum(Mk))

        # 1. Update Environment (Operator Splitting)
        # Example: Parcel cools as it rises (Dynamic T/P)
        # temp = 298.0 - (0.0065 * current_time) 
        
        # 2. Run Coagulation Step
        # The solver handles the integration from t to t+dt
        Nk, Mk = solver_jit(
            Nk, Mk, xk, 
            temp, pres, boxvol, 
            dt=dt_model, 
            icomp_nodiag=ICOMP_NODIAG
        )
        
        # 3. DIAGNOSE: Calculate rates at this new state
        # (Useful for debugging "How fast is coagulation happening right now?")
        dNdt, dMdt = get_coagulation_rates(
            Nk, Mk, xk, temp, pres, boxvol, ICOMP_NODIAG
        )

        # Store for plotting (convert to numpy to save RAM)
        history_dNdt.append(np.array(dNdt))
        history_dMdt.append(np.array(dMdt))

        # 4. Advance Time
        current_time += dt_model
        step_count += 1
        
        # 5. Logging
        total_N = jnp.sum(Nk)
        total_M = jnp.sum(Mk)
        
        if step_count % 10 == 0 or step_count == 1:
            print(f"{current_time:<10.1f} | {total_N:<15.4e} | {total_M:<15.4e} | Running")

    end_time_wall = time.time()
    
    # Store final step
    history_time.append(current_time)
    history_Nk.append(Nk)
    history_N_tot.append(jnp.sum(Nk))
    history_M_tot.append(jnp.sum(Mk))

    # =========================================================================
    # 3. Validation & Results
    # =========================================================================
    print("-" * 60)
    print("\n[Results]")
    
    # 1. Conservation Checks
    total_N_final = jnp.sum(Nk)
    total_M_final = jnp.sum(Mk)
    
    mass_error = (total_M_final - total_M_init) / total_M_init
    
    print(f"Final Total Particles: {total_N_final:.4e}")
    print(f"Final Total Mass:      {total_M_final:.4e} kg")
    print(f"Particles Lost:        {total_N_init - total_N_final:.4e} (Coagulation working)")
    
    print(f"Mass Error (Rel):      {mass_error:.4e}")
    
    if abs(mass_error) < 1e-13:
        print("✅ Mass Conservation: PERFECT")
    elif abs(mass_error) < 1e-10:
        print("✅ Mass Conservation: EXCELLENT")
    else:
        print("❌ Mass Conservation: FAILED")

    # 2. Performance Stats
    duration = end_time_wall - start_time_wall
    steps = total_time / dt_model
    print(f"\n[Performance]")
    print(f"Total Wall Time:  {duration:.4f} seconds")
    print(f"Time per Model Step: {duration/steps*1000:.2f} ms")

    # 3. Visualization (Simple ASCII Histogram)
    print("\n[Final Size Distribution]")
    # Normalize for display
    max_N = jnp.max(Nk)
    for k in range(NBINS):
        bar_len = int((Nk[k] / max_N) * 40)
        bar = "█" * bar_len
        if k % 4 == 0: # Print every 4th bin to save space
            print(f"Bin {k:02d} ({xk[k]:.1e} kg): {bar}")

    # =========================================================================
    # 4. Visualization
    # =========================================================================
    print("\nGenerating Plots...")
    
    # # 1. Size Distributions (Initial vs Final)
    # plotting.plot_size_distribution(
    #     Nk_init * boxvol, Nk, Mk_init * boxvol, Mk, xk, boxvol
    # )
    
    # # 2. Time Series (Conservation)
    # plotting.plot_time_series(
    #     history_time, history_N_tot, history_M_tot
    # )
    
    # # 3. Banana Plot
    # plotting.plot_banana(
    #     history_time, history_Nk, xk, boxvol
    # )

    # # 4. Coagulation Rates
    # plotting.plot_rates(xk, history_dNdt[-1], boxvol)
    # plotting.plot_rates(xk, history_dMdt[-1], boxvol)
    return Nk_init, Nk, Mk_init, Mk, xk, boxvol, history_time, history_N_tot, history_M_tot, history_dNdt, history_dMdt

if __name__ == "__main__":
    Nk_init, Nk, Mk_init, Mk, xk, boxvol, history_time, history_N_tot, history_M_tot, history_dNdt, history_dMdt = run_box_model()