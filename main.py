"""
TOMAS Coagulation Box Model Driver.

Demonstrates simulation execution and visualization.
"""

import time
import numpy as np

# 1. CRITICAL: Import config before JAX to enforce float64
import tomas_jax.core.config as config
from tomas_jax.core.config import NBINS, ICOMP, ICOMP_NODIAG
from tomas_jax.core.config import SRTSO4

import jax
import jax.numpy as jnp
from tomas_jax.core.state import TomasState
from tomas_jax.solvers.diffrax import diffrax_step
from tomas_jax.utils import plotting

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
    """Initialize a log-normal size distribution."""
    xk = jnp.logspace(jnp.log10(mass_min), jnp.log10(mass_max), nbins + 1)
    m_mid = jnp.sqrt(xk[:-1] * xk[1:])
    rho_init = 1770.0
    dp_mid = jnp.cbrt(m_mid / rho_init * (6.0 / np.pi))

    log_dp = jnp.log(dp_mid)
    log_gmd = jnp.log(gmd)
    log_gsd = jnp.log(gsd)
    
    dNdlogDp = (n_total / (jnp.sqrt(2 * np.pi) * log_gsd)) * \
               jnp.exp(-(log_dp - log_gmd)**2 / (2 * log_gsd**2))
    
    dlogDp = jnp.log(xk[1:] / xk[:-1]) / 3.0 
    Nk = dNdlogDp * dlogDp
    
    Mk = jnp.zeros((nbins, ICOMP))
    Mk = Mk.at[:, SRTSO4].set(Nk * m_mid)
    
    return Nk, Mk, xk

# =========================================================================
# 2. Main Simulation Loop
# =========================================================================

def run_box_model():
    print("="*60)
    print(f"🚀 TOMAS Coagulation Model - Initializing")
    print("="*60)

    # --- A. Setup Environment ---
    boxvol = 1.0e6  # 1 m^3 in cm^3
    temp = 298.0    
    pres = 101325.0 
    
    # --- B. Initialize Particles ---
    # Create two modes (Aitken and Accumulation)
    Nk_1, Mk_1, xk = create_lognormal_dist(NBINS, 2000.0, 2.0e-8, 1.4, 1e-23, 1e-12)
    Nk_2, Mk_2, _  = create_lognormal_dist(NBINS, 500.0, 2.0e-7, 1.6, 1e-23, 1e-12)
    
    Nk_init = (Nk_1 + Nk_2) * boxvol
    Mk_init = (Mk_1 + Mk_2) * boxvol
    
    Nk, Mk = Nk_init, Mk_init

    # --- C. Prepare Solver ---
    print("\n[System] JIT Compiling Solver...")
    solver_jit = jax.jit(diffrax_step, static_argnames=['icomp_nodiag'])
    # Warmup
    _ = solver_jit(Nk, Mk, xk, temp, pres, boxvol, 0.01, ICOMP_NODIAG)
    print("[System] Compilation Complete.")

    # --- D. Time Loop ---
    total_time = 7200.0   # 2 Hours
    dt_model = 60.0       # 1 minute steps
    current_time = 0.0
    
    # Data Collection for Plotting
    history_time = []
    history_Nk = []
    history_N_tot = []
    history_M_tot = []
    
    print(f"\n[Simulation] Starting {int(total_time)}s run...")
    start_time_wall = time.time()

    while current_time < total_time:
        # Store History
        history_time.append(current_time)
        history_Nk.append(Nk) # JAX array, but list append is fine
        history_N_tot.append(jnp.sum(Nk))
        history_M_tot.append(jnp.sum(Mk))

        # Run Solver
        Nk, Mk = solver_jit(
            Nk, Mk, xk, temp, pres, boxvol, dt=dt_model, icomp_nodiag=ICOMP_NODIAG
        )
        current_time += dt_model
    
    # Store final step
    history_time.append(current_time)
    history_Nk.append(Nk)
    history_N_tot.append(jnp.sum(Nk))
    history_M_tot.append(jnp.sum(Mk))

    duration = time.time() - start_time_wall
    print(f"Simulation Complete in {duration:.2f}s")
    
    # =========================================================================
    # 3. Visualization
    # =========================================================================
    print("\nGenerating Plots...")
    
    # 1. Size Distributions (Initial vs Final)
    plotting.plot_size_distribution(
        Nk_init, Nk, Mk_init, Mk, xk, boxvol
    )
    
    # 2. Time Series (Conservation)
    plotting.plot_time_series(
        history_time, history_N_tot, history_M_tot
    )
    
    # 3. Banana Plot
    plotting.plot_banana(
        history_time, history_Nk, xk, boxvol
    )

if __name__ == "__main__":
    run_box_model()