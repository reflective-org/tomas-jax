"""
TOMAS Coagulation + Condensation Box Model Driver.

This script acts as the entry point for the simulation.
It demonstrates:
1. Initialization of state (Log-normal distribution + gas phase).
2. JIT-compilation of the coagulation solver.
3. Time-stepping loop with operator splitting (coagulation + condensation).
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
from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, SRTNH4,
    MW_H2SO4, AVOGADRO
)

from tomas_jax.core.state import TomasState
from tomas_jax.solvers.diffrax import diffrax_step
from tomas_jax.solvers.condensation import condensation_step
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
    log_dp = jnp.log(dp_mid)
    log_gmd = jnp.log(gmd)
    log_gsd = jnp.log(gsd)

    dNdlogDp = (n_total / (jnp.sqrt(2 * np.pi) * log_gsd)) * \
               jnp.exp(-(log_dp - log_gmd)**2 / (2 * log_gsd**2))

    # 4. Integrate to get Number per Bin (Nk)
    dlogDp = jnp.log(xk[1:] / xk[:-1]) / 3.0 # mass ~ D^3, so dlogm = 3*dlogD

    Nk = dNdlogDp * dlogDp

    # 5. Calculate Mass per Bin (Mk)
    Mk = jnp.zeros((nbins, ICOMP))
    Mk = Mk.at[:, SRTSO4].set(Nk * m_mid)

    return Nk, Mk, xk


def molec_cm3_to_kg_gridcell(conc_molec_cm3: float, boxvol_cm3: float) -> float:
    """Convert molecular concentration [molec/cm3] to [kg/grid cell].

    Args:
        conc_molec_cm3: Concentration in molecules/cm3
        boxvol_cm3: Grid cell volume in cm3

    Returns:
        Mass in kg/grid cell
    """
    return conc_molec_cm3 * boxvol_cm3 * (MW_H2SO4 / 1000.0) / AVOGADRO


# =========================================================================
# 2. Main Simulation Loop
# =========================================================================

def run_box_model(enable_condensation: bool = True, method: str = 'tfl'):
    print("="*60)
    print(f"TOMAS Box Model - Coagulation + Condensation")
    print(f"   Bins: {NBINS}, Components: {ICOMP}")
    print(f"   Condensation: {'ON' if enable_condensation else 'OFF'}")
    print(f"   Cond. Method: {method.upper()}")
    print(f"   Precision: {'float64' if jax.config.jax_enable_x64 else 'float32'}")
    print("="*60)

    # --- A. Setup Environment ---
    boxvol = 1.0e6  # 1 m^3 in cm^3 (standard box size)
    temp = 298.0    # Kelvin
    pres = 101325.0 # Pascals
    rh = 0.5        # 50% relative humidity
    alpha = 1.0     # Accommodation coefficient

    # --- B. Initialize Particles ---
    Nk_init, Mk_init, xk = create_lognormal_dist(
        nbins=NBINS,
        n_total=100000.0,       # 100,000 particles/cm3
        gmd=1.0e-7,           # 100 nm geometric mean diameter
        gsd=1.6,              # Standard width
        mass_min=1e-23,       # ~2 nm
        mass_max=1e-12        # ~10 um
    )

    # Scale inputs by box volume
    Nk = Nk_init * boxvol
    Mk = Mk_init * boxvol

    # --- C. Initialize Gas Phase ---
    Gc = jnp.zeros(N_GAS_SPECIES)

    # H2SO4 initial concentration: 1e7 molec/cm3 (typical atmospheric)
    h2so4_init_molec_cm3 = 1.0e7
    Gc = Gc.at[SRTSO4].set(molec_cm3_to_kg_gridcell(h2so4_init_molec_cm3, boxvol))

    # H2SO4 production rate: ~1e7 molec/cm3/s (typical for polluted boundary layer)
    h2so4_prod_rate_molec_cm3_s = 1.0e7
    h2so4_prod_rate_kg_s = molec_cm3_to_kg_gridcell(h2so4_prod_rate_molec_cm3_s, boxvol)

    print(f"\nInitial H2SO4 gas: {h2so4_init_molec_cm3:.1e} molec/cm3")
    print(f"H2SO4 production:  {h2so4_prod_rate_molec_cm3_s:.1e} molec/cm3/s")

    # Calculate initial totals for conservation check
    total_N_init = jnp.sum(Nk)
    total_M_init = jnp.sum(Mk) + jnp.sum(Gc)  # aerosol + gas

    print(f"Initial Total Particles: {total_N_init:.4e}")
    print(f"Initial Aerosol Mass:    {jnp.sum(Mk):.4e} kg")

    # --- D. Prepare Coagulation Solver ---
    print("\n[System] JIT Compiling Coagulation Solver... (this happens once)")

    solver_jit = jax.jit(
        diffrax_step,
        static_argnames=['icomp_nodiag']
    )

    # Trigger compilation
    _ = solver_jit(Nk, Mk, xk, temp, pres, boxvol, 0.01, ICOMP_NODIAG)
    print("[System] Compilation Complete.")

    # --- E. Time Loop ---
    total_time = 3600.0 * 24   # 24 hour simulation
    dt_model = 60.0            # 60 second timesteps

    current_time = 0.0
    step_count = 0

    start_time_wall = time.time()

    print(f"\n[Simulation] Starting {int(total_time)}s run (dt={dt_model}s)...")
    print(f"{'Time [s]':<10} | {'Particles [#]':<15} | {'Aero Mass [kg]':<15} | {'H2SO4 gas [kg]':<15}")
    print("-" * 70)

    # Data Collection for Plotting
    history_time = []
    history_Nk = []
    history_N_tot = []
    history_M_tot = []
    history_Gc_so4 = []
    history_dNdt = []
    history_dMdt = []

    while current_time < total_time:
        # Store History
        history_time.append(current_time)
        history_Nk.append(Nk)
        history_N_tot.append(jnp.sum(Nk))
        history_M_tot.append(jnp.sum(Mk))
        history_Gc_so4.append(float(Gc[SRTSO4]))

        # 1. Add H2SO4 production (constant source)
        Gc = Gc.at[SRTSO4].set(Gc[SRTSO4] + h2so4_prod_rate_kg_s * dt_model)

        # 2. Run Coagulation Step
        Nk, Mk = solver_jit(
            Nk, Mk, xk,
            temp, pres, boxvol,
            dt=dt_model,
            icomp_nodiag=ICOMP_NODIAG
        )

        # 3. Run Condensation Step (operator split)
        if enable_condensation:
            Nk, Mk, Gc = condensation_step(
                Nk, Mk, Gc, xk,
                temp, pres, boxvol,
                rh, alpha, dt_model,
                method=method
            )

        # 4. DIAGNOSE: Calculate coagulation rates
        dNdt, dMdt = get_coagulation_rates(
            Nk, Mk, xk, temp, pres, boxvol, ICOMP_NODIAG
        )
        history_dNdt.append(np.array(dNdt))
        history_dMdt.append(np.array(dMdt))

        # 5. Advance Time
        current_time += dt_model
        step_count += 1

        # 6. Logging
        total_N = jnp.sum(Nk)
        total_M = jnp.sum(Mk)

        if step_count % 60 == 0 or step_count == 1:
            print(f"{current_time:<10.1f} | {total_N:<15.4e} | {total_M:<15.4e} | {float(Gc[SRTSO4]):<15.4e}")

    end_time_wall = time.time()

    # Store final step
    history_time.append(current_time)
    history_Nk.append(Nk)
    history_N_tot.append(jnp.sum(Nk))
    history_M_tot.append(jnp.sum(Mk))
    history_Gc_so4.append(float(Gc[SRTSO4]))

    # =========================================================================
    # 3. Validation & Results
    # =========================================================================
    print("-" * 70)
    print("\n[Results]")

    total_N_final = jnp.sum(Nk)
    total_M_final_aero = jnp.sum(Mk)
    total_M_final = total_M_final_aero + jnp.sum(Gc)

    # Total mass added by H2SO4 production
    total_h2so4_produced = h2so4_prod_rate_kg_s * total_time
    expected_total_mass = float(total_M_init) + total_h2so4_produced

    mass_error = (float(total_M_final) - expected_total_mass) / expected_total_mass

    print(f"Final Total Particles:  {total_N_final:.4e}")
    print(f"Final Aerosol Mass:     {total_M_final_aero:.4e} kg")
    print(f"Final Gas H2SO4:        {float(Gc[SRTSO4]):.4e} kg")
    print(f"Particles Lost:         {total_N_init - total_N_final:.4e} (Coagulation)")
    print(f"H2SO4 Produced:         {total_h2so4_produced:.4e} kg")
    print(f"Mass Error (Rel):       {mass_error:.4e}")

    if abs(mass_error) < 1e-6:
        print("Mass Conservation: GOOD")
    elif abs(mass_error) < 1e-3:
        print("Mass Conservation: ACCEPTABLE")
    else:
        print("Mass Conservation: CHECK NEEDED")

    # Performance
    duration = end_time_wall - start_time_wall
    steps = total_time / dt_model
    print(f"\n[Performance]")
    print(f"Total Wall Time:     {duration:.4f} seconds")
    print(f"Time per Model Step: {duration/steps*1000:.2f} ms")

    # ASCII Histogram
    print("\n[Final Size Distribution]")
    max_N = jnp.max(Nk)
    for k in range(NBINS):
        bar_len = int((Nk[k] / max_N) * 40)
        bar = "#" * bar_len
        if k % 4 == 0:
            print(f"Bin {k:02d} ({xk[k]:.1e} kg): {bar}")

    # =========================================================================
    # 4. Visualization
    # =========================================================================
    print("\nGenerating Plots...")

    # 1. Size Distributions (Initial vs Final)
    plotting.plot_size_distribution(
        Nk_init * boxvol, Nk, Mk_init * boxvol, Mk, xk, boxvol
    )

    # 2. Time Series (Conservation)
    plotting.plot_time_series(
        history_time, history_N_tot, history_M_tot
    )

    # 3. Banana Plot
    plotting.plot_banana(
        history_time, history_Nk, xk, boxvol
    )

    # 4. Coagulation Rates
    plotting.plot_rates(xk, history_dNdt[-1], boxvol)
    plotting.plot_rates(xk, history_dMdt[-1], boxvol)

    return (Nk_init, Nk, Mk_init, Mk, xk, boxvol, Gc,
            history_time, history_N_tot, history_M_tot,
            history_Gc_so4, history_dNdt, history_dMdt)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TOMAS Box Model")
    parser.add_argument('--method', choices=['tfl', 'ppm', 'ppm_jit'], default='tfl',
                        help='Condensation method: tfl (default), ppm, or ppm_jit')
    parser.add_argument('--no-condensation', action='store_true',
                        help='Disable condensation')
    args = parser.parse_args()
    results = run_box_model(
        enable_condensation=not args.no_condensation,
        method=args.method
    )
