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
    SRTSO4, SRTSO2, SRTH2O, SRTNH4,
    MW_H2SO4, MW_SO2, AVOGADRO,
)
from tomas_jax.physics.so2_chemistry import (
    so2_oxidation_step, calc_k1_so2_oh,
    calc_solar_zenith_angle, calc_oh_concentration,
)

from tomas_jax.core.state import TomasState
from tomas_jax.solvers.diffrax import diffrax_step
from tomas_jax.solvers.condensation import condensation_step, make_step
from tomas_jax.physics.nucleation import nucleation_step, zhao2024_nucleation_step, ZHAO2024_ALL_ENABLED
from tomas_jax.physics.dilution import dilute_tracer
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

def run_box_model(enable_condensation: bool = True, method: str = 'ppm_jit',
                   enable_nucleation: bool = True, use_make_step: bool = False,
                   nucl_scheme: str = 'ricco_dunne',
                   so2_init: float = 0.0, so2_emission: float = 0.0,
                   oh_conc: float = 0.0, oh_diurnal: bool = False,
                   lat: float = 45.0, lon: float = 0.0, day_of_year: int = 172,
                   dilution_rate: float = 0.0, dilution_bg: str = 'clean'):
    enable_so2 = so2_init > 0 or so2_emission > 0 or oh_conc > 0
    enable_dilution = dilution_rate > 0
    print("="*60)
    print(f"TOMAS Box Model - Coagulation + Condensation")
    print(f"   Bins: {NBINS}, Components: {ICOMP}")
    print(f"   SO2 Chemistry:{'ON' if enable_so2 else 'OFF'}"
          + (f" ({'diurnal' if oh_diurnal else 'constant'} OH)" if enable_so2 else ""))
    print(f"   Nucleation:   {'ON' if enable_nucleation else 'OFF'}"
          + (f" ({nucl_scheme})" if enable_nucleation else ""))
    print(f"   Condensation: {'ON' if enable_condensation else 'OFF'}")
    print(f"   Dilution:     {'ON' if enable_dilution else 'OFF'}"
          + (f" (kdil={dilution_rate:.1e}, bg={dilution_bg})" if enable_dilution else ""))
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

    # --- Nucleation Parameters ---
    org_conc = jnp.float64(1e7)   # Oxidized organic vapor [molec/cm3]
    nh3_conc = jnp.float64(1e9)   # NH3 [molec/cm3] (~1 ppb)
    fion = jnp.float64(3.0)       # Ion-pair production rate [pairs/cm3/s]
    fn_scale = jnp.float64(1.0)   # Nucleation rate scaling factor

    # Zhao 2024 additional precursors (only used when nucl_scheme='zhao2024')
    hno3_conc = jnp.float64(1e8)   # HNO3 [molec/cm3] (mech 5)
    ulvoc_conc = jnp.float64(5e6)  # ULVOC [molec/cm3] (mechs 6-8)
    dma_conc = jnp.float64(1e7)    # Dimethylamine [molec/cm3] (mech 9)
    hio3_conc = jnp.float64(0.0)   # HIO3 [molec/cm3] (mechs 10-11, coastal only)

    # --- SO2 Chemistry Parameters ---
    if enable_so2:
        # Initialize SO2 gas [molec/cm3 -> kg/grid cell]
        if so2_init > 0:
            so2_kg = so2_init * boxvol * (MW_SO2 / 1000.0) / AVOGADRO
            Gc = Gc.at[SRTSO2].set(so2_kg)
        # SO2 emission rate [molec/cm3/s -> kg/s]
        so2_emission_kg_s = so2_emission * boxvol * (MW_SO2 / 1000.0) / AVOGADRO
        use_diurnal = 1.0 if oh_diurnal else 0.0
    else:
        so2_emission_kg_s = 0.0
        use_diurnal = 0.0

    print(f"\nInitial H2SO4 gas: {h2so4_init_molec_cm3:.1e} molec/cm3")
    print(f"H2SO4 production:  {h2so4_prod_rate_molec_cm3_s:.1e} molec/cm3/s")
    if enable_so2:
        print(f"Initial SO2:       {so2_init:.1e} molec/cm3")
        print(f"SO2 emission:      {so2_emission:.1e} molec/cm3/s")
        print(f"OH concentration:  {oh_conc:.1e} molec/cm3"
              + (" (diurnal)" if oh_diurnal else " (constant)"))
        if oh_diurnal:
            print(f"Location:          lat={lat}N, lon={lon}E, DOY={day_of_year}")
        k1_ref = float(calc_k1_so2_oh(temp, pres, rh))
        tau_days = 1.0 / max(k1_ref * oh_conc, 1e-30) / 86400.0
        print(f"k1(SO2+OH):        {k1_ref:.3e} cm3/molec/s")
        print(f"SO2 lifetime:      {tau_days:.1f} days (at given [OH])")
    if enable_nucleation:
        print(f"Organic vapor:     {float(org_conc):.1e} molec/cm3")
        print(f"NH3:               {float(nh3_conc):.1e} molec/cm3")
        if nucl_scheme == 'zhao2024':
            print(f"HNO3:              {float(hno3_conc):.1e} molec/cm3")
            print(f"ULVOC:             {float(ulvoc_conc):.1e} molec/cm3")
            print(f"DMA:               {float(dma_conc):.1e} molec/cm3")
            print(f"HIO3:              {float(hio3_conc):.1e} molec/cm3")

    # --- Dilution Parameters ---
    if enable_dilution:
        if dilution_bg == 'ambient':
            # Use initial state as background (relaxation to initial conditions)
            Nk_bg = Nk.copy()
            Mk_bg = Mk.copy()
            Gc_bg = Gc.copy()
        else:
            # Clean air (zeros)
            Nk_bg = jnp.zeros_like(Nk)
            Mk_bg = jnp.zeros_like(Mk)
            Gc_bg = jnp.zeros_like(Gc)
        print(f"Dilution rate:     {dilution_rate:.1e} s^-1 "
              f"(tau = {1.0/dilution_rate:.0f} s = {1.0/dilution_rate/3600:.1f} h)")
    else:
        Nk_bg = None
        Mk_bg = None
        Gc_bg = None

    # Calculate initial totals for conservation check
    total_N_init = jnp.sum(Nk)
    total_M_init = jnp.sum(Mk) + jnp.sum(Gc)  # aerosol + gas

    print(f"Initial Total Particles: {total_N_init:.4e}")
    print(f"Initial Aerosol Mass:    {jnp.sum(Mk):.4e} kg")

    # --- D. Prepare Solvers ---
    if use_make_step:
        processes = []
        if enable_so2:
            processes.append('so2_chemistry')
        if enable_nucleation:
            processes.append('nucleation')
        processes.append('coagulation')
        if enable_condensation:
            processes.append('condensation')
        if enable_dilution:
            processes.append('dilution')
        step_fn = make_step(processes, cond_method=method, nucl_scheme=nucl_scheme)
        step_fn_jit = jax.jit(step_fn)
        print(f"\n[System] Using make_step({processes}, nucl_scheme='{nucl_scheme}')")
        # Warmup
        _kw = {}
        if enable_so2:
            _kw.update(oh_conc=jnp.float64(oh_conc))
        if enable_nucleation:
            _kw.update(org_conc=org_conc, nh3_conc=nh3_conc, fion=fion)
            if nucl_scheme == 'zhao2024':
                _kw.update(hno3=hno3_conc, ulvoc=ulvoc_conc,
                           dma=dma_conc, hio3=hio3_conc,
                           enable_masks=ZHAO2024_ALL_ENABLED)
        if enable_dilution:
            _kw.update(kdil=jnp.float64(dilution_rate),
                       Nk_bg=Nk_bg, Mk_bg=Mk_bg, Gc_bg=Gc_bg)
        _ = step_fn_jit(Nk, Mk, Gc, xk,
                        jnp.float64(temp), jnp.float64(pres),
                        jnp.float64(boxvol), jnp.float64(rh),
                        jnp.float64(alpha), jnp.float64(0.01), **_kw)
        print("[System] make_step JIT Compilation Complete.")
    else:
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

    # Passive tracer for dilution reference
    tracer = jnp.float64(1.0)  # starts at 1.0, decays by dilution only

    # Data Collection for Plotting
    history_time = []
    history_Nk = []
    history_N_tot = []
    history_M_tot = []
    history_Gc_so4 = []
    history_dNdt = []
    history_dMdt = []
    history_tracer = []

    while current_time < total_time:
        # Store History
        history_time.append(current_time)
        history_Nk.append(Nk)
        history_N_tot.append(jnp.sum(Nk))
        history_M_tot.append(jnp.sum(Mk))
        history_Gc_so4.append(float(Gc[SRTSO4]))
        history_tracer.append(float(tracer))

        # 1. Add H2SO4 production (constant source) and SO2 emissions
        Gc = Gc.at[SRTSO4].set(Gc[SRTSO4] + h2so4_prod_rate_kg_s * dt_model)
        if enable_so2 and so2_emission_kg_s > 0:
            Gc = Gc.at[SRTSO2].add(so2_emission_kg_s * dt_model)

        if use_make_step:
            # Composable path: single step_fn handles all enabled processes
            kw = {}
            if enable_so2:
                # Compute OH concentration (constant or diurnal)
                if oh_diurnal:
                    hour_utc = (current_time / 3600.0) % 24.0
                    cos_sza = calc_solar_zenith_angle(
                        lat, day_of_year, hour_utc, lon)
                    oh_now = calc_oh_concentration(
                        oh_conc, cos_sza, use_diurnal)
                else:
                    oh_now = jnp.float64(oh_conc)
                kw.update(oh_conc=oh_now)
            if enable_nucleation:
                kw.update(org_conc=org_conc, nh3_conc=nh3_conc, fion=fion)
                if nucl_scheme == 'zhao2024':
                    kw.update(hno3=hno3_conc, ulvoc=ulvoc_conc,
                              dma=dma_conc, hio3=hio3_conc,
                              enable_masks=ZHAO2024_ALL_ENABLED)
            if enable_dilution:
                kw.update(kdil=jnp.float64(dilution_rate),
                          Nk_bg=Nk_bg, Mk_bg=Mk_bg, Gc_bg=Gc_bg)
            Nk, Mk, Gc = step_fn_jit(
                Nk, Mk, Gc, xk,
                jnp.float64(temp), jnp.float64(pres), jnp.float64(boxvol),
                jnp.float64(rh), jnp.float64(alpha), jnp.float64(dt_model),
                **kw,
            )
        else:
            # SO2 chemistry (manual path)
            if enable_so2:
                if oh_diurnal:
                    hour_utc = (current_time / 3600.0) % 24.0
                    cos_sza = calc_solar_zenith_angle(
                        lat, day_of_year, hour_utc, lon)
                    oh_now = calc_oh_concentration(
                        oh_conc, cos_sza, use_diurnal)
                else:
                    oh_now = jnp.float64(oh_conc)
                Gc = so2_oxidation_step(
                    Gc, jnp.float64(temp), jnp.float64(pres),
                    jnp.float64(boxvol), jnp.float64(dt_model),
                    oh_now, jnp.float64(rh))

            # 2. Nucleation (creates particles, depletes gas)
            if enable_nucleation:
                if nucl_scheme == 'zhao2024':
                    Nk, Mk, Gc = zhao2024_nucleation_step(
                        Nk, Mk, Gc, xk,
                        jnp.float64(temp), jnp.float64(pres), jnp.float64(boxvol),
                        jnp.float64(dt_model),
                        org_conc, nh3_conc, fion,
                        hno3=hno3_conc, ulvoc=ulvoc_conc,
                        dma=dma_conc, hio3=hio3_conc,
                        enable_masks=ZHAO2024_ALL_ENABLED,
                        fn_scale=float(fn_scale),
                    )
                else:
                    Nk, Mk, Gc = nucleation_step(
                        Nk, Mk, Gc, xk,
                        jnp.float64(temp), jnp.float64(pres), jnp.float64(boxvol),
                        jnp.float64(dt_model),
                        org_conc, nh3_conc, fion,
                        enable_organic=1.0, enable_inorganic=1.0,
                        fn_scale=float(fn_scale),
                    )

            # 3. Run Coagulation Step
            Nk, Mk = solver_jit(
                Nk, Mk, xk,
                temp, pres, boxvol,
                dt=dt_model,
                icomp_nodiag=ICOMP_NODIAG
            )

            # 4. Run Condensation Step (operator split)
            if enable_condensation:
                Nk, Mk, Gc = condensation_step(
                    Nk, Mk, Gc, xk,
                    temp, pres, boxvol,
                    rh, alpha, dt_model,
                    method=method
                )

        # 5. Update passive tracer (dilution only, no chemistry)
        if enable_dilution:
            tracer = dilute_tracer(tracer, dt_model, dilution_rate)

        # 6. DIAGNOSE: Calculate coagulation rates
        dNdt, dMdt = get_coagulation_rates(
            Nk, Mk, xk, temp, pres, boxvol, ICOMP_NODIAG
        )
        history_dNdt.append(np.array(dNdt))
        history_dMdt.append(np.array(dMdt))

        # 7. Advance Time
        current_time += dt_model
        step_count += 1

        # 8. Logging
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
    history_tracer.append(float(tracer))

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

    if enable_dilution:
        print(f"\n[Passive Tracer]")
        print(f"Tracer (t=0):       1.0000")
        print(f"Tracer (t=final):   {float(tracer):.4f}")
        expected_tracer = np.exp(-dilution_rate * total_time)
        print(f"Expected exp(-k*t): {expected_tracer:.4f}")
        N_ratio = float(total_N_final) / float(total_N_init)
        print(f"N_total ratio:      {N_ratio:.4f}  (tracer={float(tracer):.4f})")

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
            history_Gc_so4, history_dNdt, history_dMdt,
            history_tracer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TOMAS Box Model")
    parser.add_argument('--method', choices=['tfl', 'tfl_jit', 'ppm', 'ppm_jit'], default='ppm_jit',
                        help='Condensation method: ppm_jit (default), tfl_jit, tfl, or ppm')
    parser.add_argument('--no-condensation', action='store_true',
                        help='Disable condensation')
    parser.add_argument('--no-nucleation', action='store_true',
                        help='Disable nucleation')
    parser.add_argument('--make-step', action='store_true',
                        help='Use make_step() composable API instead of manual operator splitting')
    parser.add_argument('--nucl-scheme', choices=['ricco_dunne', 'zhao2024'],
                        default='ricco_dunne',
                        help='Nucleation scheme: ricco_dunne (Riccobono+Dunne, default) '
                             'or zhao2024 (11-mechanism Zhao et al. 2024)')
    # SO2 chemistry
    parser.add_argument('--so2-init', type=float, default=0.0,
                        help='Initial SO2 concentration [molec/cm3] (e.g. 5e10)')
    parser.add_argument('--so2-emission', type=float, default=0.0,
                        help='SO2 emission rate [molec/cm3/s] (e.g. 1e7)')
    parser.add_argument('--oh-conc', type=float, default=0.0,
                        help='OH concentration [molec/cm3] (e.g. 1e6)')
    parser.add_argument('--oh-diurnal', action='store_true',
                        help='Use diurnal OH cycle (proportional to cos(SZA))')
    parser.add_argument('--lat', type=float, default=45.0,
                        help='Latitude [degrees N] for diurnal cycle (default: 45)')
    parser.add_argument('--lon', type=float, default=0.0,
                        help='Longitude [degrees E] for diurnal cycle (default: 0)')
    parser.add_argument('--day-of-year', type=int, default=172,
                        help='Day of year for diurnal cycle (default: 172, summer solstice)')
    # Dilution
    parser.add_argument('--dilution-rate', type=float, default=0.0,
                        help='Dilution rate [s^-1] (e.g. 1e-4 for BL growth)')
    parser.add_argument('--dilution-bg', choices=['clean', 'ambient'], default='clean',
                        help='Dilution background: clean (zeros) or ambient (initial state)')
    args = parser.parse_args()
    results = run_box_model(
        enable_condensation=not args.no_condensation,
        method=args.method,
        enable_nucleation=not args.no_nucleation,
        use_make_step=args.make_step,
        nucl_scheme=args.nucl_scheme,
        so2_init=args.so2_init,
        so2_emission=args.so2_emission,
        oh_conc=args.oh_conc,
        oh_diurnal=args.oh_diurnal,
        lat=args.lat,
        lon=args.lon,
        day_of_year=args.day_of_year,
        dilution_rate=args.dilution_rate,
        dilution_bg=args.dilution_bg,
    )
