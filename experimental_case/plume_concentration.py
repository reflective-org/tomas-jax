"""Aircraft SO2 release — initial plume concentration calculator.

Computes the initial SO2 concentration inside an aircraft exhaust plume
modeled as a cylinder (cross-section = circle of given radius, length
per unit time = aircraft speed).

Usage::

    # Default scenario (10 tonnes, 1 ton/min, 250 m/s, r=100m, stratosphere)
    python experimental_case/plume_concentration.py

    # Custom scenario
    python experimental_case/plume_concentration.py \\
        --total-so2 5 --release-rate 0.5 --speed 200 --radius 50 \\
        --temp 220 --pressure 7000

    # As a library
    from experimental_case.plume_concentration import plume_so2_concentration
    result = plume_so2_concentration(
        release_rate_ton_per_min=1.0,
        aircraft_speed_m_per_s=250.0,
        plume_radius_m=100.0,
        temp_K=216.0,
        pres_Pa=5500.0,
    )
    print(result['ug_per_m3'], result['ppm'], result['molec_per_cm3'])
"""

import numpy as np

# =========================================================================
# Physical constants
# =========================================================================
AVOGADRO = 6.02214076e23   # molec/mol
KB = 1.380649e-23          # J/K  (Boltzmann)
MW_SO2 = 64.066            # g/mol


def plume_so2_concentration(release_rate_ton_per_min, aircraft_speed_m_per_s,
                            plume_radius_m, temp_K, pres_Pa,
                            total_so2_ton=None):
    """Compute initial SO2 concentration in an aircraft plume.

    Parameters
    ----------
    release_rate_ton_per_min : float
        SO2 release rate [metric tonnes per minute].
    aircraft_speed_m_per_s : float
        Aircraft ground speed [m/s].
    plume_radius_m : float
        Plume cross-section radius after initial mixing [m].
    temp_K : float
        Ambient temperature [K].
    pres_Pa : float
        Ambient pressure [Pa].
    total_so2_ton : float, optional
        Total SO2 released [metric tonnes].  Used only for plume geometry
        (duration, length, volume).  Not needed for concentration.

    Returns
    -------
    dict with keys:
        ug_per_m3      : float — mass concentration [µg/m³]
        molec_per_cm3  : float — number concentration [molec/cm³]
        ppm            : float — volume mixing ratio [ppm]
        ppb            : float — volume mixing ratio [ppb]
        kg_per_m3      : float — mass concentration [kg/m³]
        n_air_per_cm3  : float — air number density [molec/cm³]
        plume_area_m2  : float — plume cross-sectional area [m²]
        mass_per_m_kg  : float — SO2 mass per meter of flight path [kg/m]
        release_rate_kg_per_s : float — release rate [kg/s]
        duration_min   : float or None — release duration [min]
        plume_length_km: float or None — total plume length [km]
        plume_volume_m3: float or None — total plume volume [m³]
    """
    # Release rate in SI
    release_rate_kg_per_s = release_rate_ton_per_min * 1e3 / 60.0

    # Plume geometry
    plume_area_m2 = np.pi * plume_radius_m**2
    mass_per_m_kg = release_rate_kg_per_s / aircraft_speed_m_per_s

    # Mass concentration
    conc_kg_per_m3 = mass_per_m_kg / plume_area_m2
    conc_ug_per_m3 = conc_kg_per_m3 * 1e9

    # Number concentration (molec/cm³)
    conc_g_per_m3 = conc_kg_per_m3 * 1e3
    conc_molec_per_cm3 = (conc_g_per_m3 / MW_SO2) * AVOGADRO * 1e-6

    # Mixing ratio (ppm, ppb)
    n_air_per_cm3 = (pres_Pa / (KB * temp_K)) * 1e-6
    mixing_ratio = conc_molec_per_cm3 / n_air_per_cm3
    conc_ppm = mixing_ratio * 1e6
    conc_ppb = mixing_ratio * 1e9

    # Optional plume geometry from total release
    duration_min = None
    plume_length_km = None
    plume_volume_m3 = None
    if total_so2_ton is not None:
        duration_min = total_so2_ton / release_rate_ton_per_min
        plume_length_km = aircraft_speed_m_per_s * duration_min * 60.0 / 1e3
        plume_volume_m3 = plume_area_m2 * plume_length_km * 1e3

    return {
        'ug_per_m3': conc_ug_per_m3,
        'molec_per_cm3': conc_molec_per_cm3,
        'ppm': conc_ppm,
        'ppb': conc_ppb,
        'kg_per_m3': conc_kg_per_m3,
        'n_air_per_cm3': n_air_per_cm3,
        'plume_area_m2': plume_area_m2,
        'mass_per_m_kg': mass_per_m_kg,
        'release_rate_kg_per_s': release_rate_kg_per_s,
        'duration_min': duration_min,
        'plume_length_km': plume_length_km,
        'plume_volume_m3': plume_volume_m3,
    }


def print_result(result, release_rate_ton_per_min, aircraft_speed_m_per_s,
                 plume_radius_m, temp_K, pres_Pa, total_so2_ton=None):
    """Pretty-print plume concentration results."""
    r = result
    print("=" * 64)
    print("Aircraft SO2 Release — Initial Plume Concentration")
    print("=" * 64)

    print("\n--- Input Parameters ---")
    print(f"  Release rate       : {release_rate_ton_per_min:.2f} ton/min "
          f"= {r['release_rate_kg_per_s']:.2f} kg/s")
    if total_so2_ton is not None:
        print(f"  Total SO2          : {total_so2_ton:.1f} tonnes")
    print(f"  Aircraft speed     : {aircraft_speed_m_per_s:.0f} m/s")
    print(f"  Plume radius       : {plume_radius_m:.0f} m")
    print(f"  Plume cross-section: {r['plume_area_m2']:,.0f} m²")

    if r['duration_min'] is not None:
        print(f"\n  Release duration   : {r['duration_min']:.1f} min")
        print(f"  Plume length       : {r['plume_length_km']:.0f} km")
        print(f"  Plume volume       : {r['plume_volume_m3']:.2e} m³")

    print("\n--- Ambient Conditions ---")
    print(f"  Temperature        : {temp_K:.0f} K")
    print(f"  Pressure           : {pres_Pa:.0f} Pa ({pres_Pa/100:.1f} hPa)")
    print(f"  Air number density : {r['n_air_per_cm3']:.3e} molec/cm³")

    print(f"\n--- SO2 per meter of flight path ---")
    print(f"  {r['mass_per_m_kg']:.4f} kg/m = {r['mass_per_m_kg']*1e3:.2f} g/m")

    print("\n" + "=" * 64)
    print("  INITIAL SO2 CONCENTRATION")
    print("=" * 64)
    print(f"  {r['ug_per_m3']:,.0f} µg/m³")
    print(f"  {r['molec_per_cm3']:.3e} molec/cm³")
    print(f"  {r['ppm']:.2f} ppm")
    print(f"  {r['ppb']:,.0f} ppb")
    print("=" * 64)


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate initial SO2 concentration in an aircraft plume')
    parser.add_argument('--total-so2', type=float, default=10.0,
                        help='Total SO2 released [tonnes] (default: 10)')
    parser.add_argument('--release-rate', type=float, default=1.0,
                        help='Release rate [ton/min] (default: 1.0)')
    parser.add_argument('--speed', type=float, default=250.0,
                        help='Aircraft speed [m/s] (default: 250)')
    parser.add_argument('--radius', type=float, default=100.0,
                        help='Plume radius [m] (default: 100)')
    parser.add_argument('--temp', type=float, default=216.0,
                        help='Ambient temperature [K] (default: 216)')
    parser.add_argument('--pressure', type=float, default=5500.0,
                        help='Ambient pressure [Pa] (default: 5500)')
    args = parser.parse_args()

    result = plume_so2_concentration(
        release_rate_ton_per_min=args.release_rate,
        aircraft_speed_m_per_s=args.speed,
        plume_radius_m=args.radius,
        temp_K=args.temp,
        pres_Pa=args.pressure,
        total_so2_ton=args.total_so2,
    )

    print_result(result, args.release_rate, args.speed, args.radius,
                 args.temp, args.pressure, args.total_so2)
