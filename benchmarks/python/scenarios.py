"""Latin Hypercube scenario generator for 24-hour benchmarks.

Generates 50 diverse scenarios spanning a wide range of atmospheric
conditions for comprehensive PPM vs TFL vs Fortran comparison.

Usage::

    python -m benchmarks.python.scenarios
"""
import os
import csv
import numpy as np
from scipy.stats.qmc import LatinHypercube

# Physical constants (match config.py)
MW_H2SO4 = 98.0
AVOGADRO = 6.02214076e23

# Parameter names and ranges: (min, max, scale)
PARAM_NAMES = [
    'N_total', 'GMD_um', 'GSD', 'temp', 'pres', 'RH', 'Gc_SO4_kg', 'h2so4_prod'
]
PARAM_RANGES = {
    'N_total':      (1e2,   1e6,   'log'),     # #/cm3
    'GMD_um':       (0.01,  1.0,   'log'),     # um
    'GSD':          (1.2,   2.5,   'linear'),  # -
    'temp':         (220.0, 320.0, 'linear'),  # K
    'pres':         (20000.0, 101325.0, 'linear'),  # Pa
    'RH':           (0.05,  0.95,  'linear'),  # fraction
    'Gc_SO4_kg':    (1e-16, 1e-10, 'log'),     # kg/cell
    'h2so4_prod':   (1e5,   1e8,   'log'),     # molec/cm3/s
}

# CSV columns for output
CSV_COLUMNS = [
    'scenario_id', 'temp', 'pres', 'N_total', 'GMD_um', 'GSD',
    'Gc_SO4_kg', 'RH', 'h2so4_prod_rate'
]


def molec_cm3_to_kg_cell(conc_molec_cm3, boxvol_cm3=1.0e6):
    """Convert molec/cm3 to kg/grid cell."""
    return conc_molec_cm3 * boxvol_cm3 * (MW_H2SO4 / 1000.0) / AVOGADRO


def generate_scenarios(n=50, seed=42):
    """Generate n Latin Hypercube scenarios.

    Args:
        n: Number of scenarios (default 50).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, each with keys: scenario_id, temp, pres, N_total,
        GMD_um, GSD, Gc_SO4_kg, RH, h2so4_prod, h2so4_prod_kg_per_s
    """
    sampler = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    samples = sampler.random(n=n)  # shape (n, 8), values in [0, 1]

    scenarios = []
    for row_idx in range(n):
        scenario = {'scenario_id': row_idx + 1}
        for col_idx, pname in enumerate(PARAM_NAMES):
            pmin, pmax, scale = PARAM_RANGES[pname]
            u = samples[row_idx, col_idx]
            if scale == 'log':
                log_min = np.log10(pmin)
                log_max = np.log10(pmax)
                scenario[pname] = float(10.0 ** (log_min + u * (log_max - log_min)))
            else:
                scenario[pname] = float(pmin + u * (pmax - pmin))

        # Derived: production rate in kg/cell/s
        scenario['h2so4_prod_kg_per_s'] = molec_cm3_to_kg_cell(scenario['h2so4_prod'])
        scenarios.append(scenario)

    return scenarios


def save_scenarios(scenarios, output_dir=None):
    """Save scenarios to CSV for Fortran harness.

    Args:
        scenarios: List of dicts from generate_scenarios().
        output_dir: Directory for output CSV. Default: benchmarks/fortran/
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'fortran')

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'scenarios.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for s in scenarios:
            writer.writerow([
                s['scenario_id'],
                f"{s['temp']:.16e}",
                f"{s['pres']:.16e}",
                f"{s['N_total']:.16e}",
                f"{s['GMD_um']:.16e}",
                f"{s['GSD']:.16e}",
                f"{s['Gc_SO4_kg']:.16e}",
                f"{s['RH']:.16e}",
                f"{s['h2so4_prod_kg_per_s']:.16e}",
            ])

    print(f"Saved {len(scenarios)} scenarios to {csv_path}")
    return csv_path


def get_scenarios(n=50, seed=42):
    """Generate scenarios and return as list of dicts for Python runner."""
    return generate_scenarios(n=n, seed=seed)


if __name__ == '__main__':
    scenarios = generate_scenarios(n=50, seed=42)
    csv_path = save_scenarios(scenarios)

    print(f"\nGenerated {len(scenarios)} scenarios")
    print(f"\nParameter ranges sampled:")
    for pname in PARAM_NAMES:
        vals = [s[pname] for s in scenarios]
        print(f"  {pname:15s}: [{min(vals):.4e}, {max(vals):.4e}]")
    print(f"\nFirst 3 scenarios:")
    for s in scenarios[:3]:
        print(f"  #{s['scenario_id']:02d}: N={s['N_total']:.1e}, "
              f"GMD={s['GMD_um']:.3f}um, GSD={s['GSD']:.2f}, "
              f"T={s['temp']:.0f}K, P={s['pres']:.0f}Pa, "
              f"RH={s['RH']:.2f}, Gc={s['Gc_SO4_kg']:.1e}, "
              f"prod={s['h2so4_prod']:.1e}")
