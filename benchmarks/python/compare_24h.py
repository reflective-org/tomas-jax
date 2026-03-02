"""3-way comparison engine for 24-hour benchmark results.

Loads Fortran CSV + JAX NPZ results and computes comparison metrics
for all 50 scenarios across coag, cond, and combined modes.

Usage::

    python -m benchmarks.python.compare_24h
    python -m benchmarks.python.compare_24h --scenario 1
"""
import os
import sys
import argparse
import numpy as np

from benchmarks.python.scenarios import get_scenarios
from benchmarks.python.utils import load_csv

NBINS = 36
ICOMP = 44
N_GAS = 43
SRTH2O = 43   # Water index — excluded from dry mass
NHOURS = 24

# Directories
_BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
FORTRAN_DIR = os.path.join(_BASE_DIR, 'fortran', 'output', '24h')
RESULTS_DIR = os.path.join(_BASE_DIR, 'results', '24h')

MODE_NAMES = {
    'coag_only': 'coag',
    'cond_only': 'cond',
    'combined': 'combined',
}


def _safe_rel_error(a, b, atol=1e-30):
    """Compute max relative error between two arrays, ignoring near-zero values."""
    denom = np.maximum(np.abs(a), atol)
    rel_err = np.abs(a - b) / denom
    return float(np.max(rel_err))


def _total_rel_error(val_a, val_b):
    """Relative error between two scalars."""
    if abs(val_a) < 1e-30 and abs(val_b) < 1e-30:
        return 0.0
    denom = max(abs(val_a), 1e-30)
    return abs(val_a - val_b) / denom


def load_fortran_hourly(scenario_id, mode_name, hour):
    """Load one hourly snapshot from Fortran CSV output.

    Args:
        scenario_id: 1-based scenario ID
        mode_name: 'coag', 'cond', or 'combined'
        hour: 1-based hour (1-24)

    Returns:
        (Nk, Mk, Gc) numpy arrays
    """
    prefix = f"s{scenario_id:02d}_{mode_name}_hour{hour:02d}"

    nk_path = os.path.join(FORTRAN_DIR, f"{prefix}_Nk.csv")
    mk_path = os.path.join(FORTRAN_DIR, f"{prefix}_Mk.csv")
    gc_path = os.path.join(FORTRAN_DIR, f"{prefix}_Gc.csv")

    Nk = load_csv(nk_path).flatten()
    Mk = load_csv(mk_path).reshape(NBINS, ICOMP)
    Gc = load_csv(gc_path).flatten()

    return Nk, Mk, Gc


def load_jax_results(scenario_id, mode, method):
    """Load JAX NPZ results for a scenario/mode/method.

    Returns:
        dict with 'Nk' (24,36), 'Mk' (24,36,44), 'Gc' (24,43),
        'N_tot' (24,), 'M_tot' (24,)
    """
    label = f"s{scenario_id:02d}_{mode}_{method}"
    npz_path = os.path.join(RESULTS_DIR, f"{label}.npz")

    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    Mk = data['Mk']

    # M_dry: fallback to recomputing from Mk if not saved
    if 'M_dry' in data:
        M_dry = data['M_dry']
    else:
        M_dry = np.sum(Mk[:, :, :SRTH2O], axis=(1, 2))

    return {
        'Nk': data['Nk'],
        'Mk': Mk,
        'Gc': data['Gc'],
        'N_tot': data['N_tot'],
        'M_tot': data['M_tot'],
        'M_dry': M_dry,
    }


def compare_scenario(scenario_id, mode):
    """3-way comparison: Fortran vs TFL vs PPM at each hourly snapshot.

    Args:
        scenario_id: 1-based scenario ID
        mode: 'coag_only', 'cond_only', or 'combined'

    Returns:
        dict with per-hour metrics, or None if data missing
    """
    fort_mode = MODE_NAMES[mode]

    # Load JAX results
    tfl_method = 'tfl'
    ppm_method = 'ppm'

    tfl_data = load_jax_results(scenario_id, mode, tfl_method)
    ppm_data = load_jax_results(scenario_id, mode, ppm_method)

    # For coag_only, both methods are identical (no condensation)
    if mode == 'coag_only':
        ppm_data = tfl_data

    # Check Fortran data exists
    fort_exists = os.path.exists(
        os.path.join(FORTRAN_DIR,
                     f"s{scenario_id:02d}_{fort_mode}_hour01_Nk.csv")
    )

    metrics = {
        'scenario_id': scenario_id,
        'mode': mode,
        'has_fortran': fort_exists,
        'has_tfl': tfl_data is not None,
        'has_ppm': ppm_data is not None,
    }

    # Per-hour arrays
    tfl_vs_fortran_Nk_relerr = np.full(NHOURS, np.nan)
    ppm_vs_fortran_Nk_relerr = np.full(NHOURS, np.nan)
    ppm_vs_tfl_Nk_relerr = np.full(NHOURS, np.nan)
    tfl_vs_fortran_Mk_relerr = np.full(NHOURS, np.nan)
    ppm_vs_fortran_Mk_relerr = np.full(NHOURS, np.nan)
    ppm_vs_tfl_Mk_relerr = np.full(NHOURS, np.nan)
    N_tot_fortran = np.full(NHOURS, np.nan)
    N_tot_tfl = np.full(NHOURS, np.nan)
    N_tot_ppm = np.full(NHOURS, np.nan)
    M_tot_fortran = np.full(NHOURS, np.nan)
    M_tot_tfl = np.full(NHOURS, np.nan)
    M_tot_ppm = np.full(NHOURS, np.nan)
    M_dry_fortran = np.full(NHOURS, np.nan)
    M_dry_tfl = np.full(NHOURS, np.nan)
    M_dry_ppm = np.full(NHOURS, np.nan)
    tfl_vs_fortran_N_tot_relerr = np.full(NHOURS, np.nan)
    ppm_vs_fortran_N_tot_relerr = np.full(NHOURS, np.nan)
    ppm_vs_tfl_N_tot_relerr = np.full(NHOURS, np.nan)
    tfl_vs_fortran_M_tot_relerr = np.full(NHOURS, np.nan)
    ppm_vs_fortran_M_tot_relerr = np.full(NHOURS, np.nan)
    ppm_vs_tfl_M_tot_relerr = np.full(NHOURS, np.nan)

    for ihour in range(NHOURS):
        hour = ihour + 1  # 1-based for Fortran files

        # Fortran data
        if fort_exists:
            try:
                Nk_f, Mk_f, Gc_f = load_fortran_hourly(
                    scenario_id, fort_mode, hour)
                N_tot_fortran[ihour] = np.sum(Nk_f)
                M_tot_fortran[ihour] = np.sum(Mk_f)
                M_dry_fortran[ihour] = np.sum(Mk_f[:, :SRTH2O])
            except Exception:
                Nk_f = Mk_f = None

        # TFL data
        if tfl_data is not None:
            Nk_tfl = tfl_data['Nk'][ihour]
            Mk_tfl = tfl_data['Mk'][ihour]
            N_tot_tfl[ihour] = tfl_data['N_tot'][ihour]
            M_tot_tfl[ihour] = tfl_data['M_tot'][ihour]
            M_dry_tfl[ihour] = tfl_data['M_dry'][ihour]
        else:
            Nk_tfl = Mk_tfl = None

        # PPM data
        if ppm_data is not None:
            Nk_ppm = ppm_data['Nk'][ihour]
            Mk_ppm = ppm_data['Mk'][ihour]
            N_tot_ppm[ihour] = ppm_data['N_tot'][ihour]
            M_tot_ppm[ihour] = ppm_data['M_tot'][ihour]
            M_dry_ppm[ihour] = ppm_data['M_dry'][ihour]
        else:
            Nk_ppm = Mk_ppm = None

        # TFL vs Fortran
        if fort_exists and Nk_f is not None and Nk_tfl is not None:
            tfl_vs_fortran_Nk_relerr[ihour] = _safe_rel_error(Nk_f, Nk_tfl)
            tfl_vs_fortran_Mk_relerr[ihour] = _safe_rel_error(Mk_f, Mk_tfl)
            tfl_vs_fortran_N_tot_relerr[ihour] = _total_rel_error(
                N_tot_fortran[ihour], N_tot_tfl[ihour])
            tfl_vs_fortran_M_tot_relerr[ihour] = _total_rel_error(
                M_tot_fortran[ihour], M_tot_tfl[ihour])

        # PPM vs Fortran
        if fort_exists and Nk_f is not None and Nk_ppm is not None:
            ppm_vs_fortran_Nk_relerr[ihour] = _safe_rel_error(Nk_f, Nk_ppm)
            ppm_vs_fortran_Mk_relerr[ihour] = _safe_rel_error(Mk_f, Mk_ppm)
            ppm_vs_fortran_N_tot_relerr[ihour] = _total_rel_error(
                N_tot_fortran[ihour], N_tot_ppm[ihour])
            ppm_vs_fortran_M_tot_relerr[ihour] = _total_rel_error(
                M_tot_fortran[ihour], M_tot_ppm[ihour])

        # PPM vs TFL
        if Nk_tfl is not None and Nk_ppm is not None:
            ppm_vs_tfl_Nk_relerr[ihour] = _safe_rel_error(Nk_tfl, Nk_ppm)
            ppm_vs_tfl_Mk_relerr[ihour] = _safe_rel_error(Mk_tfl, Mk_ppm)
            ppm_vs_tfl_N_tot_relerr[ihour] = _total_rel_error(
                N_tot_tfl[ihour], N_tot_ppm[ihour])
            ppm_vs_tfl_M_tot_relerr[ihour] = _total_rel_error(
                M_tot_tfl[ihour], M_tot_ppm[ihour])

    metrics.update({
        'tfl_vs_fortran_Nk_relerr': tfl_vs_fortran_Nk_relerr,
        'ppm_vs_fortran_Nk_relerr': ppm_vs_fortran_Nk_relerr,
        'ppm_vs_tfl_Nk_relerr': ppm_vs_tfl_Nk_relerr,
        'tfl_vs_fortran_Mk_relerr': tfl_vs_fortran_Mk_relerr,
        'ppm_vs_fortran_Mk_relerr': ppm_vs_fortran_Mk_relerr,
        'ppm_vs_tfl_Mk_relerr': ppm_vs_tfl_Mk_relerr,
        'N_tot_fortran': N_tot_fortran,
        'N_tot_tfl': N_tot_tfl,
        'N_tot_ppm': N_tot_ppm,
        'M_tot_fortran': M_tot_fortran,
        'M_tot_tfl': M_tot_tfl,
        'M_tot_ppm': M_tot_ppm,
        'M_dry_fortran': M_dry_fortran,
        'M_dry_tfl': M_dry_tfl,
        'M_dry_ppm': M_dry_ppm,
        'tfl_vs_fortran_N_tot_relerr': tfl_vs_fortran_N_tot_relerr,
        'ppm_vs_fortran_N_tot_relerr': ppm_vs_fortran_N_tot_relerr,
        'ppm_vs_tfl_N_tot_relerr': ppm_vs_tfl_N_tot_relerr,
        'tfl_vs_fortran_M_tot_relerr': tfl_vs_fortran_M_tot_relerr,
        'ppm_vs_fortran_M_tot_relerr': ppm_vs_fortran_M_tot_relerr,
        'ppm_vs_tfl_M_tot_relerr': ppm_vs_tfl_M_tot_relerr,
    })

    return metrics


def compare_all_scenarios(scenario_ids=None, modes=None):
    """Run comparisons for all scenarios and modes.

    Returns:
        List of metrics dicts.
    """
    if scenario_ids is None:
        scenario_ids = list(range(1, 51))
    if modes is None:
        modes = ['coag_only', 'cond_only', 'combined']

    all_metrics = []
    for sid in scenario_ids:
        for mode in modes:
            metrics = compare_scenario(sid, mode)
            all_metrics.append(metrics)
    return all_metrics


def print_summary(all_metrics):
    """Print a summary table of comparison results."""
    print("\n" + "=" * 80)
    print("24-Hour Benchmark Comparison Summary")
    print("=" * 80)

    for mode in ['coag_only', 'cond_only', 'combined']:
        mode_metrics = [m for m in all_metrics if m['mode'] == mode]
        if not mode_metrics:
            continue

        print(f"\n--- {mode.upper()} ---")
        print(f"{'Scen':>4s}  {'TFL-Fort N':>12s}  {'PPM-Fort N':>12s}  "
              f"{'PPM-TFL N':>12s}  {'TFL-Fort M':>12s}  {'PPM-TFL M':>12s}")

        for m in mode_metrics:
            sid = m['scenario_id']
            # Hour 24 values (index 23)
            tfn = m['tfl_vs_fortran_N_tot_relerr'][23]
            pfn = m['ppm_vs_fortran_N_tot_relerr'][23]
            ptn = m['ppm_vs_tfl_N_tot_relerr'][23]
            tfm = m['tfl_vs_fortran_M_tot_relerr'][23]
            ptm = m['ppm_vs_tfl_M_tot_relerr'][23]
            print(f"{sid:4d}  {tfn:12.4e}  {pfn:12.4e}  "
                  f"{ptn:12.4e}  {tfm:12.4e}  {ptm:12.4e}")


def save_comparison_npz(all_metrics, output_dir=None):
    """Save all comparison metrics as a single NPZ file."""
    if output_dir is None:
        output_dir = os.path.join(_BASE_DIR, 'results', '24h')
    os.makedirs(output_dir, exist_ok=True)

    # Organize by mode
    for mode in ['coag_only', 'cond_only', 'combined']:
        mode_metrics = [m for m in all_metrics if m['mode'] == mode]
        if not mode_metrics:
            continue

        n = len(mode_metrics)
        data = {}
        for key in ['tfl_vs_fortran_N_tot_relerr', 'ppm_vs_fortran_N_tot_relerr',
                     'ppm_vs_tfl_N_tot_relerr', 'tfl_vs_fortran_M_tot_relerr',
                     'ppm_vs_tfl_M_tot_relerr', 'N_tot_fortran', 'N_tot_tfl',
                     'N_tot_ppm', 'M_tot_fortran', 'M_tot_tfl', 'M_tot_ppm',
                     'M_dry_fortran', 'M_dry_tfl', 'M_dry_ppm']:
            data[key] = np.array([m[key] for m in mode_metrics])

        npz_path = os.path.join(output_dir, f"comparison_{mode}.npz")
        np.savez_compressed(npz_path, **data)
        print(f"Saved {npz_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare 24h benchmark results")
    parser.add_argument('--scenario', nargs='+', type=int, default=None,
                        help='Scenario IDs (1-based)')
    parser.add_argument('--mode', nargs='+', default=None,
                        choices=['coag_only', 'cond_only', 'combined'])
    args = parser.parse_args()

    all_metrics = compare_all_scenarios(
        scenario_ids=args.scenario,
        modes=args.mode,
    )
    print_summary(all_metrics)
    save_comparison_npz(all_metrics)
