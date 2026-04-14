"""Benchmark comparison utilities.

Functions for loading FORTRAN reference data and comparing with JAX results.
"""
import numpy as np
import os
from typing import Dict, Optional, Tuple

# Default data directory (FORTRAN output)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'fortran_output')


def _fix_fortran_float(s: str) -> str:
    """Fix Fortran floats with missing E before large exponents.

    Fortran writes e.g. '0.1024186408109169-238' instead of '0.1024186408109169E-238'
    when the exponent exceeds 3 digits. Insert 'E' before the sign.
    """
    import re
    return re.sub(r'(\d)([-+])(\d{3})', r'\1E\2\3', s)


def load_csv(path: str) -> np.ndarray:
    """Load a CSV file written by the FORTRAN benchmark harness.

    Handles FORTRAN-style E-format (e.g., 1.234E+05) and comments (#).
    Also handles missing-E format for large exponents (e.g., 0.123-238).

    Args:
        path: Path to CSV file.

    Returns:
        numpy array of float64.
    """
    try:
        return np.loadtxt(path, delimiter=',', dtype=np.float64, comments='#')
    except ValueError:
        # Fortran wrote exponents without 'E' — fix and retry
        with open(path, 'r') as f:
            text = f.read()
        fixed = _fix_fortran_float(text)
        import io
        return np.loadtxt(io.StringIO(fixed), delimiter=',', dtype=np.float64,
                          comments='#')


def compare_arrays(
    name: str,
    fortran: np.ndarray,
    jax_arr: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-30
) -> Dict:
    """Compare FORTRAN and JAX arrays element-wise.

    Args:
        name: Label for this comparison.
        fortran: FORTRAN reference array.
        jax_arr: JAX computed array.
        rtol: Relative tolerance for PASS/FAIL.
        atol: Absolute tolerance (for near-zero values).

    Returns:
        Dict with keys: name, max_rel_error, mean_rel_error, max_abs_error,
                        passed, shape_match, details.
    """
    result = {
        'name': name,
        'shape_match': fortran.shape == jax_arr.shape,
        'passed': False,
        'max_rel_error': np.inf,
        'mean_rel_error': np.inf,
        'max_abs_error': np.inf,
        'details': ''
    }

    if not result['shape_match']:
        result['details'] = (
            f"Shape mismatch: FORTRAN {fortran.shape} vs JAX {jax_arr.shape}"
        )
        return result

    # Absolute error
    abs_err = np.abs(fortran - jax_arr)
    result['max_abs_error'] = float(np.max(abs_err))

    # Relative error (avoid div by zero)
    denom = np.maximum(np.abs(fortran), atol)
    rel_err = abs_err / denom

    result['max_rel_error'] = float(np.max(rel_err))
    result['mean_rel_error'] = float(np.mean(rel_err))

    # Check pass/fail
    result['passed'] = bool(np.all(
        np.isclose(fortran, jax_arr, rtol=rtol, atol=atol)
    ))

    # Find worst element
    worst_idx = np.unravel_index(np.argmax(rel_err), rel_err.shape)
    result['details'] = (
        f"Worst at index {worst_idx}: "
        f"FORTRAN={fortran[worst_idx]:.16e}, "
        f"JAX={jax_arr[worst_idx]:.16e}, "
        f"rel_err={rel_err[worst_idx]:.4e}"
    )

    return result


def load_initial_state(data_dir: str = DEFAULT_DATA_DIR) -> Tuple[np.ndarray, ...]:
    """Load the initial state from FORTRAN benchmark output.

    Returns:
        Tuple of (Nk, Mk, xk, temp, pres, boxvol)
    """
    Nk = load_csv(os.path.join(data_dir, 'initial_Nk.csv'))
    Mk = load_csv(os.path.join(data_dir, 'initial_Mk.csv'))
    xk = load_csv(os.path.join(data_dir, 'initial_xk.csv'))

    # Scalars stored as single-value files
    params = load_csv(os.path.join(data_dir, 'params.csv'))
    temp = params[0]
    pres = params[1]
    boxvol = params[2]

    return Nk, Mk, xk, temp, pres, boxvol


def print_result(result: Dict, verbose: bool = True) -> None:
    """Pretty-print a comparison result."""
    status = "PASS" if result['passed'] else "FAIL"
    symbol = " OK " if result['passed'] else "FAIL"

    print(f"  [{symbol}] {result['name']:<30s} "
          f"max_rel={result['max_rel_error']:.4e}  "
          f"mean_rel={result['mean_rel_error']:.4e}")

    if verbose and not result['passed']:
        print(f"         {result['details']}")


def generate_report(results: list, output_path: str) -> None:
    """Generate a markdown comparison report.

    Args:
        results: List of comparison result dicts (from compare_arrays).
        output_path: Path to write the markdown report.
    """
    n_pass = sum(1 for r in results if r['passed'])
    n_total = len(results)

    lines = [
        "# TOMAS FORTRAN vs JAX Coagulation Benchmark Report",
        "",
        f"**Date:** Auto-generated",
        f"**Summary:** {n_pass}/{n_total} comparisons passed",
        "",
        "## Results",
        "",
        "| Level | Component | Max Rel Error | Mean Rel Error | Status |",
        "|-------|-----------|---------------|----------------|--------|",
    ]

    for r in results:
        status = "PASS" if r['passed'] else "**FAIL**"
        lines.append(
            f"| {r.get('level', '-')} | {r['name']} | "
            f"{r['max_rel_error']:.4e} | {r['mean_rel_error']:.4e} | {status} |"
        )

    lines.extend([
        "",
        "## Known Differences (Not Bugs)",
        "",
        "| Item | FORTRAN | JAX | Impact |",
        "|------|---------|-----|--------|",
        "| Diameter exponent | `**(0.333)` | `jnp.cbrt()` (exact 1/3) | ~1e-4 rel diff in Dpk |",
        "| Pi constant | `3.141592654` | `3.141592653589793` | ~1e-10 |",
        "| kB constant | `1.38e-23` | `1.380649e-23` | ~5e-4 |",
        "| Molar mass air | `0.0289` | `0.0289647` | ~2e-3 in mfp |",
        "",
        "## Bug Fixes Applied",
        "",
        "1. **Cunningham slip correction** (`properties.py:79`): Fixed polynomial "
        "evaluation — `Kn2 * (6 + 18*Kn)` changed to `Kn * (6 + 18*Kn)`, "
        "correcting numerator from `5+4Kn+6Kn^3+18Kn^4` to `5+4Kn+6Kn^2+18Kn^3`.",
        "",
        "2. **Density organic mass** (`density.py:44`): Added organic mass to sulfate "
        "in density calculation, matching FORTRAN `aerodens(Mk(k,srtso4)+orgmass, ...)`.",
        "",
        "3. **Density mtot** (`density.py:97`): Changed to use raw component masses "
        "instead of reconstructed compound masses, matching FORTRAN.",
        "",
        "## Integrator Comparison",
        "",
        "| Metric | FORTRAN Euler | JAX Euler | JAX Tsit5 |",
        "|--------|---------------|-----------|-----------|",
        "| Order | 1st | 1st | 5th |",
        "| Error control | CFL-like | CFL-like | Embedded RK |",
        "| Mass conservation | Depends on dt | Depends on dt | ~1e-13 |",
        "",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nReport written to: {output_path}")
