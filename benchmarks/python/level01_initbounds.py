"""Level 1: Compare bin boundaries (xk).

FORTRAN: xk(k) = Mo * 2^(k-1), where Mo = 1e-21 * 2^(-6)
JAX: jnp.logspace(log10(mass_min), log10(mass_max), nbins+1)

Expected: EXACT match if we use the same formula.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result


def compute_xk_fortran_style(nbins=36):
    """Compute xk using FORTRAN initbounds formula."""
    Mo = 1.0e-21 * 2.0**(-6)
    xk = np.array([Mo * 2.0**(k) for k in range(nbins + 1)])
    return xk


def run(data_dir, verbose=True):
    """Run Level 1 comparison."""
    results = []

    # Load FORTRAN reference
    xk_fortran = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    # JAX computation (using FORTRAN formula)
    xk_jax = compute_xk_fortran_style(nbins=36)

    result = compare_arrays("xk (bin boundaries)", xk_fortran, xk_jax,
                           rtol=1e-15, atol=1e-50)
    result['level'] = 1
    results.append(result)

    if verbose:
        print("\n=== Level 1: Bin Boundaries (xk) ===")
        print_result(result, verbose)

    return results
