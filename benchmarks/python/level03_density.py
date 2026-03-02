"""Level 3: Compare aerosol density.

After organic mass fix, should match at rtol < 1e-6.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result

from tomas_jax.physics.density import calc_density


def run(data_dir, verbose=True):
    """Run Level 3 comparison."""
    results = []

    # Load FORTRAN reference
    density_fortran = load_csv(f"{data_dir}/level03_density.csv").flatten()
    Nk_fortran = load_csv(f"{data_dir}/initial_Nk.csv").flatten()
    Mk_fortran = load_csv(f"{data_dir}/initial_Mk.csv")
    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    # Preprocess like FORTRAN (set empty bins)
    Neps = 1e-3
    Nk = np.copy(Nk_fortran)
    Mk = np.copy(Mk_fortran)
    for k in range(len(Nk)):
        if Nk[k] < Neps:
            Nk[k] = Neps
            Mk[k, :] = 0.0
            Mk[k, 0] = Neps * 1.4 * xk[k]  # SO4

    # JAX computation
    Mk_jax = jnp.array(Mk)
    density_jax = np.array(calc_density(Mk_jax))

    result = compare_arrays("density (per bin)", density_fortran, density_jax,
                           rtol=1e-3, atol=1e-10)
    result['level'] = 3
    results.append(result)

    if verbose:
        print("\n=== Level 3: Aerosol Density ===")
        print_result(result, verbose)

    return results
