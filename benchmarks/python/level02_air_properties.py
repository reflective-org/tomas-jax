"""Level 2: Compare air properties (mu, mfp).

FORTRAN uses pi=3.141592654, R=8.314, molar_mass_air=0.0289
JAX uses pi=3.141592653589793, R=8.314462618, molar_mass_air=0.0289647

Expected: rtol < 1e-4 due to constant differences.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result

from tomas_jax.physics.properties import calc_air_properties


def run(data_dir, verbose=True):
    """Run Level 2 comparison."""
    results = []

    # Load FORTRAN reference
    air_fortran = load_csv(f"{data_dir}/level02_air.csv").flatten()
    mu_fortran = air_fortran[0]
    mfp_fortran = air_fortran[1]

    # JAX computation
    temp = 298.0
    pres = 101325.0
    mu_jax, mfp_jax = calc_air_properties(temp, pres)

    result_mu = compare_arrays("mu (viscosity)",
                               np.array([mu_fortran]),
                               np.array([float(mu_jax)]),
                               rtol=1e-10, atol=1e-30)
    result_mu['level'] = 2

    result_mfp = compare_arrays("mfp (mean free path)",
                                np.array([mfp_fortran]),
                                np.array([float(mfp_jax)]),
                                rtol=2e-3, atol=1e-30)
    result_mfp['level'] = 2

    results = [result_mu, result_mfp]

    if verbose:
        print("\n=== Level 2: Air Properties ===")
        for r in results:
            print_result(r, verbose)
        print(f"  Note: mfp differs due to FORTRAN using 0.0289 vs JAX 0.0289647 for air molar mass")

    return results
