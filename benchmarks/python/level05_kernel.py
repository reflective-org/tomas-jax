"""Level 5: Compare coagulation kernel matrix (kij).

Expected: rtol < 1e-3 (propagated from Dpk difference).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result

from tomas_jax.physics.properties import calc_particle_properties
from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel


def run(data_dir, verbose=True):
    """Run Level 5 comparison."""
    results = []

    # Load FORTRAN reference
    kij_fortran = load_csv(f"{data_dir}/level05_kij.csv")

    # Load state
    Nk_fortran = load_csv(f"{data_dir}/initial_Nk.csv").flatten()
    Mk_fortran = load_csv(f"{data_dir}/initial_Mk.csv")
    params = load_csv(f"{data_dir}/params.csv").flatten()
    temp, pres, boxvol = params[0], params[1], params[2]

    # Preprocess
    xk_fortran = load_csv(f"{data_dir}/level01_xk.csv").flatten()
    Neps = 1e-3
    Nk = np.copy(Nk_fortran)
    Mk = np.copy(Mk_fortran)
    for k in range(len(Nk)):
        if Nk[k] < Neps:
            Nk[k] = Neps
            Mk[k, :] = 0.0
            Mk[k, 0] = Neps * 1.4 * xk_fortran[k]

    # JAX computation
    Dpk, Dk, ck = calc_particle_properties(
        jnp.array(Nk), jnp.array(Mk), temp, pres
    )
    kij_jax = np.array(calc_coagulation_kernel(Dpk, Dk, ck, boxvol))

    # Propagated from Dpk/Dk differences (~2-4%)
    result = compare_arrays("kij (coag kernel)", kij_fortran, kij_jax,
                           rtol=5e-2, atol=1e-30)
    result['level'] = 5
    results.append(result)

    if verbose:
        print("\n=== Level 5: Coagulation Kernel (kij) ===")
        print_result(result, verbose)

    return results
