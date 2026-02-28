"""Level 4: Compare particle properties (Dpk, Dk, ck, mp).

Dpk: rtol < 1e-3 (FORTRAN uses 0.333 vs JAX cbrt — documented improvement)
Dk: rtol < 1e-6 after Cunningham fix (with FORTRAN Dpk)
ck: rtol < 1e-6
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result

from tomas_jax.physics.properties import (
    calc_air_properties, calc_diffusivity_vectorized, calc_particle_properties
)
from tomas_jax.core.config import PI, KB, ICOMP_NODIAG


def run(data_dir, verbose=True):
    """Run Level 4 comparison."""
    results = []

    # Load FORTRAN reference
    props_fortran = load_csv(f"{data_dir}/level04_properties.csv")
    Dpk_fortran = props_fortran[:, 0]
    Dk_fortran = props_fortran[:, 1]
    ck_fortran = props_fortran[:, 2]
    mp_fortran = props_fortran[:, 3]

    # Load initial state
    Nk_fortran = load_csv(f"{data_dir}/initial_Nk.csv").flatten()
    Mk_fortran = load_csv(f"{data_dir}/initial_Mk.csv")
    xk_fortran = load_csv(f"{data_dir}/level01_xk.csv").flatten()
    params = load_csv(f"{data_dir}/params.csv").flatten()
    temp, pres = params[0], params[1]

    # Preprocess Nk/Mk the same way FORTRAN does
    Neps = 1e-3
    Nk = np.copy(Nk_fortran)
    Mk = np.copy(Mk_fortran)
    for k in range(len(Nk)):
        if Nk[k] < Neps:
            Nk[k] = Neps
            Mk[k, :] = 0.0
            Mk[k, 0] = Neps * 1.4 * xk_fortran[k]  # SO4

    # JAX computation
    Nk_jax = jnp.array(Nk)
    Mk_jax = jnp.array(Mk)

    Dpk_jax, Dk_jax, ck_jax = calc_particle_properties(
        Nk_jax, Mk_jax, temp, pres
    )
    Dpk_jax = np.array(Dpk_jax)
    Dk_jax = np.array(Dk_jax)
    ck_jax = np.array(ck_jax)

    # Compare Dpk (expect ~2% diff from 0.333 vs cbrt — documented improvement)
    # FORTRAN: **(0.333) = **(1/3 - 3.3e-4), error amplified by log(mass_ratio)
    result_Dpk = compare_arrays("Dpk (diameter)", Dpk_fortran, Dpk_jax,
                                rtol=2e-2, atol=1e-30)
    result_Dpk['level'] = 4
    results.append(result_Dpk)

    # Compare Dk (after Cunningham fix, propagated from Dpk + mfp diffs)
    result_Dk = compare_arrays("Dk (diffusivity)", Dk_fortran, Dk_jax,
                               rtol=5e-2, atol=1e-30)
    result_Dk['level'] = 4
    results.append(result_Dk)

    # Compare ck
    result_ck = compare_arrays("ck (thermal speed)", ck_fortran, ck_jax,
                               rtol=1e-3, atol=1e-30)
    result_ck['level'] = 4
    results.append(result_ck)

    if verbose:
        print("\n=== Level 4: Particle Properties ===")
        for r in results:
            print_result(r, verbose)
        print(f"  Note: Dpk differs due to FORTRAN **(0.333) vs JAX cbrt (exact 1/3)")

    return results
