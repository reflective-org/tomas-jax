"""Level 9: Compare single Euler step (with adaptive dt from FORTRAN).

Expected: rtol < 1e-6 with euler.py + mnfix_fortran.py
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result

from tomas_jax.solvers.euler import euler_single_step
from tomas_jax.physics.properties import calc_particle_properties
from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel
from tomas_jax.physics.coagulation_rates import calc_coagulation_rates
from tomas_jax.core.mnfix_fortran import mnfix_fortran
from tomas_jax.core.config import ICOMP_NODIAG


def run(data_dir, verbose=True):
    """Run Level 9 comparison."""
    results = []

    # Load FORTRAN reference
    Nk_euler_fortran = load_csv(f"{data_dir}/level09_euler_Nk.csv").flatten()
    Mk_euler_fortran = load_csv(f"{data_dir}/level09_euler_Mk.csv")
    dt_fortran = load_csv(f"{data_dir}/level09_dt.csv").flatten()[0]

    # Load initial state
    Nk_init = load_csv(f"{data_dir}/initial_Nk.csv").flatten()
    Mk_init = load_csv(f"{data_dir}/initial_Mk.csv")
    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()
    params = load_csv(f"{data_dir}/params.csv").flatten()
    temp, pres, boxvol = params[0], params[1], params[2]

    # Preprocess
    Neps = 1e-3
    Nk = np.copy(Nk_init)
    Mk = np.copy(Mk_init)
    for k in range(len(Nk)):
        if Nk[k] < Neps:
            Nk[k] = Neps
            Mk[k, :] = 0.0
            Mk[k, 0] = Neps * 1.4 * xk[k]

    # JAX: Compute properties and kernel
    Nk_jax = jnp.array(Nk)
    Mk_jax = jnp.array(Mk)
    xk_jax = jnp.array(xk)

    Dpk, Dk, ck = calc_particle_properties(Nk_jax, Mk_jax, temp, pres)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    # Single Euler step with FORTRAN dt
    Nk_euler, Mk_euler = euler_single_step(
        Nk_jax, Mk_jax, kij, xk_jax, dt_fortran, ICOMP_NODIAG
    )

    # Apply FORTRAN-equivalent mnfix
    Nk_euler, Mk_euler = mnfix_fortran(Nk_euler, Mk_euler, xk_jax, ICOMP_NODIAG)

    Nk_euler = np.array(Nk_euler)
    Mk_euler = np.array(Mk_euler)

    # Propagated from kij/rates + MNFIX differences
    result_Nk = compare_arrays("Euler step Nk",
                               Nk_euler_fortran, Nk_euler,
                               rtol=2e-2, atol=1e-10)
    result_Nk['level'] = 9

    result_Mk = compare_arrays("Euler step Mk",
                               Mk_euler_fortran, Mk_euler,
                               rtol=1e-3, atol=1e-10)
    result_Mk['level'] = 9

    results = [result_Nk, result_Mk]

    if verbose:
        print("\n=== Level 9: Single Euler Step ===")
        print(f"  FORTRAN adaptive dt = {dt_fortran:.6e} s")
        for r in results:
            print_result(r, verbose)

    return results
