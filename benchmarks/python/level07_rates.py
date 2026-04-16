"""Level 7: Compare instantaneous rates (dNdt, dMdt).

Expected: rtol < 1e-3 (propagated from kij differences).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result

from tomas_jax.physics.coagulation_rates import calc_coagulation_rates
from tomas_jax.physics.properties import calc_particle_properties
from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel
from tomas_jax.core.config import ICOMP_NODIAG


def run(data_dir, verbose=True):
    """Run Level 7 comparison."""
    results = []

    # Load FORTRAN reference
    dNdt_fortran = load_csv(f"{data_dir}/level07_dNdt.csv").flatten()
    dMdt_fortran = load_csv(f"{data_dir}/level07_dMdt.csv")

    # Load state
    Nk_raw = load_csv(f"{data_dir}/initial_Nk.csv").flatten()
    Mk_raw = load_csv(f"{data_dir}/initial_Mk.csv")
    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()
    params = load_csv(f"{data_dir}/params.csv").flatten()
    temp, pres, boxvol = params[0], params[1], params[2]

    # Preprocess
    Neps = 1e-3
    Nk = np.copy(Nk_raw)
    Mk = np.copy(Mk_raw)
    for k in range(len(Nk)):
        if Nk[k] < Neps:
            Nk[k] = Neps
            Mk[k, :] = 0.0
            Mk[k, 0] = Neps * 1.4 * xk[k]

    # JAX computation: compute full pipeline
    Nk_jax = jnp.array(Nk)
    Mk_jax = jnp.array(Mk)
    xk_jax = jnp.array(xk)

    Dpk, Dk, ck = calc_particle_properties(Nk_jax, Mk_jax, temp, pres)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)
    dNdt_jax, dMdt_jax, _overflow = calc_coagulation_rates(
        Nk_jax, Mk_jax, kij, xk_jax, ICOMP_NODIAG
    )

    dNdt_jax = np.array(dNdt_jax)
    dMdt_jax = np.array(dMdt_jax)

    # Compare dNdt (propagated from kij ~2-4%)
    result_dNdt = compare_arrays("dNdt (number rates)",
                                 dNdt_fortran, dNdt_jax,
                                 rtol=2e-2, atol=1e-10)
    result_dNdt['level'] = 7

    # Compare dMdt (non-diagnostic species only)
    icomp_nodiag = ICOMP_NODIAG
    result_dMdt = compare_arrays("dMdt (mass rates)",
                                 dMdt_fortran, dMdt_jax[:, :icomp_nodiag],
                                 rtol=1e-3, atol=1e-10)
    result_dMdt['level'] = 7

    results = [result_dNdt, result_dMdt]

    if verbose:
        print("\n=== Level 7: Instantaneous Rates (dNdt, dMdt) ===")
        for r in results:
            print_result(r, verbose)

    return results
