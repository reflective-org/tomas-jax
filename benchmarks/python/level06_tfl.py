"""Level 6: Compare TFL algorithm outputs (xbar, phi, eff).

Expected: rtol < 1e-10 (same formulas, input-propagated differences only).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result

from tomas_jax.physics.coagulation_rates import calc_xbar_phi_eff, _preprocess_concentrations
from tomas_jax.core.config import ICOMP_NODIAG


def run(data_dir, verbose=True):
    """Run Level 6 comparison."""
    results = []

    # Load FORTRAN reference
    tfl_fortran = load_csv(f"{data_dir}/level06_tfl.csv")
    xbar_fortran = tfl_fortran[:, 0]
    phi_fortran = tfl_fortran[:, 1]
    eff_fortran = tfl_fortran[:, 2]

    # Load state (use preprocessed state matching FORTRAN)
    Nk_raw = load_csv(f"{data_dir}/initial_Nk.csv").flatten()
    Mk_raw = load_csv(f"{data_dir}/initial_Mk.csv")
    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    # Preprocess like FORTRAN
    Neps = 1e-3
    Nk = np.copy(Nk_raw)
    Mk = np.copy(Mk_raw)
    for k in range(len(Nk)):
        if Nk[k] < Neps:
            Nk[k] = Neps
            Mk[k, :] = 0.0
            Mk[k, 0] = Neps * 1.4 * xk[k]

    # JAX computation
    Nk_jax = jnp.array(Nk)
    Mk_jax = jnp.array(Mk)
    xk_jax = jnp.array(xk)

    # Use the preprocessing + calc functions
    Nk_safe, Mk_safe = _preprocess_concentrations(Nk_jax, Mk_jax, xk_jax)
    xbar_jax, phi_jax, eff_jax = calc_xbar_phi_eff(
        Nk_safe, Mk_safe, xk_jax, ICOMP_NODIAG
    )

    xbar_jax = np.array(xbar_jax)
    phi_jax = np.array(phi_jax)
    eff_jax = np.array(eff_jax)

    result_xbar = compare_arrays("xbar", xbar_fortran, xbar_jax,
                                 rtol=1e-10, atol=1e-30)
    result_xbar['level'] = 6

    result_phi = compare_arrays("phi", phi_fortran, phi_jax,
                                rtol=1e-10, atol=1e-30)
    result_phi['level'] = 6

    result_eff = compare_arrays("eff", eff_fortran, eff_jax,
                                rtol=1e-10, atol=1e-30)
    result_eff['level'] = 6

    results = [result_xbar, result_phi, result_eff]

    if verbose:
        print("\n=== Level 6: TFL Algorithm (xbar, phi, eff) ===")
        for r in results:
            print_result(r, verbose)

    return results
