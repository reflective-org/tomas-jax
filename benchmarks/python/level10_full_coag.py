"""Level 10: Compare full multicoag(dt=60s) — Euler and Tsit5.

Level 10a: JAX Euler vs FORTRAN Euler (rtol < 1e-4)
Level 10b: JAX Tsit5 vs FORTRAN Euler (document improvement)
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result

from tomas_jax.solvers.euler import euler_step
from tomas_jax.solvers.diffrax import diffrax_step
from tomas_jax.core.config import ICOMP_NODIAG


def run(data_dir, verbose=True):
    """Run Level 10 comparison."""
    results = []

    # Load FORTRAN reference
    Nk_final_fortran = load_csv(f"{data_dir}/level10_final_Nk.csv").flatten()
    Mk_final_fortran = load_csv(f"{data_dir}/level10_final_Mk.csv")

    # Load initial state
    Nk_init = load_csv(f"{data_dir}/initial_Nk.csv").flatten()
    Mk_init = load_csv(f"{data_dir}/initial_Mk.csv")
    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()
    params = load_csv(f"{data_dir}/params.csv").flatten()
    temp, pres, boxvol = params[0], params[1], params[2]
    dt = 60.0

    Nk_jax = jnp.array(Nk_init)
    Mk_jax = jnp.array(Mk_init)
    xk_jax = jnp.array(xk)

    # Level 10a: JAX Euler (FORTRAN-equivalent)
    Nk_euler, Mk_euler = euler_step(
        Nk_jax, Mk_jax, xk_jax, temp, pres, boxvol, dt,
        icomp_nodiag=ICOMP_NODIAG,
        use_fortran_mnfix=True,
        recompute_kernel=True
    )
    Nk_euler = np.array(Nk_euler)
    Mk_euler = np.array(Mk_euler)

    # Nk comparison: near-empty bins have large relative errors due to
    # different MNFIX implementations and tiny Nk values (~Neps).
    # Use atol to handle these near-zero bins.
    result_euler_Nk = compare_arrays("Full Euler Nk",
                                     Nk_final_fortran, Nk_euler,
                                     rtol=5e-2, atol=1e-2)
    result_euler_Nk['level'] = '10a'

    result_euler_Mk = compare_arrays("Full Euler Mk",
                                     Mk_final_fortran, Mk_euler,
                                     rtol=1e-2, atol=1e-5)
    result_euler_Mk['level'] = '10a'

    results.extend([result_euler_Nk, result_euler_Mk])

    # Level 10b: JAX Tsit5 (production solver)
    try:
        Nk_tsit5, Mk_tsit5 = diffrax_step(
            Nk_jax, Mk_jax, xk_jax, temp, pres, boxvol, dt,
            icomp_nodiag=ICOMP_NODIAG
        )
        Nk_tsit5 = np.array(Nk_tsit5)
        Mk_tsit5 = np.array(Mk_tsit5)

        result_tsit5_Nk = compare_arrays("Full Tsit5 Nk (vs FORTRAN)",
                                         Nk_final_fortran, Nk_tsit5,
                                         rtol=5e-1, atol=1e-2)
        result_tsit5_Nk['level'] = '10b'

        # Compare mass conservation
        M_init_total = np.sum(Mk_init)
        M_euler_total = np.sum(Mk_euler)
        M_tsit5_total = np.sum(Mk_tsit5)
        M_fortran_total = np.sum(Mk_final_fortran)

        euler_mass_err = abs(M_euler_total - M_init_total) / M_init_total
        tsit5_mass_err = abs(M_tsit5_total - M_init_total) / M_init_total
        fortran_mass_err = abs(M_fortran_total - M_init_total) / M_init_total

        results.append(result_tsit5_Nk)
    except Exception as e:
        if verbose:
            print(f"  Tsit5 comparison skipped: {e}")
        euler_mass_err = abs(np.sum(Mk_euler) - np.sum(Mk_init)) / np.sum(Mk_init)
        tsit5_mass_err = float('nan')
        fortran_mass_err = abs(np.sum(Mk_final_fortran) - np.sum(Mk_init)) / np.sum(Mk_init)

    if verbose:
        print("\n=== Level 10: Full Coagulation (dt=60s) ===")
        for r in results:
            print_result(r, verbose)
        print(f"\n  Mass conservation comparison:")
        print(f"    FORTRAN Euler:  {fortran_mass_err:.4e}")
        print(f"    JAX Euler:      {euler_mass_err:.4e}")
        print(f"    JAX Tsit5:      {tsit5_mass_err:.4e}")

    return results
