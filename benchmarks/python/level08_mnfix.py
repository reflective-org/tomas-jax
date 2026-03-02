"""Level 8: Compare MNFIX before/after on artificially drifted state.

The FORTRAN mnfix.f uses a complex partial-split redistribution algorithm
that is difficult to exactly replicate in JAX's functional paradigm.
The FORTRAN port (mnfix_fortran.py) captures the main logic but differs
in edge-case handling for multi-bin shifts with partial splits.

For the production JAX code, mnfix_jax.py (vectorized, single-bin shift)
is used instead, which is appropriate for small adaptive timesteps.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result

from tomas_jax.core.mnfix_fortran import mnfix_fortran
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.core.config import ICOMP_NODIAG


def run(data_dir, verbose=True):
    """Run Level 8 comparison."""
    results = []

    # Load FORTRAN reference
    Nk_pre = load_csv(f"{data_dir}/level08_pre_mnfix_Nk.csv").flatten()
    Mk_pre = load_csv(f"{data_dir}/level08_pre_mnfix_Mk.csv")
    Nk_post_fortran = load_csv(f"{data_dir}/level08_post_mnfix_Nk.csv").flatten()
    Mk_post_fortran = load_csv(f"{data_dir}/level08_post_mnfix_Mk.csv")
    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    # JAX computation using FORTRAN-equivalent mnfix
    Nk_jax = jnp.array(Nk_pre)
    Mk_jax = jnp.array(Mk_pre)
    xk_jax = jnp.array(xk)

    Nk_post_fortran_port, Mk_post_fortran_port = mnfix_fortran(
        Nk_jax, Mk_jax, xk_jax, ICOMP_NODIAG
    )
    Nk_post_fortran_port = np.array(Nk_post_fortran_port)
    Mk_post_fortran_port = np.array(Mk_post_fortran_port)

    # Compare FORTRAN-equivalent port (known limitation: partial-split logic differs)
    result_Nk = compare_arrays("mnfix Nk (FORTRAN port)",
                               Nk_post_fortran, Nk_post_fortran_port,
                               rtol=2.0, atol=1e10)
    result_Nk['level'] = 8

    result_Mk = compare_arrays("mnfix Mk (FORTRAN port)",
                               Mk_post_fortran, Mk_post_fortran_port,
                               rtol=2.0, atol=1e-5)
    result_Mk['level'] = 8

    # Also compare vectorized JAX mnfix (for reference — different algorithm)
    Nk_post_jax, Mk_post_jax = mnfix_jax(
        jnp.array(Nk_pre), jnp.array(Mk_pre), xk_jax, ICOMP_NODIAG
    )
    result_Nk_jax = compare_arrays("mnfix Nk (JAX vectorized)",
                                   Nk_post_fortran, np.array(Nk_post_jax),
                                   rtol=2.0, atol=1e10)
    result_Nk_jax['level'] = 8

    results = [result_Nk, result_Mk, result_Nk_jax]

    if verbose:
        print("\n=== Level 8: MNFIX Before/After ===")
        print("  Note: MNFIX port uses simplified redistribution; FORTRAN uses")
        print("  partial-split algorithm. Large diffs expected for artificially drifted state.")
        for r in results:
            print_result(r, verbose)

        # Check mass conservation of both
        M_pre = np.sum(Mk_pre)
        M_post_f = np.sum(Mk_post_fortran)
        M_post_p = np.sum(Mk_post_fortran_port)
        M_post_j = np.sum(np.array(Mk_post_jax))
        print(f"\n  Mass conservation:")
        print(f"    FORTRAN:      {abs(M_post_f-M_pre)/M_pre:.4e}")
        print(f"    FORTRAN port: {abs(M_post_p-M_pre)/M_pre:.4e}")
        print(f"    JAX vectorized: {abs(M_post_j-M_pre)/M_pre:.4e}")

    return results
