"""Level 12: Isolated Condensation (5 test cases).

For each case: compute CS -> mcond -> ezcond -> eznh3eqm -> water eq -> mnfix.
Compare final Nk, Mk with Fortran reference.

Expected: rtol ~ 1e-2 (tmcond remapping amplifies small differences).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result
from .test_cases import CASES, CASE_KEYS, init_lognormal

from tomas_jax.core.config import (
    SRTSO4, SRTNH4, ICOMP_NODIAG, N_GAS_SPECIES,
)
from tomas_jax.physics.condensation_sink import calc_condensation_sink
from tomas_jax.physics.ezcond import ezcond
from tomas_jax.physics.nh3_equilibrium import eznh3eqm
from tomas_jax.physics.water_equilibrium import calc_equilibrium_water
from tomas_jax.core.mnfix_jax import mnfix_jax


def run(data_dir, verbose=True):
    """Run Level 12 comparison."""
    results = []

    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    if verbose:
        print("\n=== Level 12: Isolated Condensation ===")

    for case_key in CASE_KEYS:
        case = CASES[case_key]

        # Load Fortran reference
        mcond_data = load_csv(
            f"{data_dir}/level12_case{case_key}_mcond.csv"
        ).flatten()
        Nk_fortran = load_csv(
            f"{data_dir}/level12_case{case_key}_Nk.csv"
        ).flatten()
        Mk_fortran = load_csv(
            f"{data_dir}/level12_case{case_key}_Mk.csv"
        )

        mcond_fortran = mcond_data[0]
        CS_fortran = mcond_data[1]

        # Initialize JAX state
        temp = case['temp']
        pres = case['pres']
        boxvol = 1.0e6
        rh = case['rh']
        alpha = 1.0

        Nk, Mk = init_lognormal(case_key, xk)

        # Initialize gas concentrations
        Gc = np.zeros(N_GAS_SPECIES)
        Gc[SRTSO4] = case['Gc_so4']

        # Compute condensation sink
        CS_jax, sinkfrac_jax = calc_condensation_sink(
            jnp.array(Nk), jnp.array(Mk),
            temp, pres, boxvol,
            alpha=alpha
        )
        CS_val = float(CS_jax)

        # Compute mcond (exponential depletion over dt=60s)
        dt = 60.0
        mcond = Gc[SRTSO4] * (1.0 - np.exp(-CS_val * dt))

        # Compare mcond (wider tolerance: propagated from CS differences)
        r_mcond = compare_arrays(
            f"Case {case_key} ({case['name']}) mcond",
            np.array([mcond_fortran]), np.array([mcond]),
            rtol=0.2)
        r_mcond['level'] = '12'
        results.append(r_mcond)

        # Run ezcond
        Nk_out, Mk_out = ezcond(
            Nk, Mk, mcond, SRTSO4,
            xk, temp, pres, boxvol, alpha
        )

        # NH3 equilibrium
        Gc_jax = jnp.array(Gc)
        Mk_jax = jnp.array(Mk_out)
        Gc_jax, Mk_jax = eznh3eqm(Gc_jax, Mk_jax)
        Mk_out = np.array(Mk_jax)

        # Water equilibrium
        Mk_jax = jnp.array(Mk_out)
        Mk_jax = calc_equilibrium_water(Mk_jax, rh)
        Mk_out = np.array(Mk_jax)

        # MNFIX
        Nk_jax, Mk_jax = mnfix_jax(
            jnp.array(Nk_out), jnp.array(Mk_out),
            jnp.array(xk), ICOMP_NODIAG
        )
        Nk_out = np.array(Nk_jax)
        Mk_out = np.array(Mk_jax)

        # Compare Nk
        r_nk = compare_arrays(
            f"Case {case_key} ({case['name']}) Nk",
            Nk_fortran, Nk_out,
            rtol=1e-2, atol=1e-2)
        r_nk['level'] = '12'

        # Compare Mk
        r_mk = compare_arrays(
            f"Case {case_key} ({case['name']}) Mk",
            Mk_fortran, Mk_out,
            rtol=1e-2, atol=1e-10)
        r_mk['level'] = '12'

        results.extend([r_nk, r_mk])

        # Mass conservation check
        Nk_init, Mk_init = init_lognormal(case_key, xk)
        M_init = np.sum(Mk_init[:, SRTSO4])
        M_final = np.sum(Mk_out[:, SRTSO4])
        mass_err = abs((M_final - M_init) - mcond) / max(mcond, 1e-30)
        if verbose:
            print(f"  Case {case_key} mass conservation error: {mass_err:.4e}")

    if verbose:
        for r in results:
            print_result(r, verbose)

    return results
