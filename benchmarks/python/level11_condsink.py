"""Level 11: Gas Properties + Condensation Sink (5 test cases).

Compares gas diffusivity, mean molecular speed, mean free path,
condensation sink, and per-bin sink fractions across 5 physical
scenarios.

Expected: rtol ~ 1e-3 (known constant differences in pi, kB, M_air).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from .utils import load_csv, compare_arrays, print_result
from .test_cases import CASES, CASE_KEYS, init_lognormal

from tomas_jax.physics.gas_properties import (
    calc_gas_diffusivity,
    calc_mean_molecular_speed,
    calc_mean_free_path,
)
from tomas_jax.physics.condensation_sink import calc_condensation_sink


def run(data_dir, verbose=True):
    """Run Level 11 comparison."""
    results = []

    # Load xk (same for all cases)
    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    if verbose:
        print("\n=== Level 11: Gas Properties + Condensation Sink ===")

    for case_key in CASE_KEYS:
        case = CASES[case_key]

        # Load Fortran reference
        gas_props = load_csv(
            f"{data_dir}/level11_case{case_key}_gas_props.csv"
        ).flatten()
        condsink_data = load_csv(
            f"{data_dir}/level11_case{case_key}_condsink.csv"
        ).flatten()

        Di_fortran = gas_props[0]
        ms_fortran = gas_props[1]
        mfp_fortran = gas_props[2]
        CS_fortran = condsink_data[0]
        sinkfrac_fortran = condsink_data[1:]

        # JAX computation
        temp, pres = case['temp'], case['pres']
        Di_jax = float(calc_gas_diffusivity(temp, pres))
        ms_jax = float(calc_mean_molecular_speed(temp))
        mfp_jax = float(calc_mean_free_path(temp, pres))

        # Init case and compute CS
        Nk, Mk = init_lognormal(case_key, xk)
        CS_jax, sinkfrac_jax = calc_condensation_sink(
            jnp.array(Nk), jnp.array(Mk),
            temp, pres, 1.0e6,
            alpha=1.0
        )

        # Compare gas properties (tight: only constant diffs)
        r_di = compare_arrays(
            f"Case {case_key} ({case['name']}) Di",
            np.array([Di_fortran]), np.array([float(Di_jax)]),
            rtol=1e-3)
        r_di['level'] = '11'

        r_ms = compare_arrays(
            f"Case {case_key} ({case['name']}) ms",
            np.array([ms_fortran]), np.array([float(ms_jax)]),
            rtol=1e-3)
        r_ms['level'] = '11'

        r_mfp = compare_arrays(
            f"Case {case_key} ({case['name']}) mfp",
            np.array([mfp_fortran]), np.array([float(mfp_jax)]),
            rtol=1e-3)
        r_mfp['level'] = '11'

        # Compare condensation sink (wider tolerance due to:
        #   - Fortran getCondSink Neps=1e10 threshold gives default properties
        #     for bins with Nk<1e10, while JAX computes for all bins
        #   - Dpk: 0.333 exponent vs exact cbrt)
        r_cs = compare_arrays(
            f"Case {case_key} ({case['name']}) CS",
            np.array([CS_fortran]), np.array([float(CS_jax)]),
            rtol=0.2)
        r_cs['level'] = '11'

        r_sf = compare_arrays(
            f"Case {case_key} ({case['name']}) sinkfrac",
            sinkfrac_fortran, np.array(sinkfrac_jax),
            rtol=0.6, atol=1e-10)
        r_sf['level'] = '11'

        results.extend([r_di, r_ms, r_mfp, r_cs, r_sf])

    if verbose:
        for r in results:
            print_result(r, verbose)

    return results
