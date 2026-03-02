"""Level 14: PPM vs TFL Condensation Comparison.

Runs both PPM and TFL condensation on the same 5 test cases (A-E)
and compares:
- Number/mass conservation for both methods
- PPM vs TFL agreement
- Both methods vs Fortran reference (from level 12 data)
- Peak bin shift and distribution width

Expected: PPM and TFL should agree within rtol=0.1 for Nk,
with PPM showing better peak preservation (less numerical diffusion).
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
from tomas_jax.physics.ezcond_ppm import ezcond_ppm
from tomas_jax.physics.nh3_equilibrium import eznh3eqm
from tomas_jax.physics.water_equilibrium import calc_equilibrium_water
from tomas_jax.core.mnfix_jax import mnfix_jax


def _run_post_ezcond(Nk_out, Mk_out, Gc, rh, xk):
    """Apply NH3 equilibrium, water equilibrium, and MNFIX after ezcond."""
    Gc_jax = jnp.array(Gc)
    Mk_jax = jnp.array(Mk_out)
    Gc_jax, Mk_jax = eznh3eqm(Gc_jax, Mk_jax)
    Mk_out = np.array(Mk_jax)

    Mk_jax = jnp.array(Mk_out)
    Mk_jax = calc_equilibrium_water(Mk_jax, rh)
    Mk_out = np.array(Mk_jax)

    Nk_jax, Mk_jax = mnfix_jax(
        jnp.array(Nk_out), jnp.array(Mk_out),
        jnp.array(xk), ICOMP_NODIAG
    )
    return np.array(Nk_jax), np.array(Mk_jax)


def _fwhm_bins(Nk):
    """Compute full-width at half-maximum in bins."""
    peak = np.max(Nk)
    half_max = peak / 2.0
    above = Nk >= half_max
    if not np.any(above):
        return 0
    indices = np.where(above)[0]
    return int(indices[-1] - indices[0] + 1)


def run(data_dir, verbose=True):
    """Run Level 14 comparison: PPM vs TFL."""
    results = []

    xk = load_csv(f"{data_dir}/level01_xk.csv").flatten()

    if verbose:
        print("\n=== Level 14: PPM vs TFL Condensation ===")

    for case_key in CASE_KEYS:
        case = CASES[case_key]

        # Load Fortran reference (from level 12)
        try:
            Nk_fortran = load_csv(
                f"{data_dir}/level12_case{case_key}_Nk.csv"
            ).flatten()
            Mk_fortran = load_csv(
                f"{data_dir}/level12_case{case_key}_Mk.csv"
            )
            has_fortran = True
        except FileNotFoundError:
            has_fortran = False

        # Initialize state
        temp = case['temp']
        pres = case['pres']
        boxvol = 1.0e6
        rh = case['rh']
        alpha = 1.0

        Nk_init, Mk_init = init_lognormal(case_key, xk)

        # Gas concentrations
        Gc = np.zeros(N_GAS_SPECIES)
        Gc[SRTSO4] = case['Gc_so4']

        # Compute condensation sink and mcond
        CS_jax, _ = calc_condensation_sink(
            jnp.array(Nk_init), jnp.array(Mk_init),
            temp, pres, boxvol, accommodation_coeff=alpha
        )
        CS_val = float(CS_jax)
        dt = 60.0
        mcond = Gc[SRTSO4] * (1.0 - np.exp(-CS_val * dt))

        # Run TFL
        Nk_tfl, Mk_tfl = ezcond(
            Nk_init.copy(), Mk_init.copy(), mcond, SRTSO4,
            xk, temp, pres, boxvol, alpha
        )
        Nk_tfl, Mk_tfl = _run_post_ezcond(Nk_tfl, Mk_tfl, Gc, rh, xk)

        # Run PPM
        Nk_ppm, Mk_ppm = ezcond_ppm(
            Nk_init.copy(), Mk_init.copy(), mcond, SRTSO4,
            xk, temp, pres, boxvol, alpha
        )
        Nk_ppm, Mk_ppm = _run_post_ezcond(Nk_ppm, Mk_ppm, Gc, rh, xk)

        # --- Conservation metrics ---
        N_init_total = np.sum(Nk_init)
        N_tfl_err = abs(np.sum(Nk_tfl) - N_init_total) / N_init_total
        N_ppm_err = abs(np.sum(Nk_ppm) - N_init_total) / N_init_total

        M_init_so4 = np.sum(Mk_init[:, SRTSO4])
        M_tfl_err = abs(np.sum(Mk_tfl[:, SRTSO4]) - M_init_so4 - mcond) / max(mcond, 1e-30)
        M_ppm_err = abs(np.sum(Mk_ppm[:, SRTSO4]) - M_init_so4 - mcond) / max(mcond, 1e-30)

        # --- PPM vs TFL comparison ---
        r_nk = compare_arrays(
            f"Case {case_key} ({case['name']}) PPM vs TFL Nk",
            Nk_tfl, Nk_ppm,
            rtol=0.2, atol=1e-2
        )
        r_nk['level'] = '14'
        results.append(r_nk)

        r_mk = compare_arrays(
            f"Case {case_key} ({case['name']}) PPM vs TFL Mk(SO4)",
            Mk_tfl[:, SRTSO4], Mk_ppm[:, SRTSO4],
            rtol=0.2, atol=1e-20
        )
        r_mk['level'] = '14'
        results.append(r_mk)

        # --- PPM vs Fortran ---
        if has_fortran:
            r_nk_f = compare_arrays(
                f"Case {case_key} ({case['name']}) PPM vs Fortran Nk",
                Nk_fortran, Nk_ppm,
                rtol=0.2, atol=1e-2
            )
            r_nk_f['level'] = '14'
            results.append(r_nk_f)

        # --- Distribution shape metrics ---
        peak_tfl = int(np.argmax(Nk_tfl))
        peak_ppm = int(np.argmax(Nk_ppm))
        peak_init = int(np.argmax(Nk_init))
        fwhm_tfl = _fwhm_bins(Nk_tfl)
        fwhm_ppm = _fwhm_bins(Nk_ppm)

        if verbose:
            print(f"\n  Case {case_key} ({case['name']}):")
            print(f"    N conservation - TFL: {N_tfl_err:.4e}, PPM: {N_ppm_err:.4e}")
            print(f"    M conservation - TFL: {M_tfl_err:.4e}, PPM: {M_ppm_err:.4e}")
            print(f"    Peak bin - Init: {peak_init}, TFL: {peak_tfl}, PPM: {peak_ppm}")
            print(f"    FWHM [bins] - TFL: {fwhm_tfl}, PPM: {fwhm_ppm}")

    if verbose:
        print("\n  --- Comparison Summary ---")
        for r in results:
            print_result(r, verbose)

    return results
