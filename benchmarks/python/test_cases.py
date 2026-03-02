"""Shared test case definitions for condensation benchmarks.

Five physical scenarios covering a range of SO4 amounts, size distributions,
water content, temperature, and pressure conditions.
"""
import numpy as np

from tomas_jax.core.config import SRTSO4, NBINS, ICOMP

# Test case configurations
CASES = {
    'A': {'name': 'base',          'temp': 298.0, 'pres': 101325.0,
           'N': 1e5,   'Dp_gmd': 0.1,  'sigma': 1.6, 'Gc_so4': 1e-13, 'rh': 0.50},
    'B': {'name': 'high_so4',      'temp': 298.0, 'pres': 101325.0,
           'N': 1e5,   'Dp_gmd': 0.1,  'sigma': 1.6, 'Gc_so4': 1e-11, 'rh': 0.50},
    'C': {'name': 'small_aero',    'temp': 298.0, 'pres': 101325.0,
           'N': 1e6,   'Dp_gmd': 0.02, 'sigma': 1.8, 'Gc_so4': 1e-13, 'rh': 0.80},
    'D': {'name': 'cold_high_alt', 'temp': 240.0, 'pres': 50000.0,
           'N': 1e4,   'Dp_gmd': 0.05, 'sigma': 1.4, 'Gc_so4': 1e-14, 'rh': 0.30},
    'E': {'name': 'warm_marine',   'temp': 310.0, 'pres': 101325.0,
           'N': 500.0, 'Dp_gmd': 0.3,  'sigma': 2.0, 'Gc_so4': 1e-12, 'rh': 0.90},
}

CASE_KEYS = ['A', 'B', 'C', 'D', 'E']


def init_lognormal(case_key, xk):
    """Initialize Nk/Mk for a test case matching the Fortran harness.

    Uses the same lognormal initialization logic as benchmark_harness.f,
    including Fortran-matching constants (pi=3.141592654, exponent=0.3333).

    Args:
        case_key: One of 'A', 'B', 'C', 'D', 'E'
        xk: Bin boundaries [kg], shape (ibins+1,)

    Returns:
        Nk: Number concentrations [#/grid cell], shape (ibins,)
        Mk: Mass concentrations [kg/grid cell], shape (ibins, icomp)
    """
    case = CASES[case_key]
    pi = 3.141592654  # Match Fortran constant exactly

    N_total = case['N']
    Dp_gmd = case['Dp_gmd']
    sigma = case['sigma']
    boxvol = 1.0e6
    dens_init = 1770.0

    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))

    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk[k]) / (dens_init * pi)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk[k + 1]) / (dens_init * pi)) ** 0.3333
        Dk_init = np.sqrt(Dl * Dh)

        np_init = ((N_total * boxvol)
                   / (np.sqrt(2.0 * pi) * Dk_init * np.log(sigma))
                   * np.exp(-(np.log(Dk_init / Dp_gmd) ** 2
                              / (2.0 * np.log(sigma) ** 2)))
                   * (Dh - Dl))

        Nk[k] = np_init
        Mk[k, SRTSO4] = np_init * np.sqrt(xk[k]) * np.sqrt(xk[k + 1])

    # Neps preprocessing (match Fortran benchmark harness)
    Neps = 1.0e-3
    for k in range(NBINS):
        if Nk[k] < Neps:
            Nk[k] = Neps
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = Neps * 1.4 * xk[k]

    return Nk, Mk
