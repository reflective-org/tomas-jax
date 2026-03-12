"""Global Configuration and Constants.

This module MUST be imported before any other JAX modules to ensure
precisions are set correctly.

It serves as the 'Single Source of Truth' for model dimensions
and physical constants.
"""
import os
import jax

# =========================================================================
# 1. JAX Configuration (CRITICAL)
# =========================================================================
# Force double precision (float64) for aerosol microphysics.
# This must run before any jax.numpy usage.
jax.config.update("jax_enable_x64", True)

# Optional: Enable NaN debugging (slow, use only for dev)
# jax.config.update("jax_debug_nans", True)

# =========================================================================
# 2. Model Dimensions (TOMAS Standard)
# =========================================================================
NBINS = 40          # Number of size bins (1.7nm start, mass-doubling)
ICOMP = 44          # Total number of mass tracking components (species)
IDIAG = 2           # Number of diagnostic species (usually at the end)

# Derived counts
# Number of prognostic species (mass moves, coagulates, condenses)
ICOMP_NODIAG = ICOMP - IDIAG 

# =========================================================================
# 3. Species Indices (0-based)
# =========================================================================
# These map the 'icomp' dimension to physical chemical species.
# Adjust these if you change the TOMAS species list.

SRTSO4 = 0          # Sulfate [SO4]
SRTORG1 = 1         # First Organic Aerosol
SRTORGLAST = 41     # Last Organic Aerosol
SRTNH4 = 42         # Ammonium [NH4]
SRTH2O = 43         # Water [H2O]
# SRTNACL = 41        # Sea Salt (Assuming slot 41 based on previous context)

# =========================================================================
# 4. Physical Constants
# =========================================================================
# Centralized here to ensure consistency across kernel/property modules.

PI = 3.141592653589793
KB = 1.380649e-23       # Boltzmann constant (J/K)
R_GAS = 8.314462618     # Gas constant (J/mol K)
AVOGADRO = 6.02214076e23
MOLAR_MASS_AIR = 0.0289647  # kg/mol (Standard Atmosphere)

# =========================================================================
# 5. Condensation Parameters
# =========================================================================
IORG = 41                       # Number of organic species (indices 1-41)
N_GAS_SPECIES = ICOMP - 1      # 43 — Gc dimension (all species except water)

# Molecular weights [g/mol]
MW_H2SO4 = 98.0
MW_NH3 = 17.0
MW_NH4 = 18.0
MW_SO4 = 96.0

# Diffusion volumes (Fuller-Schettler-Giddings method)
SV_H2SO4 = 42.88               # Sum of atomic diffusion volumes for H2SO4

# Condensation numerical thresholds
NEPS_COND = 1.0e-5             # Minimum number for condensation
CS_EPS = 1.0e-20               # Minimum condensation sink

# =========================================================================
# 6. Minimum thresholds for numerical stability
# =========================================================================
TINY_N = 1.0e-20    # Threshold for number concentration
TINY_M = 1.0e-25    # Threshold for mass

# =========================================================================
# 7. Bin Boundaries
# =========================================================================
# 1.7nm start grid (Dunne 2016 nucleation cluster size)
# XK0 = (pi/6) * d^3 * rho, with d=1.7nm, rho=1770 kg/m3
import numpy as _np
_DENS_INIT = 1770.0  # kg/m3 (initial particle density)
XK0 = (_np.pi / 6.0) * (1.7e-9)**3 * _DENS_INIT  # ~4.553e-24 kg

# Legacy 36-bin grid (3.2nm start, for Fortran comparison benchmarks)
XK0_LEGACY = 1.6033e-23  # Lower boundary of old 36-bin grid [kg]
NBINS_LEGACY = 36


def xk_boundaries():
    """Return bin boundary array xk, shape (NBINS+1,).

    Uses JAX arrays for compatibility with JIT-traced code.
    """
    return make_grid(NBINS, XK0, 2.0)


def make_grid(nbins=None, xk0=None, doubling_factor=2.0):
    """Create mass-ratio bin boundary array for arbitrary resolution.

    Args:
        nbins: Number of size bins (default: NBINS=40)
        xk0: Lower boundary of first bin [kg] (default: XK0, 1.7nm start)
        doubling_factor: Mass ratio between adjacent bins
            2.0 = standard TOMAS (40 bins), sqrt(2) = high-res (80 bins)

    Returns:
        xk: Bin boundaries, shape (nbins+1,), JAX float64 array
    """
    import jax.numpy as jnp
    if nbins is None:
        nbins = NBINS
    if xk0 is None:
        xk0 = XK0
    exponents = jnp.arange(nbins + 1, dtype=jnp.float64)
    xk = xk0 * jnp.power(doubling_factor, exponents)
    return xk


def make_grid_80bin():
    """Create 80-bin high-resolution grid (1.7nm start, sqrt(2) mass ratio).

    Same diameter range as the 40-bin default but with double resolution.
    """
    return make_grid(nbins=80, xk0=XK0, doubling_factor=2.0**0.5)