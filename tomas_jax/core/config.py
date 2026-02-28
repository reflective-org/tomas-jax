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
NBINS = 36          # Number of size bins
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