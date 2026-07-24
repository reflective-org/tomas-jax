"""Configuration for the GPU-fast reduced model.

Compact species layout (contrast with core/config.py, ICOMP=44):

    Mk columns:  SRTSO4=0 (sulfate, prognostic dry mass)
                 SRTH2O=1 (water, diagnostic — re-equilibrated each step)
    Gc entries:  GH2SO4=0 (sulfuric acid vapor)
                 GSO2=1   (sulfur dioxide)

The bin grid is identical to the full model (40 bins, 1.7 nm start,
mass-doubling). Bin boundaries are mass boundaries anchored by the
1770 kg/m3 convention in core/config.py and do not vary with RH.
"""
# Import core config FIRST: it sets jax_enable_x64 before any jnp use.
from ..core.config import (  # noqa: F401  (re-exported constants)
    PI,
    KB,
    R_GAS,
    AVOGADRO,
    MOLAR_MASS_AIR,
    MW_H2SO4,
    MW_SO2,
    MW_SO4,
    SV_H2SO4,
    CS_EPS,
    NEPS_COND,
    XK0,
    make_grid,
)

# Model dimensions
NBINS = 40
ICOMP = 2            # [SO4, H2O]
IDIAG = 1            # H2O only
ICOMP_NODIAG = ICOMP - IDIAG  # = 1: only SO4 is prognostic dry mass

# Aerosol Mk column indices
SRTSO4 = 0
SRTH2O = 1

# Gas Gc indices (compact 2-wide array)
GH2SO4 = 0
GSO2 = 1
NGAS = 2

MW_H2O = 18.015


def xk_boundaries():
    """Bin boundary array xk, shape (NBINS+1,). Same grid as the full model."""
    return make_grid(NBINS, XK0, 2.0)
