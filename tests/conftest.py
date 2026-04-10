"""Shared test fixtures for TOMAS-JAX test suite."""
import pytest
import numpy as np
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO, XK0, PI,
    xk_boundaries,
)


@pytest.fixture
def xk():
    """Standard TOMAS mass-doubling bin boundaries, shape (NBINS+1,)."""
    return xk_boundaries()


def make_lognormal(xk, N_total=1000.0, GMD_um=0.05, GSD=1.8,
                   boxvol=1e6, h2so4_molec_cm3=1e7, dens=1770.0):
    """Create a lognormal aerosol distribution for testing.

    Args:
        xk: Bin boundaries, shape (nbins+1,). JAX or numpy array.
        N_total: Total number concentration [#/cm3].
        GMD_um: Geometric mean diameter [um].
        GSD: Geometric standard deviation.
        boxvol: Box volume [cm3].
        h2so4_molec_cm3: H2SO4 gas concentration [molec/cm3].
        dens: Initial particle density [kg/m3].

    Returns:
        Nk: Number per cell, shape (nbins,)
        Mk: Mass per cell, shape (nbins, ICOMP)
        Gc: Gas concentrations, shape (N_GAS_SPECIES,)
    """
    xk_np = np.array(xk)
    nbins = len(xk_np) - 1

    Nk = np.zeros(nbins)
    Mk = np.zeros((nbins, ICOMP))

    for k in range(nbins):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (dens * PI)) ** (1.0 / 3.0)
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (dens * PI)) ** (1.0 / 3.0)
        Dk = np.sqrt(Dl * Dh)
        n = ((N_total * boxvol)
             / (np.sqrt(2 * PI) * Dk * np.log(GSD))
             * np.exp(-(np.log(Dk / GMD_um) ** 2 / (2 * np.log(GSD) ** 2)))
             * (Dh - Dl))
        Nk[k] = n
        Mk[k, SRTSO4] = n * np.sqrt(xk_np[k] * xk_np[k + 1])

    # Fill empty bins with small values
    for k in range(nbins):
        if Nk[k] < 1e-3:
            Nk[k] = 1e-3
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = 1e-3 * 1.4 * xk_np[k]

    Gc = np.zeros(N_GAS_SPECIES)
    Gc[SRTSO4] = h2so4_molec_cm3 * boxvol * (MW_H2SO4 / 1000.0) / AVOGADRO

    return jnp.array(Nk), jnp.array(Mk), jnp.array(Gc)
