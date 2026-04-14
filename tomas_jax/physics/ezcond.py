"""Simple condensation driver for TOMAS-JAX.

.. deprecated::
    NOT JIT-compilable. For GPU deployment, use ``ezcond_tfl_jax``
    (``method='tfl_jit'``) or ``ezcond_ppm_jax`` (``method='ppm_jit'``).

Exact port of ezcond.f (Jeff Pierce, May 2007).

Takes a prescribed amount of condensing mass (``mcond`` [kg/grid cell])
and distributes it across the aerosol size distribution. This is the
intermediate-level condensation routine that:

    1. Applies MNFIX to ensure consistent input
    2. Computes the condensation sink and per-bin sink fractions
    3. Determines adaptive sub-stepping (limits mass increase to ~10%)
    4. For each sub-step, either:
       (a) Calls tmcond for full TFL bin redistribution (large mass)
       (b) Adds mass directly by sinkfrac (moderate mass)
       (c) Does nothing (negligible mass)
    5. Applies post-hoc mass conservation correction

The caller (``condensation_step``) is responsible for computing ``mcond``
from gas-phase depletion (exponential decay model).

Uses sequential Python loops matching Fortran line-by-line.

References:
    - ezcond.f (Jeff Pierce, May 2007)
"""
import numpy as np
from typing import Tuple

from ..core.config import ICOMP, ICOMP_NODIAG, IDIAG
from .condensation import tmcond
from .condensation_sink import calc_condensation_sink
from ..core.mnfix_jax import mnfix_jax

import jax.numpy as jnp

# Parameters matching Fortran
EPS = 1.0e-40
CS_EPS = 1.0e-20


def ezcond(
    Nki: np.ndarray,
    Mki: np.ndarray,
    mcond: float,
    spec: int,
    xk: np.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Condense a given mass across the aerosol size distribution.

    Exact port of ezcond.f.

    Args:
        Nki: Number concentration [#/grid cell], shape (ibins,)
        Mki: Mass concentration [kg/grid cell], shape (ibins, icomp)
        mcond: Total mass to condense [kg/grid cell]
        spec: Species index to condense (0-based)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        alpha: Accommodation coefficient

    Returns:
        Nkf: Updated number concentration, shape (ibins,)
        Mkf: Updated mass concentration, shape (ibins, icomp)
    """
    ibins = Nki.shape[0]
    icomp = ICOMP
    idiag = IDIAG

    tdt = 2.0 / 3.0

    # Initialize working arrays
    Nk1 = np.array(Nki, dtype=np.float64)
    Mk1 = np.array(Mki, dtype=np.float64)
    xk_np = np.array(xk, dtype=np.float64)

    # MNFIX input (line 99)
    Nk1_jax, Mk1_jax = mnfix_jax(
        jnp.array(Nk1), jnp.array(Mk1), jnp.array(xk_np), ICOMP_NODIAG
    )
    Nk1 = np.array(Nk1_jax)
    Mk1 = np.array(Mk1_jax)

    # Get condensation sink (line 102)
    CS_jax, sinkfrac_jax = calc_condensation_sink(
        jnp.array(Nk1), jnp.array(Mk1),
        temp, pres, boxvol,
        accommodation_coeff=alpha, xk=jnp.array(xk_np)
    )
    CS = float(CS_jax)
    sinkfrac = np.array(sinkfrac_jax)

    # If CS too small, dump mass in first bin (lines 105-121)
    if CS < CS_EPS:
        Nkf = np.copy(Nk1)
        Mkf = np.copy(Mk1)
        Mkf[0, spec] = Mk1[0, spec] + mcond
        Nkf[0] = Nk1[0] + mcond / np.sqrt(xk_np[0] * xk_np[1])
        return Nkf, Mkf

    # Determine nsteps (lines 128-146)
    totsinkfrac = np.sum(sinkfrac)
    nsteps = 1
    for k in range(ibins):
        if sinkfrac[k] < 1.0e-20:
            madd = 0.0
        else:
            madd = mcond * sinkfrac[k] / totsinkfrac if totsinkfrac > 0 else 0.0
        mpo = 0.0
        for j in range(icomp - idiag):
            mpo += Mk1[k, j]
        if mpo > 0.0:
            floor_val = int(madd * 0.00001 / mpo)
            ceil_val = floor_val + 1
            nsteps = max(nsteps, ceil_val)

    # Mass per sub-step (line 150)
    mconds = mcond / nsteps

    # Sub-stepping loop (lines 153-252)
    Nk2 = np.zeros(ibins)
    Mk2 = np.zeros((ibins, icomp))

    for i in range(nsteps):
        # Recalculate sink fractions (except first step)
        if i != 0:
            CS_jax, sinkfrac_jax = calc_condensation_sink(
                jnp.array(Nk1), jnp.array(Mk1),
                temp, pres, boxvol,
                accommodation_coeff=alpha, xk=jnp.array(xk_np)
            )
            sinkfrac = np.array(sinkfrac_jax)
            totsinkfrac = np.sum(sinkfrac)

        # Calculate total dry mass and species mass (lines 163-172)
        tot_m = 0.0
        tot_s = 0.0
        for k in range(ibins):
            for j in range(icomp - idiag):
                tot_m += Mk1[k, j]
                if j == spec:
                    tot_s += Mk1[k, j]

        if mconds > tot_m * 1.0e-3:
            # Path 1: Full tmcond condensation (lines 176-213)
            tau = np.zeros(ibins)
            maddp = np.zeros(ibins)
            for k in range(ibins):
                mpo = 0.0
                mpw = 0.0
                for j in range(icomp - idiag):
                    mpo += Mk1[k, j]
                for j in range(icomp):
                    mpw += Mk1[k, j]
                WR = mpw / mpo if mpo > 0.0 else 1.0

                if Nk1[k] > 0.0 and totsinkfrac > 0.0:
                    maddp[k] = mconds * sinkfrac[k] / totsinkfrac / Nk1[k]
                    mpw_pp = mpw / Nk1[k]
                    tau[k] = 1.5 * ((mpw_pp + maddp[k] * WR) ** tdt - mpw_pp ** tdt)
                else:
                    tau[k] = 0.0
                    maddp[k] = 0.0

            # MNFIX before tmcond (line 202)
            Nk1_jax, Mk1_jax = mnfix_jax(
                jnp.array(Nk1), jnp.array(Mk1), jnp.array(xk_np), ICOMP_NODIAG
            )
            Nk1 = np.array(Nk1_jax)
            Mk1 = np.array(Mk1_jax)

            # Call tmcond with zeros for moxd (SO4 condensation, line 205)
            zeros = np.zeros(ibins)
            Nk2, Mk2 = tmcond(tau, xk_np, Mk1, Nk1, spec, zeros)

        elif mconds > tot_s * 1.0e-12:
            # Path 2: Simple mass addition (lines 218-232)
            for k in range(ibins):
                if Nk1[k] > 0.0 and totsinkfrac > 0.0:
                    maddp_k = mconds * sinkfrac[k] / totsinkfrac
                else:
                    maddp_k = 0.0
                Mk2[k, spec] = Mk1[k, spec] + maddp_k
                for j in range(icomp):
                    if j != spec:
                        Mk2[k, j] = Mk1[k, j]
                Nk2[k] = Nk1[k]
            # MNFIX after simple add (line 232)
            Nk2_jax, Mk2_jax = mnfix_jax(
                jnp.array(Nk2), jnp.array(Mk2), jnp.array(xk_np), ICOMP_NODIAG
            )
            Nk2 = np.array(Nk2_jax)
            Mk2 = np.array(Mk2_jax)
        else:
            # Path 3: Do nothing (lines 233-242)
            Nk2[:] = Nk1[:]
            Mk2[:, :] = Mk1[:, :]

        # Prepare for next sub-step (lines 243-250)
        if i != nsteps - 1:
            Nk1[:] = Nk2[:]
            Mk1[:, :] = Mk2[:, :]

    # Output
    Nkf = np.copy(Nk2)
    Mkf = np.copy(Mk2)

    # Mass conservation correction (lines 261-291)
    tot_i = np.sum(Mki[:, spec])
    tot_f = np.sum(Mkf[:, spec])
    if mcond > 0.0 and abs(mcond - (tot_f - tot_i)) / mcond > 0.0:
        ratio = (tot_f - tot_i) / mcond
        if ratio > 0.0 and ratio < 2.0:  # Fortran: abs(1-ratio) < 1.0
            for k in range(ibins):
                Mkf[k, spec] = Mki[k, spec] + (Mkf[k, spec] - Mki[k, spec]) / ratio
            # MNFIX after correction (line 279)
            Nkf_jax, Mkf_jax = mnfix_jax(
                jnp.array(Nkf), jnp.array(Mkf), jnp.array(xk_np), ICOMP_NODIAG
            )
            Nkf = np.array(Nkf_jax)
            Mkf = np.array(Mkf_jax)

    return Nkf, Mkf
