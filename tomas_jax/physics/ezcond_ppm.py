"""PPM-aware simple condensation driver for TOMAS-JAX.

Analogous to ezcond.py but replaces the TFL tmcond call with the
PPM advection scheme (ppm_condensation_step). The TAU computation
uses the full mcond (not subdivided by nsteps) since PPM handles
its own CFL-based substepping internally.

Algorithm:
    1. MNFIX input
    2. Compute condensation sink + sinkfrac
    3. If CS too small -> dump mass in first bin
    4. Compute TAU per-bin from full mcond
    5. Call ppm_condensation_step (CFL substeps handled internally)
       - PPM transports ALL species (including cspecies) via upwind flux
    6. Add condensed mass proportionally to sinkfrac after transport
    7. MNFIX output

References:
    - ezcond.f (Jeff Pierce, May 2007) for the driver structure
    - condensation_ppm.py for the PPM advection algorithm
"""
import numpy as np
import jax.numpy as jnp
from typing import Tuple

from ..core.config import NBINS, ICOMP, ICOMP_NODIAG, IDIAG
from .condensation_ppm import ppm_condensation_step
from .condensation_sink import calc_condensation_sink
from ..core.mnfix_jax import mnfix_jax

# Parameters matching Fortran ezcond.f
EPS = 1.0e-40
CS_EPS = 1.0e-20


def ezcond_ppm(
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
    """Condense a given mass using PPM advection scheme.

    Same interface as ezcond() but uses PPM instead of TFL tmcond.
    TAU is computed for the full mcond (not subdivided) since PPM
    handles CFL-based substepping internally.

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
    ibins = NBINS
    icomp = ICOMP
    idiag = IDIAG

    tdt = 2.0 / 3.0

    # Initialize working arrays
    Nk1 = np.array(Nki, dtype=np.float64)
    Mk1 = np.array(Mki, dtype=np.float64)
    xk_np = np.array(xk, dtype=np.float64)

    # MNFIX input (matching ezcond.f line 99)
    Nk1_jax, Mk1_jax = mnfix_jax(
        jnp.array(Nk1), jnp.array(Mk1), jnp.array(xk_np), ICOMP_NODIAG
    )
    Nk1 = np.array(Nk1_jax)
    Mk1 = np.array(Mk1_jax)

    # Get condensation sink
    CS_jax, sinkfrac_jax = calc_condensation_sink(
        jnp.array(Nk1), jnp.array(Mk1),
        temp, pres, boxvol,
        accommodation_coeff=alpha
    )
    CS = float(CS_jax)
    sinkfrac = np.array(sinkfrac_jax)

    # If CS too small, dump mass in first bin (same as TFL path)
    if CS < CS_EPS:
        Nkf = np.copy(Nk1)
        Mkf = np.copy(Mk1)
        Mkf[0, spec] = Mk1[0, spec] + mcond
        Nkf[0] = Nk1[0] + mcond / np.sqrt(xk_np[0] * xk_np[1])
        return Nkf, Mkf

    totsinkfrac = np.sum(sinkfrac)

    # Compute TAU for the full mcond (not subdivided by nsteps)
    # PPM handles its own CFL-based substepping
    tau = np.zeros(ibins)
    for k in range(ibins):
        mpo = 0.0
        mpw = 0.0
        for j in range(icomp - idiag):
            mpo += Mk1[k, j]
        for j in range(icomp):
            mpw += Mk1[k, j]
        WR = mpw / mpo if mpo > 0.0 else 1.0

        if Nk1[k] > 0.0 and totsinkfrac > 0.0:
            maddp_k = mcond * sinkfrac[k] / totsinkfrac / Nk1[k]
            mpw_pp = mpw / Nk1[k]
            tau[k] = 1.5 * ((mpw_pp + maddp_k * WR) ** tdt - mpw_pp ** tdt)
        else:
            tau[k] = 0.0

    # Check if mcond is significant enough to run PPM
    tot_m = 0.0
    tot_s = 0.0
    for k in range(ibins):
        for j in range(icomp - idiag):
            tot_m += Mk1[k, j]
            if j == spec:
                tot_s += Mk1[k, j]

    if mcond > tot_m * 1.0e-3:
        # Full PPM condensation: transport + add mass
        # dt=1.0 since TAU already encodes the full growth forcing
        Nk2_jax, Mk2_jax = ppm_condensation_step(
            jnp.array(Nk1), jnp.array(Mk1), jnp.array(xk_np),
            jnp.array(tau), spec, dt=1.0,
            icomp_nodiag=ICOMP_NODIAG
        )
        Nk2 = np.array(Nk2_jax)
        Mk2 = np.array(Mk2_jax)

        # Add condensed mass proportionally to sinkfrac
        if totsinkfrac > 0.0:
            for k in range(ibins):
                Mk2[k, spec] += mcond * sinkfrac[k] / totsinkfrac

    elif mcond > tot_s * 1.0e-12:
        # Path 2: Simple mass addition (same as TFL ezcond)
        Nk2 = np.copy(Nk1)
        Mk2 = np.copy(Mk1)
        for k in range(ibins):
            if Nk1[k] > 0.0 and totsinkfrac > 0.0:
                maddp_k = mcond * sinkfrac[k] / totsinkfrac
            else:
                maddp_k = 0.0
            Mk2[k, spec] = Mk1[k, spec] + maddp_k
    else:
        # Path 3: Do nothing
        Nk2 = np.copy(Nk1)
        Mk2 = np.copy(Mk1)

    # Final MNFIX
    Nkf_jax, Mkf_jax = mnfix_jax(
        jnp.array(Nk2), jnp.array(Mk2), jnp.array(xk_np), ICOMP_NODIAG
    )
    Nkf = np.array(Nkf_jax)
    Mkf = np.array(Mkf_jax)

    return Nkf, Mkf
