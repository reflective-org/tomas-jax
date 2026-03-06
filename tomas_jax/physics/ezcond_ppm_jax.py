"""Pure-JAX PPM condensation driver — JIT-compilable replacement for ezcond_ppm.py.

Same algorithm as ezcond_ppm.py but with:
- All numpy replaced by jax.numpy
- Python for-loops replaced by vectorized ops
- Three-way branch via jax.lax.cond
- No internal MNFIX (caller handles it)

This makes the entire condensation pipeline traceable by JAX, enabling
JIT compilation and jax.lax.scan fusion of the 1440-step time loop.
"""
import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial

from ..core.config import ICOMP, ICOMP_NODIAG
from .condensation_ppm import ppm_condensation_step
from .condensation_sink import calc_condensation_sink


# Parameters matching Fortran ezcond.f
EPS = 1.0e-40
CS_EPS = 1.0e-20


def ezcond_ppm_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    mcond: jnp.ndarray,
    spec: int,
    xk: jnp.ndarray,
    temp: jnp.ndarray,
    pres: jnp.ndarray,
    boxvol: jnp.ndarray,
    alpha: jnp.ndarray = jnp.array(1.0),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Condense a given mass using PPM advection scheme (pure JAX, JIT-compilable).

    Same interface and algorithm as ezcond_ppm() but fully traceable.
    No internal MNFIX — the caller must call mnfix_jax before and after.

    Note: `spec` must be static when JIT-compiling (used as array index).
    The caller should use static_argnames=['spec'] or equivalent.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        mcond: Total mass to condense [kg/grid cell] (scalar)
        spec: Species index to condense (0-based, static)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        alpha: Accommodation coefficient

    Returns:
        Nk_out: Updated number concentration, shape (ibins,)
        Mk_out: Updated mass concentration, shape (ibins, icomp)
    """
    tdt = 2.0 / 3.0

    # Condensation sink (needed for outer branch decision)
    CS, sinkfrac = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        accommodation_coeff=alpha, xk=xk
    )

    # --- Dump path (CS too small): add all mass to first bin ---
    def dump_path(_):
        Nk_d = Nk.at[0].add(mcond / jnp.sqrt(xk[0] * xk[1]))
        Mk_d = Mk.at[0, spec].add(mcond)
        return Nk_d, Mk_d

    # --- Condensation path (CS significant): full TAU + PPM/simple/noop ---
    def cond_path(_):
        totsinkfrac = jnp.sum(sinkfrac)

        # Compute TAU (vectorized)
        mpo = jnp.sum(Mk[:, :ICOMP_NODIAG], axis=1)       # dry mass per bin
        mpw = jnp.sum(Mk, axis=1)                           # wet mass per bin
        safe_mpo = jnp.maximum(mpo, 1e-30)
        WR = jnp.where(mpo > 0.0, mpw / safe_mpo, 1.0)

        safe_Nk = jnp.maximum(Nk, 1e-30)
        safe_totsinkfrac = jnp.maximum(totsinkfrac, 1e-30)
        maddp = mcond * sinkfrac / safe_totsinkfrac / safe_Nk
        mpw_pp = mpw / safe_Nk

        tau = jnp.where(
            (Nk > 0.0) & (totsinkfrac > 0.0),
            1.5 * (jnp.power(mpw_pp + maddp * WR, tdt)
                   - jnp.power(mpw_pp, tdt)),
            0.0
        )

        # Significance thresholds
        tot_m = jnp.sum(Mk[:, :ICOMP_NODIAG])
        tot_s = jnp.sum(Mk[:, spec])

        # Path 1: Full PPM condensation
        def ppm_path(args):
            Nk_in, Mk_in, tau_in = args
            Nk2, Mk2 = ppm_condensation_step(
                Nk_in, Mk_in, xk, tau_in, spec, dt=1.0,
                icomp_nodiag=ICOMP_NODIAG
            )
            mass_add = mcond * sinkfrac / safe_totsinkfrac
            Mk2 = Mk2.at[:, spec].add(mass_add)
            return Nk2, Mk2

        # Path 2: Simple proportional mass add
        def simple_add_path(args):
            Nk_in, Mk_in, _ = args
            mass_add = jnp.where(
                (Nk_in > 0.0) & (totsinkfrac > 0.0),
                mcond * sinkfrac / safe_totsinkfrac,
                0.0
            )
            Mk_out = Mk_in.at[:, spec].add(mass_add)
            return Nk_in, Mk_out

        # Path 3: No-op
        def noop_path(args):
            return args[0], args[1]

        sub_args = (Nk, Mk, tau)

        def significant_path(args):
            return jax.lax.cond(
                mcond > tot_m * 1.0e-3,
                ppm_path,
                simple_add_path,
                args
            )

        return jax.lax.cond(
            mcond > tot_s * 1.0e-12,
            significant_path,
            noop_path,
            sub_args
        )

    # Outer branch: skip heavy computation when CS is negligible
    return jax.lax.cond(
        CS < CS_EPS,
        dump_path,
        cond_path,
        None
    )
