"""Dilution / entrainment — first-order relaxation toward background state.

Represents boundary layer growth, chamber ventilation, or plume entrainment.
Applies independently to Nk (number), Mk (mass), and Gc (gas concentrations).

Physics:
    dC/dt = -kdil * (C - Cbg)

    Analytical solution (exact for constant kdil and Cbg over dt):
        C(t+dt) = Cbg + (C(t) - Cbg) * exp(-kdil * dt)

    When Cbg = 0 (clean-air dilution): C(t+dt) = C(t) * exp(-kdil * dt)

All functions are JIT-compilable.
"""
import jax.numpy as jnp


def dilution_step(Nk, Mk, Gc, dt, kdil,
                  Nk_bg=None, Mk_bg=None, Gc_bg=None):
    """Apply one timestep of dilution toward background concentrations.

    Args:
        Nk: Number concentration [#/grid cell], shape (nbins,)
        Mk: Mass concentration [kg/grid cell], shape (nbins, icomp)
        Gc: Gas-phase concentrations [kg/grid cell], shape (ngas,)
        dt: Timestep [s]
        kdil: Dilution rate [s^-1]
        Nk_bg: Background number concentration (default: zeros)
        Mk_bg: Background mass concentration (default: zeros)
        Gc_bg: Background gas concentration (default: zeros)

    Returns:
        (Nk_new, Mk_new, Gc_new): Diluted concentrations
    """
    # Default backgrounds: clean air (zeros)
    if Nk_bg is None:
        Nk_bg = jnp.zeros_like(Nk)
    if Mk_bg is None:
        Mk_bg = jnp.zeros_like(Mk)
    if Gc_bg is None:
        Gc_bg = jnp.zeros_like(Gc)

    decay = jnp.exp(-kdil * dt)

    Nk_new = Nk_bg + (Nk - Nk_bg) * decay
    Mk_new = Mk_bg + (Mk - Mk_bg) * decay
    Gc_new = Gc_bg + (Gc - Gc_bg) * decay

    return Nk_new, Mk_new, Gc_new


def dilute_tracer(tracer, dt, kdil, tracer_bg=0.0):
    """Dilute a passive scalar tracer (same physics as dilution_step).

    A passive tracer undergoes only dilution — no chemistry, deposition,
    or microphysics. Useful as a reference to isolate the dilution signal
    from reactive species.

    Args:
        tracer: Current tracer value (scalar or array)
        dt: Timestep [s]
        kdil: Dilution rate [s^-1]
        tracer_bg: Background tracer value (default: 0.0)

    Returns:
        Diluted tracer value
    """
    decay = jnp.exp(-kdil * dt)
    return tracer_bg + (tracer - tracer_bg) * decay
