"""Dilution / entrainment — volume expansion approach.

Represents boundary layer growth, chamber ventilation, or plume entrainment
by expanding the box volume.  Concentrations decrease as 1/V while total
per-cell mass is conserved (for clean-air dilution).

Physics:
    dV/dt = kdil * V

    Analytical solution:
        V(t+dt) = V(t) * exp(kdil * dt)

    For clean air (no background): total Nk, Mk, Gc unchanged.
        Concentrations decrease as C(t) = C(0) * V(0)/V(t)
                                        = C(0) * exp(-kdil * t).

    For ambient background: entrained air brings mass at background
    concentration, so total per-cell mass increases by c_bg * dV.

Reference: TRACER_SOM-TOMAS box.f — BOXVOL = initial_boxvol * dilt_rate_new

All functions are JIT-compilable.
"""
import jax.numpy as jnp


def dilution_step(boxvol, Nk, Mk, Gc, dt, kdil,
                  Nk_bg_conc=None, Mk_bg_conc=None, Gc_bg_conc=None):
    """Apply one timestep of dilution by expanding box volume.

    The box volume grows exponentially.  For clean-air dilution (background
    concentrations = 0), total per-cell mass is exactly conserved —
    only concentrations (#/cm³, kg/cm³) decrease.  For ambient entrainment,
    the entrained volume brings in background mass.

    Args:
        boxvol: Current box volume [cm³]
        Nk: Number [#/grid cell], shape (nbins,)
        Mk: Mass [kg/grid cell], shape (nbins, icomp)
        Gc: Gas concentrations [kg/grid cell], shape (ngas,)
        dt: Timestep [s]
        kdil: Dilution rate [s⁻¹]
        Nk_bg_conc: Background number concentration [#/cm³] (default: zeros)
        Mk_bg_conc: Background mass concentration [kg/cm³] (default: zeros)
        Gc_bg_conc: Background gas concentration [kg/cm³] (default: zeros)

    Returns:
        (boxvol_new, Nk_new, Mk_new, Gc_new)
    """
    boxvol_new = boxvol * jnp.exp(kdil * dt)
    dV = boxvol_new - boxvol

    # Default: clean air (zero background concentrations)
    if Nk_bg_conc is None:
        Nk_bg_conc = jnp.zeros_like(Nk)
    if Mk_bg_conc is None:
        Mk_bg_conc = jnp.zeros_like(Mk)
    if Gc_bg_conc is None:
        Gc_bg_conc = jnp.zeros_like(Gc)

    # Add mass from entrained volume (zero for clean air)
    Nk_new = Nk + Nk_bg_conc * dV
    Mk_new = Mk + Mk_bg_conc * dV
    Gc_new = Gc + Gc_bg_conc * dV

    return boxvol_new, Nk_new, Mk_new, Gc_new


def dilute_tracer(tracer, dt, kdil, tracer_bg=0.0):
    """Dilute a passive scalar tracer.

    Tracks concentration decrease due to volume expansion:
        C(t) / C(0) = V(0) / V(t) = exp(-kdil * t)

    For non-zero background the tracer relaxes toward tracer_bg.

    Args:
        tracer: Current tracer value (scalar or array)
        dt: Timestep [s]
        kdil: Dilution rate [s⁻¹]
        tracer_bg: Background tracer value (default: 0.0)

    Returns:
        Diluted tracer value
    """
    decay = jnp.exp(-kdil * dt)
    return tracer_bg + (tracer - tracer_bg) * decay
