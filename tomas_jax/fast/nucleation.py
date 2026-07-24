"""Dunne (2016) neutral-binary H2SO4-H2O nucleation, batched + analytic.

Only the binary-neutral channel (Jbn) of the four Dunne mechanisms —
no ions (Jbi/Jti need fion), no NH3 (Jtn), no organics (Riccobono).

Deliberate deviations from physics/nucleation.nucleation_step, documented
in docs/gpu_fast.md:

  - 100% of the cluster mass is deposited as SO4 (the full model deposits
    90% SO4 / 10% organic, with the organic fraction created from nothing).

  - Gas depletion is sulfur-conserving: dGc(H2SO4) = dM(SO4) * 98/96.
    The full model's unclamped path removes H2SO4 kg equal to the SO4 kg
    added, creating sulfur at the 2% level; its clamped path already used
    the 98/96 conversion. With this convention the total-S budget closes
    exactly (see tests/test_fast_step.py).

  - EXACT time integration instead of substepping. With T fixed over the
    step (operator splitting) and every nucleated cluster carrying mass
    _MNUC, the gas obeys

        dG/dt = -A * G^p,   p = 3.95
        A = kbn(T) * (1e-6 * conv)^p * _MNUC * boxvol * (98/96) * fn_scale
        conv = kg -> molec/cm^3 factor

    whose closed form is G(t) = (G0^(1-p) + (p-1) A t)^(1/(1-p)). This is
    the converged limit of the full model's adaptive Euler substep loop
    (validated in tests/test_fast_nucleation.py against a 10^4-substep
    reference), never over-consumes the gas (G(t) > 0 always, so no clamp
    branch), and eliminates both the 1..20 traced loop and the per-substep
    MNFIX. The deposit sits near the bin-0 geometric mean (the 1.7 nm grid
    start was chosen for the Dunne cluster), so one MNFIX after the step
    suffices.
"""
import jax.numpy as jnp

# Cluster mass and Jbn constants shared with the full model for exact
# rate comparability (Dunne 2016 Table S1; Kulmala 2013 cluster).
from ..physics.nucleation import _MNUC, _PBN, _UBN, _VBN, _WBN
from .config import AVOGADRO, GH2SO4, MW_H2SO4, MW_SO4, SRTSO4

_G_TINY = 1.0e-40  # below this, no nucleation (also guards the power law)


def dunne_binary_neutral_rate(temp, h2so4):
    """Dunne 2016 binary-neutral rate Jbn [cm^-3 s^-1].

    Jbn = exp(u - exp(v*(T/1000 - w))) * (h2so4/1e6)^p

    Args:
        temp: Temperature [K], scalar or (C,)
        h2so4: Gas-phase H2SO4 [molec/cm^3], scalar or (C,)
    """
    cna = h2so4 * 1e-6
    kbn = jnp.exp(_UBN - jnp.exp(_VBN * (temp / 1000.0 - _WBN)))
    return kbn * cna**_PBN


def nucleation_step(Nk, Mk, Gc, temp, boxvol, dt, fn_scale=1.0):
    """Jbn-only nucleation over a timestep, integrated exactly.

    Args:
        Nk: (..., nbins); Mk: (..., nbins, 2); Gc: (..., 2)
        temp: [K] (...,); boxvol: [cm^3] (...,); dt: full step [s]
        fn_scale: rate scaling factor

    Returns:
        (Nk, Mk, Gc). Caller runs MNFIX once after this step.
    """
    p = _PBN
    gc0 = Gc[..., GH2SO4]

    conv = 1000.0 / MW_H2SO4 * AVOGADRO / boxvol  # kg/cell -> molec/cm^3
    kbn = jnp.exp(_UBN - jnp.exp(_VBN * (temp / 1000.0 - _WBN)))
    A = (
        kbn * fn_scale
        * (1e-6 * conv) ** p
        * _MNUC * boxvol * (MW_H2SO4 / MW_SO4)
    )

    g_safe = jnp.maximum(gc0, _G_TINY)
    g_new = (g_safe ** (1.0 - p) + (p - 1.0) * A * dt) ** (1.0 / (1.0 - p))
    g_new = jnp.minimum(g_new, gc0)  # numerical safety

    has_nuc = gc0 > _G_TINY
    d_gas = jnp.where(has_nuc, gc0 - g_new, 0.0)
    dM_so4 = d_gas * (MW_SO4 / MW_H2SO4)
    dN = dM_so4 / _MNUC

    Nk = Nk.at[..., 0].add(dN)
    Mk = Mk.at[..., 0, SRTSO4].add(dM_so4)
    Gc = Gc.at[..., GH2SO4].add(-d_gas)
    return Nk, Mk, Gc
