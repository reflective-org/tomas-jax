"""One reduced-model outer step, batched over cells.

Process order (canonical, matching the full model's make_step):
    SO2+OH chemistry -> nucleation -> MNFIX -> water -> coagulation
    -> condensation -> water -> MNFIX

Water equilibrium runs before coagulation (the kernel sees the current
wet size, including freshly nucleated mass — the full model uses the
previous step's water there) and again after condensation, in the
full-model position (before the closing MNFIX).
"""
from functools import partial

import jax
import jax.numpy as jnp

from ..physics.so2_chemistry import calc_k1_so2_oh
from .coagulation import coagulation_step
from .condensation import condensation_step
from .config import GH2SO4, GSO2, ICOMP_NODIAG, MW_H2SO4, MW_SO2
from .mnfix import mnfix
from .nucleation import nucleation_step
from .water import equilibrium_water


def so2_oxidation(Gc, temp, pres, rh, dt, oh_conc):
    """Batched SO2 + OH -> H2SO4 with the compact 2-wide gas array.

    Same analytic pseudo-first-order solution as
    physics/so2_chemistry.so2_oxidation_step (the kg <-> molec/cm^3
    round-trip there cancels algebraically and is omitted).
    """
    k1 = calc_k1_so2_oh(temp, pres, rh)
    so2_kg = Gc[..., GSO2]
    d_so2 = so2_kg * (1.0 - jnp.exp(-k1 * oh_conc * dt))
    d_h2so4 = d_so2 * (MW_H2SO4 / MW_SO2)
    Gc = Gc.at[..., GSO2].add(-d_so2)
    Gc = Gc.at[..., GH2SO4].add(d_h2so4)
    return Gc


def fast_step(
    state,
    dt,
    oh_conc=0.0,
    h2so4_prod=0.0,
    so2_prod=0.0,
    alpha=1.0,
    fn_scale=1.0,
    cond_sub_cap=40,
    coag_sub_cap=256,
    coag_c_max=0.1,
    coag_pallas=False,
):
    """Advance a FastState by one outer step of dt seconds.

    Args:
        state: FastState with leading cell axis C.
        dt: Outer timestep [s].
        oh_conc: OH [molec/cm^3], scalar or (C,).
        h2so4_prod / so2_prod: production rates [kg/cell/s], scalar or (C,).
        alpha: accommodation coefficient (scalar).
        fn_scale: nucleation rate scale.
        cond_sub_cap / coag_sub_cap: static caps for the shared adaptive
            substep counts (PPM CFL / coagulation stability).
        coag_c_max: coagulation stability Courant factor.
        coag_pallas: opt-in fused Triton kernel for the coagulation
            substep loop (GPU only; ~3x faster at high substep counts,
            reassociation-level differences — see fast/coagulation_pallas).

    Returns:
        (state, diag) — diag dict with per-step diagnostics:
        coag_overflow (C, 2) mass lost past the top bin this step,
        cond_cap_hit / coag_cap_hit (bool) substep caps exceeded,
        coag_n_sub (int) the shared coagulation substep count run.
    """
    Nk, Mk, Gc = state.Nk, state.Mk, state.Gc
    xk = state.xk
    temp, pres, boxvol, rh = state.temp, state.pres, state.boxvol, state.rh

    # 0. Gas production (emissions / host-model tendencies)
    Gc = Gc.at[..., GH2SO4].add(h2so4_prod * dt)
    Gc = Gc.at[..., GSO2].add(so2_prod * dt)

    # 1. SO2 + OH chemistry
    Gc = so2_oxidation(Gc, temp, pres, rh, dt, oh_conc)

    # 2. Nucleation (analytic Jbn) + MNFIX
    Nk, Mk, Gc = nucleation_step(Nk, Mk, Gc, temp, boxvol, dt, fn_scale)
    Nk, Mk = mnfix(Nk, Mk, xk, ICOMP_NODIAG)

    # 3. Water equilibrium so coagulation sees current wet sizes
    Mk = equilibrium_water(Mk, temp, rh)

    # 4. Coagulation (adaptive-capped Euler substeps + MNFIX)
    if coag_pallas:
        from .coagulation_pallas import coagulation_step_pallas as coag_fn
    else:
        coag_fn = coagulation_step
    Nk, Mk, coag_overflow, coag_cap_hit, coag_n_sub = coag_fn(
        Nk, Mk, xk, temp, pres, boxvol, dt,
        c_max=coag_c_max, n_sub_cap=coag_sub_cap,
    )

    # 5. Condensation (PPM, global-max capped CFL substeps)
    Nk, Mk, Gc, cond_cap_hit = condensation_step(
        Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
        alpha=alpha, n_sub_cap=cond_sub_cap,
    )

    # 6. Water equilibrium + closing MNFIX (full-model order)
    Mk = equilibrium_water(Mk, temp, rh)
    Nk, Mk = mnfix(Nk, Mk, xk, ICOMP_NODIAG)

    diag = {
        "coag_overflow": coag_overflow,
        "cond_cap_hit": cond_cap_hit,
        "coag_cap_hit": coag_cap_hit,
        "coag_n_sub": coag_n_sub,
    }
    return state.update(Nk=Nk, Mk=Mk, Gc=Gc), diag


def make_fast_step(dt=360.0, jit=True, **step_kwargs):
    """Bind step options and (optionally) JIT.

    Returns step(state, oh_conc=..., h2so4_prod=..., so2_prod=...)
    -> (state, diag). All per-call inputs may be scalars or (C,) arrays.
    """
    fn = partial(fast_step, dt=dt, **step_kwargs)
    return jax.jit(fn) if jit else fn
