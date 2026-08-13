"""Batched coagulation: frozen Brownian kernel + adaptive-capped Euler substeps.

Mirrors solvers/diffrax.coag_euler_step (kernel frozen per outer step,
forward Euler + MNFIX per substep, top-bin overflow accumulated) but with
the substep count chosen adaptively per outer step from a stability /
accuracy criterion, shared across the whole batch:

    lambda_k = kij_kk * N_k  +  sum_{j>k} kij_kj * N_j  +  2 * K1M_k / xk_k
    n_sub    = clip(ceil(dt * max_over_cells_bins(lambda) / c_max), 1, cap)

The three terms are the per-particle loss frequencies of the TFL scheme:
self-coagulation, collision with larger bins, and promotion by the mass
flux arriving from below (K1M_k = sum_{i<k} kij_ki * M_i; the 2/xk factor
is the TFL phi/eff scale). Forward Euler is only accurate while
dt_sub * lambda is small; the default c_max = 0.1 gives sig-bin errors
indistinguishable from 0.05 (p50 7e-7, p99 1.2e-3 vs a 0.0125/cap-4096
converged reference on the stiffest 1M-benchmark cells — the 256-substep
cap, not c_max, limits those cells) at ~half the substep demand of
non-capped chunks. Fixed coarse substeps lose >10% of mass to the
positivity clamp in burst scenarios, which motivated the adaptive
policy (original 0.05 calibration in docs/gpu_fast.md).
Realistic distributions need n_sub ~ 1-15 per 360 s; nucleation-burst
cells can demand ~100+. The cap bounds wall time; hitting it degrades the
stiffest cells toward coarser accuracy and is reported via cap_hit.
Manage cost at large C with sorted cell chunking (run_fast
sort_by_coag_cost), not by lowering the cap.

The per-cell math (particle properties, kernel, TFL rates) is reused from
the full model via jax.vmap; only the density call is swapped.
"""
import jax
import jax.numpy as jnp

from ..physics.coagulation_kernel import calc_coagulation_kernel
from ..physics.coagulation_rates import calc_coagulation_rates
from ..physics.properties import calc_air_properties, calc_diffusivity_vectorized
from .config import ICOMP_NODIAG, KB, PI, SRTSO4
from .density import calc_density
from .mnfix import mnfix


def _particle_properties_cell(Nk, Mk, temp, pres):
    """Per-cell copy of physics/properties.calc_particle_properties with
    the binary H2SO4/H2O density."""
    mu, mfp = calc_air_properties(temp, pres)
    rho_p = calc_density(Mk)

    total_mass = jnp.sum(Mk, axis=-1)
    safe_Nk = jnp.maximum(Nk, 1e-20)
    mp = jnp.where(Nk > 1e-20, total_mass / safe_Nk, 1e-25)

    Dpk = jnp.cbrt(mp / rho_p * (6.0 / PI))
    ck = jnp.sqrt(8.0 * KB * temp / (PI * mp))
    Dk = calc_diffusivity_vectorized(temp, mu, Dpk, mfp)
    return Dpk, Dk, ck


def _kernel_cell(Nk, Mk, temp, pres, boxvol):
    Dpk, Dk, ck = _particle_properties_cell(Nk, Mk, temp, pres)
    return calc_coagulation_kernel(Dpk, Dk, ck, boxvol)


def _loss_frequency_cell(Nk, Mk, kij, xk):
    """Per-bin loss frequency [1/s] bounding the Euler substep size.

    All bins count, including near-empty ones: a bin with negligible
    standing content can still carry the cell's whole promotion mass flux
    (2*K1M/xk), and under-resolving it corrupts the downstream cascade
    (calibration: masking such bins loses 3% of mass and distorts
    populated bins ~7x on adversarial states).
    """
    in_term = jnp.triu(kij, k=1) @ Nk
    k1m = jnp.tril(kij, k=-1) @ Mk[:, SRTSO4]
    return jnp.diag(kij) * Nk + in_term + 2.0 * k1m / xk[:-1]


def coagulation_step(
    Nk, Mk, xk, temp, pres, boxvol, dt, c_max=0.1, n_sub_cap=256
):
    """One coagulation step with adaptive-capped shared substeps.

    Args:
        Nk: (C, nbins); Mk: (C, nbins, 2)
        xk: (nbins+1,); temp/pres/boxvol: (C,); dt: [s]
        c_max: max fraction of a bin's content lost per substep (accuracy
            knob; 0.05 keeps significant bins within ~1% of converged)
        n_sub_cap: static cap on the shared substep count

    Returns:
        (Nk, Mk, overflow, cap_hit, n_sub) — overflow is the mass
        [kg/cell] lost past the top bin this step, shape (C, 2); cap_hit
        is a bool scalar; n_sub the shared substep count actually run.
    """
    kij = jax.vmap(_kernel_cell)(Nk, Mk, temp, pres, boxvol)

    lam = jax.vmap(_loss_frequency_cell, in_axes=(0, 0, 0, None))(
        Nk, Mk, kij, xk
    )
    n_raw = jnp.ceil(dt * jnp.max(lam) / c_max).astype(jnp.int32)
    n_sub = jnp.clip(n_raw, 1, n_sub_cap)
    cap_hit = n_raw > n_sub_cap
    dt_sub = dt / n_sub

    # The kernel is frozen for the whole outer step: build its triangular
    # decomposition once so the substep loop only does the batched matvecs
    # (rebuilding tril/triu masks per substep re-materializes (C, B, B)).
    kij_parts = (
        jnp.tril(kij, k=-1),
        jnp.triu(kij, k=1),
        jnp.diagonal(kij, axis1=-2, axis2=-1),
    )

    rates = jax.vmap(
        lambda n, m, kl, ku, kd: calc_coagulation_rates(
            n, m, None, xk, ICOMP_NODIAG, kij_parts=(kl, ku, kd)
        )
    )

    def substep(_, carry):
        Nk_c, Mk_c, ovf = carry
        dNdt, dMdt, dOvf = rates(Nk_c, Mk_c, *kij_parts)
        Nk_n = jnp.maximum(Nk_c + dt_sub * dNdt, 0.0)
        Mk_n = jnp.maximum(Mk_c + dt_sub * dMdt, 0.0)
        ovf = ovf + dt_sub * dOvf
        Nk_n, Mk_n = mnfix(Nk_n, Mk_n, xk, ICOMP_NODIAG)
        return (Nk_n, Mk_n, ovf)

    overflow0 = jnp.zeros(Mk.shape[:-2] + (Mk.shape[-1],))
    Nk_f, Mk_f, overflow = jax.lax.fori_loop(
        0, n_sub, substep, (Nk, Mk, overflow0)
    )
    return Nk_f, Mk_f, overflow, cap_hit, n_sub
