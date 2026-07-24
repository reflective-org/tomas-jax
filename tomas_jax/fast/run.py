"""Scan-fused multi-hour driver for the reduced model.

The time loop is compiled as jit(scan) segments (default 10 outer steps
per segment) with the state buffer donated between segments, so a 6-hour
run at dt=360 s is 6 XLA program invocations with no host round-trips
inside a segment. Cells are independent, so optional chunking over the
cell axis is exact and cuts peak memory proportionally.
"""
from functools import partial

import jax
import jax.numpy as jnp

from .config import GH2SO4, GSO2, SRTSO4
from .state import FastState
from .step import fast_step


def run_fast(
    state,
    n_steps,
    dt=360.0,
    oh_conc=0.0,
    h2so4_prod=0.0,
    so2_prod=0.0,
    steps_per_segment=10,
    n_cell_chunks=1,
    sort_by_coag_cost=False,
    **step_kwargs,
):
    """Advance a FastState by n_steps outer steps of dt seconds.

    Args:
        state: FastState with leading cell axis C.
        n_steps: Total outer steps (e.g. 60 for 6 h at dt=360 s).
        dt: Outer timestep [s].
        oh_conc / h2so4_prod / so2_prod: scalars or (C,) arrays.
        steps_per_segment: outer steps fused per jit(scan) segment.
        n_cell_chunks: split the cell axis into this many sequential
            chunks (exact — cells are independent). REQUIRED at large C:
            the per-cell coagulation kernel (C, 40, 40) and its triangular
            copies exceed GPU memory near C=1M; use 4-8 chunks there.
        sort_by_coag_cost: with chunking, order cells by their initial
            coagulation loss frequency first (and restore order at the
            end). The adaptive substep count is a batch-max per chunk, so
            sorting confines expensive counts to the chunks that need
            them instead of letting one violent cell slow every chunk.
        step_kwargs: forwarded to fast_step (alpha, fn_scale, caps, ...).

    Returns:
        (state, diags) — diags is a dict of per-step time series stacked
        over all n_steps:
          N_tot (n_steps,): total particle number over all cells
          M_dry (n_steps,): total dry SO4 aerosol mass [kg]
          Gc_h2so4 / Gc_so2 (n_steps,): total gas [kg]
          overflow_so4 (n_steps,): coag top-bin SO4 mass lost this step [kg]
          cond_cap_hit / coag_cap_hit (n_steps,): bool substep-cap flags
    """
    if n_cell_chunks > 1:
        if sort_by_coag_cost:
            return _run_sorted(
                state, n_steps, dt, oh_conc, h2so4_prod, so2_prod,
                steps_per_segment, n_cell_chunks, **step_kwargs,
            )
        return _run_chunked(
            state, n_steps, dt, oh_conc, h2so4_prod, so2_prod,
            steps_per_segment, n_cell_chunks, **step_kwargs,
        )

    # Defensive copy: segments donate their input buffers for in-place
    # reuse; without this the CALLER's arrays would be deleted by the
    # first segment call.
    state = jax.tree_util.tree_map(jnp.array, state)

    def _percell(v):
        return jnp.broadcast_to(
            jnp.asarray(v, dtype=jnp.float64), (state.ncells,)
        )

    oh_conc = _percell(oh_conc)
    h2so4_prod = _percell(h2so4_prod)
    so2_prod = _percell(so2_prod)

    def scan_body(carry, _):
        st, ys = carry  # ys unused; scan carries only the state
        st, diag = fast_step(
            st, dt,
            oh_conc=oh_conc, h2so4_prod=h2so4_prod, so2_prod=so2_prod,
            **step_kwargs,
        )
        out = {
            "N_tot": jnp.sum(st.Nk),
            "M_dry": jnp.sum(st.Mk[..., SRTSO4]),
            "Gc_h2so4": jnp.sum(st.Gc[..., GH2SO4]),
            "Gc_so2": jnp.sum(st.Gc[..., GSO2]),
            "overflow_so4": jnp.sum(diag["coag_overflow"][..., SRTSO4]),
            "cond_cap_hit": diag["cond_cap_hit"],
            "coag_cap_hit": diag["coag_cap_hit"],
        }
        return (st, ys), out

    @partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
    def run_segment(st, length):
        (st, _), ys = jax.lax.scan(
            scan_body, (st, None), None, length=length
        )
        return st, ys

    diags_list = []
    remaining = n_steps
    while remaining > 0:
        length = min(steps_per_segment, remaining)
        state, ys = run_segment(state, length)
        diags_list.append(jax.device_get(ys))
        remaining -= length

    import numpy as np
    diags = {
        k: np.concatenate([d[k] for d in diags_list])
        for k in diags_list[0]
    }
    return state, diags


def _permute_state(state, idx):
    return FastState(
        Nk=state.Nk[idx], Mk=state.Mk[idx], Gc=state.Gc[idx], xk=state.xk,
        temp=state.temp[idx], pres=state.pres[idx],
        boxvol=state.boxvol[idx], rh=state.rh[idx],
    )


def _run_sorted(
    state, n_steps, dt, oh_conc, h2so4_prod, so2_prod,
    steps_per_segment, n_cell_chunks, **step_kwargs,
):
    """Sort cells by initial coagulation loss frequency, run chunked,
    restore the original cell order."""
    import numpy as np

    from .coagulation import _kernel_cell, _loss_frequency_cell

    @jax.jit
    def _lam(st):
        kij = jax.vmap(_kernel_cell)(st.Nk, st.Mk, st.temp, st.pres, st.boxvol)
        lam = jax.vmap(_loss_frequency_cell, in_axes=(0, 0, 0, None))(
            st.Nk, st.Mk, kij, st.xk
        )
        return jnp.max(lam, axis=-1)

    def _maybe_permute(v, idx):
        arr = jnp.asarray(v, dtype=jnp.float64)
        return arr[idx] if arr.ndim > 0 else arr

    order = jnp.argsort(_lam(state))
    inverse = jnp.argsort(order)
    out, diags = _run_chunked(
        _permute_state(state, order), n_steps, dt,
        _maybe_permute(oh_conc, order),
        _maybe_permute(h2so4_prod, order),
        _maybe_permute(so2_prod, order),
        steps_per_segment, n_cell_chunks, **step_kwargs,
    )
    return _permute_state(out, inverse), diags


def _run_chunked(
    state, n_steps, dt, oh_conc, h2so4_prod, so2_prod,
    steps_per_segment, n_cell_chunks, **step_kwargs,
):
    import numpy as np

    C = state.ncells
    bounds = np.linspace(0, C, n_cell_chunks + 1).astype(int)

    def _slice_forcing(v, lo, hi):
        arr = jnp.asarray(v, dtype=jnp.float64)
        return arr[lo:hi] if arr.ndim > 0 else arr

    out_states = []
    diags_acc = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        chunk = FastState(
            Nk=state.Nk[lo:hi], Mk=state.Mk[lo:hi], Gc=state.Gc[lo:hi],
            xk=state.xk, temp=state.temp[lo:hi], pres=state.pres[lo:hi],
            boxvol=state.boxvol[lo:hi], rh=state.rh[lo:hi],
        )
        chunk, diags = run_fast(
            chunk, n_steps, dt,
            oh_conc=_slice_forcing(oh_conc, lo, hi),
            h2so4_prod=_slice_forcing(h2so4_prod, lo, hi),
            so2_prod=_slice_forcing(so2_prod, lo, hi),
            steps_per_segment=steps_per_segment,
            n_cell_chunks=1,
            **step_kwargs,
        )
        out_states.append(jax.device_get(chunk))
        if diags_acc is None:
            diags_acc = diags
        else:
            for k in ("N_tot", "M_dry", "Gc_h2so4", "Gc_so2", "overflow_so4"):
                diags_acc[k] = diags_acc[k] + diags[k]
            for k in ("cond_cap_hit", "coag_cap_hit"):
                diags_acc[k] = diags_acc[k] | diags[k]

    import numpy as np
    merged = FastState(
        Nk=jnp.concatenate([s.Nk for s in out_states]),
        Mk=jnp.concatenate([s.Mk for s in out_states]),
        Gc=jnp.concatenate([s.Gc for s in out_states]),
        xk=state.xk,
        temp=state.temp, pres=state.pres,
        boxvol=state.boxvol, rh=state.rh,
    )
    return merged, diags_acc
