"""Scan-fused multi-hour driver for the reduced model.

The time loop is compiled as jit(scan) segments (default 10 outer steps
per segment) with the state buffer donated between segments, so a 6-hour
run at dt=360 s is 6 XLA program invocations with no host round-trips
inside a segment. Cells are independent, so optional chunking over the
cell axis is exact and cuts peak memory proportionally.

Forcings (oh_conc, h2so4_prod, so2_prod) accept, independently of each
other:
    scalar          — constant in time, uniform over cells
    (C,)            — constant in time, per cell
    (n_steps, 1)    — time profile, uniform over cells
    (n_steps, C)    — time profile, per cell
Time-varying forcings are fed through the scan (one row per outer step).
"""
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from .config import GH2SO4, GSO2, SRTSO4
from .state import FastState
from .step import fast_step

_FORCING_NAMES = ("oh_conc", "h2so4_prod", "so2_prod")

# How each diags key combines across cell chunks. Every key emitted by
# the scan body MUST appear here — _merge_diags raises KeyError on an
# unlisted key rather than silently returning first-chunk-only values.
_DIAG_MERGE = {
    "N_tot": "sum",
    "M_dry": "sum",
    "Gc_h2so4": "sum",
    "Gc_so2": "sum",
    "overflow_so4": "sum",
    "cond_cap_hit": "or",
    "coag_cap_hit": "or",
    "coag_n_sub": "max",
}


def _merge_diags(acc, d):
    """Merge one chunk's diags into the accumulator (host numpy)."""
    if acc is None:
        return dict(d)
    for k in d:
        rule = _DIAG_MERGE[k]
        if rule == "sum":
            acc[k] = acc[k] + d[k]
        elif rule == "or":
            acc[k] = acc[k] | d[k]
        else:
            acc[k] = np.maximum(acc[k], d[k])
    return acc


def _empty_diags():
    return {
        k: np.zeros(0, dtype=bool if rule == "or" else
                    (np.int32 if rule == "max" else np.float64))
        for k, rule in _DIAG_MERGE.items()
    }


def _normalize_forcing(name, v, n_steps, ncells):
    """Return ((C,) array, time_varying=False) or ((n_steps, C), True)."""
    arr = jnp.asarray(v, dtype=jnp.float64)
    if arr.ndim <= 1:
        return jnp.broadcast_to(arr, (ncells,)), False
    if arr.ndim == 2:
        if arr.shape[0] != n_steps:
            raise ValueError(
                f"{name}: time-varying forcing must have shape "
                f"(n_steps={n_steps}, C or 1), got {arr.shape}"
            )
        return jnp.broadcast_to(arr, (n_steps, ncells)), True
    raise ValueError(f"{name}: expected ndim <= 2, got shape {arr.shape}")


@lru_cache(maxsize=None)
def _segment_runner(dt, step_kwargs_items):
    """Build (once per (dt, step_kwargs)) the jitted segment program.

    Module-level cache so every chunk, segment, and run_fast call reuses
    the same compiled executable — a fresh jax.jit closure per call
    means re-tracing AND re-compiling (~seconds each on GPU), which
    dominated chunked runs (8 chunks x 6 segments = 48 recompiles of an
    identical program at 1M cells).

    All forcings are passed as per-step scan rows (length, C) for a
    fixed signature; `length` is static (one compile per distinct
    segment length — the final short segment at most).
    """
    step_kwargs = dict(step_kwargs_items)

    def scan_body(st, xs):
        st, diag = fast_step(st, dt, **xs, **step_kwargs)
        out = {
            "N_tot": jnp.sum(st.Nk),
            "M_dry": jnp.sum(st.Mk[..., SRTSO4]),
            "Gc_h2so4": jnp.sum(st.Gc[..., GH2SO4]),
            "Gc_so2": jnp.sum(st.Gc[..., GSO2]),
            "overflow_so4": jnp.sum(diag["coag_overflow"][..., SRTSO4]),
            "cond_cap_hit": diag["cond_cap_hit"],
            "coag_cap_hit": diag["coag_cap_hit"],
            "coag_n_sub": diag["coag_n_sub"],
        }
        return st, out

    @partial(jax.jit, donate_argnums=(0,), static_argnums=(2,))
    def run_segment(st, xs, length):
        return jax.lax.scan(scan_body, st, xs, length=length)

    return run_segment


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
        oh_conc / h2so4_prod / so2_prod: scalar, (C,), (n_steps, 1), or
            (n_steps, C) — see module docstring. Row t applies to step t.
        steps_per_segment: outer steps fused per jit(scan) segment.
        n_cell_chunks: split the cell axis into this many sequential
            chunks (exact — cells are independent). REQUIRED at large C:
            the per-cell coagulation kernel (C, 40, 40) and its triangular
            copies exceed GPU memory near C=1M; use 4-8 chunks there.
        sort_by_coag_cost: with chunking, re-order cells by their current
            coagulation loss frequency before every segment (and restore
            order at the end). The adaptive substep count is a batch-max
            per chunk, so sorting confines expensive counts to the chunks
            that need them instead of letting one violent cell slow every
            chunk; re-sorting each segment tracks stiffness as it evolves
            (nucleation bursts).
        step_kwargs: forwarded to fast_step (alpha, fn_scale, caps, ...).

    Returns:
        (state, diags) — diags is a dict of per-step time series stacked
        over all n_steps:
          N_tot (n_steps,): total particle number over all cells
          M_dry (n_steps,): total dry SO4 aerosol mass [kg]
          Gc_h2so4 / Gc_so2 (n_steps,): total gas [kg]
          overflow_so4 (n_steps,): coag top-bin SO4 mass lost this step [kg]
          cond_cap_hit / coag_cap_hit (n_steps,): bool substep-cap flags
          coag_n_sub (n_steps,): shared coagulation substep count run
              this step; with chunking, the MAX over chunks (stiffest
              chunk), not the count every cell ran
    """
    if n_steps == 0:
        return state, _empty_diags()

    forcings = {
        "oh_conc": oh_conc, "h2so4_prod": h2so4_prod, "so2_prod": so2_prod
    }

    if n_cell_chunks > 1:
        runner = _run_sorted if sort_by_coag_cost else _run_chunked
        return runner(
            state, n_steps, dt, forcings,
            steps_per_segment, n_cell_chunks, **step_kwargs,
        )

    # Defensive copy: segments donate their input buffers for in-place
    # reuse; without this the CALLER's arrays would be deleted by the
    # first segment call.
    state = jax.tree_util.tree_map(jnp.array, state)

    # Constant forcings stay (C,) for the whole call and are broadcast
    # to per-step rows one SEGMENT at a time (steps_per_segment x C —
    # never a dense (n_steps, C) buffer); time-varying forcings are
    # already (n_steps, C) and sliced per segment. Either way the
    # cached segment runner sees one fixed (length, C) signature.
    C = state.ncells
    norm = {
        name: _normalize_forcing(name, forcings[name], n_steps, C)
        for name in _FORCING_NAMES
    }

    try:
        run_segment = _segment_runner(
            float(dt), tuple(sorted(step_kwargs.items()))
        )
    except TypeError:
        # Unhashable step kwarg (e.g. an array): fall back to an
        # uncached runner; recompiles per run_fast call.
        run_segment = _segment_runner.__wrapped__(
            float(dt), tuple(step_kwargs.items())
        )

    diags_list = []
    done = 0
    while done < n_steps:
        length = min(steps_per_segment, n_steps - done)
        xs = {
            k: (arr[done:done + length] if is_varying
                else jnp.broadcast_to(arr, (length, C)))
            for k, (arr, is_varying) in norm.items()
        }
        state, ys = run_segment(state, xs, length)
        diags_list.append(jax.device_get(ys))
        done += length

    diags = {
        k: np.concatenate([d[k] for d in diags_list])
        for k in diags_list[0]
    }
    return state, diags


def _permute_state(state, idx):
    """Index the cell axis of every per-cell field (int array or slice)."""
    return FastState(
        Nk=state.Nk[idx], Mk=state.Mk[idx], Gc=state.Gc[idx], xk=state.xk,
        temp=state.temp[idx], pres=state.pres[idx],
        boxvol=state.boxvol[idx], rh=state.rh[idx],
    )


def _run_sorted(
    state, n_steps, dt, forcings,
    steps_per_segment, n_cell_chunks, **step_kwargs,
):
    """Re-sort cells by current coagulation loss frequency before every
    segment, run each segment chunked, restore original cell order.

    Stiffness evolves during the run (nucleation bursts), so a single
    t=0 sort stops confining the expensive substep counts to the stiff
    chunks after a few segments — measured 2.7x slowdown at 1M cells.
    The state stays in sorted order between segments; `perm` tracks the
    composite permutation (current position -> original cell index).
    """
    from .coagulation import _kernel_cell, _loss_frequency_cell

    C = state.ncells
    bounds = np.linspace(0, C, n_cell_chunks + 1).astype(int)

    @jax.jit
    def _lam_chunk(Nk, Mk, temp, pres, boxvol, xk):
        kij = jax.vmap(_kernel_cell)(Nk, Mk, temp, pres, boxvol)
        lam = jax.vmap(_loss_frequency_cell, in_axes=(0, 0, 0, None))(
            Nk, Mk, kij, xk
        )
        return jnp.max(lam, axis=-1)

    def _lam(st):
        # chunked so the (C, B, B) kernel stays within the chunk budget
        return jnp.concatenate([
            _lam_chunk(st.Nk[lo:hi], st.Mk[lo:hi], st.temp[lo:hi],
                       st.pres[lo:hi], st.boxvol[lo:hi], st.xk)
            for lo, hi in zip(bounds[:-1], bounds[1:])
        ])

    # Normalize (and upload) every forcing exactly once, in ORIGINAL
    # cell order: (C,) if constant, (n_steps, C) if time-varying.
    norm = {
        name: _normalize_forcing(name, forcings[name], n_steps, C)
        for name in _FORCING_NAMES
    }

    perm = jnp.arange(C)
    diags_segments = []
    done = 0
    while done < n_steps:
        length = min(steps_per_segment, n_steps - done)

        order = jnp.argsort(_lam(state))
        state = _permute_state(state, order)
        perm = perm[order]

        # Time rows for this segment, sliced BEFORE any cell gather so
        # per-chunk work is (length, chunk) — not (n_steps, chunk).
        seg_rows = {
            k: (arr[done:done + length] if is_varying else arr)
            for k, (arr, is_varying) in norm.items()
        }

        seg_states = []
        diags_acc = None
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            chunk = _permute_state(state, slice(lo, hi))
            # forcings live in ORIGINAL cell order: route this chunk's
            # original indices through the current permutation
            idx = perm[lo:hi]
            chunk_forcings = {
                k: (seg_rows[k][:, idx] if norm[k][1] else seg_rows[k][idx])
                for k in norm
            }
            chunk, diags = run_fast(
                chunk, length, dt,
                steps_per_segment=steps_per_segment,
                n_cell_chunks=1,
                **chunk_forcings,
                **step_kwargs,
            )
            seg_states.append(chunk)
            diags_acc = _merge_diags(diags_acc, diags)
        diags_segments.append(diags_acc)

        # Reassemble in sorted order. Env fields (temp/pres/boxvol/rh)
        # are taken from the pre-run permuted state: fast_step evolves
        # only Nk/Mk/Gc. If a process ever makes an env field
        # prognostic, take it from the chunk outputs here.
        state = FastState(
            Nk=jnp.concatenate([s.Nk for s in seg_states]),
            Mk=jnp.concatenate([s.Mk for s in seg_states]),
            Gc=jnp.concatenate([s.Gc for s in seg_states]),
            xk=state.xk,
            temp=state.temp, pres=state.pres,
            boxvol=state.boxvol, rh=state.rh,
        )
        seg_states.clear()
        done += length

    diags = {
        k: np.concatenate([d[k] for d in diags_segments])
        for k in diags_segments[0]
    }
    inverse = jnp.argsort(perm)
    return _permute_state(state, inverse), diags


def _run_chunked(
    state, n_steps, dt, forcings,
    steps_per_segment, n_cell_chunks, **step_kwargs,
):
    C = state.ncells
    bounds = np.linspace(0, C, n_cell_chunks + 1).astype(int)

    # Normalize (and upload) every forcing exactly once.
    norm = {
        name: _normalize_forcing(name, forcings[name], n_steps, C)
        for name in _FORCING_NAMES
    }

    out_states = []
    diags_acc = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        chunk = _permute_state(state, slice(lo, hi))
        chunk_forcings = {
            k: (arr[:, lo:hi] if is_varying else arr[lo:hi])
            for k, (arr, is_varying) in norm.items()
        }
        chunk, diags = run_fast(
            chunk, n_steps, dt,
            steps_per_segment=steps_per_segment,
            n_cell_chunks=1,
            **chunk_forcings,
            **step_kwargs,
        )
        out_states.append(jax.device_get(chunk))
        diags_acc = _merge_diags(diags_acc, diags)

    # Env fields from the input state: fast_step evolves only Nk/Mk/Gc
    # (see note in _run_sorted).
    merged = FastState(
        Nk=jnp.concatenate([s.Nk for s in out_states]),
        Mk=jnp.concatenate([s.Mk for s in out_states]),
        Gc=jnp.concatenate([s.Gc for s in out_states]),
        xk=state.xk,
        temp=state.temp, pres=state.pres,
        boxvol=state.boxvol, rh=state.rh,
    )
    return merged, diags_acc
