"""Fused Pallas (Triton) coagulation substep loop — GPU fast path.

Drop-in alternative to fast/coagulation.coagulation_step with identical
signature and returns. The frozen per-cell kernel kij (nbins x nbins f64,
12.8 KB) is uploaded ONCE per outer step and every Euler substep (TFL
rates + Euler + positivity clamp + full 2-sweep MNFIX) runs inside a
single Pallas kernel per cell. The XLA path instead re-reads the
(C, nbins, nbins) kernel decomposition from HBM every substep via two
GEMMs plus transpose/reduce fusions — the dominant measured cost of the
fast model (docs/gpu_fast_optimization.md).

Architecture (persistent-CTA grid — each CTA processes one cell at a
time, cells pid, pid+P, ...; bin axis padded nbins -> BP = next pow2
>= nbins+1, a Triton shape requirement):

  * All per-bin state lives in (BP,) register vectors. There are NO
    (BP, BP) register tiles: a first fused-kernel attempt that held kij
    and one-hot masks as (64, 64) tiles was 5-25x SLOWER than XLA
    because ~20 tile-sized temporaries per substep spill to local
    memory (see the optimization log). Everything cross-bin is instead:
  * TFL matvecs: a FULLY UNROLLED loop over source bins; each iteration
    loads one kij column ((BP,) contiguous row of the transposed
    kernel, L1/L2-resident after the first substep) and 3 scalars, and
    updates 4 accumulators with masked FMAs. Unrolling lets all the
    loads issue independently — with a fori_loop the ~120 L2-latency
    scalar loads per substep serialize and dominate.
  * shift_right and the MNFIX deposit: a CTA-private GMEM scratch row
    + CTA barrier, using masked gather loads (shift) and f64 atomic_add
    scatter (deposit, lax.cond-skipped on the CTA-uniform "no lane
    shifted" fast path). CTA-private scratch means barriers fully order
    the traffic; it stays L1/L2-resident.
  * MNFIX shift-target bins: exact ceil/floor(log2) via IEEE-754
    exponent-field extraction (no libdevice log); the new average mass
    xnew is gathered from the true xk table so bin boundaries are
    bit-identical to the reference (the grid is NOT assumed to be exact
    powers of two — jnp.power gives 1-ulp deviations on some bins).

Numerical deviations from the XLA path (all reassociation-level,
<< 1e-10 relative; verified in tests/test_fast_coag_pallas.py):
  * matvec accumulation order (serial j vs cuBLAS GEMM order),
  * deposit summation order (atomic_add vs XLA scatter-add — the XLA
    GPU scatter is itself non-bitwise-deterministic),
  * exact integer log2 vs the reference's float ceil(log(r)/log(p)):
    these can pick a different MNFIX target bin only when log(r)/log(p)
    sits within ~1 ulp of an integer (measure-zero; requires p = 2
    mass-doubling grids, which is the only fast-model grid).

Padded lanes are kept inert: kij columns are zero there (masked loads),
their bin boundaries are clones of the top bin so MNFIX never shifts
them, and they are re-zeroed after each Euler update so the mass
promoted past the top bin (lane nbins receives bin nbins-1's TFL shift)
never survives — exactly reproducing the reference, where shift_right
drops that mass and only the analytic overflow term accounts for it.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pl_triton

from .coagulation import _kernel_cell, _loss_frequency_cell
from .config import SRTSO4

# Constants mirrored from the reference implementations (do not retune):
_NEPS_RATES = 1.0e-3   # physics/coagulation_rates.NEPS
_NEPS_MNFIX = 1.0e-5   # fast/mnfix._NEPS
_EPS = 1.0e-40         # fast/mnfix._EPS

_MANT_MASK = (1 << 52) - 1


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _ceil_log2(r):
    """Exact ceil(log2(r)) for finite r >= 1, via the IEEE-754 exponent."""
    b = jax.lax.bitcast_convert_type(r, jnp.int64)
    e = ((b >> 52) & 0x7FF) - 1023
    frac_nonzero = (b & _MANT_MASK) != 0
    return (e + jnp.where(frac_nonzero, 1, 0)).astype(jnp.int32)


def _floor_log2(r):
    """Exact floor(log2(r)) for finite r >= 1."""
    b = jax.lax.bitcast_convert_type(r, jnp.int64)
    return (((b >> 52) & 0x7FF) - 1023).astype(jnp.int32)


def _make_kernel(nbins, bp, n_sweeps, use_barrier, n_cells, n_prog):
    """Build the fused substep-loop kernel (static shapes).

    Persistent-CTA layout: a fixed grid of n_prog programs; program pid
    processes cells pid, pid+n_prog, ... sequentially (n_cells is padded
    to a multiple of n_prog by the wrapper). Keeping the concurrent cell
    count ~2-3k keeps the per-cell kij slices L2-resident instead of
    re-streaming (C, nbins, nbins) from HBM every substep.

    Scratch row usage (s_ref, shape (n_prog, 8, bp), CTA-private):
      rows 0-2: matvec source vectors [ms, xbar, Ns]
      rows 3-4: shift_right sources [growthN, E = A + B*xk[k+1]]
      rows 5-7: MNFIX deposit accumulators [recv_N, recv_m, recv_w]
    Distinct rows per phase keep the barrier schedule minimal: 2 CTA
    barriers per substep, plus 3 per MNFIX sweep only when some lane
    actually shifts (lax.cond on a CTA-uniform predicate).
    """
    assert n_cells % n_prog == 0
    trips = n_cells // n_prog

    def barrier():
        if use_barrier:
            pl_triton.debug_barrier()

    def kernel(nsub_ref, scal_ref, xklo_ref, xkhi_ref, xkgeo_ref, xkg_ref,
               nk_ref, ms_ref, mw_ref, kijT_ref,
               nk_out, ms_out, mw_out, ovf_out, s_ref):
        # int64 indices so mixed-type dynamic_slice doesn't trip the
        # interpret-mode discharge (x64 default index type is int64).
        pid = pl.program_id(0).astype(jnp.int64)
        n_sub = nsub_ref[0]
        dt_sub = scal_ref[0]
        zeta = scal_ref[1]
        zeta3 = scal_ref[2]
        xk0 = scal_ref[4]
        xk_top = scal_ref[5]

        k1 = jax.lax.broadcasted_iota(jnp.int32, (bp,), 0)
        valid = k1 < nbins
        last = k1 == nbins - 1

        xk_lo = xklo_ref[...]     # xk[k]   (top-bin clone on padded lanes)
        xk_hi = xkhi_ref[...]     # xk[k+1] (   "   )
        xk_geo = xkgeo_ref[...]   # sqrt(xk[k]*xk[k+1]) ( " )

        def _row(row):
            # traced index: interpret-mode discharge rejects Python ints
            return jnp.asarray(row, jnp.int64)

        def store_row(row, vec):
            pl.store(s_ref, (pid, _row(row), pl.ds(0, bp)), vec)

        def load_row(row):
            return pl.load(s_ref, (pid, _row(row), pl.ds(0, bp)))

        def shift_row(row):
            # shift_right via scratch: out[k] = row[k-1], 0 at k=0
            km1 = jnp.maximum(k1 - 1, 0)
            return pl.load(s_ref, (pid, _row(row), km1), mask=k1 > 0,
                           other=0.0)

        def rates(c, kd, N, m):
            # physics/coagulation_rates.calc_coagulation_rates,
            # specialized to icomp_nodiag=1, in shift-late form.
            empty = N < _NEPS_RATES
            Ns = jnp.where(empty, _NEPS_RATES, N)
            msafe = jnp.where(empty, _NEPS_RATES * xk_geo, m)

            xbar = msafe / Ns
            ratio = xbar / xk_lo
            factor = 2.0 * Ns / xk_lo
            p = xk_hi / xk_lo
            eff = factor * (p - ratio) / (p - 1.0)
            phi = factor * (ratio - 1.0) / (p - 1.0)
            low = xbar < xk_lo
            eff = jnp.where(low, factor, eff)
            phi = jnp.where(low, 0.0, phi)
            high = xbar > xk_hi
            phi = jnp.where(high, factor, phi)
            eff = jnp.where(high, 0.0, eff)

            # TFL triangular matvecs: serial over source bin j, parallel
            # over receiving bin k (the lanes). kij column j is row j of
            # the transposed kernel — one contiguous (bp,) load.
            # No entry barrier: the previous consumers of rows 0-2 (the
            # previous substep's unrolled matvec loads) are ordered by
            # that substep's post-shift-store barrier.
            store_row(0, msafe)
            store_row(1, xbar)
            store_row(2, Ns)
            barrier()

            # Fully unrolled over source bins so all loads are issued
            # independently (a fori_loop here serializes ~120 L2-latency
            # scalar loads and dominates the substep).
            z = jnp.zeros((bp,), jnp.float64)
            k1m, k1mx, k1mx2, in_term = z, z, z, z
            for j in range(nbins):
                jj = _row(j)
                col = pl.load(kijT_ref, (c, jj, pl.ds(0, bp)),
                              mask=valid, other=0.0)
                msj = pl.load(s_ref, (pid, _row(0), jj))
                xbj = pl.load(s_ref, (pid, _row(1), jj))
                nsj = pl.load(s_ref, (pid, _row(2), jj))
                # same products (same operands/order) as the reference
                # GEMM columns [m, m*xbar, m*xbar^2]
                c1j = msj * xbj
                c2j = msj * (xbj * xbj)
                lo = jnp.where(j < k1, col, 0.0)
                up = jnp.where(j > k1, col, 0.0)
                k1m = k1m + lo * msj
                k1mx = k1mx + lo * c1j
                k1mx2 = k1mx2 + lo * c2j
                in_term = in_term + up * nsj

            two_xk = 2.0 * xk_lo
            dNdt_curr = (
                -kd * Ns ** 2
                - phi * k1m
                - zeta * (eff - phi) / two_xk * k1mx
                - Ns * in_term
            )
            dMdt_curr = (
                Ns * k1m
                - kd * Ns * msafe
                - msafe * in_term
                - phi * xk_hi * k1m
                - 0.5 * zeta * eff * k1mx
                + zeta3 * (phi - eff) / two_xk * k1mx2
            )
            # k-1 ("prev") contributions, computed per-bin then shifted:
            #   dNdt_prev = shift(growthN)
            #   dMdt_prev = shift(A) + shift(B)*xk[k]   (B's factor uses
            #   the RECEIVING bin's lower boundary — multicoag.f)
            growthN = (
                0.5 * kd * Ns ** 2
                + phi * k1m
                + zeta * (eff - phi) / two_xk * k1mx
            )
            A = (
                kd * Ns * msafe
                + 0.5 * zeta * eff * k1mx
                - zeta3 * (phi - eff) / two_xk * k1mx2
            )
            B = phi * k1m
            # E_k = A_k + B_k*xk_hi_k, so shift(E)[k] = A[k-1] +
            # B[k-1]*xk[k]: one shifted vector covers both dMdt_prev
            # terms (B's factor is the receiving bin's lower boundary
            # = the source bin's upper boundary), and E at the top bin
            # IS the overflow rate.
            E = A + B * xk_hi
            store_row(3, growthN)
            store_row(4, E)
            barrier()
            dNdt = dNdt_curr + shift_row(3)
            dMdt = dMdt_curr + shift_row(4)
            # Top-bin overflow = the bin nbins-1 "prev" contribution that
            # shift_right drops, with xk[nbins] as the receiving boundary.
            dOvf = jnp.where(last, E, 0.0)
            return dNdt, dMdt, dOvf

        def sweep(N, m, w):
            # fast/mnfix._drift_sweep, icomp=2 (SO4 prognostic, H2O diag)
            dry = m
            avg = dry / (N + _EPS)
            avg = jnp.where((N == 0.0) & (dry <= 0.0), xk_geo, avg)
            fm = m / (dry + _EPS)
            fw = w / (dry + _EPS)

            needs_up = (avg > xk_hi) & (k1 < nbins - 1)
            ratio_up = avg * 1.1 / (xk0 + _EPS)
            kk_up_raw = _ceil_log2(jnp.maximum(ratio_up, 1.0)) - 1
            kk_up = jnp.clip(kk_up_raw, k1 + 1, nbins - 1)

            needs_down = (avg < xk_lo) & (k1 > 0)
            ratio_dn = avg / (1.1 * xk0 + _EPS)
            kk_dn_raw = _floor_log2(jnp.maximum(ratio_dn, 1.0))
            kk_dn = jnp.clip(kk_dn_raw, 0, jnp.maximum(k1 - 1, 0))

            shifted = needs_up | needs_down
            kk = jnp.where(needs_up, kk_up, jnp.where(needs_down, kk_dn, k1))
            # xnew from the true xk table (bit-exact bin boundaries)
            g_idx = jnp.where(needs_up, jnp.minimum(kk_up + 1, nbins), kk_dn)
            xg = pl.load(xkg_ref, (g_idx,))
            xnew = jnp.where(needs_up, xg / 1.1, xg * 1.1)

            nshift = jnp.clip(
                (dry - xk_geo * N) / (xnew - xk_geo + _EPS), 0.0, N
            )
            nshift = jnp.where(shifted, nshift, 0.0)
            mshift = xnew * nshift
            n_remain = N - nshift

            Nb = jnp.where(shifted, n_remain, N)
            mb = jnp.where(shifted, (xk_geo * n_remain) * fm, m)
            wb = jnp.where(shifted, (xk_geo * n_remain) * fw, w)

            # Conservative deposit: f64 atomic scatter-add into scratch
            # rows 5-7. Skipped entirely (zeroing, barriers, readback)
            # when no lane shifts — the predicate is a CTA-uniform
            # reduction, so the in-branch barriers cannot diverge.
            def deposit():
                zv = jnp.zeros((bp,), jnp.float64)
                barrier()  # previous sweep's readback of rows 5-7 done
                store_row(5, zv)
                store_row(6, zv)
                store_row(7, zv)
                barrier()
                kk64 = kk.astype(jnp.int64)
                # Only shifting lanes fire an atomic (a 0.0 add is a
                # no-op, so masking changes nothing numerically). The
                # interpret path has no masked-atomic rule; adding the
                # zeros there is equivalent.
                amask = shifted if use_barrier else None
                pl.atomic_add(s_ref, (pid, _row(5), kk64), nshift,
                              mask=amask)
                pl.atomic_add(s_ref, (pid, _row(6), kk64), mshift * fm,
                              mask=amask)
                pl.atomic_add(s_ref, (pid, _row(7), kk64), mshift * fw,
                              mask=amask)
                barrier()
                return load_row(5), load_row(6), load_row(7)

            def no_deposit():
                zv = jnp.zeros((bp,), jnp.float64)
                return zv, zv, zv

            n_shifting = jnp.sum(shifted.astype(jnp.int32))
            recv_N, recv_m, recv_w = jax.lax.cond(
                n_shifting > 0, deposit, no_deposit
            )
            return Nb + recv_N, mb + recv_m, wb + recv_w

        def mnfix_(N, m, w):
            # fast/mnfix.mnfix phases 1-3
            N = jnp.maximum(N, 0.0)
            m = jnp.maximum(m, 0.0)
            w = jnp.maximum(w, 0.0)
            empty = N < _NEPS_MNFIX
            N = jnp.where(empty, _NEPS_MNFIX, N)
            m = jnp.where(empty, _NEPS_MNFIX * xk_geo, m)
            w = jnp.where(empty, 0.0, w)

            dry = m
            avg = dry / (N + _EPS)
            too_high = avg > xk_top
            scale = (N * xk_top / 1.2) / (dry + _EPS)
            m = jnp.where(too_high, m * scale, m)
            w = jnp.where(too_high, w * scale, w)

            dry = m
            too_low = dry / (N + _EPS) < xk0
            N = jnp.where(too_low, dry / (xk0 * 1.2), N)

            for _ in range(n_sweeps):
                N, m, w = sweep(N, m, w)
            return N, m, w

        def substep_factory(c, kd):
            def body(_, carry):
                N, m, w, ovf_acc = carry
                dNdt, dMdt, dOvf = rates(c, kd, N, m)
                N = jnp.maximum(N + dt_sub * dNdt, 0.0)
                m = jnp.maximum(m + dt_sub * dMdt, 0.0)
                w = jnp.maximum(w, 0.0)  # dMdt is 0 for diagnostic H2O
                ovf_acc = ovf_acc + dt_sub * dOvf
                # Kill the promoted mass that landed on padded lanes
                # (lane `nbins` is the reference's dropped shift_right
                # output).
                N = jnp.where(valid, N, 0.0)
                m = jnp.where(valid, m, 0.0)
                w = jnp.where(valid, w, 0.0)
                N, m, w = mnfix_(N, m, w)
                return N, m, w, ovf_acc
            return body

        def cell_body(t, _):
            c = pid + t.astype(jnp.int64) * n_prog
            # Frozen-kernel diagonal, gathered once per cell.
            kd = pl.load(kijT_ref, (c, k1, k1), mask=valid, other=0.0)
            N0 = pl.load(nk_ref, (c, pl.ds(0, bp)), mask=valid, other=0.0)
            m0 = pl.load(ms_ref, (c, pl.ds(0, bp)), mask=valid, other=0.0)
            w0 = pl.load(mw_ref, (c, pl.ds(0, bp)), mask=valid, other=0.0)

            z = jnp.zeros((bp,), jnp.float64)
            Nf, mf, wf, ovf_acc = jax.lax.fori_loop(
                0, n_sub, substep_factory(c, kd), (N0, m0, w0, z)
            )
            pl.store(nk_out, (c, pl.ds(0, bp)), Nf, mask=valid)
            pl.store(ms_out, (c, pl.ds(0, bp)), mf, mask=valid)
            pl.store(mw_out, (c, pl.ds(0, bp)), wf, mask=valid)
            pl.store(ovf_out, (c,), jnp.sum(ovf_acc))
            return _  # dummy traced carry (Triton must yield a Value)

        jax.lax.fori_loop(0, trips, cell_body, jnp.zeros((), jnp.int32))

    return kernel


@functools.lru_cache(maxsize=None)
def _get_pallas_fn(nbins, bp, n_sweeps, num_warps, interpret, n_cells,
                   n_prog):
    # interpret mode runs the kernel body as sequential JAX ops, where
    # program order already serializes the scratch traffic; the Triton
    # barrier primitive has no interpret lowering, so drop it there.
    kernel = _make_kernel(nbins, bp, n_sweeps, use_barrier=not interpret,
                          n_cells=n_cells, n_prog=n_prog)
    kwargs = {}
    if not interpret and num_warps is not None:
        kwargs["compiler_params"] = pl_triton.CompilerParams(
            num_warps=num_warps
        )

    def call(nsub, scal, xklo, xkhi, xkgeo, xkg, Nk, ms, mw, kijT):
        out_shape = [
            jax.ShapeDtypeStruct((n_cells, nbins), jnp.float64),  # Nk
            jax.ShapeDtypeStruct((n_cells, nbins), jnp.float64),  # M so4
            jax.ShapeDtypeStruct((n_cells, nbins), jnp.float64),  # M h2o
            jax.ShapeDtypeStruct((n_cells,), jnp.float64),   # overflow so4
            jax.ShapeDtypeStruct((n_prog, 8, bp), jnp.float64),  # scratch
        ]
        return pl.pallas_call(
            kernel,
            grid=(n_prog,),
            out_shape=out_shape,
            interpret=interpret,
            **kwargs,
        )(nsub, scal, xklo, xkhi, xkgeo, xkg, Nk, ms, mw, kijT)

    return call


def _padded_grid_arrays(xk, nbins, bp):
    """Per-lane bin-boundary vectors, padded lanes cloned from the top bin
    (mnfix-inert), plus the gather table (true xk, finite extension)."""
    xk_geo = jnp.sqrt(xk[:-1] * xk[1:])  # same expression as the references
    npad = bp - nbins
    xk_lo = jnp.concatenate([xk[:-1], jnp.full((npad,), xk[nbins - 1])])
    xk_hi = jnp.concatenate([xk[1:], jnp.full((npad,), xk[nbins])])
    xk_geo_p = jnp.concatenate([xk_geo, jnp.full((npad,), xk_geo[-1])])
    p = xk[1] / xk[0]
    ext = xk[-1] * p ** jnp.arange(1, bp - nbins, dtype=jnp.float64)
    xk_g = jnp.concatenate([xk, ext])  # (bp,) — indices 0..nbins are true xk
    return xk_lo, xk_hi, xk_geo_p, xk_g


def coagulation_step_pallas(
    Nk, Mk, xk, temp, pres, boxvol, dt, c_max=0.1, n_sub_cap=256,
    num_warps=2, n_persistent=2640, interpret=False,
):
    """Drop-in replacement for fast/coagulation.coagulation_step.

    Same args and returns: (Nk, Mk, overflow, cap_hit, n_sub). Extra
    static knobs: num_warps (Triton), n_persistent (persistent-CTA grid
    size; ~20 per SM keeps the concurrent kij slices L2-resident),
    interpret (run the kernel as JAX ops — CPU-compatible, for testing).
    GPU-only unless interpret=True.
    """
    nbins = Nk.shape[-1]
    bp = _next_pow2(nbins + 1)
    C = Nk.shape[0]
    n_prog = min(n_persistent, C)
    # Pad the cell axis to a multiple of the persistent grid so the
    # in-kernel cell loop needs no bounds masking (padded cells compute
    # into padded output slots, dropped below).
    C_pad = ((C + n_prog - 1) // n_prog) * n_prog

    # Frozen kernel + shared adaptive substep count: identical to the
    # reference path (same functions, same XLA ops).
    kij = jax.vmap(_kernel_cell)(Nk, Mk, temp, pres, boxvol)
    lam = jax.vmap(_loss_frequency_cell, in_axes=(0, 0, 0, None))(
        Nk, Mk, kij, xk
    )
    n_raw = jnp.ceil(dt * jnp.max(lam) / c_max).astype(jnp.int32)
    n_sub = jnp.clip(n_raw, 1, n_sub_cap)
    cap_hit = n_raw > n_sub_cap
    dt_sub = dt / n_sub

    # The in-kernel matvec streams kij COLUMNS: transpose once per outer
    # step so each column is a contiguous row.
    kijT = jnp.swapaxes(kij, -1, -2)

    def pad(a):
        if C_pad == C:
            return a
        widths = [(0, C_pad - C)] + [(0, 0)] * (a.ndim - 1)
        return jnp.pad(a, widths)

    xk_lo, xk_hi, xk_geo, xk_g = _padded_grid_arrays(xk, nbins, bp)

    # Scalars, computed with the same XLA expressions as the references:
    p_ratio = xk[1] / xk[0]
    zeta = 0.5 * (1.0 + (p_ratio + 1.0) ** 2 / (4.0 * p_ratio))
    zeta3 = zeta ** 3
    log_p = jnp.log(xk[1] / xk[0])  # layout slot kept; kernel uses exact log2
    scal = jnp.stack([
        dt_sub, zeta, zeta3, log_p, xk[0], xk[nbins],
        jnp.zeros(()), jnp.zeros(()),
    ]).astype(jnp.float64)
    nsub_arr = jnp.reshape(n_sub, (1,))

    call = _get_pallas_fn(nbins, bp, 2, num_warps, interpret, C_pad,
                          n_prog)
    Nk_f, ms_f, mw_f, ovf, _scratch = call(
        nsub_arr, scal, xk_lo, xk_hi, xk_geo, xk_g,
        pad(Nk), pad(Mk[..., 0]), pad(Mk[..., 1]), pad(kijT),
    )
    Nk_f, ms_f, mw_f, ovf = (
        Nk_f[:C], ms_f[:C], mw_f[:C], ovf[:C]
    )
    Mk_f = jnp.stack([ms_f, mw_f], axis=-1)
    overflow = jnp.zeros(Mk.shape[:-2] + (Mk.shape[-1],))
    overflow = overflow.at[..., SRTSO4].set(ovf)
    return Nk_f, Mk_f, overflow, cap_hit, n_sub
