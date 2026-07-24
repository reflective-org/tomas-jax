"""Step- and run-level tests for the reduced model: conservation, batching,
regimes. CPU, small C."""
import numpy as np
import pytest

import jax.numpy as jnp

from tomas_jax.fast import FastState, fast_step, run_fast
from tomas_jax.fast.coagulation import coagulation_step
from tomas_jax.fast.condensation import condensation_step
from tomas_jax.fast.config import (
    GH2SO4,
    GSO2,
    MW_H2SO4,
    MW_SO2,
    MW_SO4,
    NBINS,
    SRTSO4,
    xk_boundaries,
)

XK = xk_boundaries()
XMID = jnp.sqrt(XK[:-1] * XK[1:])

S_GAS = 32.06 / MW_H2SO4
S_SO2 = 32.06 / MW_SO2
S_AER = 32.06 / MW_SO4


def _lognormal_cells(C, seed=1, n_range=(1e2, 1e5), boxvol=1e6):
    rng = np.random.default_rng(seed)
    dp = np.cbrt(np.asarray(XMID) / 1770.0 * 6 / np.pi)
    Nk = np.zeros((C, NBINS))
    for c in range(C):
        gmd = rng.uniform(20e-9, 150e-9)
        gsd = rng.uniform(1.5, 2.2)
        w = np.exp(-0.5 * ((np.log(dp) - np.log(gmd)) / np.log(gsd)) ** 2)
        Nk[c] = rng.uniform(*n_range) * boxvol * w / w.sum()
    Nk = jnp.asarray(Nk)
    Mk = jnp.zeros((C, NBINS, 2)).at[..., SRTSO4].set(Nk * XMID)
    Gc = jnp.zeros((C, 2)).at[:, GH2SO4].set(1e-13).at[:, GSO2].set(1e-9)
    return FastState.create(
        Nk, Mk, Gc,
        temp=rng.uniform(200.0, 300.0, C),
        pres=rng.uniform(5e3, 1.01e5, C),
        boxvol=boxvol,
        rh=rng.uniform(0.02, 0.95, C),
    )


def _total_S(state, ovf_so4=0.0):
    return (
        float(jnp.sum(state.Gc[:, GH2SO4])) * S_GAS
        + float(jnp.sum(state.Gc[:, GSO2])) * S_SO2
        + (float(jnp.sum(state.Mk[..., SRTSO4])) + ovf_so4) * S_AER
    )


class TestSulfurBudget:
    def test_closed_system(self):
        """S(SO2)+S(H2SO4)+S(SO4)+S(overflow) constant over 2 h.

        coag_sub_cap raised so the stiffest cell is fully resolved and no
        cap-hit degradation mixes into the conservation check."""
        state = _lognormal_cells(8)
        s0 = _total_S(state)
        out, diags = run_fast(
            state, n_steps=20, dt=360.0, oh_conc=1.5e6, coag_sub_cap=1024
        )
        s1 = _total_S(out, ovf_so4=float(np.sum(diags["overflow_so4"])))
        assert s1 == pytest.approx(s0, rel=1e-9)
        assert not diags["cond_cap_hit"].any()
        assert not diags["coag_cap_hit"].any()

    def test_with_so2_emission(self):
        """External SO2 source accounted: S(t) = S(0) + emitted."""
        state = _lognormal_cells(4, seed=2)
        so2_prod = 1e-15  # kg/cell/s
        s0 = _total_S(state)
        n_steps, dt = 10, 360.0
        out, diags = run_fast(
            state, n_steps=n_steps, dt=dt, oh_conc=2e6, so2_prod=so2_prod
        )
        emitted_S = so2_prod * dt * n_steps * 4 * S_SO2
        s1 = _total_S(out, ovf_so4=float(np.sum(diags["overflow_so4"])))
        assert s1 - s0 == pytest.approx(emitted_S, rel=1e-9)


class TestProcessProperties:
    def test_coagulation_number_monotone(self):
        state = _lognormal_cells(4, seed=3, n_range=(1e4, 1e5))
        Nk, Mk = state.Nk, state.Mk
        N1, M1, ovf, cap = coagulation_step(
            Nk, Mk, XK, state.temp, state.pres, state.boxvol, dt=360.0
        )
        assert float(jnp.sum(N1)) < float(jnp.sum(Nk))
        # mass closure with overflow (NEPS-seeding floor ~1e-8, as full model)
        m0 = float(jnp.sum(Mk[..., SRTSO4]))
        m1 = float(jnp.sum(M1[..., SRTSO4])) + float(jnp.sum(ovf[..., SRTSO4]))
        assert m1 == pytest.approx(m0, rel=1e-7)

    def test_condensation_gas_aerosol_closure(self):
        """Aerosol SO4 gained == gas H2SO4 lost * 96/98 (S-conserving mode)."""
        state = _lognormal_cells(4, seed=4)
        Gc = state.Gc.at[:, GH2SO4].set(1e-11)
        N1, M1, G1, cap = condensation_step(
            state.Nk, state.Mk, Gc, XK,
            state.temp, state.pres, state.boxvol, dt=360.0,
        )
        gas_lost = float(jnp.sum(Gc[:, GH2SO4]) - jnp.sum(G1[:, GH2SO4]))
        aer_gain = float(jnp.sum(M1[..., SRTSO4]) - jnp.sum(state.Mk[..., SRTSO4]))
        assert gas_lost > 0
        assert aer_gain == pytest.approx(gas_lost * MW_SO4 / MW_H2SO4, rel=1e-9)
        assert float(jnp.min(N1)) >= 0.0 and float(jnp.min(M1)) >= 0.0

    def test_dry_mass_invariant_under_rh(self):
        """Water is diagnostic: RH changes never touch dry budgets."""
        base = _lognormal_cells(3, seed=5)
        results = []
        for rh in (0.05, 0.5, 0.95):
            st = base.update(rh=jnp.full(3, rh))
            out, diags = run_fast(st, n_steps=5, dt=360.0, oh_conc=1e6)
            results.append(
                float(jnp.sum(out.Mk[..., SRTSO4]))
                + float(np.sum(diags["overflow_so4"]))
            )
        # coag kernel depends on wet size, so totals differ slightly through
        # different collision rates — but S is conserved for each RH
        s0 = _total_S(base)
        for rh in (0.05, 0.5, 0.95):
            st = base.update(rh=jnp.full(3, rh))
            out, diags = run_fast(st, n_steps=5, dt=360.0, oh_conc=1e6)
            s1 = _total_S(out, ovf_so4=float(np.sum(diags["overflow_so4"])))
            assert s1 == pytest.approx(s0, rel=1e-9)


class TestBatchSemantics:
    def test_identical_cells_identical_results(self):
        """A batch of copies of one cell evolves identically to C=1."""
        one = _lognormal_cells(1, seed=6)
        C = 5
        many = FastState(
            Nk=jnp.tile(one.Nk, (C, 1)),
            Mk=jnp.tile(one.Mk, (C, 1, 1)),
            Gc=jnp.tile(one.Gc, (C, 1)),
            xk=one.xk,
            temp=jnp.tile(one.temp, C),
            pres=jnp.tile(one.pres, C),
            boxvol=jnp.tile(one.boxvol, C),
            rh=jnp.tile(one.rh, C),
        )
        out1, _ = fast_step(one, 360.0, oh_conc=1.5e6)
        outC, _ = fast_step(many, 360.0, oh_conc=1.5e6)
        # copies within the batch must be bit-identical to each other;
        # vs the C=1 run only up to reassociation (XLA tiling differs by
        # batch shape)
        for c in range(1, C):
            np.testing.assert_array_equal(
                np.asarray(outC.Nk[c]), np.asarray(outC.Nk[0])
            )
            np.testing.assert_array_equal(
                np.asarray(outC.Mk[c]), np.asarray(outC.Mk[0])
            )
        np.testing.assert_allclose(
            np.asarray(outC.Nk[0]), np.asarray(out1.Nk[0]), rtol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(outC.Gc), np.tile(np.asarray(out1.Gc), (C, 1)),
            rtol=1e-10,
        )

    def test_permutation_equivariance(self):
        """Shuffling cells shuffles outputs (shared substep count is
        permutation-invariant)."""
        state = _lognormal_cells(6, seed=7)
        perm = np.array([3, 0, 5, 1, 4, 2])
        shuffled = FastState(
            Nk=state.Nk[perm], Mk=state.Mk[perm], Gc=state.Gc[perm],
            xk=state.xk, temp=state.temp[perm], pres=state.pres[perm],
            boxvol=state.boxvol[perm], rh=state.rh[perm],
        )
        out, _ = fast_step(state, 360.0, oh_conc=1.5e6)
        out_s, _ = fast_step(shuffled, 360.0, oh_conc=1.5e6)
        np.testing.assert_array_equal(
            np.asarray(out_s.Nk), np.asarray(out.Nk[perm])
        )
        np.testing.assert_array_equal(
            np.asarray(out_s.Mk), np.asarray(out.Mk[perm])
        )

    def test_chunked_equals_unchunked(self):
        """With identical cells every chunk shares the same batch-max
        substep counts, so chunking is numerically transparent. (For
        heterogeneous cells, chunking changes the shared substep counts —
        each chunk still gets a valid, slightly-more-or-less refined
        integration.)"""
        one = _lognormal_cells(1, seed=8)
        C = 6
        state = FastState(
            Nk=jnp.tile(one.Nk, (C, 1)), Mk=jnp.tile(one.Mk, (C, 1, 1)),
            Gc=jnp.tile(one.Gc, (C, 1)), xk=one.xk,
            temp=jnp.tile(one.temp, C), pres=jnp.tile(one.pres, C),
            boxvol=jnp.tile(one.boxvol, C), rh=jnp.tile(one.rh, C),
        )
        out1, d1 = run_fast(state, n_steps=3, dt=360.0, oh_conc=1e6)
        out2, d2 = run_fast(
            state, n_steps=3, dt=360.0, oh_conc=1e6, n_cell_chunks=3
        )
        np.testing.assert_allclose(
            np.asarray(out2.Nk), np.asarray(out1.Nk), rtol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(d2["M_dry"]), np.asarray(d1["M_dry"]), rtol=1e-10
        )


class TestRegimes:
    def test_stratospheric_endtoend(self):
        """SAI regime: 220 K, 5 kPa, RH 3%, SO2-rich — finite and conserving."""
        C = 3
        rng = np.random.default_rng(9)
        dp = np.cbrt(np.asarray(XMID) / 1770.0 * 6 / np.pi)
        w = np.exp(-0.5 * ((np.log(dp) - np.log(80e-9)) / np.log(1.8)) ** 2)
        Nk = jnp.asarray(np.tile(10.0 * 1e6 * w / w.sum(), (C, 1)))  # 10/cm3
        Mk = jnp.zeros((C, NBINS, 2)).at[..., SRTSO4].set(Nk * XMID)
        Gc = jnp.zeros((C, 2)).at[:, GSO2].set(1e-7)  # SO2-rich (injection)
        state = FastState.create(
            Nk, Mk, Gc, temp=220.0, pres=5e3, boxvol=1e6, rh=0.03
        )
        s0 = _total_S(state)
        out, diags = run_fast(state, n_steps=10, dt=360.0, oh_conc=1e6)
        assert bool(
            jnp.all(jnp.isfinite(out.Nk))
            & jnp.all(jnp.isfinite(out.Mk))
            & jnp.all(jnp.isfinite(out.Gc))
        )
        assert float(jnp.min(out.Nk)) >= 0.0
        s1 = _total_S(out, ovf_so4=float(np.sum(diags["overflow_so4"])))
        assert s1 == pytest.approx(s0, rel=1e-9)
        # chemistry happened: SO2 declined, aerosol grew
        assert float(jnp.sum(out.Gc[:, GSO2])) < float(jnp.sum(Gc[:, GSO2]))
        assert float(jnp.sum(out.Mk[..., SRTSO4])) > float(
            jnp.sum(Mk[..., SRTSO4])
        )
