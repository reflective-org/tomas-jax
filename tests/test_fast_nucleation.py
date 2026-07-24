"""Tests for Jbn-only nucleation: rate extraction, S conservation, exactness."""
import numpy as np
import pytest

import jax.numpy as jnp

from tomas_jax.fast.config import (
    GH2SO4,
    MW_H2SO4,
    MW_SO4,
    NBINS,
    SRTH2O,
    SRTSO4,
    xk_boundaries,
)
from tomas_jax.fast.mnfix import mnfix
from tomas_jax.fast.nucleation import dunne_binary_neutral_rate, nucleation_step
from tomas_jax.physics.nucleation import _MNUC, dunne_nucleation_rate

XK = xk_boundaries()

S_FRAC_GAS = 32.06 / MW_H2SO4   # S mass fraction of H2SO4
S_FRAC_AER = 32.06 / MW_SO4     # S mass fraction of SO4


def _total_sulfur(Mk, Gc):
    return float(
        jnp.sum(Gc[..., GH2SO4] * S_FRAC_GAS)
        + jnp.sum(Mk[..., SRTSO4] * S_FRAC_AER)
    )


def _base_state(C=3, gc_h2so4=1e-12):
    Nk = jnp.full((C, NBINS), 100.0)
    xmid = jnp.sqrt(XK[:-1] * XK[1:])
    Mk = jnp.zeros((C, NBINS, 2)).at[..., SRTSO4].set(100.0 * xmid)
    Gc = jnp.zeros((C, 2)).at[..., GH2SO4].set(gc_h2so4)
    return Nk, Mk, Gc


class TestRateExtraction:
    def test_jbn_matches_dunne(self):
        """dunne_binary_neutral_rate pins the Jbn channel of the full model."""
        for T in [208.0, 230.0, 260.0, 298.0]:
            for h2so4 in [1e5, 1e6, 1e7, 1e8, 1e9]:
                _, Jbn, _, _, _ = dunne_nucleation_rate(
                    T, 3.0, h2so4, 1e9, 2.5e19
                )
                got = float(dunne_binary_neutral_rate(T, h2so4))
                assert got == pytest.approx(float(Jbn), rel=1e-14)


class TestNucleationStep:
    def test_sulfur_conserved(self):
        """Total S exact through nucleation (the S-conserving 98/96 deviation)."""
        Nk, Mk, Gc = _base_state(gc_h2so4=1e-9)
        temp = jnp.full(3, 220.0)
        boxvol = jnp.full(3, 1e6)
        s0 = _total_sulfur(Mk, Gc)
        Nk2, Mk2, Gc2 = nucleation_step(Nk, Mk, Gc, temp, boxvol, dt=360.0)
        assert _total_sulfur(Mk2, Gc2) == pytest.approx(s0, rel=1e-13)
        assert float(jnp.sum(Nk2)) > float(jnp.sum(Nk))  # actually nucleated

    def test_sulfur_conserved_gas_exhaustion(self):
        """Extreme rates consume nearly all gas without overshooting."""
        Nk, Mk, Gc = _base_state(gc_h2so4=1e-14)
        temp = jnp.full(3, 208.0)
        boxvol = jnp.full(3, 1e6)
        s0 = _total_sulfur(Mk, Gc)
        Nk2, Mk2, Gc2 = nucleation_step(
            Nk, Mk, Gc, temp, boxvol, dt=360.0, fn_scale=1e12
        )
        assert _total_sulfur(Mk2, Gc2) == pytest.approx(s0, rel=1e-13)
        assert float(jnp.min(Gc2[..., GH2SO4])) >= 0.0
        # analytic solution asymptotes to zero gas but never below
        assert float(jnp.max(Gc2[..., GH2SO4])) < 1e-14

    def test_zero_gas_is_noop(self):
        Nk, Mk, Gc = _base_state(gc_h2so4=0.0)
        Nk2, Mk2, Gc2 = nucleation_step(
            Nk, Mk, Gc, jnp.full(3, 208.0), jnp.full(3, 1e6), dt=360.0
        )
        np.testing.assert_array_equal(np.asarray(Nk2), np.asarray(Nk))
        np.testing.assert_array_equal(np.asarray(Mk2), np.asarray(Mk))
        np.testing.assert_array_equal(np.asarray(Gc2), np.asarray(Gc))

    def test_water_column_untouched(self):
        Nk, Mk, Gc = _base_state()
        Mk = Mk.at[..., SRTH2O].set(1e-16)
        _, Mk2, _ = nucleation_step(
            Nk, Mk, Gc, jnp.full(3, 220.0), jnp.full(3, 1e6), dt=360.0
        )
        np.testing.assert_array_equal(
            np.asarray(Mk2[..., SRTH2O]), np.asarray(Mk[..., SRTH2O])
        )

    def test_deposit_goes_to_bin0_so4_only(self):
        Nk, Mk, Gc = _base_state(gc_h2so4=1e-9)
        Nk2, Mk2, _ = nucleation_step(
            Nk, Mk, Gc, jnp.full(3, 220.0), jnp.full(3, 1e6), dt=360.0
        )
        np.testing.assert_array_equal(
            np.asarray(Nk2[:, 1:]), np.asarray(Nk[:, 1:])
        )
        np.testing.assert_array_equal(
            np.asarray(Mk2[:, 1:, :]), np.asarray(Mk[:, 1:, :])
        )

    def test_analytic_matches_brute_force_euler(self):
        """Closed form == converged limit of the full model's substep loop.

        Reference: 10^5 forward-Euler substeps of dG/dt = -A G^p in numpy,
        in a burst scenario where a 4-substep Euler errs by ~50%.
        """
        gc0 = 1e-11
        temp, boxvol, dt = 208.0, 1e6, 360.0

        # Brute-force Euler on the gas ODE (same constants as the module)
        conv = 1000.0 / MW_H2SO4 * 6.02214076e23 / boxvol
        kbn = float(dunne_binary_neutral_rate(temp, conv * gc0) / (conv * gc0 * 1e-6) ** 3.95)
        A = kbn * (1e-6 * conv) ** 3.95 * _MNUC * boxvol * (MW_H2SO4 / MW_SO4)
        n_ref = 100_000
        g = gc0
        for _ in range(n_ref):
            g = g - A * g**3.95 * (dt / n_ref)
        d_gas_ref = gc0 - g

        Nk, Mk, Gc = _base_state(C=1, gc_h2so4=gc0)
        _, Mk2, Gc2 = nucleation_step(
            Nk, Mk, Gc, jnp.full(1, temp), jnp.full(1, boxvol), dt
        )
        d_gas = gc0 - float(Gc2[0, GH2SO4])
        assert d_gas == pytest.approx(d_gas_ref, rel=1e-4)
        assert float(Mk2[0, 0, SRTSO4]) - float(Mk[0, 0, SRTSO4]) == pytest.approx(
            d_gas_ref * MW_SO4 / MW_H2SO4, rel=1e-4
        )

    def test_mnfix_bin0_renormalization_pinned(self):
        """MNFIX phase 2 renormalizes sub-bin0 nucleated number by
        _MNUC*1.2/xk[0] when bin 0 is otherwise empty — regression-pins the
        documented cluster-mass / grid interaction."""
        C = 1
        Nk = jnp.zeros((C, NBINS))
        Mk = jnp.zeros((C, NBINS, 2))
        Gc = jnp.zeros((C, 2)).at[..., GH2SO4].set(1e-9)
        Nk2, Mk2, _ = nucleation_step(
            Nk, Mk, Gc, jnp.full(C, 208.0), jnp.full(C, 1e6), dt=60.0
        )
        Nf, Mf = mnfix(Nk2, Mk2, XK, icomp_nodiag=1)
        expected_ratio = _MNUC / (float(XK[0]) * 1.2)
        got_ratio = float(Nf[0, 0] / Nk2[0, 0])
        assert got_ratio == pytest.approx(expected_ratio, rel=1e-6)
        # dry mass conserved by the renormalization
        assert float(Mf[0, 0, 0]) == pytest.approx(float(Mk2[0, 0, 0]), rel=1e-12)
