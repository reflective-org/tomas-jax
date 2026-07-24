"""Tests for the fast-branch Tabazadeh water equilibrium and binary density."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tomas_jax.fast.config import NBINS, SRTSO4, SRTH2O
from tomas_jax.fast.density import calc_density
from tomas_jax.fast.water import equilibrium_water, h2so4_equilibrium_wt
from tomas_jax.physics.radiative_forcing import (
    h2so4_equilibrium_wt as h2so4_equilibrium_wt_numpy,
)


class TestTabazadehJax:
    def test_matches_numpy_reference_in_range(self):
        """JAX port agrees with the numpy reference over the fitted range."""
        for T in [185, 190, 200, 210, 220, 230, 240, 250, 260]:
            for rh in [0.5, 1, 5, 10, 30, 50, 70, 90]:
                ref = h2so4_equilibrium_wt_numpy(T, rh)
                got = float(h2so4_equilibrium_wt(T, rh))
                assert got == pytest.approx(ref, abs=1e-10), f"T={T}, RH={rh}"

    def test_typical_stratospheric(self):
        wt = float(h2so4_equilibrium_wt(220.0, 5.0))
        assert 55 < wt < 70

    def test_monotonic_in_rh(self):
        wts = h2so4_equilibrium_wt(jnp.full(3, 220.0), jnp.array([5.0, 10.0, 50.0]))
        assert wts[0] > wts[1] > wts[2]

    def test_bounds_low_rh(self):
        """RH -> 0 clamps to the 80 wt% table ceiling."""
        assert float(h2so4_equilibrium_wt(220.0, 0.0)) == 80.0

    def test_bounds_high_rh(self):
        """RH >= 100 clamps to the 10 wt% table floor (NOT 0 like the numpy
        reference — a 0 wt% would imply infinite water uptake)."""
        assert float(h2so4_equilibrium_wt(220.0, 100.0)) == 10.0
        assert float(h2so4_equilibrium_wt(220.0, 150.0)) == 10.0

    def test_valid_range(self):
        for T in [190, 200, 210, 220, 230, 240, 250]:
            for rh in [1, 5, 10, 50]:
                wt = float(h2so4_equilibrium_wt(float(T), float(rh)))
                assert 10 <= wt <= 80

    def test_warm_extrapolation_sanity(self):
        """Extrapolation above 260 K stays physical (~43 wt% at 298 K/RH50,
        Giauque / Clegg-Brimblecombe isopiestic data)."""
        wt_50 = float(h2so4_equilibrium_wt(298.0, 50.0))
        assert 38 <= wt_50 <= 48
        wt_80 = float(h2so4_equilibrium_wt(298.0, 80.0))
        assert 22 <= wt_80 <= 34
        # Smooth in T: no jump across the 260 K validity edge
        for rh in [10.0, 50.0, 90.0]:
            w260 = float(h2so4_equilibrium_wt(260.0, rh))
            w300 = float(h2so4_equilibrium_wt(300.0, rh))
            assert abs(w260 - w300) < 5.0

    def test_batched_equals_scalar(self):
        T = np.array([200.0, 220.0, 250.0, 298.0])
        rh = np.array([5.0, 50.0, 90.0, 30.0])
        batched = np.asarray(h2so4_equilibrium_wt(jnp.asarray(T), jnp.asarray(rh)))
        scalar = np.array([float(h2so4_equilibrium_wt(t, r)) for t, r in zip(T, rh)])
        np.testing.assert_array_equal(batched, scalar)

    def test_jit_and_grad_finite(self):
        f = jax.jit(h2so4_equilibrium_wt)
        assert np.isfinite(float(f(220.0, 5.0)))
        g = jax.grad(lambda t: h2so4_equilibrium_wt(t, 50.0))(220.0)
        assert np.isfinite(float(g))


class TestEquilibriumWater:
    def test_water_acid_ratio(self):
        """water/acid == 100/wt - 1 exactly."""
        Mk = jnp.zeros((NBINS, 2)).at[:, SRTSO4].set(1e-15)
        temp, rh = 220.0, 0.05
        Mk_new = equilibrium_water(Mk, temp, rh)
        wt = float(h2so4_equilibrium_wt(temp, rh * 100.0))
        acid = 1e-15 * 98.0 / 96.0
        expected = acid * (100.0 / wt - 1.0)
        np.testing.assert_allclose(np.asarray(Mk_new[:, SRTH2O]), expected, rtol=1e-12)

    def test_zero_so4_gives_zero_water(self):
        Mk = jnp.zeros((NBINS, 2))
        Mk_new = equilibrium_water(Mk, 220.0, 0.5)
        assert float(jnp.max(jnp.abs(Mk_new[:, SRTH2O]))) == 0.0

    def test_batched_shapes(self):
        C = 7
        Mk = jnp.ones((C, NBINS, 2)) * 1e-15
        temp = jnp.linspace(200.0, 298.0, C)
        rh = jnp.linspace(0.05, 0.9, C)
        Mk_new = equilibrium_water(Mk, temp, rh)
        assert Mk_new.shape == (C, NBINS, 2)
        # dry column untouched
        np.testing.assert_array_equal(
            np.asarray(Mk_new[..., SRTSO4]), np.asarray(Mk[..., SRTSO4])
        )


class TestBinaryDensity:
    def test_matches_full_model_limit(self):
        """Binary density == full calc_density with NH4/organics zeroed."""
        from tomas_jax.physics.density import calc_density as calc_density_full
        from tomas_jax.core.config import ICOMP, SRTH2O as SRTH2O_FULL

        rng = np.random.default_rng(0)
        mso4 = rng.uniform(0.0, 1e-10, NBINS)
        mh2o = rng.uniform(0.0, 1e-10, NBINS)

        Mk2 = jnp.stack([jnp.asarray(mso4), jnp.asarray(mh2o)], axis=1)
        Mk44 = jnp.zeros((NBINS, ICOMP))
        Mk44 = Mk44.at[:, 0].set(mso4).at[:, SRTH2O_FULL].set(mh2o)

        np.testing.assert_allclose(
            np.asarray(calc_density(Mk2)),
            np.asarray(calc_density_full(Mk44)),
            rtol=1e-12,
        )

    def test_reference_values(self):
        # Pure water limit
        Mk = jnp.array([[0.0, 1e-10]])
        assert float(calc_density(Mk)[0]) == pytest.approx(997.1, abs=1.0)
        # ~43 wt% H2SO4 (x is on the acid basis: mso4 s.t. 100*mso4*98/96/(tot)=43)
        mso4 = 0.43 * 96.0 / 98.0
        Mk = jnp.array([[mso4, 1.0 - mso4]]) * 1e-10
        assert 1310 < float(calc_density(Mk)[0]) < 1340

    def test_empty_bin_fallback(self):
        Mk = jnp.zeros((3, 2))
        np.testing.assert_array_equal(np.asarray(calc_density(Mk)), 1000.0)

    def test_monotone_in_wt(self):
        frac = jnp.linspace(0.05, 0.95, 19)
        Mk = jnp.stack([frac, 1.0 - frac], axis=1) * 1e-10
        d = np.asarray(calc_density(Mk))
        assert np.all(np.diff(d) > 0)
