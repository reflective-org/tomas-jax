"""Tests for dilution / entrainment process."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tomas_jax.core.config import ICOMP, N_GAS_SPECIES, SRTSO4, NBINS
from tomas_jax.physics.dilution import dilution_step, dilute_tracer


@pytest.fixture
def setup():
    """Common test setup: 40-bin aerosol + gas state."""
    Nk = jnp.ones(NBINS) * 1e6
    Mk = jnp.ones((NBINS, ICOMP)) * 1e-15
    Gc = jnp.ones(N_GAS_SPECIES) * 1e-10
    return Nk, Mk, Gc


class TestDilution:

    def test_clean_air_decay(self, setup):
        """kdil > 0, Cbg = 0 -> C(t) = C0 * exp(-kdil*t)."""
        Nk, Mk, Gc = setup
        kdil = 1e-4
        dt = 3600.0  # 1 hour

        Nk_new, Mk_new, Gc_new = dilution_step(
            Nk, Mk, Gc, dt, kdil,
            jnp.zeros_like(Nk), jnp.zeros_like(Mk), jnp.zeros_like(Gc))

        expected_decay = np.exp(-kdil * dt)
        np.testing.assert_allclose(Nk_new, Nk * expected_decay, rtol=1e-12)
        np.testing.assert_allclose(Mk_new, Mk * expected_decay, rtol=1e-12)
        np.testing.assert_allclose(Gc_new, Gc * expected_decay, rtol=1e-12)

    def test_relaxation_to_background(self, setup):
        """kdil > 0, Cbg != 0 -> C approaches Cbg."""
        Nk, Mk, Gc = setup
        kdil = 1e-3
        dt = 3600.0

        Nk_bg = jnp.ones(NBINS) * 500.0
        Mk_bg = jnp.ones((NBINS, ICOMP)) * 1e-18
        Gc_bg = jnp.ones(N_GAS_SPECIES) * 1e-12

        Nk_new, Mk_new, Gc_new = dilution_step(
            Nk, Mk, Gc, dt, kdil, Nk_bg, Mk_bg, Gc_bg)

        decay = np.exp(-kdil * dt)
        expected_Nk = Nk_bg + (Nk - Nk_bg) * decay
        np.testing.assert_allclose(Nk_new, expected_Nk, rtol=1e-12)

        # After dilution, Nk should be closer to Nk_bg than before
        dist_before = jnp.sum(jnp.abs(Nk - Nk_bg))
        dist_after = jnp.sum(jnp.abs(Nk_new - Nk_bg))
        assert float(dist_after) < float(dist_before)

    def test_zero_kdil_noop(self, setup):
        """kdil = 0 -> no change."""
        Nk, Mk, Gc = setup

        Nk_new, Mk_new, Gc_new = dilution_step(
            Nk, Mk, Gc, 3600.0, 0.0,
            jnp.zeros_like(Nk), jnp.zeros_like(Mk), jnp.zeros_like(Gc))

        np.testing.assert_array_equal(Nk_new, Nk)
        np.testing.assert_array_equal(Mk_new, Mk)
        np.testing.assert_array_equal(Gc_new, Gc)

    def test_mass_species_independence(self, setup):
        """Each Mk species should dilute independently."""
        Nk, _, Gc = setup
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[:, 0].set(1e-12)   # SO4 only in species 0
        Mk = Mk.at[:, 1].set(1e-14)   # Organics in species 1
        kdil = 1e-4
        dt = 3600.0

        _, Mk_new, _ = dilution_step(
            Nk, Mk, Gc, dt, kdil,
            jnp.zeros_like(Nk), jnp.zeros_like(Mk), jnp.zeros_like(Gc))

        decay = np.exp(-kdil * dt)
        np.testing.assert_allclose(Mk_new[:, 0], 1e-12 * decay, rtol=1e-12)
        np.testing.assert_allclose(Mk_new[:, 1], 1e-14 * decay, rtol=1e-12)
        # Other species remain zero
        np.testing.assert_allclose(Mk_new[:, 2], 0.0, atol=1e-30)

    def test_steady_state(self, setup):
        """C0 = Cbg -> no change regardless of kdil."""
        Nk, Mk, Gc = setup

        Nk_new, Mk_new, Gc_new = dilution_step(
            Nk, Mk, Gc, 3600.0, 1e-2,
            Nk_bg=Nk, Mk_bg=Mk, Gc_bg=Gc)

        np.testing.assert_allclose(Nk_new, Nk, rtol=1e-14)
        np.testing.assert_allclose(Mk_new, Mk, rtol=1e-14)
        np.testing.assert_allclose(Gc_new, Gc, rtol=1e-14)

    def test_jit_compilable(self, setup):
        """dilution_step should be JIT-compilable."""
        Nk, Mk, Gc = setup
        f = jax.jit(dilution_step)
        Nk_new, Mk_new, Gc_new = f(
            Nk, Mk, Gc, 3600.0, 1e-4,
            jnp.zeros_like(Nk), jnp.zeros_like(Mk), jnp.zeros_like(Gc))
        assert float(jnp.sum(Nk_new)) < float(jnp.sum(Nk))


class TestPassiveTracer:

    def test_tracer_clean_air_decay(self):
        """Tracer with bg=0 should decay as exp(-kdil*t)."""
        tracer = jnp.float64(1.0)
        kdil = 1e-4
        dt = 3600.0
        for _ in range(10):
            tracer = dilute_tracer(tracer, dt, kdil)
        expected = np.exp(-kdil * dt * 10)
        np.testing.assert_allclose(float(tracer), expected, rtol=1e-12)

    def test_tracer_relaxation(self):
        """Tracer with bg != 0 should relax toward background."""
        tracer = jnp.float64(1.0)
        tracer_bg = 0.5
        kdil = 1e-3
        dt = 3600.0
        tracer_new = dilute_tracer(tracer, dt, kdil, tracer_bg)
        expected = tracer_bg + (1.0 - tracer_bg) * np.exp(-kdil * dt)
        np.testing.assert_allclose(float(tracer_new), expected, rtol=1e-12)

    def test_tracer_zero_kdil(self):
        """Tracer with kdil=0 should be unchanged."""
        tracer = jnp.float64(0.75)
        tracer_new = dilute_tracer(tracer, 3600.0, 0.0)
        np.testing.assert_allclose(float(tracer_new), 0.75)

    def test_tracer_jit_compilable(self):
        """dilute_tracer should be JIT-compilable."""
        f = jax.jit(dilute_tracer)
        result = f(jnp.float64(1.0), 3600.0, 1e-4)
        assert float(result) < 1.0
