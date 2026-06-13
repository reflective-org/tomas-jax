"""Tests for volume-based dilution / entrainment process."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tomas_jax.core.config import ICOMP, N_GAS_SPECIES, SRTSO4, NBINS
from tomas_jax.physics.dilution import dilution_step, dilute_tracer


BOXVOL = 1.0e6  # cm³


@pytest.fixture
def setup():
    """Common test setup: 40-bin aerosol + gas state."""
    Nk = jnp.ones(NBINS) * 1e6
    Mk = jnp.ones((NBINS, ICOMP)) * 1e-15
    Gc = jnp.ones(N_GAS_SPECIES) * 1e-10
    return Nk, Mk, Gc


class TestVolumeBasedDilution:

    def test_clean_air_mass_conserved(self, setup):
        """Clean-air dilution: total per-cell mass is conserved."""
        Nk, Mk, Gc = setup
        kdil = 1e-4
        dt = 3600.0  # 1 hour

        boxvol_new, Nk_new, Mk_new, Gc_new = dilution_step(
            jnp.float64(BOXVOL), Nk, Mk, Gc, dt, kdil)

        # Per-cell totals unchanged (no background)
        np.testing.assert_array_equal(Nk_new, Nk)
        np.testing.assert_array_equal(Mk_new, Mk)
        np.testing.assert_array_equal(Gc_new, Gc)

    def test_boxvol_grows(self, setup):
        """Box volume grows as exp(+kdil*dt)."""
        Nk, Mk, Gc = setup
        kdil = 1e-4
        dt = 3600.0

        boxvol_new, _, _, _ = dilution_step(
            jnp.float64(BOXVOL), Nk, Mk, Gc, dt, kdil)

        expected_boxvol = BOXVOL * np.exp(kdil * dt)
        np.testing.assert_allclose(float(boxvol_new), expected_boxvol, rtol=1e-12)

    def test_concentration_decreases(self, setup):
        """Concentration [#/cm³] decreases as exp(-kdil*dt)."""
        Nk, Mk, Gc = setup
        kdil = 1e-4
        dt = 3600.0

        boxvol_new, Nk_new, _, _ = dilution_step(
            jnp.float64(BOXVOL), Nk, Mk, Gc, dt, kdil)

        # N per cell unchanged, but concentration = N/boxvol decreases
        conc_before = float(jnp.sum(Nk)) / BOXVOL
        conc_after = float(jnp.sum(Nk_new)) / float(boxvol_new)
        expected_ratio = np.exp(-kdil * dt)
        np.testing.assert_allclose(conc_after / conc_before, expected_ratio, rtol=1e-12)

    def test_ambient_background_adds_mass(self, setup):
        """Ambient background: entrained air adds mass."""
        Nk, Mk, Gc = setup
        kdil = 1e-3
        dt = 3600.0

        # Background at half the initial concentration
        Nk_bg_conc = 0.5 * Nk / BOXVOL   # [#/cm³]
        Mk_bg_conc = 0.5 * Mk / BOXVOL   # [kg/cm³]
        Gc_bg_conc = 0.5 * Gc / BOXVOL   # [kg/cm³]

        boxvol_new, Nk_new, Mk_new, Gc_new = dilution_step(
            jnp.float64(BOXVOL), Nk, Mk, Gc, dt, kdil,
            Nk_bg_conc, Mk_bg_conc, Gc_bg_conc)

        dV = float(boxvol_new) - BOXVOL

        # Mass should increase: original + background * dV
        expected_Nk = Nk + Nk_bg_conc * dV
        np.testing.assert_allclose(Nk_new, expected_Nk, rtol=1e-12)
        assert float(jnp.sum(Nk_new)) > float(jnp.sum(Nk))

    def test_zero_kdil_noop(self, setup):
        """kdil = 0 -> no change."""
        Nk, Mk, Gc = setup

        boxvol_new, Nk_new, Mk_new, Gc_new = dilution_step(
            jnp.float64(BOXVOL), Nk, Mk, Gc, 3600.0, 0.0)

        np.testing.assert_allclose(float(boxvol_new), BOXVOL, rtol=1e-14)
        np.testing.assert_array_equal(Nk_new, Nk)
        np.testing.assert_array_equal(Mk_new, Mk)
        np.testing.assert_array_equal(Gc_new, Gc)

    def test_multi_step_volume_accumulation(self, setup):
        """Volume accumulates correctly over multiple steps."""
        Nk, Mk, Gc = setup
        kdil = 1e-4
        dt = 60.0  # 1 minute
        nsteps = 1440  # 24 hours

        boxvol = jnp.float64(BOXVOL)
        for _ in range(nsteps):
            boxvol, Nk, Mk, Gc = dilution_step(boxvol, Nk, Mk, Gc, dt, kdil)

        expected_boxvol = BOXVOL * np.exp(kdil * nsteps * dt)
        np.testing.assert_allclose(float(boxvol), expected_boxvol, rtol=1e-10)

        # Per-cell mass still conserved (clean air)
        expected_Nk = jnp.ones(NBINS) * 1e6
        np.testing.assert_allclose(Nk, expected_Nk, rtol=1e-12)

    def test_jit_compilable(self, setup):
        """dilution_step should be JIT-compilable."""
        Nk, Mk, Gc = setup
        f = jax.jit(dilution_step)
        boxvol_new, Nk_new, Mk_new, Gc_new = f(
            jnp.float64(BOXVOL), Nk, Mk, Gc, 3600.0, 1e-4)
        assert float(boxvol_new) > BOXVOL

    def test_species_independence(self, setup):
        """Each Mk species should be treated independently."""
        Nk, _, Gc = setup
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[:, 0].set(1e-12)
        Mk = Mk.at[:, 1].set(1e-14)
        kdil = 1e-4
        dt = 3600.0

        # Clean air: all species preserved
        _, _, Mk_new, _ = dilution_step(
            jnp.float64(BOXVOL), Nk, Mk, Gc, dt, kdil)
        np.testing.assert_allclose(Mk_new[:, 0], 1e-12, rtol=1e-14)
        np.testing.assert_allclose(Mk_new[:, 1], 1e-14, rtol=1e-14)
        np.testing.assert_allclose(Mk_new[:, 2], 0.0, atol=1e-30)


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

    def test_tracer_matches_volume_ratio(self):
        """Passive tracer decay matches inverse of volume expansion."""
        kdil = 1e-4
        dt = 60.0
        nsteps = 100

        # Volume expansion
        boxvol = BOXVOL
        for _ in range(nsteps):
            boxvol *= np.exp(kdil * dt)
        vol_ratio = BOXVOL / boxvol  # V0/V(t)

        # Tracer decay
        tracer = 1.0
        for _ in range(nsteps):
            tracer *= np.exp(-kdil * dt)

        np.testing.assert_allclose(tracer, vol_ratio, rtol=1e-12)


class TestMakeStepWithDilution:

    def test_make_step_returns_4_values(self, setup):
        """make_step() with dilution returns (Nk, Mk, Gc, boxvol)."""
        from tomas_jax.solvers.condensation import make_step
        from tomas_jax.core.config import make_grid, XK0

        Nk, Mk, Gc = setup
        xk = make_grid(NBINS, XK0, 2.0)

        step_fn = make_step(['condensation', 'dilution'], cond_method='ppm_jit')
        result = step_fn(Nk, Mk, Gc, xk,
                         jnp.float64(298.0), jnp.float64(101325.0),
                         jnp.float64(BOXVOL), jnp.float64(0.5),
                         jnp.float64(1.0), jnp.float64(60.0),
                         kdil=jnp.float64(1e-4))
        assert len(result) == 4
        Nk_new, Mk_new, Gc_new, boxvol_new = result
        assert float(boxvol_new) > BOXVOL

    def test_make_step_without_dilution_returns_4_values(self, setup):
        """make_step() without dilution also returns 4 values (boxvol unchanged)."""
        from tomas_jax.solvers.condensation import make_step
        from tomas_jax.core.config import make_grid, XK0

        Nk, Mk, Gc = setup
        xk = make_grid(NBINS, XK0, 2.0)

        step_fn = make_step(['condensation'], cond_method='ppm_jit')
        result = step_fn(Nk, Mk, Gc, xk,
                         jnp.float64(298.0), jnp.float64(101325.0),
                         jnp.float64(BOXVOL), jnp.float64(0.5),
                         jnp.float64(1.0), jnp.float64(60.0))
        assert len(result) == 4
        _, _, _, boxvol_out = result
        np.testing.assert_allclose(float(boxvol_out), BOXVOL, rtol=1e-14)

    def test_clean_air_mass_budget_conserved(self, setup):
        """Clean-air dilution: total gas + aerosol mass is conserved."""
        from tomas_jax.solvers.condensation import make_step
        from tomas_jax.core.config import make_grid, XK0, SRTH2O

        Nk, Mk, Gc = setup
        xk = make_grid(NBINS, XK0, 2.0)

        step_fn = make_step(['dilution'], cond_method='ppm_jit')

        init_mass = float(jnp.sum(Mk[:, :SRTH2O])) + float(jnp.sum(Gc))

        boxvol = jnp.float64(BOXVOL)
        for _ in range(100):
            Nk, Mk, Gc, boxvol = step_fn(
                Nk, Mk, Gc, xk,
                jnp.float64(298.0), jnp.float64(101325.0),
                boxvol, jnp.float64(0.5), jnp.float64(1.0), jnp.float64(60.0),
                kdil=jnp.float64(1e-3))

        final_mass = float(jnp.sum(Mk[:, :SRTH2O])) + float(jnp.sum(Gc))
        np.testing.assert_allclose(final_mass, init_mass, rtol=1e-12)
