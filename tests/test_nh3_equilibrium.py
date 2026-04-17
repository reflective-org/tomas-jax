"""Unit tests for NH3/NH4 equilibrium (eznh3eqm.f port).

Tests cover:
    - Ammonia-limited case (all NH3 goes to particles)
    - Ammonia-excess case (particles filled to 2:1 stoichiometry)
    - Total nitrogen conservation (gas + particle)
    - Distribution proportional to sulfate per bin
    - Edge cases: zero NH3, zero SO4, empty bins
    - JIT compatibility
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import (
    NBINS, ICOMP, SRTSO4, SRTNH4, MW_NH3, MW_NH4, MW_SO4,
    N_GAS_SPECIES,
)
from tomas_jax.physics.nh3_equilibrium import eznh3eqm


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def standard_grid():
    """Standard 40-bin grid with some sulfate in bins 10-20."""
    Mk = jnp.zeros((NBINS, ICOMP))
    # Put SO4 in bins 10-20 with varying amounts
    for k in range(10, 21):
        Mk = Mk.at[k, SRTSO4].set(1e-15 * (k - 9))
    return Mk


@pytest.fixture
def standard_gc():
    """Gas array with some NH3."""
    Gc = jnp.zeros(N_GAS_SPECIES)
    return Gc


# =========================================================================
# Ammonia-limited case
# =========================================================================

class TestAmmoniaLimited:
    """When tot_nh3/2 < tot_so4, all ammonia goes to particles."""

    def test_no_gas_phase_nh3_remains(self, standard_grid, standard_gc):
        """Gas-phase NH3 should be zero after equilibrium."""
        Mk = standard_grid
        # Total SO4 moles = sum(Mk[:, SRTSO4]) / MW_SO4
        tot_so4_mol = float(jnp.sum(Mk[:, SRTSO4])) / MW_SO4
        # Add NH3 that gives tot_nh3/2 < tot_so4 (i.e., tot_nh3 < 2*tot_so4)
        nh3_mass = tot_so4_mol * 0.5 * MW_NH3  # half of what's needed for full neutralization
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        Gc_new, Mk_new = eznh3eqm(Gc, Mk)
        assert float(Gc_new[SRTNH4]) == pytest.approx(0.0, abs=1e-30)

    def test_all_nh3_to_particles(self, standard_grid, standard_gc):
        """All NH3 should be distributed to particle phase."""
        Mk = standard_grid
        tot_so4_mol = float(jnp.sum(Mk[:, SRTSO4])) / MW_SO4
        nh3_mass = tot_so4_mol * 0.5 * MW_NH3
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        Gc_new, Mk_new = eznh3eqm(Gc, Mk)
        # Total particle NH4 should equal total NH3 input (converted to NH4 MW)
        tot_nh3_input_mol = nh3_mass / MW_NH3
        tot_nh4_particle_mol = float(jnp.sum(Mk_new[:, SRTNH4])) / MW_NH4
        assert tot_nh4_particle_mol == pytest.approx(tot_nh3_input_mol, rel=1e-10)

    def test_distributed_proportional_to_so4(self, standard_grid, standard_gc):
        """NH4 should be distributed in proportion to SO4 in each bin."""
        Mk = standard_grid
        tot_so4_mol = float(jnp.sum(Mk[:, SRTSO4])) / MW_SO4
        nh3_mass = tot_so4_mol * 0.5 * MW_NH3
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        _, Mk_new = eznh3eqm(Gc, Mk)

        # sfrac = SO4[k] / sum(SO4)
        so4_total = jnp.sum(Mk[:, SRTSO4])
        sfrac = Mk[:, SRTSO4] / so4_total

        # NH4 in each bin should be proportional to sfrac
        nh4_total = jnp.sum(Mk_new[:, SRTNH4])
        nh4_frac = Mk_new[:, SRTNH4] / jnp.maximum(nh4_total, 1e-30)

        np.testing.assert_allclose(
            np.array(nh4_frac), np.array(sfrac), atol=1e-12
        )


# =========================================================================
# Ammonia-excess case
# =========================================================================

class TestAmmoniaExcess:
    """When tot_nh3/2 >= tot_so4, particles fully neutralized, excess stays gas."""

    def test_particles_at_2to1_stoichiometry(self, standard_grid, standard_gc):
        """Each bin should have NH4/SO4 molar ratio = 2:1."""
        Mk = standard_grid
        tot_so4_mol = float(jnp.sum(Mk[:, SRTSO4])) / MW_SO4
        # 10x excess ammonia
        nh3_mass = tot_so4_mol * 10.0 * MW_NH3
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        _, Mk_new = eznh3eqm(Gc, Mk)

        # For bins with SO4, check 2:1 molar ratio
        for k in range(10, 21):
            so4_mol = float(Mk_new[k, SRTSO4]) / MW_SO4
            nh4_mol = float(Mk_new[k, SRTNH4]) / MW_NH4
            if so4_mol > 1e-30:
                assert nh4_mol / so4_mol == pytest.approx(2.0, rel=1e-10)

    def test_excess_nh3_in_gas_phase(self, standard_grid, standard_gc):
        """Excess ammonia should remain in gas phase."""
        Mk = standard_grid
        tot_so4_mol = float(jnp.sum(Mk[:, SRTSO4])) / MW_SO4
        nh3_mass = tot_so4_mol * 10.0 * MW_NH3
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        Gc_new, _ = eznh3eqm(Gc, Mk)

        # Gas NH3 should be positive (excess)
        assert float(Gc_new[SRTNH4]) > 0.0

    def test_gas_phase_equals_excess(self, standard_grid, standard_gc):
        """Gas NH3 should equal total - 2*SO4 (in moles)."""
        Mk = standard_grid
        tot_so4_mol = float(jnp.sum(Mk[:, SRTSO4])) / MW_SO4
        nh3_mass = tot_so4_mol * 10.0 * MW_NH3
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        Gc_new, _ = eznh3eqm(Gc, Mk)

        expected_gas_mol = (nh3_mass / MW_NH3 + 0.0) - tot_so4_mol * 2.0
        actual_gas_mol = float(Gc_new[SRTNH4]) / MW_NH3
        assert actual_gas_mol == pytest.approx(expected_gas_mol, rel=1e-10)


# =========================================================================
# Nitrogen conservation
# =========================================================================

class TestNitrogenConservation:
    """Total nitrogen (gas NH3 + particle NH4) must be conserved."""

    @pytest.mark.parametrize("nh3_factor", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_total_n_conserved(self, standard_grid, standard_gc, nh3_factor):
        """Total N moles before = total N moles after."""
        Mk = standard_grid
        tot_so4_mol = float(jnp.sum(Mk[:, SRTSO4])) / MW_SO4
        nh3_mass = tot_so4_mol * nh3_factor * MW_NH3
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        # Initial particle NH4 is zero
        tot_n_before = nh3_mass / MW_NH3

        Gc_new, Mk_new = eznh3eqm(Gc, Mk)

        tot_n_after = (float(Gc_new[SRTNH4]) / MW_NH3
                       + float(jnp.sum(Mk_new[:, SRTNH4])) / MW_NH4)
        assert tot_n_after == pytest.approx(tot_n_before, rel=1e-10)

    def test_conservation_with_initial_particle_nh4(self, standard_grid, standard_gc):
        """Conservation holds when particles already have NH4."""
        Mk = standard_grid
        # Pre-load some particle NH4
        for k in range(10, 21):
            Mk = Mk.at[k, SRTNH4].set(1e-16)
        nh3_mass = 1e-13
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        tot_n_before = (nh3_mass / MW_NH3
                        + float(jnp.sum(Mk[:, SRTNH4])) / MW_NH4)

        Gc_new, Mk_new = eznh3eqm(Gc, Mk)

        tot_n_after = (float(Gc_new[SRTNH4]) / MW_NH3
                       + float(jnp.sum(Mk_new[:, SRTNH4])) / MW_NH4)
        assert tot_n_after == pytest.approx(tot_n_before, rel=1e-10)


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    """Edge cases: zero inputs, uniform bins, single bin."""

    def test_zero_nh3(self, standard_grid, standard_gc):
        """Zero gas NH3 and zero particle NH4: nothing changes."""
        Mk = standard_grid
        Gc = standard_gc  # NH3 = 0

        Gc_new, Mk_new = eznh3eqm(Gc, Mk)

        np.testing.assert_allclose(np.array(Gc_new), np.array(Gc), atol=1e-30)
        np.testing.assert_allclose(
            np.array(Mk_new[:, SRTNH4]), 0.0, atol=1e-30
        )

    def test_zero_so4(self, standard_gc):
        """No sulfate: all NH3 stays gas phase."""
        Mk = jnp.zeros((NBINS, ICOMP))
        nh3_mass = 1e-12
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        Gc_new, Mk_new = eznh3eqm(Gc, Mk)

        # No particles to absorb, all stays gas
        # (ammonia-limited since tot_so4=0, tot_nh3/2 > 0 = tot_so4 → excess path)
        # In excess path: particles get 2*SO4=0, gas gets remainder
        assert float(Gc_new[SRTNH4]) == pytest.approx(nh3_mass, rel=1e-10)
        np.testing.assert_allclose(
            np.array(Mk_new[:, SRTNH4]), 0.0, atol=1e-30
        )

    def test_single_bin_with_so4(self, standard_gc):
        """Single populated bin gets all the NH4."""
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[15, SRTSO4].set(1e-14)
        nh3_mass = 5e-16
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        Gc_new, Mk_new = eznh3eqm(Gc, Mk)

        # All NH4 should be in bin 15
        tot_nh4 = float(jnp.sum(Mk_new[:, SRTNH4]))
        assert float(Mk_new[15, SRTNH4]) == pytest.approx(tot_nh4, rel=1e-12)

    def test_so4_unchanged(self, standard_grid, standard_gc):
        """SO4 should not be modified by NH3 equilibrium."""
        Mk = standard_grid
        nh3_mass = 1e-13
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        _, Mk_new = eznh3eqm(Gc, Mk)

        np.testing.assert_allclose(
            np.array(Mk_new[:, SRTSO4]),
            np.array(Mk[:, SRTSO4]),
            atol=1e-30,
        )

    def test_other_species_unchanged(self, standard_grid, standard_gc):
        """Species other than NH4 should not be modified."""
        Mk = standard_grid
        # Add some organic mass
        Mk = Mk.at[15, 1].set(1e-15)
        nh3_mass = 1e-13
        Gc = standard_gc.at[SRTNH4].set(nh3_mass)

        _, Mk_new = eznh3eqm(Gc, Mk)

        # Check organics unchanged
        assert float(Mk_new[15, 1]) == pytest.approx(float(Mk[15, 1]), rel=1e-12)
        # Check water unchanged (it's zero)
        np.testing.assert_allclose(
            np.array(Mk_new[:, 43]),  # SRTH2O
            np.array(Mk[:, 43]),
            atol=1e-30,
        )

    def test_other_gc_species_unchanged(self, standard_grid, standard_gc):
        """Gas species other than NH3/NH4 should not be modified."""
        Mk = standard_grid
        Gc = standard_gc.at[SRTNH4].set(1e-13)
        Gc = Gc.at[0].set(5e-12)  # some H2SO4 gas

        Gc_new, _ = eznh3eqm(Gc, Mk)

        assert float(Gc_new[0]) == pytest.approx(5e-12, rel=1e-12)


# =========================================================================
# JIT compatibility
# =========================================================================

class TestJIT:
    """Verify JIT compilation works."""

    def test_jit_compiles(self, standard_grid, standard_gc):
        """eznh3eqm should be JIT-compilable."""
        Mk = standard_grid
        Gc = standard_gc.at[SRTNH4].set(1e-13)

        jitted = jax.jit(eznh3eqm)
        Gc_new, Mk_new = jitted(Gc, Mk)

        assert jnp.all(jnp.isfinite(Gc_new))
        assert jnp.all(jnp.isfinite(Mk_new))

    def test_jit_matches_eager(self, standard_grid, standard_gc):
        """JIT output should match eager execution."""
        Mk = standard_grid
        Gc = standard_gc.at[SRTNH4].set(1e-13)

        Gc_eager, Mk_eager = eznh3eqm(Gc, Mk)
        Gc_jit, Mk_jit = jax.jit(eznh3eqm)(Gc, Mk)

        np.testing.assert_allclose(
            np.array(Gc_jit), np.array(Gc_eager), atol=1e-20
        )
        np.testing.assert_allclose(
            np.array(Mk_jit), np.array(Mk_eager), atol=1e-20
        )
