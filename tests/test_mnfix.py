"""Unit tests for MNFIX mass-number drift correction (mnfix.f port).

Tests cover:
    - Phase 1: Empty bin fix (Nk < NEPS)
    - Phase 2: Extreme out-of-range (avg > xk[nbins] or avg < xk[0])
    - Phase 3: Partial transfer (avg outside bin → split to neighbor)
    - Number and mass conservation
    - In-range bins unmodified
    - Multi-bin jumps (analytical log2 target finding)
    - JIT compatibility
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, SRTSO4, make_grid,
)
from tomas_jax.core.mnfix_jax import mnfix_jax


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def xk():
    return make_grid()


@pytest.fixture
def empty_distribution():
    """Empty distribution."""
    Nk = jnp.zeros(NBINS)
    Mk = jnp.zeros((NBINS, ICOMP))
    return Nk, Mk


def make_in_range_bin(xk, k, n=1e8):
    """Create a single bin with avg mass in range [xk[k], xk[k+1]]."""
    Nk = jnp.zeros(NBINS)
    Mk = jnp.zeros((NBINS, ICOMP))
    mp = jnp.sqrt(xk[k] * xk[k + 1])  # geometric mean
    Nk = Nk.at[k].set(n)
    Mk = Mk.at[k, SRTSO4].set(n * mp)
    return Nk, Mk


# =========================================================================
# Phase 1: Empty bin fix
# =========================================================================

class TestPhase1EmptyBins:
    """Bins with Nk < NEPS (1e-5) get fixed to tiny values."""

    def test_zero_bins_get_neps(self, xk):
        """Empty bins should get Nk = NEPS = 1e-5."""
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)

        # All bins should have Nk >= NEPS
        NEPS = 1e-5
        assert jnp.all(Nk_new >= NEPS - 1e-10)

    def test_empty_bins_get_so4_at_geometric_mean(self, xk):
        """Empty bins should get SO4 = NEPS * sqrt(xk[k]*xk[k+1])."""
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)

        NEPS = 1e-5
        for k in range(NBINS):
            expected_so4 = NEPS * float(jnp.sqrt(xk[k] * xk[k + 1]))
            assert float(Mk_new[k, SRTSO4]) == pytest.approx(
                expected_so4, rel=1e-6
            )

    def test_populated_bins_unchanged_by_phase1(self, xk):
        """Bins with Nk >> NEPS should not be affected by Phase 1."""
        Nk, Mk = make_in_range_bin(xk, 15, n=1e8)
        n_before = float(Nk[15])
        m_before = float(Mk[15, SRTSO4])

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)

        assert float(Nk_new[15]) == pytest.approx(n_before, rel=1e-6)
        assert float(Mk_new[15, SRTSO4]) == pytest.approx(m_before, rel=1e-6)


# =========================================================================
# Phase 2: Extreme out-of-range
# =========================================================================

class TestPhase2Extreme:
    """Bins with avg mass way above or below the grid get trimmed."""

    def test_avg_above_grid_trimmed(self, xk):
        """If avg > xk[NBINS], mass should be trimmed."""
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        k = NBINS - 1  # top bin
        Nk = Nk.at[k].set(1e8)
        # Set mass way above the top boundary
        Mk = Mk.at[k, SRTSO4].set(1e8 * xk[NBINS] * 100.0)

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)

        # After correction, avg mass should be within grid
        drymass = float(jnp.sum(Mk_new[k, :ICOMP_NODIAG]))
        avg = drymass / (float(Nk_new[k]) + 1e-40)
        assert avg <= float(xk[NBINS]) * 2.0  # some headroom

    def test_avg_below_grid_trimmed(self, xk):
        """If avg < xk[0], number should be trimmed."""
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        k = 0  # bottom bin
        Nk = Nk.at[k].set(1e15)  # way too many particles for the mass
        Mk = Mk.at[k, SRTSO4].set(1e15 * xk[0] * 0.01)

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)

        # After correction, avg mass should be >= xk[0]
        drymass = float(jnp.sum(Mk_new[k, :ICOMP_NODIAG]))
        avg = drymass / (float(Nk_new[k]) + 1e-40)
        assert avg >= float(xk[0]) * 0.5  # some tolerance


# =========================================================================
# Phase 3: Partial transfer
# =========================================================================

class TestPhase3PartialTransfer:
    """When avg mass drifts outside bin bounds, partial transfer to neighbor."""

    def test_upward_shift(self, xk):
        """Avg mass above xk[k+1] should shift some particles upward."""
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        n = 1e8
        # Set avg mass = 1.5 * xk[k+1] (above upper boundary)
        Nk = Nk.at[k].set(n)
        Mk = Mk.at[k, SRTSO4].set(n * 1.5 * xk[k + 1])

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)

        # Some particles should have moved to bin k+1 or higher
        assert float(Nk_new[k]) < float(Nk[k])
        assert float(Nk_new[k + 1]) > 0.0 or any(
            float(Nk_new[j]) > 1e-4 for j in range(k + 1, NBINS)
        )

    def test_downward_shift(self, xk):
        """Avg mass below xk[k] should shift some particles downward."""
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        n = 1e8
        # Set avg mass = 0.5 * xk[k] (below lower boundary)
        Nk = Nk.at[k].set(n)
        Mk = Mk.at[k, SRTSO4].set(n * 0.5 * xk[k])

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)

        # Some particles should have moved to bin k-1 or lower
        assert float(Nk_new[k]) < float(Nk[k])

    def test_partial_not_total(self, xk):
        """Transfer should be partial, not moving all particles out."""
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        n = 1e8
        # Set avg mass slightly above xk[k+1]
        Nk = Nk.at[k].set(n)
        Mk = Mk.at[k, SRTSO4].set(n * 1.1 * xk[k + 1])

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)

        # Source bin should still have particles (partial transfer)
        assert float(Nk_new[k]) > 0.1 * n

    def test_in_range_bin_unchanged(self, xk):
        """Bin with avg mass in [xk[k], xk[k+1]] should not be shifted."""
        k = 15
        Nk, Mk = make_in_range_bin(xk, k, n=1e8)

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)

        # Bin k should keep essentially all its particles
        assert float(Nk_new[k]) == pytest.approx(float(Nk[k]), rel=0.01)


# =========================================================================
# Conservation
# =========================================================================

class TestConservation:
    """MNFIX should conserve total number and total mass."""

    def test_number_conservation_upward(self, xk):
        """Total number conserved during upward shift."""
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        n = 1e8
        Nk = Nk.at[k].set(n)
        Mk = Mk.at[k, SRTSO4].set(n * 1.5 * xk[k + 1])

        Nk_new, _ = mnfix_jax(Nk, Mk, xk)

        # Total N should be preserved (minus empty-bin NEPS additions)
        NEPS = 1e-5
        n_total_before = float(jnp.sum(Nk)) + NEPS * (NBINS - 1)  # approx
        n_total_after = float(jnp.sum(Nk_new))
        assert n_total_after == pytest.approx(n_total_before, rel=0.01)

    def test_mass_conservation_upward(self, xk):
        """Total dry mass conserved during upward shift."""
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        n = 1e8
        mass = n * 1.5 * xk[k + 1]
        Nk = Nk.at[k].set(n)
        Mk = Mk.at[k, SRTSO4].set(mass)

        _, Mk_new = mnfix_jax(Nk, Mk, xk)

        m_total_before = float(mass)
        m_total_after = float(jnp.sum(Mk_new[:, SRTSO4]))
        # Allow some tolerance for empty-bin SO4 additions
        assert m_total_after == pytest.approx(m_total_before, rel=0.01)

    def test_multispecies_conservation(self, xk):
        """Conservation holds when multiple species are present."""
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        n = 1e8
        Nk = Nk.at[k].set(n)
        # Mix of SO4 and organic, total above bin boundary
        so4_mass = n * 0.8 * xk[k + 1]
        org_mass = n * 0.7 * xk[k + 1]
        Mk = Mk.at[k, SRTSO4].set(so4_mass)
        Mk = Mk.at[k, 1].set(org_mass)  # organic species 1

        _, Mk_new = mnfix_jax(Nk, Mk, xk)

        # Total mass of each species should be ~conserved
        so4_total = float(jnp.sum(Mk_new[:, SRTSO4]))
        org_total = float(jnp.sum(Mk_new[:, 1]))
        assert so4_total == pytest.approx(float(so4_mass), rel=0.02)
        assert org_total == pytest.approx(float(org_mass), rel=0.02)

    def test_composition_fractions_preserved(self, xk):
        """Species fractions should be preserved during transfer."""
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        n = 1e8
        Nk = Nk.at[k].set(n)
        # 60% SO4, 40% organic
        total_mass = n * 1.5 * xk[k + 1]
        Mk = Mk.at[k, SRTSO4].set(0.6 * total_mass)
        Mk = Mk.at[k, 1].set(0.4 * total_mass)

        _, Mk_new = mnfix_jax(Nk, Mk, xk)

        # Destination bin should have similar composition fractions
        for j in range(k, min(k + 3, NBINS)):
            total_j = float(jnp.sum(Mk_new[j, :ICOMP_NODIAG]))
            if total_j > 1e-20:
                so4_frac = float(Mk_new[j, SRTSO4]) / total_j
                org_frac = float(Mk_new[j, 1]) / total_j
                if so4_frac > 0.01 and org_frac > 0.01:
                    assert so4_frac == pytest.approx(0.6, abs=0.05)
                    assert org_frac == pytest.approx(0.4, abs=0.05)


# =========================================================================
# Multi-bin jumps
# =========================================================================

class TestMultiBinJump:
    """Test large mass shifts that skip multiple bins."""

    def test_large_upward_jump(self, xk):
        """Very heavy avg mass should jump multiple bins upward."""
        k = 5
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        n = 1e8
        # avg mass = xk[15] (10 bins above)
        Nk = Nk.at[k].set(n)
        Mk = Mk.at[k, SRTSO4].set(n * xk[15])

        Nk_new, _ = mnfix_jax(Nk, Mk, xk)

        # Particles should have moved well above bin 5
        # There should be significant particles in bins >> 5
        upper_total = float(jnp.sum(Nk_new[k + 3:]))
        assert upper_total > 0.1 * n

    def test_large_downward_jump(self, xk):
        """Very light avg mass should jump multiple bins downward."""
        k = 25
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        n = 1e8
        # avg mass = xk[15] (10 bins below)
        Nk = Nk.at[k].set(n)
        Mk = Mk.at[k, SRTSO4].set(n * xk[15])

        Nk_new, _ = mnfix_jax(Nk, Mk, xk)

        # Particles should have moved to lower bins
        lower_total = float(jnp.sum(Nk_new[:k - 2]))
        assert lower_total > 0.1 * n


# =========================================================================
# Positivity
# =========================================================================

class TestPositivity:
    """MNFIX should maintain physical positivity."""

    def test_nk_nonnegative(self, xk):
        """Nk should never go negative."""
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        Nk = Nk.at[k].set(1e8)
        Mk = Mk.at[k, SRTSO4].set(1e8 * 1.5 * xk[k + 1])

        Nk_new, _ = mnfix_jax(Nk, Mk, xk)
        assert jnp.all(Nk_new >= 0.0)

    def test_mk_nonnegative(self, xk):
        """Mk should never go negative."""
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        Nk = Nk.at[k].set(1e8)
        Mk = Mk.at[k, SRTSO4].set(1e8 * 1.5 * xk[k + 1])

        _, Mk_new = mnfix_jax(Nk, Mk, xk)
        assert jnp.all(Mk_new >= 0.0)

    def test_no_nan_output(self, xk):
        """Output should never contain NaN."""
        # Stress test with mixed in/out-of-range bins
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        for k in range(0, NBINS, 3):
            Nk = Nk.at[k].set(1e8)
            # Alternate: in-range, above, below
            if k % 9 == 0:
                Mk = Mk.at[k, SRTSO4].set(1e8 * jnp.sqrt(xk[k] * xk[k + 1]))
            elif k % 9 == 3:
                Mk = Mk.at[k, SRTSO4].set(1e8 * 2.0 * xk[k + 1])
            else:
                Mk = Mk.at[k, SRTSO4].set(1e8 * 0.3 * xk[k])

        Nk_new, Mk_new = mnfix_jax(Nk, Mk, xk)
        assert jnp.all(jnp.isfinite(Nk_new))
        assert jnp.all(jnp.isfinite(Mk_new))


# =========================================================================
# JIT compatibility
# =========================================================================

class TestMNFIXJIT:
    """Verify JIT compilation works."""

    def test_jit_compiles(self, xk):
        Nk, Mk = make_in_range_bin(xk, 15, n=1e8)
        mnfix_jit = jax.jit(lambda n, m: mnfix_jax(n, m, xk))
        Nk_new, Mk_new = mnfix_jit(Nk, Mk)
        assert jnp.all(jnp.isfinite(Nk_new))
        assert jnp.all(jnp.isfinite(Mk_new))

    def test_jit_matches_eager(self, xk):
        k = 15
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        Nk = Nk.at[k].set(1e8)
        Mk = Mk.at[k, SRTSO4].set(1e8 * 1.5 * xk[k + 1])

        Nk_eager, Mk_eager = mnfix_jax(Nk, Mk, xk)
        mnfix_jit = jax.jit(lambda n, m: mnfix_jax(n, m, xk))
        Nk_jit, Mk_jit = mnfix_jit(Nk, Mk)

        np.testing.assert_allclose(
            np.array(Nk_jit), np.array(Nk_eager), atol=1e-15
        )
        np.testing.assert_allclose(
            np.array(Mk_jit), np.array(Mk_eager), atol=1e-15
        )
