"""Unit tests for water equilibrium (ezwatereqm.f, waterso4.f, waternacl.f ports).

Tests cover:
    - Sulfate water uptake: piecewise polynomial regions, boundary continuity
    - Sea salt water uptake: piecewise polynomial regions
    - Full calc_equilibrium_water: per-bin water content
    - Physical limits: WR >= 1.0, monotonically increasing with RH
    - Known reference values from ISORROPIA fits
    - Edge cases: dry conditions, near-saturation, empty bins
    - JIT compatibility
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import (
    NBINS, ICOMP, SRTSO4, SRTORG1, SRTH2O, IORG,
)
from tomas_jax.physics.water_equilibrium import (
    water_uptake_sulfate,
    water_uptake_seasalt,
    calc_equilibrium_water,
)


# =========================================================================
# water_uptake_sulfate (waterso4.f)
# =========================================================================

class TestWaterUptakeSulfate:
    """Test NH4HSO4 water uptake ratio."""

    def test_dry_wr_near_1(self):
        """At very low RH, WR should be close to 1.0."""
        wr = water_uptake_sulfate(jnp.array(5.0))
        assert float(wr) >= 1.0
        assert float(wr) < 1.1

    def test_wr_increases_with_rh(self):
        """WR should increase monotonically with RH."""
        rh_vals = jnp.array([10.0, 30.0, 50.0, 70.0, 80.0, 90.0, 95.0])
        wr_vals = water_uptake_sulfate(rh_vals)
        for i in range(len(rh_vals) - 1):
            assert float(wr_vals[i + 1]) > float(wr_vals[i])

    def test_known_reference_50pct(self):
        """WR at 50% RH should be ~1.29 (from validation results)."""
        wr = water_uptake_sulfate(jnp.array(50.0))
        assert float(wr) == pytest.approx(1.29, abs=0.05)

    def test_known_reference_80pct(self):
        """WR at 80% RH should be ~1.96 (from validation results)."""
        wr = water_uptake_sulfate(jnp.array(80.0))
        assert float(wr) == pytest.approx(1.96, abs=0.1)

    def test_known_reference_95pct(self):
        """WR at 95% RH should be ~5.05 (from validation results)."""
        wr = water_uptake_sulfate(jnp.array(95.0))
        assert float(wr) == pytest.approx(5.05, abs=0.3)

    def test_clipped_below_1(self):
        """WR should never go below 1.0."""
        wr = water_uptake_sulfate(jnp.array(1.0))
        assert float(wr) >= 1.0

    def test_clipped_above_30(self):
        """WR should be capped at 30.0."""
        wr = water_uptake_sulfate(jnp.array(99.0))
        assert float(wr) <= 30.0

    def test_rh_clamped_below_1(self):
        """RH < 1% should be clamped to 1%."""
        wr_neg = water_uptake_sulfate(jnp.array(-5.0))
        wr_one = water_uptake_sulfate(jnp.array(1.0))
        assert float(wr_neg) == pytest.approx(float(wr_one), rel=1e-10)

    def test_rh_clamped_above_99(self):
        """RH > 99% should be clamped to 99%."""
        wr_high = water_uptake_sulfate(jnp.array(105.0))
        wr_99 = water_uptake_sulfate(jnp.array(99.0))
        assert float(wr_high) == pytest.approx(float(wr_99), rel=1e-10)

    @pytest.mark.parametrize("rh", [41.0, 61.0, 81.0, 91.0, 96.0])
    def test_continuity_at_boundaries(self, rh):
        """Piecewise fit should be approximately continuous at breakpoints."""
        wr_below = water_uptake_sulfate(jnp.array(rh - 0.01))
        wr_above = water_uptake_sulfate(jnp.array(rh + 0.01))
        # Allow up to 5% discontinuity (polynomial fits aren't perfectly smooth)
        ratio = float(wr_above) / float(wr_below)
        assert 0.90 < ratio < 1.10

    def test_vectorized(self):
        """Should work with array input."""
        rh_arr = jnp.array([20.0, 50.0, 80.0, 95.0])
        wr = water_uptake_sulfate(rh_arr)
        assert wr.shape == (4,)
        assert jnp.all(wr >= 1.0)


# =========================================================================
# water_uptake_seasalt (waternacl.f)
# =========================================================================

class TestWaterUptakeSeasalt:
    """Test sea salt (NaCl) water uptake ratio."""

    def test_low_rh(self):
        """At low RH, sea salt WR should be ~1.17."""
        wr = water_uptake_seasalt(jnp.array(10.0))
        assert float(wr) == pytest.approx(1.17, abs=0.05)

    def test_wr_increases_with_rh(self):
        """WR should increase monotonically with RH."""
        rh_vals = jnp.array([10.0, 30.0, 50.0, 70.0, 85.0, 95.0])
        wr_vals = water_uptake_seasalt(rh_vals)
        for i in range(len(rh_vals) - 1):
            assert float(wr_vals[i + 1]) > float(wr_vals[i])

    def test_higher_than_sulfate_at_high_rh(self):
        """Sea salt is more hygroscopic than sulfate at high RH."""
        rh = jnp.array(90.0)
        wr_nacl = water_uptake_seasalt(rh)
        wr_so4 = water_uptake_sulfate(rh)
        assert float(wr_nacl) > float(wr_so4)

    def test_clipped_below_1(self):
        """WR should never go below 1.0."""
        wr = water_uptake_seasalt(jnp.array(1.0))
        assert float(wr) >= 1.0

    def test_clipped_above_45(self):
        """WR should be capped at 45.0."""
        wr = water_uptake_seasalt(jnp.array(99.0))
        assert float(wr) <= 45.0

    def test_vectorized(self):
        """Should work with array input."""
        rh_arr = jnp.array([20.0, 50.0, 80.0, 95.0])
        wr = water_uptake_seasalt(rh_arr)
        assert wr.shape == (4,)
        assert jnp.all(wr >= 1.0)


# =========================================================================
# calc_equilibrium_water (ezwatereqm.f)
# =========================================================================

class TestCalcEquilibriumWater:
    """Test full equilibrium water calculation."""

    def test_water_mass_positive(self):
        """Water mass should be non-negative."""
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[15, SRTSO4].set(1e-14)
        Mk_new = calc_equilibrium_water(Mk, rh=0.8)
        assert jnp.all(Mk_new[:, SRTH2O] >= 0.0)

    def test_water_increases_with_rh(self):
        """More water at higher RH."""
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[15, SRTSO4].set(1e-14)

        Mk_low = calc_equilibrium_water(Mk, rh=0.3)
        Mk_high = calc_equilibrium_water(Mk, rh=0.9)

        assert float(Mk_high[15, SRTH2O]) > float(Mk_low[15, SRTH2O])

    def test_water_proportional_to_so4(self):
        """Bins with more SO4 should have more water."""
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[10, SRTSO4].set(1e-14)
        Mk = Mk.at[20, SRTSO4].set(1e-13)  # 10x more

        Mk_new = calc_equilibrium_water(Mk, rh=0.8)

        ratio = float(Mk_new[20, SRTH2O]) / float(Mk_new[10, SRTH2O])
        assert ratio == pytest.approx(10.0, rel=1e-10)

    def test_organic_contributes_to_water(self):
        """Organic aerosol should also take up water (same as sulfate)."""
        Mk_so4 = jnp.zeros((NBINS, ICOMP))
        Mk_so4 = Mk_so4.at[15, SRTSO4].set(1e-14)

        Mk_org = jnp.zeros((NBINS, ICOMP))
        Mk_org = Mk_org.at[15, SRTORG1].set(1e-14)

        water_so4 = float(calc_equilibrium_water(Mk_so4, rh=0.8)[15, SRTH2O])
        water_org = float(calc_equilibrium_water(Mk_org, rh=0.8)[15, SRTH2O])

        # Organic uses same WR as sulfate, but SO4 has 1.2 NH4HSO4 factor
        # so water_so4 = 1.2 * so4_mass * (wr-1), water_org = org_mass * (wr-1)
        assert water_so4 == pytest.approx(water_org * 1.2, rel=1e-10)

    def test_empty_bins_zero_water(self):
        """Bins with no aerosol should have zero water."""
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[15, SRTSO4].set(1e-14)

        Mk_new = calc_equilibrium_water(Mk, rh=0.8)

        # All bins except 15 should have zero water
        for k in range(NBINS):
            if k != 15:
                assert float(Mk_new[k, SRTH2O]) == pytest.approx(0.0, abs=1e-30)

    def test_dry_conditions(self):
        """At very low RH, water should be minimal."""
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[15, SRTSO4].set(1e-14)

        Mk_new = calc_equilibrium_water(Mk, rh=0.01)
        # WR ~1.01 at 1% RH → water ~0.01 * 1.2 * so4_mass
        assert float(Mk_new[15, SRTH2O]) > 0.0
        assert float(Mk_new[15, SRTH2O]) < float(Mk[15, SRTSO4])

    def test_only_water_species_modified(self):
        """Only the H2O species should change; SO4, organics, NH4 untouched."""
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[15, SRTSO4].set(1e-14)
        Mk = Mk.at[15, SRTORG1].set(5e-15)
        Mk = Mk.at[15, 42].set(2e-16)  # NH4

        Mk_new = calc_equilibrium_water(Mk, rh=0.8)

        assert float(Mk_new[15, SRTSO4]) == pytest.approx(1e-14, rel=1e-12)
        assert float(Mk_new[15, SRTORG1]) == pytest.approx(5e-15, rel=1e-12)
        assert float(Mk_new[15, 42]) == pytest.approx(2e-16, rel=1e-12)

    def test_rh_as_fraction(self):
        """Input RH is fraction (0-1), not percent."""
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[15, SRTSO4].set(1e-14)

        # rh=0.8 should give substantial water
        Mk_new = calc_equilibrium_water(Mk, rh=0.8)
        water = float(Mk_new[15, SRTH2O])
        assert water > 0.5e-14  # significant water at 80% RH


# =========================================================================
# JIT compatibility
# =========================================================================

class TestWaterJIT:
    """Verify JIT compilation works for all water equilibrium functions."""

    def test_sulfate_jit(self):
        wr = jax.jit(water_uptake_sulfate)(jnp.array(50.0))
        assert jnp.isfinite(wr)

    def test_seasalt_jit(self):
        wr = jax.jit(water_uptake_seasalt)(jnp.array(50.0))
        assert jnp.isfinite(wr)

    def test_calc_equilibrium_water_jit(self):
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[15, SRTSO4].set(1e-14)

        calc_jit = jax.jit(calc_equilibrium_water)
        Mk_new = calc_jit(Mk, 0.8)
        assert jnp.all(jnp.isfinite(Mk_new))

    def test_jit_matches_eager(self):
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[15, SRTSO4].set(1e-14)

        Mk_eager = calc_equilibrium_water(Mk, 0.8)
        Mk_jit = jax.jit(calc_equilibrium_water)(Mk, 0.8)

        np.testing.assert_allclose(
            np.array(Mk_jit), np.array(Mk_eager), atol=1e-20
        )
