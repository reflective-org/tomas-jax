"""Tests for radiative forcing calculation.

Validates:
    1. Bohren-Huffman Mie scattering against known values
    2. Upscatter fraction physical bounds
    3. Chylek & Wong RF equation: sign, scaling, magnitude
    4. Mass scattering efficiency
    5. Scattering efficiency vs radius (Pierce et al. 2010 Fig 1 consistency)
"""
import numpy as np
import pytest

from tomas_jax.core.config import PI, make_grid, XK0, NBINS, ICOMP
from tomas_jax.physics.bhmie import bhmie
from tomas_jax.physics.radiative_forcing import (
    precompute_mie_properties,
    compute_optical_depth,
    compute_rf,
    compute_mass_scattering_efficiency,
    compute_rf_efficiency,
    upscatter_fraction,
    scattering_efficiency_vs_radius,
    h2so4_equilibrium_wt,
    h2so4_solution_density,
    MieProperties,
    DENSITY_SULFATE,
    WAVELENGTH_DEFAULT,
    REFINDEX_SULFATE,
)


# =========================================================================
# Mie scattering tests
# =========================================================================
class TestBhmie:
    """Tests for Bohren-Huffman Mie scattering."""

    def test_small_particle_rayleigh_limit(self):
        """For x << 1 (Rayleigh regime), Qsca ~ x^4."""
        x_small = 0.01
        refrel = complex(1.5, 0.0)
        _, _, qext, qsca, _, gsca = bhmie(x_small, refrel, 2)

        # In Rayleigh limit, Qsca should be very small (proportional to x^4)
        assert qsca > 0
        assert qsca < 1e-4
        # gsca should be near 0 for tiny particles (isotropic scattering)
        assert abs(gsca) < 0.1

    def test_large_particle_geometric_limit(self):
        """For x >> 1, Qext -> 2 (extinction paradox)."""
        x_large = 100.0
        refrel = complex(1.5, 0.0)
        _, _, qext, qsca, _, gsca = bhmie(x_large, refrel, 2)

        # Qext approaches 2 for large non-absorbing spheres
        assert abs(qext - 2.0) < 0.5
        # gsca should be forward-peaked (close to 1)
        assert gsca > 0.5

    def test_qext_geq_qsca(self):
        """Extinction efficiency >= scattering efficiency (always)."""
        for x in [0.1, 1.0, 5.0, 20.0]:
            _, _, qext, qsca, _, _ = bhmie(x, complex(1.4, 0.01), 2)
            assert qext >= qsca

    def test_non_absorbing_qext_equals_qsca(self):
        """For non-absorbing particles (Im(n)=0), Qext ~ Qsca."""
        x = 2.0
        refrel = complex(1.4, 0.0)
        _, _, qext, qsca, _, _ = bhmie(x, refrel, 2)
        # Should be very close for no absorption
        assert abs(qext - qsca) / qext < 0.01

    def test_known_value_sulfate(self):
        """Check Mie calculation for a typical sulfate aerosol particle.

        r=0.2μm at λ=0.55μm → x=2.28. Sulfate refindex 1.4+1e-8j.
        """
        r = 0.2e-6  # 200 nm
        wl = 0.55e-6
        x = 2.0 * PI * r / wl
        _, _, qext, qsca, _, gsca = bhmie(x, complex(1.4, 1e-8), 2)

        # For x~2.3, sulfate: Qsca should be ~1-3 range (resonance peak)
        assert 0.5 < qsca < 5.0
        # Asymmetry parameter should be moderate
        assert 0.0 < gsca < 0.9


# =========================================================================
# Upscatter fraction tests
# =========================================================================
class TestUpscatterFraction:
    """Tests for Wiscombe & Grams upscatter fraction."""

    def test_isotropic_scattering(self):
        """For g=0 (isotropic), upscatter fraction ~ 0.5."""
        beta = upscatter_fraction(0.0, np.radians(30.0))
        assert 0.4 < beta < 0.6

    def test_forward_scattering_low_upscatter(self):
        """For g close to 1 (forward), upscatter should be small."""
        beta = upscatter_fraction(0.8, np.radians(30.0))
        assert beta < 0.3

    def test_upscatter_bounds(self):
        """Upscatter fraction should be in [0, 1]."""
        for g in [0.0, 0.3, 0.6, 0.9]:
            for sza in [10., 30., 60., 80.]:
                beta = upscatter_fraction(g, np.radians(sza))
                assert 0.0 <= beta <= 1.0, f"beta={beta} for g={g}, sza={sza}"

    def test_high_zenith_angle_zero(self):
        """At SZA >= 90°, no sunlight → upscatter = 0."""
        beta = upscatter_fraction(0.5, np.radians(91.0))
        assert beta == 0.0


# =========================================================================
# Mie properties for TOMAS bins
# =========================================================================
class TestPrecomputeMie:
    """Tests for precompute_mie_properties."""

    def test_returns_correct_shape(self):
        """Output arrays match number of bins."""
        xk = make_grid(40, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)
        assert mie.radii.shape == (40,)
        assert mie.Qsca.shape == (40,)
        assert mie.gsca.shape == (40,)

    def test_radii_increasing(self):
        """Bin radii should increase monotonically."""
        xk = make_grid(40, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)
        assert np.all(np.diff(mie.radii) > 0)

    def test_radii_range(self):
        """First bin should be ~nm, last bin should be ~μm."""
        xk = make_grid(40, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)
        # First bin: ~1-2 nm radius
        assert 0.5e-9 < mie.radii[0] < 5e-9
        # Last bin: ~5-20 μm radius
        assert 1e-6 < mie.radii[-1] < 50e-6

    def test_qsca_positive(self):
        """All scattering efficiencies should be positive."""
        xk = make_grid(40, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)
        assert np.all(mie.Qsca > 0)

    def test_gsca_range(self):
        """Asymmetry parameter should be in [-1, 1]."""
        xk = make_grid(40, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)
        assert np.all(mie.gsca >= -1.0)
        assert np.all(mie.gsca <= 1.0)

    def test_with_upscatter(self):
        """Test that global average upscatter is computed when requested."""
        # Use 10 bins for speed
        xk = make_grid(10, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=True)
        assert not np.any(np.isnan(mie.upscatter_avg))
        assert np.all(mie.upscatter_avg >= 0.0)
        assert np.all(mie.upscatter_avg <= 1.0)


# =========================================================================
# Optical depth tests
# =========================================================================
class TestOpticalDepth:
    """Tests for optical depth calculation."""

    def test_linear_in_number(self):
        """Optical depth should scale linearly with particle number."""
        xk = make_grid(40, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)
        Nk = np.zeros(40)
        Nk[20] = 1e6
        tau1 = compute_optical_depth(Nk, mie, column_area=1.0)
        Nk[20] = 2e6
        tau2 = compute_optical_depth(Nk, mie, column_area=1.0)
        np.testing.assert_allclose(tau2, 2.0 * tau1)

    def test_inversely_proportional_to_area(self):
        """Optical depth ~ 1/column_area."""
        xk = make_grid(40, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)
        Nk = np.ones(40) * 1e6
        tau1 = compute_optical_depth(Nk, mie, column_area=1.0)
        tau2 = compute_optical_depth(Nk, mie, column_area=2.0)
        np.testing.assert_allclose(tau1, 2.0 * tau2)

    def test_zero_number_zero_tau(self):
        """No particles → no optical depth."""
        xk = make_grid(40, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)
        Nk = np.zeros(40)
        tau = compute_optical_depth(Nk, mie, column_area=1.0)
        np.testing.assert_allclose(tau, 0.0)


# =========================================================================
# Radiative forcing tests
# =========================================================================
class TestComputeRF:
    """Tests for the Chylek & Wong RF calculation."""

    @pytest.fixture
    def mie_10bin(self):
        """10-bin Mie properties with upscatter (faster than 40-bin)."""
        xk = make_grid(10, XK0, 2.0)
        return precompute_mie_properties(xk, global_avg_upscatter=True)

    def test_rf_negative_cooling(self, mie_10bin):
        """Scattering aerosol should produce negative RF (cooling)."""
        Nk = np.ones(10) * 1e8
        rf_total, rf_per_bin = compute_rf(Nk, mie_10bin, column_area=1e10)
        assert rf_total < 0.0, "Scattering aerosol RF should be negative (cooling)"

    def test_rf_linear_in_number(self, mie_10bin):
        """RF should scale linearly with particle number (thin-aerosol limit)."""
        Nk1 = np.ones(10) * 1e6
        Nk2 = np.ones(10) * 2e6
        rf1, _ = compute_rf(Nk1, mie_10bin, column_area=1e10)
        rf2, _ = compute_rf(Nk2, mie_10bin, column_area=1e10)
        np.testing.assert_allclose(rf2, 2.0 * rf1, rtol=1e-10)

    def test_rf_zero_for_empty(self, mie_10bin):
        """No particles → zero RF."""
        Nk = np.zeros(10)
        rf_total, _ = compute_rf(Nk, mie_10bin, column_area=1e10)
        assert rf_total == 0.0

    def test_rf_decreases_with_albedo(self, mie_10bin):
        """Higher albedo → less RF (aerosol over bright surface is less effective)."""
        Nk = np.ones(10) * 1e8
        rf_low_alb, _ = compute_rf(Nk, mie_10bin, column_area=1e10, albedo=0.1)
        rf_high_alb, _ = compute_rf(Nk, mie_10bin, column_area=1e10, albedo=0.6)
        # Both negative; low albedo should have larger magnitude
        assert abs(rf_low_alb) > abs(rf_high_alb)

    def test_rf_per_bin_sums_to_total(self, mie_10bin):
        """Per-bin contributions should sum to total."""
        Nk = np.random.default_rng(42).uniform(1e4, 1e8, size=10)
        rf_total, rf_per_bin = compute_rf(Nk, mie_10bin, column_area=1e10)
        np.testing.assert_allclose(np.sum(rf_per_bin), rf_total, rtol=1e-12)


# =========================================================================
# Mass scattering efficiency
# =========================================================================
class TestMassScatteringEfficiency:
    """Tests for MSE calculation."""

    def test_mse_positive(self):
        """MSE should be positive for any nonzero distribution."""
        xk = make_grid(10, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)
        Nk = np.ones(10) * 1e6
        Mk = np.ones((10, ICOMP)) * 1e-15
        mse, _ = compute_mass_scattering_efficiency(Nk, Mk, mie)
        assert mse > 0

    def test_smaller_particles_higher_mse(self):
        """Smaller particles should have higher MSE (more surface area per mass)."""
        xk = make_grid(10, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=False)

        # Distribution concentrated in small bins
        Nk_small = np.zeros(10)
        Nk_small[3] = 1e6
        Mk_small = np.zeros((10, ICOMP))
        Mk_small[3, 0] = 1e-15

        # Distribution concentrated in large bins
        Nk_large = np.zeros(10)
        Nk_large[8] = 1e6
        Mk_large = np.zeros((10, ICOMP))
        Mk_large[8, 0] = 1e-10

        mse_small, _ = compute_mass_scattering_efficiency(Nk_small, Mk_small, mie)
        mse_large, _ = compute_mass_scattering_efficiency(Nk_large, Mk_large, mie)
        # Small particles scatter more per unit mass (in Rayleigh-to-Mie transition)
        # This is valid when both are in the small-particle regime
        assert mse_small > mse_large


# =========================================================================
# Scattering efficiency vs radius (Pierce Fig 1 validation)
# =========================================================================
class TestScatteringEfficiencyVsRadius:
    """Test scattering efficiency curve shape (Pierce et al. 2010 Fig 1)."""

    def test_peak_in_submicron_range(self):
        """Peak scattering efficiency should be for r ~ 0.1-0.5 μm."""
        radii, rf_per_burden = scattering_efficiency_vs_radius(
            n_radii=50, r_min=10e-9, r_max=5e-6
        )
        # Find radius of peak cooling (most negative RF)
        peak_idx = np.argmin(rf_per_burden)
        peak_r = radii[peak_idx]
        # Pierce Fig 1 shows peak near 0.1-0.3 μm
        assert 50e-9 < peak_r < 1e-6, (
            f"Peak at r={peak_r*1e6:.3f} μm, expected 0.05-1.0 μm"
        )

    def test_all_cooling(self):
        """All values should be negative (cooling) for purely scattering aerosol."""
        radii, rf_per_burden = scattering_efficiency_vs_radius(
            n_radii=30, r_min=10e-9, r_max=5e-6
        )
        assert np.all(rf_per_burden <= 0.0)

    def test_large_particles_less_efficient(self):
        """Very large particles (r > 5μm) should be less efficient than peak."""
        radii, rf_per_burden = scattering_efficiency_vs_radius(
            n_radii=50, r_min=10e-9, r_max=10e-6
        )
        peak_rf = np.min(rf_per_burden)
        # Find RF at r = 5 μm
        idx_5um = np.argmin(np.abs(radii - 5e-6))
        rf_5um = rf_per_burden[idx_5um]
        assert abs(rf_5um) < abs(peak_rf)


# =========================================================================
# RF efficiency
# =========================================================================
class TestRFEfficiency:
    """Tests for RF per unit burden."""

    def test_rf_efficiency_negative(self):
        """RF efficiency should be negative (cooling per unit burden)."""
        xk = make_grid(10, XK0, 2.0)
        mie = precompute_mie_properties(xk, global_avg_upscatter=True)
        Nk = np.ones(10) * 1e6
        Mk = np.ones((10, ICOMP)) * 1e-15
        rf_eff = compute_rf_efficiency(Nk, Mk, mie)
        assert rf_eff < 0.0


class TestTabazadehComposition:
    """Tests for Tabazadeh et al. (1997) H₂SO₄/H₂O equilibrium."""

    def test_typical_stratospheric(self):
        """At T=220K, RH=5%: expect ~60-70 wt% H₂SO₄."""
        wt = h2so4_equilibrium_wt(220, 5)
        assert 55 < wt < 70

    def test_monotonic_in_rh(self):
        """Higher RH → more dilute solution (lower wt%)."""
        wt_5 = h2so4_equilibrium_wt(220, 5)
        wt_10 = h2so4_equilibrium_wt(220, 10)
        wt_50 = h2so4_equilibrium_wt(220, 50)
        assert wt_5 > wt_10 > wt_50

    def test_bounds_low_rh(self):
        """At RH=0, return maximum tabulated wt% (80%)."""
        wt = h2so4_equilibrium_wt(220, 0)
        assert wt == 80.0

    def test_bounds_high_rh(self):
        """At RH=100, return 0 wt% (pure water)."""
        wt = h2so4_equilibrium_wt(220, 100)
        assert wt == 0.0

    def test_valid_range(self):
        """Composition within 10-80% for reasonable stratospheric conditions."""
        for T in [190, 200, 210, 220, 230, 240, 250]:
            for rh in [1, 5, 10, 50]:
                wt = h2so4_equilibrium_wt(T, rh)
                assert 10 <= wt <= 80, f"T={T}, RH={rh}: wt={wt}"

    def test_density_increasing_with_wt(self):
        """Density increases with H₂SO₄ weight percent."""
        rho_30 = h2so4_solution_density(30)
        rho_60 = h2so4_solution_density(60)
        rho_80 = h2so4_solution_density(80)
        assert rho_30 < rho_60 < rho_80

    def test_density_pure_water(self):
        """Density at 0 wt% ≈ 998 kg/m³."""
        rho = h2so4_solution_density(0)
        assert abs(rho - 998) < 5

    def test_density_reasonable_range(self):
        """Solution density between water and pure H₂SO₄."""
        for wt in [10, 30, 50, 70, 90]:
            rho = h2so4_solution_density(wt)
            assert 990 < rho < 1850
