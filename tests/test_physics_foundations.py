"""Unit tests for foundational physics modules: density, properties, gas_properties.

These modules had zero test coverage. Tests verify against known reference
values and physical limits.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from tomas_jax.core.config import (
    NBINS, ICOMP, SRTSO4, SRTNH4, SRTH2O, SRTORG1, PI, KB,
)
from tomas_jax.physics.density import calc_density
from tomas_jax.physics.properties import (
    calc_air_properties, calc_particle_properties,
)
from tomas_jax.physics.gas_properties import (
    calc_gas_diffusivity, calc_mean_molecular_speed,
    calc_mean_free_path, calc_knudsen_number,
    calc_fuchs_sutugin_correction, calc_air_viscosity,
)


# =========================================================================
# density.py tests
# =========================================================================

class TestCalcDensity:
    """Test aerosol density calculation (Tang 1997 fits)."""

    def test_pure_sulfate(self):
        """Pure H2SO4 solution density should be > 1000 kg/m3."""
        Mk = jnp.zeros((1, ICOMP))
        # Need mass above empty-bin threshold (1e-15)
        Mk = Mk.at[0, SRTSO4].set(1e-12)
        rho = calc_density(Mk)
        # Pure sulfuric acid (100 wt%) density ~ 1830 kg/m3
        assert float(rho[0]) > 1500.0
        assert float(rho[0]) < 2500.0

    def test_with_water(self):
        """Adding water should decrease density toward 1000 kg/m3."""
        Mk_dry = jnp.zeros((1, ICOMP))
        Mk_dry = Mk_dry.at[0, SRTSO4].set(1e-12)
        rho_dry = calc_density(Mk_dry)

        # Add water equal to sulfate mass (~50 wt% solution)
        Mk_wet = Mk_dry.at[0, SRTH2O].set(1e-12)
        rho_wet = calc_density(Mk_wet)

        # Wet particle density closer to water (1000 kg/m3)
        assert float(rho_wet[0]) < float(rho_dry[0])

    def test_empty_bins(self):
        """Empty bins should return default density (~1400 kg/m3)."""
        Mk = jnp.zeros((5, ICOMP))
        rho = calc_density(Mk)
        # Should not NaN and should give reasonable default
        assert jnp.all(jnp.isfinite(rho))
        assert jnp.all(rho > 500.0)
        assert jnp.all(rho < 3000.0)

    def test_vectorized_multiple_bins(self):
        """Should handle multiple bins in parallel."""
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[:, SRTSO4].set(jnp.linspace(1e-20, 1e-15, NBINS))
        rho = calc_density(Mk)
        assert rho.shape == (NBINS,)
        assert jnp.all(jnp.isfinite(rho))

    def test_with_ammonium(self):
        """SO4 + NH4 should give (NH4)2SO4-like density."""
        Mk = jnp.zeros((1, ICOMP))
        Mk = Mk.at[0, SRTSO4].set(1e-12)
        # Molar ratio NH4:SO4 = 2:1 for (NH4)2SO4
        Mk = Mk.at[0, SRTNH4].set(1e-12 * (2 * 18.0 / 96.0))
        rho = calc_density(Mk)
        # (NH4)2SO4 density ~ 1770 kg/m3
        assert 1200.0 < float(rho[0]) < 2500.0


# =========================================================================
# properties.py tests
# =========================================================================

class TestCalcAirProperties:
    """Test air viscosity and mean free path calculations."""

    def test_stp_viscosity(self):
        """Air viscosity at 298K should be ~1.8e-5 Pa.s."""
        mu, mfp = calc_air_properties(298.0, 101325.0)
        assert 1.7e-5 < float(mu) < 1.9e-5

    def test_stp_mean_free_path(self):
        """Mean free path at STP should be ~65 nm."""
        mu, mfp = calc_air_properties(298.0, 101325.0)
        assert 50e-9 < float(mfp) < 80e-9  # 50-80 nm

    def test_viscosity_increases_with_temp(self):
        """Viscosity should increase with temperature."""
        mu_cold, _ = calc_air_properties(200.0, 101325.0)
        mu_hot, _ = calc_air_properties(350.0, 101325.0)
        assert float(mu_hot) > float(mu_cold)

    def test_mfp_increases_at_low_pressure(self):
        """MFP should increase when pressure decreases (fewer molecules)."""
        _, mfp_sealevel = calc_air_properties(298.0, 101325.0)
        _, mfp_tropopause = calc_air_properties(220.0, 20000.0)
        assert float(mfp_tropopause) > float(mfp_sealevel)


class TestCalcParticleProperties:
    """Test particle diameter, diffusivity, thermal speed."""

    def test_known_diameter(self):
        """100nm particle with known mass should give ~100nm Dpk."""
        # Pure sulfate particle, density ~1770 kg/m3
        # For Dp=100nm: V = pi/6 * (100e-9)^3 = 5.24e-22 m^3
        # m = V * rho = 5.24e-22 * 1770 = 9.27e-19 kg
        Nk = jnp.ones(1) * 1e6
        Mk = jnp.zeros((1, ICOMP))
        Mk = Mk.at[0, SRTSO4].set(1e6 * 9.27e-19)

        Dpk, Dk, ck = calc_particle_properties(Nk, Mk, 298.0, 101325.0)
        Dp_nm = float(Dpk[0]) * 1e9  # convert to nm
        assert 80.0 < Dp_nm < 120.0, f"Expected ~100nm, got {Dp_nm:.1f}nm"

    def test_diffusivity_reasonable(self):
        """Particle diffusivity for 100nm particles at STP."""
        Nk = jnp.ones(1) * 1e6
        Mk = jnp.zeros((1, ICOMP))
        Mk = Mk.at[0, SRTSO4].set(1e6 * 1e-18)

        Dpk, Dk, ck = calc_particle_properties(Nk, Mk, 298.0, 101325.0)
        # 100nm particles: Dk ~ 5e-10 to 1e-8 m2/s
        assert 1e-11 < float(Dk[0]) < 1e-7

    def test_thermal_speed(self):
        """Thermal speed should scale as sqrt(T/m)."""
        Nk = jnp.ones(1) * 1e6
        Mk = jnp.zeros((1, ICOMP))
        Mk = Mk.at[0, SRTSO4].set(1e6 * 1e-18)

        _, _, ck_cold = calc_particle_properties(Nk, Mk, 200.0, 101325.0)
        _, _, ck_hot = calc_particle_properties(Nk, Mk, 400.0, 101325.0)
        # ck ~ sqrt(T), so ck(400)/ck(200) ~ sqrt(2) ~ 1.41
        ratio = float(ck_hot[0]) / float(ck_cold[0])
        np.testing.assert_allclose(ratio, np.sqrt(2.0), rtol=0.01)

    def test_empty_bins_no_nan(self):
        """Empty bins should produce finite values (not NaN)."""
        Nk = jnp.zeros(NBINS)
        Mk = jnp.zeros((NBINS, ICOMP))
        Dpk, Dk, ck = calc_particle_properties(Nk, Mk, 298.0, 101325.0)
        assert jnp.all(jnp.isfinite(Dpk))
        assert jnp.all(jnp.isfinite(Dk))
        assert jnp.all(jnp.isfinite(ck))


# =========================================================================
# gas_properties.py tests
# =========================================================================

class TestGasDiffusivity:
    """Test Fuller-Schettler-Giddings gas diffusivity."""

    def test_h2so4_at_298k(self):
        """H2SO4 diffusivity at 298K, 1atm should be ~1.15e-5 m2/s."""
        Di = calc_gas_diffusivity(298.0, 101325.0)
        np.testing.assert_allclose(float(Di), 1.15e-5, rtol=0.05)

    def test_increases_with_temperature(self):
        """Diffusivity increases with T (more energetic collisions)."""
        Di_cold = calc_gas_diffusivity(200.0, 101325.0)
        Di_hot = calc_gas_diffusivity(350.0, 101325.0)
        assert float(Di_hot) > float(Di_cold)

    def test_decreases_with_pressure(self):
        """Diffusivity increases at lower pressure (fewer collisions)."""
        Di_high_p = calc_gas_diffusivity(298.0, 101325.0)
        Di_low_p = calc_gas_diffusivity(298.0, 50000.0)
        assert float(Di_low_p) > float(Di_high_p)


class TestMeanMolecularSpeed:
    """Test mean molecular speed."""

    def test_h2so4_at_298k(self):
        """H2SO4 mean molecular speed at 298K."""
        from tomas_jax.core.config import MW_H2SO4
        ms = calc_mean_molecular_speed(298.0, MW_H2SO4)
        # sqrt(8RT/piM) = sqrt(8*8.314*298/(pi*0.098)) ~ 251 m/s
        assert 200.0 < float(ms) < 300.0

    def test_lighter_molecule_faster(self):
        """Lighter molecules should move faster at same T."""
        ms_heavy = calc_mean_molecular_speed(298.0, 98.0)  # H2SO4
        ms_light = calc_mean_molecular_speed(298.0, 18.0)  # H2O
        assert float(ms_light) > float(ms_heavy)


class TestFuchsSutuginCorrection:
    """Test transition-regime correction factor."""

    def test_continuum_limit(self):
        """Kn → 0: beta → 1 (continuum regime)."""
        Kn = jnp.array([1e-5, 1e-3])
        beta = calc_fuchs_sutugin_correction(Kn)
        np.testing.assert_allclose(np.array(beta), 1.0, atol=0.01)

    def test_free_molecular_limit(self):
        """Kn → ∞: beta → alpha/(2*Kn) → small."""
        Kn = jnp.array([100.0, 1000.0])
        beta = calc_fuchs_sutugin_correction(Kn)
        # For large Kn: beta ≈ 1/(2*Kn) → very small
        assert jnp.all(beta < 0.01)
        assert jnp.all(beta > 0.0)

    def test_accommodation_coefficient(self):
        """Lower alpha → lower beta (less efficient uptake)."""
        Kn = jnp.array([1.0])
        beta_full = calc_fuchs_sutugin_correction(Kn, alpha=1.0)
        beta_half = calc_fuchs_sutugin_correction(Kn, alpha=0.5)
        assert float(beta_half[0]) < float(beta_full[0])

    def test_monotonic_in_kn(self):
        """Beta should decrease monotonically with Kn."""
        Kn = jnp.logspace(-2, 2, 50)
        beta = calc_fuchs_sutugin_correction(Kn)
        # Check monotonically decreasing
        diffs = np.diff(np.array(beta))
        assert np.all(diffs <= 0), "Beta should decrease with Kn"
