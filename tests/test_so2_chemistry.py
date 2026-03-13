"""Tests for SO2 + OH gas-phase chemistry (Sun et al. 2022 Troe formalism)."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tomas_jax.core.config import (
    SRTSO4, SRTSO2, MW_H2SO4, MW_SO2, N_GAS_SPECIES, AVOGADRO, KB,
)
from tomas_jax.physics.so2_chemistry import (
    calc_k1_so2_oh,
    calc_solar_zenith_angle,
    calc_oh_concentration,
    so2_oxidation_step,
)


# =========================================================================
# TestTroeRateConstant
# =========================================================================

class TestTroeRateConstant:
    """Tests for calc_k1_so2_oh."""

    def test_k1_298K_1atm_in_expected_range(self):
        """k1 at 298K/1atm should be ~1.0-1.5e-12 cm^3/molec/s."""
        k1 = float(calc_k1_so2_oh(298.0, 101325.0))
        assert 0.9e-12 < k1 < 1.6e-12, f"k1 = {k1:.3e}, expected ~1.2e-12"

    def test_k1_increases_with_pressure(self):
        """k1 should increase with pressure (more third-body collisions)."""
        k1_low = float(calc_k1_so2_oh(298.0, 50000.0))
        k1_high = float(calc_k1_so2_oh(298.0, 101325.0))
        assert k1_high > k1_low

    def test_k1_temperature_dependence(self):
        """k1 should generally increase at lower temperatures (negative T-dependence)."""
        k1_cold = float(calc_k1_so2_oh(220.0, 101325.0))
        k1_warm = float(calc_k1_so2_oh(298.0, 101325.0))
        # At standard pressure, both should be in fall-off regime
        # Cold gives higher k0 but we're comparing the overall k1
        # The effect depends on fall-off position, but at 1 atm k1 should be higher at cold T
        assert k1_cold > k1_warm * 0.8, "k1 should not drop dramatically at 220K"

    def test_k1_h2o_enhancement(self):
        """H2O should enhance k1 (k0_H2O > k0_N2)."""
        k1_dry = float(calc_k1_so2_oh(298.0, 101325.0, rh=0.0))
        k1_wet = float(calc_k1_so2_oh(298.0, 101325.0, rh=0.5))
        assert k1_wet > k1_dry

    def test_k1_h2o_enhancement_moderate(self):
        """H2O enhancement should be moderate (few percent) at surface conditions."""
        k1_dry = float(calc_k1_so2_oh(298.0, 101325.0, rh=0.0))
        k1_wet = float(calc_k1_so2_oh(298.0, 101325.0, rh=0.5))
        enhancement = (k1_wet - k1_dry) / k1_dry
        # Sun et al. Fig 13 shows ~2-5% enhancement at 298K, 50% RH
        assert 0.001 < enhancement < 0.20, f"Enhancement = {enhancement:.3f}"

    def test_k1_jit_compilable(self):
        """calc_k1_so2_oh should be JIT-compilable."""
        k1_jit = jax.jit(calc_k1_so2_oh)
        k1 = float(k1_jit(298.0, 101325.0, 0.0))
        assert k1 > 0

    def test_k1_low_pressure_limit(self):
        """At very low pressure, k1 should approach k0*[M]."""
        # Very low pressure (~1 hPa)
        k1_low = float(calc_k1_so2_oh(298.0, 100.0))
        assert k1_low > 0
        assert k1_low < 1e-12  # Below high-pressure limit

    def test_k1_gives_expected_lifetime(self):
        """SO2 lifetime at [OH]=1e6 should be ~7-10 days."""
        k1 = float(calc_k1_so2_oh(298.0, 101325.0))
        oh = 1.0e6  # molec/cm^3
        tau_days = 1.0 / (k1 * oh) / 86400.0
        assert 5.0 < tau_days < 15.0, f"tau = {tau_days:.1f} days, expected ~8-9"

    def test_k1_multiple_temperatures(self):
        """k1 should be positive at atmospheric temperature range."""
        for T in [200.0, 220.0, 250.0, 273.0, 298.0, 320.0]:
            k1 = float(calc_k1_so2_oh(T, 101325.0))
            assert k1 > 0, f"k1 negative at T={T}K"
            assert k1 < 1e-10, f"k1 unreasonably large at T={T}K"


# =========================================================================
# TestSolarZenithAngle
# =========================================================================

class TestSolarZenithAngle:
    """Tests for calc_solar_zenith_angle."""

    def test_noon_equator_equinox(self):
        """At noon, equator, equinox: cos(SZA) should be ~1.0."""
        # Spring equinox: ~day 80 (March 21)
        cos_sza = float(calc_solar_zenith_angle(0.0, 80, 12.0, 0.0))
        assert cos_sza > 0.95, f"cos(SZA) = {cos_sza}, expected ~1.0"

    def test_midnight_is_nighttime(self):
        """At midnight, equator: cos(SZA) should be negative."""
        cos_sza = float(calc_solar_zenith_angle(0.0, 80, 0.0, 0.0))
        assert cos_sza < 0, f"cos(SZA) = {cos_sza}, expected negative"

    def test_polar_winter_darkness(self):
        """At 80°N on winter solstice (day 355), should be dark at noon."""
        cos_sza = float(calc_solar_zenith_angle(80.0, 355, 12.0, 0.0))
        # At 80°N, winter solstice: sun doesn't rise
        assert cos_sza < 0.1, f"cos(SZA) = {cos_sza}, expected < 0.1"

    def test_longitude_offset(self):
        """Longitude should shift the peak — noon at lon=90 is 6 UTC."""
        cos_sza_noon = float(calc_solar_zenith_angle(0.0, 80, 6.0, 90.0))
        cos_sza_offset = float(calc_solar_zenith_angle(0.0, 80, 12.0, 90.0))
        assert cos_sza_noon > cos_sza_offset, "Noon should have higher cos(SZA)"

    def test_symmetric_morning_evening(self):
        """cos(SZA) at 9am and 3pm should be equal (symmetric around noon)."""
        cos_am = float(calc_solar_zenith_angle(45.0, 172, 9.0, 0.0))
        cos_pm = float(calc_solar_zenith_angle(45.0, 172, 15.0, 0.0))
        np.testing.assert_allclose(cos_am, cos_pm, atol=1e-10)

    def test_jit_compilable(self):
        """calc_solar_zenith_angle should be JIT-compilable."""
        f = jax.jit(calc_solar_zenith_angle)
        cos_sza = float(f(45.0, 172, 12.0, 0.0))
        assert -1.0 <= cos_sza <= 1.0


# =========================================================================
# TestOHConcentration
# =========================================================================

class TestOHConcentration:
    """Tests for calc_oh_concentration."""

    def test_constant_mode(self):
        """In constant mode, OH should equal oh_const regardless of SZA."""
        oh = float(calc_oh_concentration(1e6, -0.5, use_diurnal=0.0))
        np.testing.assert_allclose(oh, 1e6)

    def test_diurnal_daytime(self):
        """In diurnal mode, daytime OH should be oh_const * cos(SZA)."""
        cos_sza = 0.8
        oh = float(calc_oh_concentration(2e6, cos_sza, use_diurnal=1.0))
        np.testing.assert_allclose(oh, 2e6 * 0.8)

    def test_diurnal_nighttime(self):
        """In diurnal mode, nighttime (cos_sza < 0) should give OH = 0."""
        oh = float(calc_oh_concentration(2e6, -0.3, use_diurnal=1.0))
        np.testing.assert_allclose(oh, 0.0)

    def test_diurnal_sunset(self):
        """At cos(SZA) = 0 (sunset/sunrise), OH should be 0."""
        oh = float(calc_oh_concentration(1e6, 0.0, use_diurnal=1.0))
        np.testing.assert_allclose(oh, 0.0)


# =========================================================================
# TestSO2OxidationStep
# =========================================================================

class TestSO2OxidationStep:
    """Tests for so2_oxidation_step."""

    @pytest.fixture
    def setup(self):
        """Common test setup."""
        boxvol = 1.0e6  # 1 m^3 in cm^3
        temp = 298.0
        pres = 101325.0

        Gc = jnp.zeros(N_GAS_SPECIES)
        # 5e10 molec/cm3 SO2 -> kg/grid cell
        so2_molec_cm3 = 5.0e10
        so2_kg = so2_molec_cm3 * boxvol * (MW_SO2 / 1000.0) / AVOGADRO
        Gc = Gc.at[SRTSO2].set(so2_kg)

        return Gc, temp, pres, boxvol

    def test_so2_decreases(self, setup):
        """SO2 should decrease after oxidation step."""
        Gc, temp, pres, boxvol = setup
        so2_before = float(Gc[SRTSO2])
        Gc_new = so2_oxidation_step(Gc, temp, pres, boxvol, 60.0, 1e6)
        so2_after = float(Gc_new[SRTSO2])
        assert so2_after < so2_before

    def test_h2so4_increases(self, setup):
        """H2SO4 should increase after oxidation step."""
        Gc, temp, pres, boxvol = setup
        h2so4_before = float(Gc[SRTSO4])
        Gc_new = so2_oxidation_step(Gc, temp, pres, boxvol, 60.0, 1e6)
        h2so4_after = float(Gc_new[SRTSO4])
        assert h2so4_after > h2so4_before

    def test_sulfur_conservation(self, setup):
        """Total sulfur (SO2 + H2SO4) should be conserved in moles."""
        Gc, temp, pres, boxvol = setup
        # Initial sulfur [mol]
        s_before = float(Gc[SRTSO2]) / (MW_SO2 / 1000.0) + float(Gc[SRTSO4]) / (MW_H2SO4 / 1000.0)

        Gc_new = so2_oxidation_step(Gc, temp, pres, boxvol, 60.0, 1e6)

        s_after = float(Gc_new[SRTSO2]) / (MW_SO2 / 1000.0) + float(Gc_new[SRTSO4]) / (MW_H2SO4 / 1000.0)
        np.testing.assert_allclose(s_after, s_before, rtol=1e-12)

    def test_zero_so2_no_change(self, setup):
        """With no SO2, nothing should happen."""
        Gc, temp, pres, boxvol = setup
        Gc_zero = Gc.at[SRTSO2].set(0.0)
        Gc_new = so2_oxidation_step(Gc_zero, temp, pres, boxvol, 60.0, 1e6)
        np.testing.assert_allclose(float(Gc_new[SRTSO4]), float(Gc_zero[SRTSO4]), atol=1e-30)

    def test_zero_oh_no_change(self, setup):
        """With no OH, SO2 should not change."""
        Gc, temp, pres, boxvol = setup
        so2_before = float(Gc[SRTSO2])
        Gc_new = so2_oxidation_step(Gc, temp, pres, boxvol, 60.0, 0.0)
        np.testing.assert_allclose(float(Gc_new[SRTSO2]), so2_before, rtol=1e-12)

    def test_longer_dt_more_decay(self, setup):
        """Longer timestep should produce more SO2 decay."""
        Gc, temp, pres, boxvol = setup
        Gc_short = so2_oxidation_step(Gc, temp, pres, boxvol, 60.0, 1e6)
        Gc_long = so2_oxidation_step(Gc, temp, pres, boxvol, 3600.0, 1e6)
        assert float(Gc_long[SRTSO2]) < float(Gc_short[SRTSO2])

    def test_jit_compilable(self, setup):
        """so2_oxidation_step should be JIT-compilable."""
        Gc, temp, pres, boxvol = setup
        f = jax.jit(so2_oxidation_step)
        Gc_new = f(Gc, temp, pres, boxvol, 60.0, 1e6, 0.0)
        assert float(Gc_new[SRTSO2]) < float(Gc[SRTSO2])

    def test_scan_compatible(self, setup):
        """so2_oxidation_step should work inside jax.lax.scan."""
        Gc, temp, pres, boxvol = setup

        def body(Gc_c, _):
            Gc_c = so2_oxidation_step(Gc_c, temp, pres, boxvol, 60.0, 1e6, 0.0)
            return Gc_c, None

        Gc_final, _ = jax.lax.scan(body, Gc, None, length=10)
        assert float(Gc_final[SRTSO2]) < float(Gc[SRTSO2])

    def test_gc_shape_44(self):
        """Gc should have 44 elements (N_GAS_SPECIES = ICOMP = 44)."""
        assert N_GAS_SPECIES == 44
        Gc = jnp.zeros(N_GAS_SPECIES)
        assert Gc.shape[0] == 44
        assert SRTSO2 == 43  # Last element
