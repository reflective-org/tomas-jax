"""Tests for VBS/SOA condensation modules.

Tests:
    - VBS config: C* temperature correction, unit conversions
    - Kelvin effect: magnitude, monotonicity, large-particle limit
    - Driving force: positive/negative/zero, equilibrium mass
    - SOA condensation: mass conservation, pure condensation, evaporation,
      equilibrium correction, integration with make_step
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, SRTSO4, SRTORG1, SRTORGLAST, SRTH2O,
    N_GAS_SPECIES, R_GAS, KB, PI, AVOGADRO, make_grid,
)
from tomas_jax.core.state import TomasState
from tomas_jax.physics.vbs_config import (
    DEFAULT_VBS_CONFIG, VBSConfig, N_VBS_BINS,
    CSTAR_REF_UG, DELTA_HVAP_KJ, MW_ORG, T_REF,
    calc_Cstar_T, cstar_ug_to_Pa, cstar_ug_to_molec_cm3,
)
from tomas_jax.physics.kelvin_effect import (
    calc_kelvin_factor, calc_diameters_from_xk,
)
from tomas_jax.physics.vbs_driving_force import (
    calc_boxmass, calc_ambient_pressure, calc_equilibrium_pressure,
    calc_driving_force, calc_equilibrium_mass,
)
from tomas_jax.physics.soa_condensation import (
    soa_condensation_step, _condense_one_species,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_test_state(nbins=NBINS, n_total=1e10, gmd_um=0.1, gsd=1.5,
                     temp=298.0, pres=101325.0, rh=0.5):
    """Create a lognormal test distribution."""
    xk = make_grid(nbins)
    xmid = jnp.sqrt(xk[:-1] * xk[1:])
    rho = 1400.0
    dp_mid = jnp.cbrt(xmid / rho * 6.0 / PI)
    gmd_m = gmd_um * 1e-6
    sigma_g = jnp.log(gsd)
    ln_dp = jnp.log(dp_mid)
    ln_gmd = jnp.log(gmd_m)
    Nk = n_total * jnp.exp(-0.5 * ((ln_dp - ln_gmd) / sigma_g) ** 2)
    Nk = Nk / jnp.sum(Nk) * n_total
    Mk = jnp.zeros((nbins, ICOMP))
    Mk = Mk.at[:, SRTSO4].set(Nk * xmid * 0.5)
    for j in range(N_VBS_BINS):
        Mk = Mk.at[:, SRTORG1 + j].set(Nk * xmid * 0.05)
    boxvol = 1e6  # 1 cm³
    Gc = jnp.zeros(N_GAS_SPECIES)
    return Nk, Mk, Gc, xk, temp, pres, boxvol, rh


# =========================================================================
# VBS Config Tests
# =========================================================================

class TestVBSConfig:
    def test_default_config_shape(self):
        cfg = DEFAULT_VBS_CONFIG
        assert cfg.n_bins == 6
        assert cfg.cstar_ref_ug.shape == (6,)
        assert cfg.delta_Hvap_kJ.shape == (6,)
        assert cfg.mw.shape == (6,)
        assert len(cfg.species_indices) == 6

    def test_species_indices(self):
        cfg = DEFAULT_VBS_CONFIG
        for j in range(6):
            assert cfg.species_indices[j] == SRTORG1 + j

    def test_cstar_T_at_reference(self):
        """C* at T_ref should equal C*_ref."""
        cstar_T = calc_Cstar_T(CSTAR_REF_UG, DELTA_HVAP_KJ, T_REF)
        np.testing.assert_allclose(cstar_T, CSTAR_REF_UG, rtol=1e-12)

    def test_cstar_T_lower_temp(self):
        """Lower T → lower C* (more condensation)."""
        cstar_298 = calc_Cstar_T(CSTAR_REF_UG, DELTA_HVAP_KJ, 298.0)
        cstar_273 = calc_Cstar_T(CSTAR_REF_UG, DELTA_HVAP_KJ, 273.0)
        assert jnp.all(cstar_273 < cstar_298)

    def test_cstar_T_higher_temp(self):
        """Higher T → higher C* (more evaporation)."""
        cstar_298 = calc_Cstar_T(CSTAR_REF_UG, DELTA_HVAP_KJ, 298.0)
        cstar_320 = calc_Cstar_T(CSTAR_REF_UG, DELTA_HVAP_KJ, 320.0)
        assert jnp.all(cstar_320 > cstar_298)

    def test_cstar_Pa_conversion(self):
        """Round-trip: ug/m3 -> Pa -> back."""
        cstar_ug = jnp.array(1.0)
        mw = 200.0
        temp = 298.0
        cstar_Pa = cstar_ug_to_Pa(cstar_ug, mw, temp)
        # C* = 1 µg/m³ of MW=200 at 298K → very small Pa
        assert cstar_Pa > 0
        # Back-convert: Pa → µg/m³
        cstar_kg_m3 = cstar_Pa * (mw * 1e-3) / (R_GAS * temp)
        cstar_ug_back = cstar_kg_m3 * 1e9
        np.testing.assert_allclose(float(cstar_ug_back), 1.0, rtol=1e-10)

    def test_cstar_molec_cm3_conversion(self):
        cstar_ug = jnp.array(1.0)
        mw = 200.0
        c_molec = cstar_ug_to_molec_cm3(cstar_ug, mw)
        # 1 µg/m³ = 1e-12 g/cm³ → (1e-12/200) * 6.022e23 ≈ 3.01e9 molec/cm³
        np.testing.assert_allclose(float(c_molec), 3.011e9, rtol=0.01)


# =========================================================================
# Kelvin Effect Tests
# =========================================================================

class TestKelvinEffect:
    def test_kelvin_large_particles(self):
        """Kelvin factor → 1 for large particles."""
        Dpk = jnp.array([1e-6, 1e-5, 1e-4])  # 1, 10, 100 µm
        Ke = calc_kelvin_factor(Dpk, 0.025, 200.0, 1200.0, 298.0)
        np.testing.assert_allclose(Ke, 1.0, atol=0.01)

    def test_kelvin_small_particles(self):
        """Kelvin factor > 1 for small particles."""
        Dpk = jnp.array([3e-9, 5e-9, 10e-9])  # 3, 5, 10 nm
        Ke = calc_kelvin_factor(Dpk, 0.025, 200.0, 1200.0, 298.0)
        assert jnp.all(Ke > 1.0)
        # 3nm should have largest Kelvin factor
        assert float(Ke[0]) > float(Ke[1]) > float(Ke[2])

    def test_kelvin_monotonic_with_Dp(self):
        """Kelvin factor decreases monotonically with diameter."""
        Dpk = jnp.logspace(-9, -5, 20)
        Ke = calc_kelvin_factor(Dpk, 0.025, 200.0, 1200.0, 298.0)
        assert jnp.all(jnp.diff(Ke) <= 0)

    def test_kelvin_always_geq_one(self):
        """Kelvin factor is always >= 1."""
        Dpk = jnp.logspace(-10, -4, 50)
        Ke = calc_kelvin_factor(Dpk, 0.025, 200.0, 1200.0, 298.0)
        assert jnp.all(Ke >= 1.0)

    def test_calc_diameters(self):
        """Diameters should be physical (nm to µm range)."""
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        Dpk = calc_diameters_from_xk(xk, Nk, Mk, ICOMP_NODIAG)
        assert Dpk.shape == (NBINS,)
        assert jnp.all(Dpk > 0)
        assert jnp.all(Dpk < 1e-3)  # < 1 mm


# =========================================================================
# Driving Force Tests
# =========================================================================

class TestDrivingForce:
    def test_boxmass(self):
        """Box mass for 1 cm³ at STP should be ~1.2e-6 kg."""
        bm = calc_boxmass(298.0, 101325.0, 1.0)  # 1 cm³
        np.testing.assert_allclose(float(bm), 1.18e-6, rtol=0.05)

    def test_ambient_pressure_positive(self):
        """Non-zero gas gives positive pamb."""
        Gc_species = jnp.array(1e-15)  # 1e-15 kg
        pamb = calc_ambient_pressure(Gc_species, 200.0, 298.0, 101325.0, 1e6)
        assert float(pamb) > 0

    def test_ambient_pressure_zero_gas(self):
        """Zero gas gives zero pamb."""
        pamb = calc_ambient_pressure(jnp.array(0.0), 200.0, 298.0, 101325.0, 1e6)
        assert float(pamb) == 0.0

    def test_equilibrium_pressure_empty_bins(self):
        """Empty bins (Mtot=0) give psat=0 (pure condensation)."""
        nbins = 5
        psat = calc_equilibrium_pressure(
            jnp.array(0.1),
            jnp.zeros(nbins),
            jnp.zeros(nbins),
            jnp.ones(nbins),
        )
        np.testing.assert_allclose(psat, 0.0, atol=1e-30)

    def test_driving_force_positive(self):
        """pamb > psat → condensation (dp > 0)."""
        dp = calc_driving_force(jnp.array(1.0), jnp.array([0.5, 0.3, 0.1]))
        assert jnp.all(dp > 0)

    def test_driving_force_negative(self):
        """pamb < psat → evaporation (dp < 0)."""
        dp = calc_driving_force(jnp.array(0.1), jnp.array([0.5, 0.3, 0.2]))
        assert jnp.all(dp < 0)

    def test_driving_force_zero(self):
        """pamb = psat → equilibrium (dp = 0)."""
        dp = calc_driving_force(jnp.array(0.5), jnp.array([0.5, 0.5]))
        np.testing.assert_allclose(dp, 0.0, atol=1e-15)

    def test_equilibrium_mass_no_particles(self):
        """With no particles, equilibrium mass change is zero."""
        nbins = 5
        maddEQ = calc_equilibrium_mass(
            jnp.array(0.1),        # pamb
            jnp.array(1.0),        # cstar_Pa
            jnp.zeros(nbins),      # Mtot_org
            jnp.ones(nbins),       # kelvin
            jnp.zeros(nbins),      # Mk_species
            jnp.zeros(nbins),      # Nk
        )
        np.testing.assert_allclose(maddEQ, 0.0, atol=1e-30)


# =========================================================================
# SOA Condensation Step Tests
# =========================================================================

class TestSOACondensation:
    def test_mass_conservation(self):
        """Total mass (gas + particle) is conserved."""
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        alpha = 1.0
        dt = 60.0

        # Add organic gas for condensation
        for j in range(N_VBS_BINS):
            Gc = Gc.at[SRTORG1 + j].set(1e-15)

        # Total mass before
        total_before = jnp.sum(Gc[SRTORG1:SRTORG1 + N_VBS_BINS]) + \
                       jnp.sum(Mk[:, SRTORG1:SRTORG1 + N_VBS_BINS])

        Nk_new, Mk_new, Gc_new = soa_condensation_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        )

        total_after = jnp.sum(Gc_new[SRTORG1:SRTORG1 + N_VBS_BINS]) + \
                      jnp.sum(Mk_new[:, SRTORG1:SRTORG1 + N_VBS_BINS])

        np.testing.assert_allclose(float(total_after), float(total_before),
                                   rtol=1e-10)

    def test_pure_condensation_low_cstar(self):
        """Very low C* species: gas added above equilibrium should condense."""
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        alpha = 1.0
        dt = 3600.0  # 1 hour

        # Zero out initial organics in particles so no evaporation
        Mk = Mk.at[:, SRTORG1].set(0.0)

        # Put gas in first VBS bin (C*=0.01 µg/m³ — essentially non-volatile)
        Gc = Gc.at[SRTORG1].set(1e-12)
        gas_before = float(Gc[SRTORG1])

        # Single-species config (only bin 0)
        from tomas_jax.physics.vbs_config import VBSConfig
        single_cfg = VBSConfig(
            n_bins=1,
            cstar_ref_ug=CSTAR_REF_UG[:1],
            delta_Hvap_kJ=DELTA_HVAP_KJ[:1],
            mw=MW_ORG[:1],
            sigma=0.025,
            rho=1200.0,
            sv=120.0,
            t_ref=298.0,
            species_indices=(SRTORG1,),
        )

        Nk_new, Mk_new, Gc_new = soa_condensation_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
            vbs_config=single_cfg,
        )

        gas_after = float(Gc_new[SRTORG1])
        # With very low C* and no initial particle organics, gas should decrease
        assert gas_after < gas_before * 0.9

    def test_high_cstar_less_condensation(self):
        """High C* species condenses less than low C*."""
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        alpha = 1.0
        dt = 600.0

        # Low C* species
        from tomas_jax.physics.vbs_config import VBSConfig
        low_cfg = VBSConfig(
            n_bins=1,
            cstar_ref_ug=jnp.array([0.01]),
            delta_Hvap_kJ=jnp.array([150.0]),
            mw=jnp.array([200.0]),
            sigma=0.025, rho=1200.0, sv=120.0, t_ref=298.0,
            species_indices=(SRTORG1,),
        )

        Gc_low = Gc.at[SRTORG1].set(1e-15)
        _, _, Gc_low_after = soa_condensation_step(
            Nk, Mk, Gc_low, xk, temp, pres, boxvol, rh, alpha, dt,
            vbs_config=low_cfg,
        )
        condensed_low = float(Gc_low[SRTORG1]) - float(Gc_low_after[SRTORG1])

        # High C* species
        high_cfg = VBSConfig(
            n_bins=1,
            cstar_ref_ug=jnp.array([1000.0]),
            delta_Hvap_kJ=jnp.array([80.0]),
            mw=jnp.array([200.0]),
            sigma=0.025, rho=1200.0, sv=120.0, t_ref=298.0,
            species_indices=(SRTORG1,),
        )

        Gc_high = Gc.at[SRTORG1].set(1e-15)
        _, _, Gc_high_after = soa_condensation_step(
            Nk, Mk, Gc_high, xk, temp, pres, boxvol, rh, alpha, dt,
            vbs_config=high_cfg,
        )
        condensed_high = float(Gc_high[SRTORG1]) - float(Gc_high_after[SRTORG1])

        # Low C* should condense more
        assert condensed_low > condensed_high

    def test_no_gas_no_particles_no_change(self):
        """With zero gas and zero particle organics, nothing changes."""
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        # Zero out organic particle mass AND gas
        for j in range(N_VBS_BINS):
            Mk = Mk.at[:, SRTORG1 + j].set(0.0)
        Mk_before = Mk.copy()

        Nk_new, Mk_new, Gc_new = soa_condensation_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, 1.0, 60.0,
        )

        np.testing.assert_allclose(Mk_new[:, SRTORG1:SRTORG1 + N_VBS_BINS],
                                   Mk_before[:, SRTORG1:SRTORG1 + N_VBS_BINS],
                                   atol=1e-30)

    def test_evaporation_with_no_gas(self):
        """With organics in particles but zero gas, evaporation occurs."""
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        # Gc has zero organic gas, but particles have organics → evaporation
        org_before = float(jnp.sum(Mk[:, SRTORG1:SRTORG1 + N_VBS_BINS]))

        Nk_new, Mk_new, Gc_new = soa_condensation_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, 1.0, 600.0,
        )

        org_after = float(jnp.sum(Mk_new[:, SRTORG1:SRTORG1 + N_VBS_BINS]))
        gas_after = float(jnp.sum(Gc_new[SRTORG1:SRTORG1 + N_VBS_BINS]))

        # Particle organics should decrease (evaporation)
        assert org_after < org_before
        # Gas should increase
        assert gas_after > 0

    def test_number_conserved_ppm(self):
        """With PPM, total number is conserved (Nk redistributed)."""
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        for j in range(N_VBS_BINS):
            Gc = Gc.at[SRTORG1 + j].set(1e-15)

        Nk_new, Mk_new, Gc_new = soa_condensation_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, 1.0, 60.0,
            use_ppm=True,
        )

        # PPM redistributes Nk across bins, but total is conserved
        np.testing.assert_allclose(
            float(jnp.sum(Nk_new)), float(jnp.sum(Nk)), rtol=1e-8)

    def test_number_unchanged_no_ppm(self):
        """Without PPM, Nk is unchanged (direct mass addition only)."""
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        for j in range(N_VBS_BINS):
            Gc = Gc.at[SRTORG1 + j].set(1e-15)

        Nk_new, Mk_new, Gc_new = soa_condensation_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, 1.0, 60.0,
            use_ppm=False,
        )

        np.testing.assert_allclose(Nk_new, Nk, rtol=1e-14)

    def test_temperature_dependence(self):
        """Colder temperature → more condensation (lower C*)."""
        alpha = 1.0
        dt = 600.0

        from tomas_jax.physics.vbs_config import VBSConfig
        cfg = VBSConfig(
            n_bins=1,
            cstar_ref_ug=jnp.array([10.0]),  # semi-volatile
            delta_Hvap_kJ=jnp.array([108.0]),
            mw=jnp.array([200.0]),
            sigma=0.025, rho=1200.0, sv=120.0, t_ref=298.0,
            species_indices=(SRTORG1,),
        )

        # Warm case (298K)
        Nk_w, Mk_w, Gc_w, xk, _, pres, boxvol, rh = _make_test_state(temp=298.0)
        Gc_w = Gc_w.at[SRTORG1].set(1e-14)
        _, _, Gc_warm_after = soa_condensation_step(
            Nk_w, Mk_w, Gc_w, xk, 298.0, pres, boxvol, rh, alpha, dt,
            vbs_config=cfg,
        )

        # Cold case (240K)
        Nk_c, Mk_c, Gc_c, _, _, _, _, _ = _make_test_state(temp=240.0)
        Gc_c = Gc_c.at[SRTORG1].set(1e-14)
        _, _, Gc_cold_after = soa_condensation_step(
            Nk_c, Mk_c, Gc_c, xk, 240.0, pres, boxvol, rh, alpha, dt,
            vbs_config=cfg,
        )

        condensed_warm = float(Gc_w[SRTORG1]) - float(Gc_warm_after[SRTORG1])
        condensed_cold = float(Gc_c[SRTORG1]) - float(Gc_cold_after[SRTORG1])

        # Cold should condense more
        assert condensed_cold > condensed_warm

    def test_non_negative_particle_mass(self):
        """Particle mass should never go negative after evaporation."""
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        alpha = 1.0
        dt = 3600.0

        # Set very small particle organics (so evaporation could go negative)
        Mk = Mk.at[:, SRTORG1].set(1e-30)
        # No gas → should try to evaporate from particles
        Gc = Gc.at[SRTORG1].set(0.0)

        Nk_new, Mk_new, Gc_new = soa_condensation_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
            vbs_config=VBSConfig(
                n_bins=1,
                cstar_ref_ug=jnp.array([1000.0]),
                delta_Hvap_kJ=jnp.array([80.0]),
                mw=jnp.array([200.0]),
                sigma=0.025, rho=1200.0, sv=120.0, t_ref=298.0,
                species_indices=(SRTORG1,),
            ),
        )

        assert jnp.all(Mk_new[:, SRTORG1] >= 0.0)


# =========================================================================
# Integration Tests
# =========================================================================

class TestMakeStepIntegration:
    def test_make_step_soa_only(self):
        """make_step with soa_condensation alone."""
        from tomas_jax.solvers.condensation import make_step
        step = make_step(['soa_condensation'], cond_method='ppm_jit')

        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        for j in range(N_VBS_BINS):
            Gc = Gc.at[SRTORG1 + j].set(1e-15)

        Nk_out, Mk_out, Gc_out, bv_out = step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, 1.0, 60.0,
        )
        assert Nk_out.shape == Nk.shape
        assert Mk_out.shape == Mk.shape
        assert Gc_out.shape == Gc.shape

    def test_make_step_full_chain(self):
        """make_step with condensation + soa_condensation."""
        from tomas_jax.solvers.condensation import make_step
        step = make_step(['condensation', 'soa_condensation'],
                         cond_method='ppm_jit')

        Nk, Mk, Gc, xk, temp, pres, boxvol, rh = _make_test_state()
        # Zero out particle organics to avoid evaporation
        for j in range(N_VBS_BINS):
            Mk = Mk.at[:, SRTORG1 + j].set(0.0)
        Gc = Gc.at[SRTSO4].set(1e-14)  # H2SO4 gas
        for j in range(N_VBS_BINS):
            Gc = Gc.at[SRTORG1 + j].set(1e-12)  # organic gas

        Nk_out, Mk_out, Gc_out, bv_out = step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, 1.0, 60.0,
        )
        # H2SO4 should decrease
        assert float(Gc_out[SRTSO4]) < float(Gc[SRTSO4])
        # Total organic mass (gas + particle) should be conserved
        total_org_before = float(jnp.sum(Gc[SRTORG1:SRTORG1 + N_VBS_BINS])) + \
                           float(jnp.sum(Mk[:, SRTORG1:SRTORG1 + N_VBS_BINS]))
        total_org_after = float(jnp.sum(Gc_out[SRTORG1:SRTORG1 + N_VBS_BINS])) + \
                          float(jnp.sum(Mk_out[:, SRTORG1:SRTORG1 + N_VBS_BINS]))
        np.testing.assert_allclose(total_org_after, total_org_before, rtol=1e-8)

    def test_make_step_invalid_process(self):
        """Unknown process should raise ValueError."""
        from tomas_jax.solvers.condensation import make_step
        with pytest.raises(ValueError, match="Unknown process"):
            make_step(['soa_condensation', 'bogus'])
