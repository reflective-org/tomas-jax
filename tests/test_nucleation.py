"""Tests for nucleation parameterizations and driver."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tomas_jax.core.config import (
    NBINS, ICOMP, N_GAS_SPECIES,
    SRTSO4, SRTORGLAST, SRTH2O,
    AVOGADRO, XK0, PI,
)
from tomas_jax.physics.nucleation import (
    ricco_nucleation_rate,
    dunne_nucleation_rate,
    nucleation_step,
    zhao2024_synergistic_hno3_rate,
    kirkby2016_pure_organic_rate,
    zhao2024_organic_h2so4_rate,
    zhao2024_amine_h2so4_rate,
    zhao2024_iodine_oxoacid_rate,
    zhao2024_nucleation_step,
    _compute_ionc,
    _MNUC, _KM, _DELTA_H, _KB_NUC, _T0,
    _K16_T_REF, _K16_T_SCALE,
    _ORG_SA_K,
    ZHAO2024_ALL_ENABLED, ZHAO2024_DUNNE_ONLY,
)


@pytest.fixture
def base_state(xk):
    """Create a minimal aerosol state for testing."""
    Nk = jnp.ones(NBINS) * 1e6
    Mk = jnp.zeros((NBINS, ICOMP))
    Mk = Mk.at[:, SRTSO4].set(Nk * jnp.sqrt(xk[:-1] * xk[1:]))
    Gc = jnp.zeros(N_GAS_SPECIES)
    # Set ~1e7 molec/cm3 H2SO4 in a 1 m^3 box
    boxvol = 1e6  # cm^3
    h2so4_molec_cm3 = 1e7
    gc_so4 = h2so4_molec_cm3 * boxvol * (98.0 / 1000.0) / AVOGADRO
    Gc = Gc.at[SRTSO4].set(gc_so4)
    return Nk, Mk, Gc, xk, boxvol


# =========================================================================
# Riccobono 2014 tests
# =========================================================================

class TestRiccobono:
    def test_known_value_at_T0(self):
        """At T=278K (T0), ft=1, so J = km * h2so4^2 * org."""
        h2so4 = 1e7
        org = 1e7
        fn = ricco_nucleation_rate(jnp.float64(278.0), jnp.float64(h2so4), jnp.float64(org))
        expected = _KM * h2so4**2 * org  # 3.27e-21 * 1e14 * 1e7 = 3.27
        np.testing.assert_allclose(float(fn), expected, rtol=1e-10)

    def test_temperature_dependence(self):
        """Lower T should increase rate (delta_H > 0)."""
        h2so4 = jnp.float64(1e7)
        org = jnp.float64(1e7)
        fn_cold = ricco_nucleation_rate(jnp.float64(260.0), h2so4, org)
        fn_warm = ricco_nucleation_rate(jnp.float64(300.0), h2so4, org)
        assert float(fn_cold) > float(fn_warm)

    def test_zero_h2so4(self):
        fn = ricco_nucleation_rate(jnp.float64(278.0), jnp.float64(0.0), jnp.float64(1e7))
        assert float(fn) == 0.0

    def test_zero_org(self):
        fn = ricco_nucleation_rate(jnp.float64(278.0), jnp.float64(1e7), jnp.float64(0.0))
        assert float(fn) == 0.0

    def test_scales_with_h2so4_squared(self):
        org = jnp.float64(1e7)
        T = jnp.float64(278.0)
        fn1 = ricco_nucleation_rate(T, jnp.float64(1e7), org)
        fn2 = ricco_nucleation_rate(T, jnp.float64(2e7), org)
        np.testing.assert_allclose(float(fn2) / float(fn1), 4.0, rtol=1e-10)

    def test_scales_linearly_with_org(self):
        h2so4 = jnp.float64(1e7)
        T = jnp.float64(278.0)
        fn1 = ricco_nucleation_rate(T, h2so4, jnp.float64(1e7))
        fn2 = ricco_nucleation_rate(T, h2so4, jnp.float64(3e7))
        np.testing.assert_allclose(float(fn2) / float(fn1), 3.0, rtol=1e-10)


# =========================================================================
# Dunne 2016 tests
# =========================================================================

class TestDunne:
    def test_binary_neutral_only(self):
        """With nh3=0, fion=0: only Jbn > 0."""
        T = jnp.float64(280.0)
        Mair = jnp.float64(2.5e19)
        fn, Jbn, Jtn, Jbi, Jti = dunne_nucleation_rate(
            T, jnp.float64(0.0), jnp.float64(1e7), jnp.float64(0.0), Mair
        )
        assert float(Jbn) > 0.0
        assert float(Jtn) == 0.0
        # fion=0 -> ionc=0 -> Jbi=0, Jti=0
        assert float(Jbi) == 0.0
        assert float(Jti) == 0.0
        np.testing.assert_allclose(float(fn), float(Jbn), rtol=1e-10)

    def test_all_four_mechanisms(self):
        """With nh3>0 and fion>0, all 4 mechanisms should contribute."""
        T = jnp.float64(270.0)
        Mair = jnp.float64(2.5e19)
        fn, Jbn, Jtn, Jbi, Jti = dunne_nucleation_rate(
            T, jnp.float64(3.0), jnp.float64(1e7), jnp.float64(1e9), Mair
        )
        assert float(Jbn) > 0.0
        assert float(Jtn) > 0.0
        assert float(Jbi) > 0.0
        assert float(Jti) > 0.0
        np.testing.assert_allclose(float(fn), float(Jbn + Jtn + Jbi + Jti), rtol=1e-10)

    def test_nh3_threshold(self):
        """Below threshold nh3, ternary rates should be zero."""
        T = jnp.float64(280.0)
        Mair = jnp.float64(2.5e19)
        # nh3 = 1e-5 molec/cm3 -> scaled = 1e-11 < 1e-10
        fn, Jbn, Jtn, Jbi, Jti = dunne_nucleation_rate(
            T, jnp.float64(3.0), jnp.float64(1e7), jnp.float64(1e-5), Mair
        )
        assert float(Jtn) == 0.0
        assert float(Jti) == 0.0
        assert float(Jbn) > 0.0

    def test_temperature_sensitivity(self):
        """Colder temperatures should generally increase nucleation."""
        Mair = jnp.float64(2.5e19)
        h2so4 = jnp.float64(1e7)
        nh3 = jnp.float64(1e9)
        fion = jnp.float64(3.0)
        fn_cold, *_ = dunne_nucleation_rate(jnp.float64(240.0), fion, h2so4, nh3, Mair)
        fn_warm, *_ = dunne_nucleation_rate(jnp.float64(300.0), fion, h2so4, nh3, Mair)
        assert float(fn_cold) > float(fn_warm)

    def test_h2so4_dependence(self):
        """Higher H2SO4 should increase all rates."""
        T = jnp.float64(270.0)
        Mair = jnp.float64(2.5e19)
        fn_lo, *_ = dunne_nucleation_rate(
            T, jnp.float64(3.0), jnp.float64(1e6), jnp.float64(1e9), Mair
        )
        fn_hi, *_ = dunne_nucleation_rate(
            T, jnp.float64(3.0), jnp.float64(1e8), jnp.float64(1e9), Mair
        )
        assert float(fn_hi) > float(fn_lo)


# =========================================================================
# nucleation_step driver tests
# =========================================================================

class TestNucleationStep:
    def test_particles_added_to_bin0(self, base_state):
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk0_before = float(Nk[0])
        Nk_new, Mk_new, Gc_new = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),  # dt=60s
            jnp.float64(1e7),   # org_conc
            jnp.float64(0.0),   # nh3
            jnp.float64(0.0),   # fion
            enable_organic=1.0, enable_inorganic=0.0,
        )
        assert float(Nk_new[0]) > Nk0_before

    def test_only_bin0_modified(self, base_state):
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk_new, Mk_new, Gc_new = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(1e7), jnp.float64(0.0), jnp.float64(0.0),
            enable_organic=1.0, enable_inorganic=0.0,
        )
        # Bins 1-35 unchanged
        np.testing.assert_array_equal(np.array(Nk_new[1:]), np.array(Nk[1:]))
        np.testing.assert_array_equal(np.array(Mk_new[1:]), np.array(Mk[1:]))

    def test_so4_mass_balance(self, base_state):
        """Gas H2SO4 lost = aerosol SO4 gained (Fortran: no MW correction)."""
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk_new, Mk_new, Gc_new = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(1e7), jnp.float64(0.0), jnp.float64(0.0),
            enable_organic=1.0, enable_inorganic=0.0,
        )
        dM_so4 = float(jnp.sum(Mk_new[:, SRTSO4]) - jnp.sum(Mk[:, SRTSO4]))
        dGc = float(Gc[SRTSO4] - Gc_new[SRTSO4])
        # Fortran depletes Gc by SO4 mass directly (no 98/96 MW correction)
        np.testing.assert_allclose(dGc, dM_so4, rtol=1e-6)

    def test_organic_mass_added(self, base_state):
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk_new, Mk_new, Gc_new = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(1e7), jnp.float64(0.0), jnp.float64(0.0),
            enable_organic=1.0, enable_inorganic=0.0,
        )
        dM_org = float(Mk_new[0, SRTORGLAST] - Mk[0, SRTORGLAST])
        assert dM_org > 0.0

    def test_so4_org_ratio(self, base_state):
        """90% SO4, 10% organic by mass in nucleated clusters."""
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk_new, Mk_new, Gc_new = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(1e7), jnp.float64(0.0), jnp.float64(0.0),
            enable_organic=1.0, enable_inorganic=0.0,
        )
        dM_so4 = float(Mk_new[0, SRTSO4] - Mk[0, SRTSO4])
        dM_org = float(Mk_new[0, SRTORGLAST] - Mk[0, SRTORGLAST])
        np.testing.assert_allclose(dM_so4 / dM_org, 9.0, rtol=1e-10)

    def test_gas_clamping(self, xk):
        """When gas would go negative, clamp and use all available."""
        Nk = jnp.ones(NBINS) * 1e6
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[:, SRTSO4].set(Nk * jnp.sqrt(xk[:-1] * xk[1:]))
        Gc = jnp.zeros(N_GAS_SPECIES)
        # Very small gas: ~1e3 molec/cm3
        boxvol = 1e6
        gc_tiny = 1e3 * boxvol * 98.0 / 1000.0 / AVOGADRO
        Gc = Gc.at[SRTSO4].set(gc_tiny)

        Nk_new, Mk_new, Gc_new = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(3600.0),  # long dt to exhaust gas
            jnp.float64(1e15), jnp.float64(1e9), jnp.float64(3.0),
            enable_organic=1.0, enable_inorganic=1.0,
        )
        # Gas should be exactly zero after clamping
        assert float(Gc_new[SRTSO4]) == 0.0

    def test_zero_gas_noop(self, xk):
        """No gas -> no nucleation."""
        Nk = jnp.ones(NBINS) * 1e6
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[:, SRTSO4].set(Nk * jnp.sqrt(xk[:-1] * xk[1:]))
        Gc = jnp.zeros(N_GAS_SPECIES)
        boxvol = 1e6

        Nk_new, Mk_new, Gc_new = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(1e7), jnp.float64(0.0), jnp.float64(0.0),
            enable_organic=1.0, enable_inorganic=0.0,
        )
        # h2so4=0 -> fn_org = km * 0 * org = 0 -> no nucleation
        np.testing.assert_array_equal(np.array(Nk_new), np.array(Nk))

    def test_enable_disable_organic(self, base_state):
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk_on, _, _ = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(1e7), jnp.float64(0.0), jnp.float64(0.0),
            enable_organic=1.0, enable_inorganic=0.0,
        )
        Nk_off, _, _ = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(1e7), jnp.float64(0.0), jnp.float64(0.0),
            enable_organic=0.0, enable_inorganic=0.0,
        )
        assert float(Nk_on[0]) > float(Nk_off[0])
        np.testing.assert_array_equal(np.array(Nk_off), np.array(Nk))

    def test_enable_disable_inorganic(self, base_state):
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk_on, _, _ = nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(0.0), jnp.float64(0.0), jnp.float64(0.0),
            enable_organic=0.0, enable_inorganic=1.0,
        )
        # binary-neutral only (fion=0, nh3=0)
        assert float(Nk_on[0]) > float(Nk[0])

    def test_fn_scale(self, base_state):
        """fn_scale=2 should double the nucleation effect."""
        Nk, Mk, Gc, xk, boxvol = base_state
        args = (
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(1e7), jnp.float64(0.0), jnp.float64(0.0),
        )
        Nk_1x, _, _ = nucleation_step(
            Nk, Mk, Gc, xk, *args,
            enable_organic=1.0, enable_inorganic=0.0, fn_scale=1.0,
        )
        Nk_2x, _, _ = nucleation_step(
            Nk, Mk, Gc, xk, *args,
            enable_organic=1.0, enable_inorganic=0.0, fn_scale=2.0,
        )
        dN_1x = float(Nk_1x[0] - Nk[0])
        dN_2x = float(Nk_2x[0] - Nk[0])
        np.testing.assert_allclose(dN_2x / dN_1x, 2.0, rtol=1e-10)

    def test_jit_compilation(self, base_state):
        """nucleation_step should JIT-compile and produce correct results."""
        Nk, Mk, Gc, xk, boxvol = base_state
        nuc_jit = jax.jit(nucleation_step, static_argnames=[])
        Nk_new, Mk_new, Gc_new = nuc_jit(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            jnp.float64(1e7), jnp.float64(0.0), jnp.float64(0.0),
            1.0, 0.0, 1.0,
        )
        # Should have more particles than before
        assert float(Nk_new[0]) > float(Nk[0])
        # Gas should decrease
        assert float(Gc_new[SRTSO4]) < float(Gc[SRTSO4])

    def test_cluster_mass_value(self):
        """Verify mnuc = 4/3*pi*rnuc^3 * rho."""
        expected = 4.0 / 3.0 * PI * (0.85e-9)**3 * 1350.0
        np.testing.assert_allclose(_MNUC, expected, rtol=1e-10)

    def test_cluster_in_bin0(self, xk):
        """Cluster mass should be less than xk[0] (always goes to bin 0)."""
        assert _MNUC < float(xk[0])


# =========================================================================
# Zhao 2024 — Mechanism 5: H2SO4-HNO3-NH3 synergistic
# =========================================================================

class TestSynergisticHNO3:
    def test_cold_temperature_high_rate(self):
        """At 220K (well below cutoff), rate should be large."""
        J = zhao2024_synergistic_hno3_rate(
            jnp.float64(220.0), jnp.float64(1e7),
            jnp.float64(1e9), jnp.float64(1e9),
        )
        assert float(J) > 1.0

    def test_warm_temperature_zero(self):
        """At 300K (well above 238K cutoff), sigmoid kills rate."""
        J = zhao2024_synergistic_hno3_rate(
            jnp.float64(300.0), jnp.float64(1e7),
            jnp.float64(1e9), jnp.float64(1e9),
        )
        assert float(J) < 1e-10

    def test_zero_hno3(self):
        J = zhao2024_synergistic_hno3_rate(
            jnp.float64(220.0), jnp.float64(1e7),
            jnp.float64(0.0), jnp.float64(1e9),
        )
        assert float(J) == 0.0

    def test_h2so4_cubic_scaling(self):
        """J scales as [H2SO4]^3."""
        T = jnp.float64(220.0)
        hno3 = jnp.float64(1e9)
        nh3 = jnp.float64(1e9)
        J1 = zhao2024_synergistic_hno3_rate(T, jnp.float64(1e6), hno3, nh3)
        J2 = zhao2024_synergistic_hno3_rate(T, jnp.float64(2e6), hno3, nh3)
        np.testing.assert_allclose(float(J2) / float(J1), 8.0, rtol=1e-10)


# =========================================================================
# Zhao 2024 — Mechanisms 6-7: Pure-organic (Kirkby 2016)
# =========================================================================

class TestKirkby2016:
    def test_neutral_positive_at_278K(self):
        """At reference T, neutral rate should be positive for ulvoc > 0."""
        Jn, Ji = kirkby2016_pure_organic_rate(
            jnp.float64(278.0), jnp.float64(1e7), jnp.float64(700.0),
        )
        assert float(Jn) > 0.0
        assert float(Ji) > 0.0

    def test_zero_ulvoc(self):
        """Zero ULVOC -> zero rates."""
        Jn, Ji = kirkby2016_pure_organic_rate(
            jnp.float64(278.0), jnp.float64(0.0), jnp.float64(700.0),
        )
        assert float(Jn) == 0.0
        assert float(Ji) == 0.0

    def test_temperature_factor_at_ref(self):
        """At T_ref=278K, f_T should be 1.0."""
        import math
        f_T = math.exp(-(278.0 - _K16_T_REF) / _K16_T_SCALE)
        np.testing.assert_allclose(f_T, 1.0, rtol=1e-10)

    def test_cold_increases_rate(self):
        """Lower T -> higher rate (Zhao 2024 temperature factor)."""
        ulvoc = jnp.float64(1e7)
        ionc = jnp.float64(700.0)
        Jn_cold, _ = kirkby2016_pure_organic_rate(jnp.float64(265.0), ulvoc, ionc)
        Jn_warm, _ = kirkby2016_pure_organic_rate(jnp.float64(290.0), ulvoc, ionc)
        assert float(Jn_cold) > float(Jn_warm)

    def test_ion_scales_with_ionc(self):
        """Ion-induced rate scales linearly with ion concentration."""
        T = jnp.float64(278.0)
        ulvoc = jnp.float64(1e7)
        _, Ji1 = kirkby2016_pure_organic_rate(T, ulvoc, jnp.float64(500.0))
        _, Ji2 = kirkby2016_pure_organic_rate(T, ulvoc, jnp.float64(1000.0))
        np.testing.assert_allclose(float(Ji2) / float(Ji1), 2.0, rtol=1e-10)


# =========================================================================
# Zhao 2024 — Mechanism 8: Organic-H2SO4
# =========================================================================

class TestOrganicH2SO4:
    def test_bilinear_form(self):
        """J = k * [H2SO4] * [ORG] * f_T at reference temperature."""
        T = jnp.float64(278.0)
        h2so4 = jnp.float64(1e7)
        ulvoc = jnp.float64(1e7)
        J = zhao2024_organic_h2so4_rate(T, h2so4, ulvoc)
        # At T_ref, f_T = 1.0
        expected = _ORG_SA_K * 1e7 * 1e7
        np.testing.assert_allclose(float(J), expected, rtol=1e-10)

    def test_linear_in_h2so4(self):
        T = jnp.float64(278.0)
        ulvoc = jnp.float64(1e7)
        J1 = zhao2024_organic_h2so4_rate(T, jnp.float64(1e7), ulvoc)
        J2 = zhao2024_organic_h2so4_rate(T, jnp.float64(3e7), ulvoc)
        np.testing.assert_allclose(float(J2) / float(J1), 3.0, rtol=1e-10)

    def test_linear_in_ulvoc(self):
        T = jnp.float64(278.0)
        h2so4 = jnp.float64(1e7)
        J1 = zhao2024_organic_h2so4_rate(T, h2so4, jnp.float64(1e6))
        J2 = zhao2024_organic_h2so4_rate(T, h2so4, jnp.float64(5e6))
        np.testing.assert_allclose(float(J2) / float(J1), 5.0, rtol=1e-10)


# =========================================================================
# Zhao 2024 — Mechanism 9: Amine-H2SO4
# =========================================================================

class TestAmineH2SO4:
    def test_positive_rate(self):
        J = zhao2024_amine_h2so4_rate(
            jnp.float64(278.0), jnp.float64(1e7), jnp.float64(1e8),
        )
        assert float(J) > 0.0

    def test_h2so4_cubic_scaling(self):
        """J scales as [H2SO4]^3."""
        T = jnp.float64(278.0)
        dma = jnp.float64(1e8)
        J1 = zhao2024_amine_h2so4_rate(T, jnp.float64(1e7), dma)
        J2 = zhao2024_amine_h2so4_rate(T, jnp.float64(2e7), dma)
        np.testing.assert_allclose(float(J2) / float(J1), 8.0, rtol=1e-10)

    def test_dma_1p5_scaling(self):
        """J scales as [DMA]^1.5."""
        T = jnp.float64(278.0)
        h2so4 = jnp.float64(1e7)
        J1 = zhao2024_amine_h2so4_rate(T, h2so4, jnp.float64(1e8))
        J2 = zhao2024_amine_h2so4_rate(T, h2so4, jnp.float64(4e8))
        # (4)^1.5 = 8.0
        np.testing.assert_allclose(float(J2) / float(J1), 8.0, rtol=1e-10)

    def test_temperature_function_at_278K(self):
        """Cai 2021 f_T(278K) should be close to 1.0."""
        import math
        f_T = (1.576 * math.exp(-((278.0 - 250.6) / 23.18)**2)
             + 0.6956 * math.exp(-((278.0 - 273.1) / 13.01)**2))
        np.testing.assert_allclose(f_T, 1.0, atol=0.01)

    def test_zero_dma(self):
        J = zhao2024_amine_h2so4_rate(
            jnp.float64(278.0), jnp.float64(1e7), jnp.float64(0.0),
        )
        assert float(J) == 0.0


# =========================================================================
# Zhao 2024 — Mechanisms 10-11: Iodine oxoacids
# =========================================================================

class TestIodineOxoacids:
    def test_positive_rates(self):
        Jn, Ji = zhao2024_iodine_oxoacid_rate(
            jnp.float64(290.0), jnp.float64(1e8), jnp.float64(700.0),
        )
        assert float(Jn) > 0.0
        assert float(Ji) > 0.0

    def test_zero_hio3(self):
        Jn, Ji = zhao2024_iodine_oxoacid_rate(
            jnp.float64(290.0), jnp.float64(0.0), jnp.float64(700.0),
        )
        assert float(Jn) == 0.0
        assert float(Ji) == 0.0

    def test_cold_increases_neutral_rate(self):
        """Lower T -> higher rate (Arrhenius with positive E_a)."""
        hio3 = jnp.float64(1e8)
        ionc = jnp.float64(700.0)
        Jn_cold, _ = zhao2024_iodine_oxoacid_rate(jnp.float64(270.0), hio3, ionc)
        Jn_warm, _ = zhao2024_iodine_oxoacid_rate(jnp.float64(300.0), hio3, ionc)
        assert float(Jn_cold) > float(Jn_warm)

    def test_neutral_temp_clamp_at_263K(self):
        """Below 263K, neutral rate should plateau (T clamped)."""
        hio3 = jnp.float64(1e8)
        ionc = jnp.float64(700.0)
        Jn_250, _ = zhao2024_iodine_oxoacid_rate(jnp.float64(250.0), hio3, ionc)
        Jn_263, _ = zhao2024_iodine_oxoacid_rate(jnp.float64(263.0), hio3, ionc)
        np.testing.assert_allclose(float(Jn_250), float(Jn_263), rtol=1e-10)

    def test_ion_temp_clamp_at_283K(self):
        """Below 283K, ion-induced rate should plateau (T clamped)."""
        hio3 = jnp.float64(1e8)
        ionc = jnp.float64(700.0)
        _, Ji_270 = zhao2024_iodine_oxoacid_rate(jnp.float64(270.0), hio3, ionc)
        _, Ji_283 = zhao2024_iodine_oxoacid_rate(jnp.float64(283.0), hio3, ionc)
        np.testing.assert_allclose(float(Ji_270), float(Ji_283), rtol=1e-10)

    def test_ion_scales_with_ionc(self):
        """Ion-induced rate scales linearly with ion concentration."""
        T = jnp.float64(290.0)
        hio3 = jnp.float64(1e8)
        _, Ji1 = zhao2024_iodine_oxoacid_rate(T, hio3, jnp.float64(350.0))
        _, Ji2 = zhao2024_iodine_oxoacid_rate(T, hio3, jnp.float64(700.0))
        np.testing.assert_allclose(float(Ji2) / float(Ji1), 2.0, rtol=1e-10)


# =========================================================================
# zhao2024_nucleation_step driver tests
# =========================================================================

class TestZhao2024Step:
    def test_particles_added_to_bin0(self, base_state):
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk_new, Mk_new, Gc_new = zhao2024_nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            org_conc=1e7, nh3_conc=1e8, fion=3.0,
            ulvoc=1e7, dma=1e8,
        )
        assert float(Nk_new[0]) > float(Nk[0])

    def test_only_bin0_modified(self, base_state):
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk_new, Mk_new, Gc_new = zhao2024_nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            org_conc=1e7, nh3_conc=1e8, fion=3.0,
        )
        np.testing.assert_array_equal(np.array(Nk_new[1:]), np.array(Nk[1:]))

    def test_so4_mass_balance(self, base_state):
        """Gas H2SO4 lost = aerosol SO4 gained."""
        Nk, Mk, Gc, xk, boxvol = base_state
        # Use high org+nh3 for a larger nucleation signal (better precision)
        Nk_new, Mk_new, Gc_new = zhao2024_nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            org_conc=1e7, nh3_conc=1e8, fion=3.0,
            ulvoc=1e7, dma=1e8,
        )
        # Check via bin 0 only (avoids catastrophic cancellation from sum)
        dM_so4 = float(Mk_new[0, SRTSO4] - Mk[0, SRTSO4])
        dGc = float(Gc[SRTSO4] - Gc_new[SRTSO4])
        np.testing.assert_allclose(dGc, dM_so4, rtol=1e-6)

    def test_enable_masks_dunne_only(self, base_state):
        """DUNNE_ONLY masks should give fewer particles than ALL_ENABLED."""
        Nk, Mk, Gc, xk, boxvol = base_state
        args = (
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
        )
        kw = dict(org_conc=1e7, nh3_conc=1e8, fion=3.0,
                  ulvoc=1e7, dma=1e8, hio3=1e8)
        Nk_all, _, _ = zhao2024_nucleation_step(
            Nk, Mk, Gc, xk, *args, enable_masks=ZHAO2024_ALL_ENABLED, **kw,
        )
        Nk_dunne, _, _ = zhao2024_nucleation_step(
            Nk, Mk, Gc, xk, *args, enable_masks=ZHAO2024_DUNNE_ONLY, **kw,
        )
        assert float(Nk_all[0]) > float(Nk_dunne[0])

    def test_gas_clamping(self, xk):
        """When gas is exhausted, Gc[SO4] -> 0."""
        Nk = jnp.ones(NBINS) * 1e6
        Mk = jnp.zeros((NBINS, ICOMP))
        Mk = Mk.at[:, SRTSO4].set(Nk * jnp.sqrt(xk[:-1] * xk[1:]))
        Gc = jnp.zeros(N_GAS_SPECIES)
        boxvol = 1e6
        gc_tiny = 1e3 * boxvol * 98.0 / 1000.0 / AVOGADRO
        Gc = Gc.at[SRTSO4].set(gc_tiny)

        Nk_new, Mk_new, Gc_new = zhao2024_nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(3600.0),
            org_conc=1e15, nh3_conc=1e9, fion=3.0,
            ulvoc=1e7, dma=1e8,
        )
        assert float(Gc_new[SRTSO4]) == 0.0

    def test_fn_scale(self, base_state):
        """fn_scale=2 should double the nucleation effect."""
        Nk, Mk, Gc, xk, boxvol = base_state
        args = (
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
        )
        kw = dict(org_conc=1e7, nh3_conc=1e8, fion=3.0,
                  enable_masks=ZHAO2024_DUNNE_ONLY)
        Nk_1x, _, _ = zhao2024_nucleation_step(
            Nk, Mk, Gc, xk, *args, fn_scale=1.0, **kw,
        )
        Nk_2x, _, _ = zhao2024_nucleation_step(
            Nk, Mk, Gc, xk, *args, fn_scale=2.0, **kw,
        )
        dN_1x = float(Nk_1x[0] - Nk[0])
        dN_2x = float(Nk_2x[0] - Nk[0])
        np.testing.assert_allclose(dN_2x / dN_1x, 2.0, rtol=1e-10)

    def test_jit_compilation(self, base_state):
        """zhao2024_nucleation_step should JIT-compile."""
        Nk, Mk, Gc, xk, boxvol = base_state
        nuc_jit = jax.jit(zhao2024_nucleation_step,
                          static_argnames=['enable_masks'])
        Nk_new, Mk_new, Gc_new = nuc_jit(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            org_conc=1e7, nh3_conc=1e8, fion=3.0,
            ulvoc=1e7, dma=1e8,
            enable_masks=ZHAO2024_ALL_ENABLED,
        )
        assert float(Nk_new[0]) > float(Nk[0])
        assert float(Gc_new[SRTSO4]) < float(Gc[SRTSO4])

    def test_so4_org_ratio(self, base_state):
        """Nucleated clusters maintain 90% SO4, 10% organic."""
        Nk, Mk, Gc, xk, boxvol = base_state
        Nk_new, Mk_new, Gc_new = zhao2024_nucleation_step(
            Nk, Mk, Gc, xk,
            jnp.float64(278.0), jnp.float64(101325.0), jnp.float64(boxvol),
            jnp.float64(60.0),
            org_conc=1e7, nh3_conc=1e8, fion=3.0,
            enable_masks=ZHAO2024_DUNNE_ONLY,
        )
        dM_so4 = float(Mk_new[0, SRTSO4] - Mk[0, SRTSO4])
        dM_org = float(Mk_new[0, SRTORGLAST] - Mk[0, SRTORGLAST])
        np.testing.assert_allclose(dM_so4 / dM_org, 9.0, rtol=1e-10)


# =========================================================================
# Ion concentration helper
# =========================================================================

class TestComputeIonc:
    def test_positive(self):
        ionc = _compute_ionc(jnp.float64(3.0), jnp.float64(278.0), jnp.float64(2.5e19))
        assert float(ionc) > 0.0

    def test_zero_fion(self):
        ionc = _compute_ionc(jnp.float64(0.0), jnp.float64(278.0), jnp.float64(2.5e19))
        assert float(ionc) == 0.0

    def test_typical_value(self):
        """GCR: fion~3, T~278K -> ionc ~ 500-2000 cm^-3."""
        ionc = _compute_ionc(jnp.float64(3.0), jnp.float64(278.0), jnp.float64(2.5e19))
        val = float(ionc)
        assert 500.0 < val < 2000.0
