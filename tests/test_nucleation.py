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
from tomas_jax.core.config import xk_boundaries
from tomas_jax.physics.nucleation import (
    ricco_nucleation_rate,
    dunne_nucleation_rate,
    nucleation_step,
    _MNUC, _KM, _DELTA_H, _KB_NUC, _T0,
)


@pytest.fixture
def xk():
    return xk_boundaries()


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
