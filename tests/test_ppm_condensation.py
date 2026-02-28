"""Comprehensive tests for PPM condensation algorithm.

Test hierarchy (implement in order):
1. Core utilities (dmdt_int, WR, edge velocity)
2. PPM reconstruction
3. Flux computation
4. Number-only advection
5. Passive species transport
6. Dry mass integral
7. Closure update
8. Full stepper
9. Comparison with TFL

Conservation invariants enforced:
- Total number conserved (closed boundary)
- Non-condensing dry mass conserved
- No NaN/Inf in outputs
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

# Import the module under test
from tomas_jax.physics.condensation_ppm import (
    # Constants
    DELTA_XI, EPSN, I0, I1, I2,
    # Core utilities
    dmdt_int,
    compute_wr,
    compute_edge_velocity,
    # PPM reconstruction
    ppm_reconstruct,
    # Flux computation
    ppm_flux,
    _integrate_parabola_left,
    _integrate_parabola_right,
    # Update
    advect_totals,
    passive_flux,
    # Dry mass integral
    dry_mass_from_ppm_number,
    dry_mass_from_ppm_number_quadrature,
    # Closure
    closure_update,
    # Steppers
    compute_substeps,
    ppm_advect_number_only,
    ppm_condensation_step,
)

# Test configuration
NBINS = 36
ICOMP = 44
ICOMP_NODIAG = 42  # Excluding water and diagnostics


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def standard_grid():
    """Standard TOMAS mass-doubling grid."""
    xk = jnp.logspace(-21, -12, NBINS + 1)
    return xk


@pytest.fixture
def uniform_distribution(standard_grid):
    """Uniform number distribution across all bins."""
    Nk = jnp.ones(NBINS) * 1e6
    return Nk


@pytest.fixture
def gaussian_distribution(standard_grid):
    """Gaussian-like distribution peaked at bin 18."""
    peak_bin = 18
    width = 5.0
    bins = jnp.arange(NBINS)
    Nk = 1e8 * jnp.exp(-0.5 * ((bins - peak_bin) / width) ** 2)
    return Nk


@pytest.fixture
def single_bin_distribution(standard_grid):
    """All particles in a single bin (bin 18)."""
    Nk = jnp.zeros(NBINS)
    Nk = Nk.at[18].set(1e8)
    return Nk


@pytest.fixture
def standard_mass(standard_grid, uniform_distribution):
    """Standard mass distribution with geometric mean mass per particle."""
    xk = standard_grid
    Nk = uniform_distribution
    mass_per_particle = jnp.sqrt(xk[:-1] * xk[1:])
    Mk = jnp.zeros((NBINS, ICOMP))
    # Distribute mass equally among first few dry species
    for j in range(min(5, ICOMP_NODIAG)):
        Mk = Mk.at[:, j].set(Nk * mass_per_particle / 5)
    return Mk


@pytest.fixture
def frozen_wr():
    """Frozen WR = 1 (all dry, no water)."""
    return jnp.ones(NBINS)


# =============================================================================
# 1.2 DMDT_INT TESTS
# =============================================================================

class TestDmdtInt:
    """Tests for analytic growth solution."""

    def test_zero_tau_identity(self):
        """tau=0 should return input unchanged."""
        m0 = jnp.array([1e-15, 1e-14, 1e-13])
        tau = jnp.zeros(3)
        wr = jnp.ones(3)

        m_final = dmdt_int(m0, tau, wr)

        np.testing.assert_allclose(m_final, m0, rtol=1e-10)

    def test_positive_tau_growth(self):
        """Positive tau should cause growth (m_final >= m0)."""
        m0 = jnp.array([1e-15, 1e-14, 1e-13])
        tau = jnp.array([1e-16, 1e-15, 1e-14])
        wr = jnp.ones(3)

        m_final = dmdt_int(m0, tau, wr)

        assert jnp.all(m_final >= m0), f"Growth failed: {m_final} < {m0}"

    def test_negative_tau_evaporation(self):
        """Negative tau should cause shrinkage (m_final <= m0)."""
        m0 = jnp.array([1e-13, 1e-14, 1e-15])
        tau = jnp.array([-1e-14, -1e-15, -1e-16])
        wr = jnp.ones(3)

        m_final = dmdt_int(m0, tau, wr)

        assert jnp.all(m_final <= m0), f"Evaporation failed: {m_final} > {m0}"

    def test_no_nan_inf(self):
        """Output should never contain NaN or Inf."""
        test_cases = [
            (jnp.array([1e-20]), jnp.array([1e-25]), jnp.array([1.0])),
            (jnp.array([1e-10]), jnp.array([1e-8]), jnp.array([1.5])),
            (jnp.array([1e-15]), jnp.array([-1e-10]), jnp.array([1.0])),
            (jnp.array([1e-15]), jnp.array([0.0]), jnp.array([2.0])),
        ]

        for m0, tau, wr in test_cases:
            m_final = dmdt_int(m0, tau, wr)
            assert jnp.all(jnp.isfinite(m_final)), f"NaN/Inf for m0={m0}, tau={tau}, wr={wr}"

    def test_wet_ratio_effect(self):
        """Higher WR should affect growth rate."""
        m0 = jnp.array([1e-15])
        tau = jnp.array([1e-14])

        m_final_dry = dmdt_int(m0, tau, jnp.array([1.0]))
        m_final_wet = dmdt_int(m0, tau, jnp.array([1.5]))

        rel_diff = jnp.abs(m_final_dry - m_final_wet) / m_final_dry
        assert rel_diff > 1e-6, \
            f"WR should affect growth, but rel_diff = {rel_diff}"


# =============================================================================
# 1.3 FROZEN WR TESTS
# =============================================================================

class TestComputeWR:
    """Tests for WR computation."""

    def test_dry_only(self):
        """All dry mass should give WR = 1."""
        Mk = jnp.ones((NBINS, ICOMP)) * 1e-15
        Mk = Mk.at[:, ICOMP_NODIAG:].set(0.0)

        WR = compute_wr(Mk, ICOMP_NODIAG)

        np.testing.assert_allclose(WR, 1.0, rtol=1e-10)

    def test_with_water(self):
        """With water, WR should be > 1."""
        Mk = jnp.ones((NBINS, ICOMP)) * 1e-15
        Mk = Mk.at[:, -1].set(0.5e-15)

        WR = compute_wr(Mk, ICOMP_NODIAG)

        assert jnp.all(WR > 1.0), "WR should be > 1 with water"

    def test_empty_bins(self):
        """Empty bins should not cause NaN/Inf."""
        Mk = jnp.zeros((NBINS, ICOMP))

        WR = compute_wr(Mk, ICOMP_NODIAG)

        assert jnp.all(jnp.isfinite(WR)), "WR should be finite for empty bins"
        assert jnp.all(WR >= 1.0), "WR should be >= 1"


# =============================================================================
# 1.4 EDGE VELOCITY TESTS
# =============================================================================

class TestEdgeVelocity:
    """Tests for upwind-consistent edge velocity."""

    def test_zero_tau_zero_velocity(self, standard_grid, frozen_wr):
        """tau=0 should give zero velocity everywhere."""
        xk = standard_grid
        TAU_sub = jnp.zeros(NBINS)
        dt_sub = 1.0

        u_edges = compute_edge_velocity(xk, TAU_sub, frozen_wr, dt_sub)

        np.testing.assert_allclose(u_edges, 0.0, atol=1e-14)

    def test_positive_tau_positive_velocity(self, standard_grid, frozen_wr):
        """Positive tau should give positive velocity (growth)."""
        xk = standard_grid
        TAU_sub = jnp.ones(NBINS) * 1e-16
        dt_sub = 1.0

        u_edges = compute_edge_velocity(xk, TAU_sub, frozen_wr, dt_sub)

        assert jnp.all(u_edges[1:-1] >= 0), "Growth should give non-negative velocity"

    def test_boundary_clipping(self, standard_grid, frozen_wr):
        """Velocity at boundaries should be clipped (zero or inward)."""
        xk = standard_grid
        TAU_sub = jnp.ones(NBINS) * 1e-10
        dt_sub = 1.0

        u_edges = compute_edge_velocity(xk, TAU_sub, frozen_wr, dt_sub)

        assert u_edges[-1] == 0.0 or u_edges[-1] <= 0.0, \
            "Upper boundary should have zero or negative velocity"

    def test_no_nan_inf(self, standard_grid, frozen_wr):
        """Edge velocities should never be NaN/Inf."""
        xk = standard_grid
        test_taus = [
            jnp.zeros(NBINS),
            jnp.ones(NBINS) * 1e-16,
            jnp.ones(NBINS) * -1e-16,
            jnp.linspace(-1e-16, 1e-16, NBINS),
        ]

        for TAU_sub in test_taus:
            u_edges = compute_edge_velocity(xk, TAU_sub, frozen_wr, 1.0)
            assert jnp.all(jnp.isfinite(u_edges)), f"NaN/Inf for TAU_sub={TAU_sub[:5]}..."


# =============================================================================
# 1.5 PPM RECONSTRUCTION TESTS
# =============================================================================

class TestPPMReconstruct:
    """Tests for PPM parabolic reconstruction."""

    def test_constant_profile(self):
        """Constant input should give q_L = q_R = constant, q_6 = 0."""
        q_bar = jnp.ones(NBINS) * 5.0

        q_L, q_R, q_6 = ppm_reconstruct(q_bar)

        np.testing.assert_allclose(q_L, 5.0, rtol=1e-10)
        np.testing.assert_allclose(q_R, 5.0, rtol=1e-10)
        np.testing.assert_allclose(q_6, 0.0, atol=1e-10)

    def test_integral_preservation(self):
        """Parabola integral should equal cell average (interior cells)."""
        q_bar = jnp.exp(-((jnp.arange(NBINS) - 18) / 5.0) ** 2) * 10.0

        q_L, q_R, q_6 = ppm_reconstruct(q_bar)

        dq = q_R - q_L
        integral = q_L + 0.5 * dq + q_6 / 6.0

        # Check interior cells (skip first and last few where limiting is active)
        np.testing.assert_allclose(integral[3:-3], q_bar[3:-3], rtol=1e-10)

    def test_monotonicity_preserved(self):
        """Monotonically increasing input should give monotonic edges."""
        q_bar = jnp.arange(NBINS, dtype=float) + 1.0

        q_L, q_R, q_6 = ppm_reconstruct(q_bar)

        edges = jnp.concatenate([q_L[:1], 0.5*(q_R[:-1] + q_L[1:]), q_R[-1:]])

        diffs = jnp.diff(edges)
        assert jnp.all(diffs >= -1e-10), "Edges should be monotonic for monotonic input"

    def test_no_new_extrema(self):
        """Parabola should not create new extrema inside cells (smooth profile)."""
        q_bar = jnp.exp(-((jnp.arange(NBINS) - 18) / 5.0) ** 2) * 10.0

        q_L, q_R, q_6 = ppm_reconstruct(q_bar)

        dq = q_R - q_L
        q_mid = q_L + 0.5 * (dq + q_6 * 0.5)

        for i in range(2, NBINS - 2):
            q_min = min(q_L[i], q_R[i])
            q_max = max(q_L[i], q_R[i])
            assert q_mid[i] >= q_min - 1e-6, f"Bin {i}: q_mid below minimum"
            assert q_mid[i] <= q_max + 1e-6, f"Bin {i}: q_mid above maximum"


# =============================================================================
# 1.6 FLUX COMPUTATION TESTS
# =============================================================================

class TestPPMFlux:
    """Tests for PPM flux computation."""

    def test_zero_velocity_zero_flux(self):
        """Zero velocity should give zero flux."""
        q_L = jnp.ones(NBINS) * 5.0
        q_R = jnp.ones(NBINS) * 5.0
        q_6 = jnp.zeros(NBINS)
        u_edges = jnp.zeros(NBINS + 1)
        dt_sub = 1.0

        F = ppm_flux(q_L, q_R, q_6, u_edges, dt_sub)

        np.testing.assert_allclose(F, 0.0, atol=1e-15)

    def test_constant_profile_uniform_velocity(self):
        """Constant q with uniform u should give flux ~ u * q * dxi."""
        q_val = 5.0
        q_L = jnp.ones(NBINS) * q_val
        q_R = jnp.ones(NBINS) * q_val
        q_6 = jnp.zeros(NBINS)
        u_val = 0.1
        u_edges = jnp.ones(NBINS + 1) * u_val
        dt_sub = 0.5

        F = ppm_flux(q_L, q_R, q_6, u_edges, dt_sub)

        expected = u_val * q_val
        np.testing.assert_allclose(F[1:-1], expected, rtol=0.1)

    def test_flux_direction(self):
        """Positive velocity should give positive flux from left cell."""
        q_L = jnp.ones(NBINS) * 5.0
        q_R = jnp.ones(NBINS) * 5.0
        q_6 = jnp.zeros(NBINS)
        u_edges = jnp.ones(NBINS + 1) * 0.1
        dt_sub = 0.5

        F = ppm_flux(q_L, q_R, q_6, u_edges, dt_sub)

        assert jnp.all(F[1:-1] > 0), "Positive velocity should give positive flux"

    def test_boundary_flux_zero(self):
        """Boundary fluxes should be zero."""
        q_L = jnp.ones(NBINS) * 5.0
        q_R = jnp.ones(NBINS) * 5.0
        q_6 = jnp.zeros(NBINS)
        u_edges = jnp.ones(NBINS + 1) * 0.1
        dt_sub = 0.5

        F = ppm_flux(q_L, q_R, q_6, u_edges, dt_sub)

        assert F[0] == 0.0, "Left boundary flux should be zero"
        assert F[-1] == 0.0, "Right boundary flux should be zero"


# =============================================================================
# 1.7 ADVECTION UPDATE TESTS
# =============================================================================

class TestAdvectTotals:
    """Tests for conservative flux-divergence update."""

    def test_conservation_closed_domain(self):
        """With zero boundary flux, total should be conserved."""
        Q = jnp.exp(-((jnp.arange(NBINS) - 18) / 5.0) ** 2) * 1e8
        F = jnp.zeros(NBINS + 1)
        dt_sub = 1.0

        Q_new = advect_totals(Q, F, dt_sub)

        np.testing.assert_allclose(jnp.sum(Q_new), jnp.sum(Q), rtol=1e-10)

    def test_flux_causes_transfer(self):
        """Non-zero internal flux should transfer mass between cells."""
        Q = jnp.zeros(NBINS)
        Q = Q.at[10].set(100.0)

        F = jnp.zeros(NBINS + 1)
        F = F.at[11].set(10.0)
        dt_sub = 1.0

        Q_new = advect_totals(Q, F, dt_sub)

        assert Q_new[10] == 90.0
        assert Q_new[11] == 10.0
        assert jnp.sum(Q_new) == jnp.sum(Q)


# =============================================================================
# 1.8 PASSIVE SPECIES TRANSPORT TESTS
# =============================================================================

class TestPassiveFlux:
    """Tests for consistent passive species transport."""

    def test_constant_ratio_moves_with_number(self):
        """If r_j = const, mass should move exactly like number."""
        Nk = jnp.ones(NBINS) * 1e6
        ratio = 1e-15
        Mk_j = Nk * ratio
        F_N = jnp.zeros(NBINS + 1)
        F_N = F_N.at[18].set(1e4)
        u_edges = jnp.zeros(NBINS + 1)
        u_edges = u_edges.at[18].set(0.1)

        F_M_j = passive_flux(F_N, Mk_j, Nk, u_edges)

        expected_mass_flux = F_N * ratio
        np.testing.assert_allclose(F_M_j[18], expected_mass_flux[18], rtol=1e-10)

    def test_empty_donor_no_blowup(self):
        """Empty donor cell should not cause NaN/Inf."""
        Nk = jnp.zeros(NBINS)
        Mk_j = jnp.zeros(NBINS)
        F_N = jnp.ones(NBINS + 1)
        u_edges = jnp.ones(NBINS + 1) * 0.1

        F_M_j = passive_flux(F_N, Mk_j, Nk, u_edges)

        assert jnp.all(jnp.isfinite(F_M_j)), "Should handle empty donor without NaN"


# =============================================================================
# 1.9 DRY MASS INTEGRAL TESTS
# =============================================================================

class TestDryMassIntegral:
    """Tests for exact dry mass integral from PPM number."""

    def test_closed_form_matches_quadrature(self, standard_grid):
        """Closed form should match 3-point Gauss quadrature."""
        xk = standard_grid

        n_L = jnp.exp(-((jnp.arange(NBINS) - 15) / 5.0) ** 2) * 1e6
        n_R = jnp.exp(-((jnp.arange(NBINS) - 20) / 5.0) ** 2) * 1e6
        n_6 = jnp.sin(jnp.arange(NBINS) * 0.2) * 1e5

        M_closed = dry_mass_from_ppm_number(n_L, n_R, n_6, xk)
        M_quad = dry_mass_from_ppm_number_quadrature(n_L, n_R, n_6, xk)

        np.testing.assert_allclose(M_closed, M_quad, rtol=1e-4)

    def test_positivity(self, standard_grid):
        """If n(xi) >= 0, M_dry should be >= 0."""
        xk = standard_grid

        n_bar = jnp.ones(NBINS) * 1e6
        n_L, n_R, n_6 = ppm_reconstruct(n_bar)

        M_dry = dry_mass_from_ppm_number(n_L, n_R, n_6, xk)

        assert jnp.all(M_dry >= -1e-20), "Dry mass should be non-negative"

    def test_scaling_with_number(self, standard_grid):
        """Doubling n should double M_dry."""
        xk = standard_grid
        n_L = jnp.ones(NBINS) * 1e6
        n_R = jnp.ones(NBINS) * 1e6
        n_6 = jnp.zeros(NBINS)

        M1 = dry_mass_from_ppm_number(n_L, n_R, n_6, xk)
        M2 = dry_mass_from_ppm_number(2*n_L, 2*n_R, 2*n_6, xk)

        np.testing.assert_allclose(M2, 2*M1, rtol=1e-10)


# =============================================================================
# 1.10 CLOSURE UPDATE TESTS
# =============================================================================

class TestClosureUpdate:
    """Tests for CSPECIES closure update."""

    def test_noncondensing_unchanged(self, standard_grid, standard_mass):
        """Non-condensing species should not be modified."""
        Mk = standard_mass
        M_dry_total = jnp.sum(Mk[:, :ICOMP_NODIAG], axis=1) * 1.1
        cspecies = 0

        Mk_new = closure_update(Mk, M_dry_total, cspecies, ICOMP_NODIAG)

        for j in range(ICOMP_NODIAG):
            if j != cspecies:
                np.testing.assert_allclose(Mk_new[:, j], Mk[:, j], rtol=1e-10)

    def test_dry_mass_balance(self, standard_grid, standard_mass):
        """Sum of dry masses should equal M_dry_total after closure."""
        Mk = standard_mass
        M_dry_total = jnp.sum(Mk[:, :ICOMP_NODIAG], axis=1) * 1.1
        cspecies = 0

        Mk_new = closure_update(Mk, M_dry_total, cspecies, ICOMP_NODIAG)

        M_dry_sum = jnp.sum(Mk_new[:, :ICOMP_NODIAG], axis=1)
        np.testing.assert_allclose(M_dry_sum, M_dry_total, rtol=1e-10)

    def test_cspecies_increases_with_growth(self, standard_grid, standard_mass):
        """Growth (M_dry_total increases) should increase CSPECIES."""
        Mk = standard_mass
        M_dry_old = jnp.sum(Mk[:, :ICOMP_NODIAG], axis=1)
        M_dry_total = M_dry_old * 1.1
        cspecies = 0

        Mk_new = closure_update(Mk, M_dry_total, cspecies, ICOMP_NODIAG)

        assert jnp.sum(Mk_new[:, cspecies]) > jnp.sum(Mk[:, cspecies])


# =============================================================================
# 2.1 SUBSTEP COUNT TESTS
# =============================================================================

class TestComputeSubsteps:
    """Tests for CFL-based substep calculation."""

    def test_small_tau_one_substep(self, standard_grid, frozen_wr):
        """Small tau should require only 1 substep."""
        xk = standard_grid
        TAU = jnp.ones(NBINS) * 1e-20
        dt = 1.0

        n_sub = compute_substeps(xk, TAU, frozen_wr, dt, C_max=0.8)

        assert n_sub == 1

    def test_large_tau_multiple_substeps(self, standard_grid, frozen_wr):
        """Large tau should require multiple substeps."""
        xk = standard_grid
        TAU = jnp.ones(NBINS) * 1e-10
        dt = 1.0

        n_sub = compute_substeps(xk, TAU, frozen_wr, dt, C_max=0.8)

        assert n_sub > 1


# =============================================================================
# NUMBER-ONLY ADVECTION TESTS
# =============================================================================

class TestNumberOnlyAdvection:
    """Tests for PPM number-only advection (validation phase)."""

    def test_zero_tau_identity(self, standard_grid, gaussian_distribution, frozen_wr):
        """Zero tau should return distribution unchanged."""
        xk = standard_grid
        Nk = gaussian_distribution
        TAU = jnp.zeros(NBINS)
        dt = 1.0

        Nk_new = ppm_advect_number_only(Nk, xk, TAU, frozen_wr, dt)

        np.testing.assert_allclose(Nk_new, Nk, rtol=1e-10)

    def test_number_conservation(self, standard_grid, gaussian_distribution, frozen_wr):
        """Total number should be conserved."""
        xk = standard_grid
        Nk = gaussian_distribution
        TAU = jnp.ones(NBINS) * 1e-16
        dt = 10.0

        Nk_new = ppm_advect_number_only(Nk, xk, TAU, frozen_wr, dt)

        np.testing.assert_allclose(jnp.sum(Nk_new), jnp.sum(Nk), rtol=1e-8)

    def test_positivity(self, standard_grid, gaussian_distribution, frozen_wr):
        """Number should remain non-negative."""
        xk = standard_grid
        Nk = gaussian_distribution
        TAU = jnp.ones(NBINS) * 1e-15
        dt = 10.0

        Nk_new = ppm_advect_number_only(Nk, xk, TAU, frozen_wr, dt)

        assert jnp.all(Nk_new >= 0), "Number should be non-negative"

    def test_growth_shifts_distribution(self, standard_grid, single_bin_distribution, frozen_wr):
        """Positive tau should shift distribution to higher bins."""
        xk = standard_grid
        Nk = single_bin_distribution
        initial_peak = 18
        TAU = jnp.ones(NBINS) * 5e-16
        dt = 10.0

        Nk_new = ppm_advect_number_only(Nk, xk, TAU, frozen_wr, dt)

        new_peak = jnp.argmax(Nk_new)

        assert new_peak >= initial_peak, f"Peak should shift up: {initial_peak} -> {new_peak}"

    def test_jit_compatible(self, standard_grid, gaussian_distribution, frozen_wr):
        """Function should be JIT-compilable."""
        xk = standard_grid
        Nk = gaussian_distribution
        TAU = jnp.ones(NBINS) * 1e-16
        dt = 1.0

        advect_jit = jax.jit(ppm_advect_number_only)

        Nk_new = advect_jit(Nk, xk, TAU, frozen_wr, dt)

        assert jnp.all(jnp.isfinite(Nk_new))


# =============================================================================
# FULL STEPPER TESTS
# =============================================================================

class TestFullStepper:
    """Tests for complete PPM condensation stepper."""

    def test_zero_tau_identity(self, standard_grid, standard_mass, uniform_distribution):
        """Zero tau should return state unchanged."""
        xk = standard_grid
        Nk = uniform_distribution
        Mk = standard_mass
        TAU = jnp.zeros(NBINS)
        cspecies = 0
        dt = 1.0

        Nk_new, Mk_new = ppm_condensation_step(
            Nk, Mk, xk, TAU, cspecies, dt, ICOMP_NODIAG
        )

        np.testing.assert_allclose(Nk_new, Nk, rtol=1e-10)

    def test_number_conservation(self, standard_grid, standard_mass, gaussian_distribution):
        """Total number should be conserved."""
        xk = standard_grid
        Nk = gaussian_distribution
        Mk = standard_mass
        TAU = jnp.ones(NBINS) * 1e-16
        cspecies = 0
        dt = 10.0

        Nk_new, _ = ppm_condensation_step(
            Nk, Mk, xk, TAU, cspecies, dt, ICOMP_NODIAG
        )

        np.testing.assert_allclose(jnp.sum(Nk_new), jnp.sum(Nk), rtol=1e-8)

    def test_noncondensing_mass_conservation(self, standard_grid, standard_mass, gaussian_distribution):
        """Non-condensing dry species mass should be conserved."""
        xk = standard_grid
        Nk = gaussian_distribution
        Mk = standard_mass
        TAU = jnp.ones(NBINS) * 1e-16
        cspecies = 0
        dt = 10.0

        _, Mk_new = ppm_condensation_step(
            Nk, Mk, xk, TAU, cspecies, dt, ICOMP_NODIAG
        )

        for j in range(ICOMP_NODIAG):
            if j != cspecies:
                M_before = jnp.sum(Mk[:, j])
                M_after = jnp.sum(Mk_new[:, j])
                np.testing.assert_allclose(M_after, M_before, rtol=1e-6,
                    err_msg=f"Species {j} mass not conserved")

    def test_cspecies_conserved_by_transport(self, standard_grid, standard_mass, gaussian_distribution):
        """PPM transport alone should conserve CSPECIES mass (addition is done by ezcond_ppm)."""
        xk = standard_grid
        Nk = gaussian_distribution
        Mk = standard_mass
        TAU = jnp.ones(NBINS) * 1e-15
        cspecies = 0
        dt = 10.0

        _, Mk_new = ppm_condensation_step(
            Nk, Mk, xk, TAU, cspecies, dt, ICOMP_NODIAG
        )

        M_cs_before = jnp.sum(Mk[:, cspecies])
        M_cs_after = jnp.sum(Mk_new[:, cspecies])

        np.testing.assert_allclose(
            M_cs_after, M_cs_before, rtol=1e-10,
            err_msg="CSPECIES mass should be conserved by transport"
        )

    def test_jit_compatible(self, standard_grid, standard_mass, uniform_distribution):
        """Full stepper should be JIT-compilable."""
        xk = standard_grid
        Nk = uniform_distribution
        Mk = standard_mass
        TAU = jnp.ones(NBINS) * 1e-16
        cspecies = 0
        dt = 1.0

        step_jit = jax.jit(lambda Nk, Mk: ppm_condensation_step(
            Nk, Mk, xk, TAU, cspecies, dt, ICOMP_NODIAG
        ))

        Nk_new, Mk_new = step_jit(Nk, Mk)

        assert jnp.all(jnp.isfinite(Nk_new))
        assert jnp.all(jnp.isfinite(Mk_new))


# =============================================================================
# PPM vs TFL COMPARISON
# =============================================================================

class TestPPMvsTFL:
    """Comparison tests between PPM and TFL ezcond drivers."""

    def test_small_mcond_agreement(self):
        """For small mcond, PPM and TFL should produce similar distributions."""
        from tomas_jax.core.config import SRTSO4, NBINS, ICOMP, N_GAS_SPECIES
        from tomas_jax.physics.ezcond import ezcond
        from tomas_jax.physics.ezcond_ppm import ezcond_ppm
        from tomas_jax.physics.condensation_sink import calc_condensation_sink

        # Use test case A parameters
        xk = np.logspace(-23, -12, NBINS + 1)
        boxvol = 1.0e6
        temp = 298.0
        pres = 101325.0

        # Simple lognormal initialization
        m_mid = np.sqrt(xk[:-1] * xk[1:])
        dp_mid = (6.0 * m_mid / (1770.0 * np.pi)) ** (1.0/3.0) * 1e6
        Nk = 1e5 * boxvol * np.exp(-0.5 * ((np.log(dp_mid) - np.log(0.1)) / np.log(1.6)) ** 2)
        Nk = np.maximum(Nk, 1e-3)
        Mk = np.zeros((NBINS, ICOMP))
        Mk[:, SRTSO4] = Nk * m_mid

        # Small mcond
        mcond = 1e-15

        # Run TFL
        Nk_tfl, Mk_tfl = ezcond(
            Nk, Mk, mcond, SRTSO4, xk, temp, pres, boxvol
        )

        # Run PPM
        Nk_ppm, Mk_ppm = ezcond_ppm(
            Nk, Mk, mcond, SRTSO4, xk, temp, pres, boxvol
        )

        # Both should conserve total number approximately
        np.testing.assert_allclose(
            np.sum(Nk_ppm), np.sum(Nk_tfl), rtol=0.05,
            err_msg="Total number should roughly match between PPM and TFL"
        )


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
