"""Tests for JIT-compiled TFL condensation pipeline.

Validates that the pure-JAX TFL implementation (condensation_tfl_jax.py)
matches the sequential TFL implementation (condensation.py + ezcond.py)
and that JIT compilation succeeds.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO, XK0,
)
from tomas_jax.physics.condensation_tfl_jax import tmcond_jax, ezcond_tfl_jax
from tomas_jax.solvers.condensation import (
    condensation_step,
    condensation_step_tfl_jax,
    condensation_step_tfl_jit,
    run_condensation_scan_tfl,
)

# --- Fixtures ---

BOXVOL = 1.0e6
DENS = 1770.0
PI_F = 3.141592654

@pytest.fixture
def xk():
    x = np.zeros(NBINS + 1)
    x[0] = XK0
    for k in range(NBINS):
        x[k + 1] = 2.0 * x[k]
    return jnp.array(x)


@pytest.fixture
def lognormal_state(xk):
    """Standard lognormal initial condition."""
    xk_np = np.array(xk)
    N_total, GMD, GSD = 1000.0, 0.05, 1.8
    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))
    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS * PI_F)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS * PI_F)) ** 0.3333
        Dk = np.sqrt(Dl * Dh)
        n = ((N_total * BOXVOL)
             / (np.sqrt(2 * PI_F) * Dk * np.log(GSD))
             * np.exp(-(np.log(Dk / GMD) ** 2 / (2 * np.log(GSD) ** 2)))
             * (Dh - Dl))
        Nk[k] = n
        Mk[k, SRTSO4] = n * np.sqrt(xk_np[k] * xk_np[k + 1])
    for k in range(NBINS):
        if Nk[k] < 1e-3:
            Nk[k] = 1e-3
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = 1e-3 * 1.4 * xk_np[k]
    Gc = np.zeros(N_GAS_SPECIES)
    h2so4_conc = 1e7
    Gc[SRTSO4] = h2so4_conc * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO
    return jnp.array(Nk), jnp.array(Mk), jnp.array(Gc)


# --- tmcond_jax tests ---

class TestTmcondJax:
    def test_zero_tau_preserves_state(self, xk, lognormal_state):
        Nk, Mk, _ = lognormal_state
        tau = jnp.zeros(NBINS)
        Nk_out, Mk_out = tmcond_jax(tau, xk, Mk, Nk, SRTSO4)
        np.testing.assert_allclose(np.array(Nk_out), np.array(Nk), rtol=1e-12)
        np.testing.assert_allclose(np.array(Mk_out), np.array(Mk), rtol=1e-12)

    def test_positive_tau_shifts_distribution(self, xk, lognormal_state):
        Nk, Mk, _ = lognormal_state
        tau = jnp.full(NBINS, 1e-16)
        Nk_out, Mk_out = tmcond_jax(tau, xk, Mk, Nk, SRTSO4)
        # Distribution should shift to larger sizes
        peak_before = jnp.argmax(Nk)
        peak_after = jnp.argmax(Nk_out)
        assert peak_after >= peak_before

    def test_jit_compilable(self, xk, lognormal_state):
        Nk, Mk, _ = lognormal_state
        tau = jnp.full(NBINS, 1e-16)
        fn = jax.jit(tmcond_jax, static_argnums=(4, 5))
        Nk_out, Mk_out = fn(tau, xk, Mk, Nk, SRTSO4, ICOMP_NODIAG)
        assert not jnp.any(jnp.isnan(Nk_out))
        assert not jnp.any(jnp.isnan(Mk_out))


# --- ezcond_tfl_jax tests ---

class TestEzcondTflJax:
    def test_zero_mcond_preserves_state(self, xk, lognormal_state):
        Nk, Mk, _ = lognormal_state
        mcond = jnp.array(0.0)
        Nk_out, Mk_out = ezcond_tfl_jax(
            Nk, Mk, mcond, SRTSO4, xk,
            jnp.array(278.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(1.0)
        )
        np.testing.assert_allclose(np.array(Nk_out), np.array(Nk), rtol=1e-10)

    def test_jit_compilable(self, xk, lognormal_state):
        Nk, Mk, _ = lognormal_state
        mcond = jnp.array(1e-15)
        fn = jax.jit(ezcond_tfl_jax, static_argnums=(3, 9))
        Nk_out, Mk_out = fn(
            Nk, Mk, mcond, SRTSO4, xk,
            jnp.array(278.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(1.0), ICOMP_NODIAG
        )
        assert not jnp.any(jnp.isnan(Nk_out))


# --- condensation_step_tfl_jax tests ---

class TestCondensationStepTflJax:
    def test_no_nan(self, xk, lognormal_state):
        Nk, Mk, Gc = lognormal_state
        Nk2, Mk2, Gc2 = condensation_step_tfl_jax(
            Nk, Mk, Gc, xk,
            jnp.array(278.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.8),
            jnp.array(1.0), jnp.array(60.0)
        )
        assert not jnp.any(jnp.isnan(Nk2))
        assert not jnp.any(jnp.isnan(Mk2))
        assert not jnp.any(jnp.isnan(Gc2))

    def test_gas_depletes(self, xk, lognormal_state):
        Nk, Mk, Gc = lognormal_state
        Nk2, Mk2, Gc2 = condensation_step_tfl_jax(
            Nk, Mk, Gc, xk,
            jnp.array(278.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.8),
            jnp.array(1.0), jnp.array(60.0)
        )
        assert float(Gc2[SRTSO4]) < float(Gc[SRTSO4])

    def test_mass_conservation(self, xk, lognormal_state):
        Nk, Mk, Gc = lognormal_state
        M_dry_0 = float(jnp.sum(Mk[:, :SRTH2O]))
        G_0 = float(jnp.sum(Gc))
        total_0 = M_dry_0 + G_0

        Nk2, Mk2, Gc2 = condensation_step_tfl_jax(
            Nk, Mk, Gc, xk,
            jnp.array(278.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.8),
            jnp.array(1.0), jnp.array(60.0)
        )
        M_dry_f = float(jnp.sum(Mk2[:, :SRTH2O]))
        G_f = float(jnp.sum(Gc2))
        total_f = M_dry_f + G_f

        rel_err = abs(total_f - total_0) / abs(total_0)
        assert rel_err < 1e-6, f"Mass conservation error: {rel_err:.2e}"

    def test_jit_compiles(self, xk, lognormal_state):
        Nk, Mk, Gc = lognormal_state
        # Should not raise
        Nk2, Mk2, Gc2 = condensation_step_tfl_jit(
            Nk, Mk, Gc, xk,
            jnp.array(278.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.8),
            jnp.array(1.0), jnp.array(60.0)
        )
        assert not jnp.any(jnp.isnan(Nk2))

    def test_matches_sequential_tfl_1step(self, xk, lognormal_state):
        """TFL JIT should match sequential TFL after 1 step."""
        Nk, Mk, Gc = lognormal_state
        args = (278.0, 101325.0, BOXVOL, 0.8, 1.0, 60.0)

        Nk_tfl, Mk_tfl, Gc_tfl = condensation_step(
            Nk, Mk, Gc, xk, *args, method='tfl'
        )
        Nk_jit, Mk_jit, Gc_jit = condensation_step_tfl_jit(
            Nk, Mk, Gc, xk,
            *[jnp.array(a) for a in args]
        )

        N_tfl = float(jnp.sum(Nk_tfl))
        N_jit = float(jnp.sum(Nk_jit))
        rel_N = abs(N_tfl - N_jit) / max(abs(N_tfl), 1e-30)
        assert rel_N < 1e-6, f"N_tot mismatch: {rel_N:.2e}"

        M_tfl = float(jnp.sum(Mk_tfl[:, :SRTH2O]))
        M_jit = float(jnp.sum(Mk_jit[:, :SRTH2O]))
        rel_M = abs(M_tfl - M_jit) / max(abs(M_tfl), 1e-30)
        assert rel_M < 1e-6, f"M_dry mismatch: {rel_M:.2e}"

    def test_matches_sequential_tfl_60steps(self, xk, lognormal_state):
        """TFL JIT should match sequential TFL after 60 steps (1 hour)."""
        Nk, Mk, Gc = lognormal_state
        prod_rate = float(Gc[SRTSO4]) * 0.001

        Nk_t, Mk_t, Gc_t = Nk, Mk, Gc
        Nk_j, Mk_j, Gc_j = Nk, Mk, Gc

        for _ in range(60):
            Gc_t = Gc_t.at[SRTSO4].add(prod_rate * 60.0)
            Nk_t, Mk_t, Gc_t = condensation_step(
                Nk_t, Mk_t, Gc_t, xk,
                278.0, 101325.0, BOXVOL, 0.8, 1.0, 60.0, method='tfl'
            )
            Gc_j = Gc_j.at[SRTSO4].add(prod_rate * 60.0)
            Nk_j, Mk_j, Gc_j = condensation_step_tfl_jit(
                Nk_j, Mk_j, Gc_j, xk,
                jnp.array(278.0), jnp.array(101325.0),
                jnp.array(BOXVOL), jnp.array(0.8),
                jnp.array(1.0), jnp.array(60.0)
            )

        N_tfl = float(jnp.sum(Nk_t))
        N_jit = float(jnp.sum(Nk_j))
        rel_N = abs(N_tfl - N_jit) / max(abs(N_tfl), 1e-30)
        assert rel_N < 1e-4, f"N_tot mismatch after 60 steps: {rel_N:.2e}"

        M_tfl = float(jnp.sum(Mk_t[:, :SRTH2O]))
        M_jit = float(jnp.sum(Mk_j[:, :SRTH2O]))
        rel_M = abs(M_tfl - M_jit) / max(abs(M_tfl), 1e-30)
        assert rel_M < 1e-4, f"M_dry mismatch after 60 steps: {rel_M:.2e}"


# --- Scan-fused loop tests ---

class TestScanTfl:
    def test_scan_runs(self, xk, lognormal_state):
        Nk, Mk, Gc = lognormal_state
        prod_rate = jnp.array(float(Gc[SRTSO4]) * 0.001)
        Nk_f, Mk_f, Gc_f, hist = run_condensation_scan_tfl(
            Nk, Mk, Gc, xk,
            jnp.array(278.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.8),
            jnp.array(1.0), jnp.array(60.0),
            nsteps=10,
            prod_rate=prod_rate,
        )
        assert hist.shape == (10, 3)
        assert not jnp.any(jnp.isnan(Nk_f))

    def test_scan_vs_loop(self, xk, lognormal_state):
        """Scan-fused loop should match step-by-step loop."""
        Nk, Mk, Gc = lognormal_state
        prod_rate = float(Gc[SRTSO4]) * 0.001
        nsteps = 10

        # Step-by-step
        Nk_l, Mk_l, Gc_l = Nk, Mk, Gc
        for _ in range(nsteps):
            Gc_l = Gc_l.at[SRTSO4].add(prod_rate * 60.0)
            Nk_l, Mk_l, Gc_l = condensation_step_tfl_jit(
                Nk_l, Mk_l, Gc_l, xk,
                jnp.array(278.0), jnp.array(101325.0),
                jnp.array(BOXVOL), jnp.array(0.8),
                jnp.array(1.0), jnp.array(60.0)
            )

        # Scan
        Nk_s, Mk_s, Gc_s, _ = run_condensation_scan_tfl(
            Nk, Mk, Gc, xk,
            jnp.array(278.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.8),
            jnp.array(1.0), jnp.array(60.0),
            nsteps=nsteps,
            prod_rate=jnp.array(prod_rate),
        )

        np.testing.assert_allclose(np.array(Nk_s), np.array(Nk_l), rtol=1e-10)
        np.testing.assert_allclose(np.array(Mk_s), np.array(Mk_l), rtol=1e-10)

    def test_n_conservation(self, xk, lognormal_state):
        """Cond-only should conserve N (no nucleation)."""
        Nk, Mk, Gc = lognormal_state
        N0 = float(jnp.sum(Nk))
        prod_rate = jnp.array(float(Gc[SRTSO4]) * 0.001)

        Nk_f, Mk_f, Gc_f, _ = run_condensation_scan_tfl(
            Nk, Mk, Gc, xk,
            jnp.array(278.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.8),
            jnp.array(1.0), jnp.array(60.0),
            nsteps=60,
            prod_rate=prod_rate,
        )
        Nf = float(jnp.sum(Nk_f))
        ratio = Nf / N0
        assert 0.999 < ratio < 1.001, f"N ratio = {ratio:.6f}"


# --- Dispatcher test ---

class TestDispatcher:
    def test_tfl_jit_dispatches(self, xk, lognormal_state):
        Nk, Mk, Gc = lognormal_state
        Nk2, Mk2, Gc2 = condensation_step(
            Nk, Mk, Gc, xk,
            278.0, 101325.0, BOXVOL, 0.8, 1.0, 60.0,
            method='tfl_jit'
        )
        assert not jnp.any(jnp.isnan(Nk2))
