"""Tests for JIT-compiled PPM condensation pipeline.

Validates:
1. Numerical equivalence: ppm_jit vs ppm within tolerance
2. Mass conservation: aerosol gained = gas lost
3. N conservation for cond-only
4. JIT compilation succeeds (no retracing)
5. Scan-fused loop vs step-by-step equivalence
"""
import pytest
import time
import numpy as np

import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO, XK0,
)
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics.ezcond_ppm_jax import ezcond_ppm_jax
from tomas_jax.physics.ezcond_ppm import ezcond_ppm
from tomas_jax.solvers.condensation import (
    condensation_step,
    condensation_step_jax,
    condensation_step_jit,
    run_condensation_scan,
)

# =========================================================================
# Fixtures
# =========================================================================

BOXVOL = 1.0e6  # cm^3
DENS_INIT = 1770.0
PI_F = 3.141592654


def make_xk():
    """Mass-doubling bin boundaries — delegates to config.xk_boundaries()."""
    from tomas_jax.core.config import xk_boundaries
    return xk_boundaries()


def make_lognormal(N_total=1e4, GMD_um=0.1, GSD=1.8, xk=None):
    """Create a lognormal distribution for testing."""
    if xk is None:
        xk = make_xk()
    xk_np = np.array(xk)

    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))

    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI_F)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI_F)) ** 0.3333
        Dk = np.sqrt(Dl * Dh)
        np_init = ((N_total * BOXVOL)
                   / (np.sqrt(2.0 * PI_F) * Dk * np.log(GSD))
                   * np.exp(-(np.log(Dk / GMD_um) ** 2
                              / (2.0 * np.log(GSD) ** 2)))
                   * (Dh - Dl))
        Nk[k] = np_init
        Mk[k, SRTSO4] = np_init * np.sqrt(xk_np[k]) * np.sqrt(xk_np[k + 1])

    NEPS = 1e-3
    for k in range(NBINS):
        if Nk[k] < NEPS:
            Nk[k] = NEPS
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = NEPS * 1.4 * xk_np[k]

    return jnp.array(Nk), jnp.array(Mk)


def make_gas(h2so4_molec_cm3=1e7):
    """Create gas array with H2SO4."""
    Gc = jnp.zeros(N_GAS_SPECIES)
    gc_so4 = h2so4_molec_cm3 * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO
    return Gc.at[SRTSO4].set(gc_so4)


# =========================================================================
# Test: ezcond_ppm_jax vs ezcond_ppm numerical equivalence
# =========================================================================

class TestEzcondPpmJax:
    """Test the pure-JAX ezcond replacement against the numpy version."""

    def test_basic_equivalence(self):
        """ezcond_ppm_jax should match ezcond_ppm within tolerance."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        mcond = 1e-15  # small condensation mass

        # Numpy version
        Nk_np, Mk_np = ezcond_ppm(
            np.array(Nk), np.array(Mk), mcond, SRTSO4,
            np.array(xk), 298.0, 101325.0, BOXVOL, 1.0
        )

        # JAX version (no internal MNFIX, so apply MNFIX before/after to match)
        Nk_fix, Mk_fix = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)
        Nk_jax, Mk_jax = ezcond_ppm_jax(
            Nk_fix, Mk_fix, jnp.array(mcond), SRTSO4,
            xk, jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(1.0)
        )
        Nk_jax, Mk_jax = mnfix_jax(Nk_jax, Mk_jax, xk, ICOMP_NODIAG)

        # Check relative tolerance
        N_tot_np = np.sum(Nk_np)
        N_tot_jax = float(jnp.sum(Nk_jax))
        rel_N = abs(N_tot_jax - N_tot_np) / max(N_tot_np, 1e-30)
        assert rel_N < 0.01, f"N total mismatch: {rel_N:.4e}"

        M_tot_np = np.sum(Mk_np[:, :ICOMP_NODIAG])
        M_tot_jax = float(jnp.sum(Mk_jax[:, :ICOMP_NODIAG]))
        rel_M = abs(M_tot_jax - M_tot_np) / max(M_tot_np, 1e-30)
        assert rel_M < 0.01, f"M dry total mismatch: {rel_M:.4e}"

    def test_no_nan(self):
        """Output should contain no NaN or Inf."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        mcond = 1e-14

        Nk_out, Mk_out = ezcond_ppm_jax(
            Nk, Mk, jnp.array(mcond), SRTSO4,
            xk, jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(1.0)
        )
        assert not jnp.any(jnp.isnan(Nk_out)), "NaN in Nk output"
        assert not jnp.any(jnp.isnan(Mk_out)), "NaN in Mk output"
        assert not jnp.any(jnp.isinf(Nk_out)), "Inf in Nk output"
        assert not jnp.any(jnp.isinf(Mk_out)), "Inf in Mk output"

    def test_jit_compiles(self):
        """ezcond_ppm_jax should JIT-compile without error."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        mcond = jnp.array(1e-15)

        ezcond_jit = jax.jit(ezcond_ppm_jax, static_argnums=(3,))
        Nk_out, Mk_out = ezcond_jit(
            Nk, Mk, mcond, SRTSO4,
            xk, jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(1.0)
        )
        assert Nk_out.shape == (NBINS,)
        assert Mk_out.shape == (NBINS, ICOMP)

    def test_zero_mcond_noop(self):
        """Zero mcond should leave state unchanged."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)

        Nk_out, Mk_out = ezcond_ppm_jax(
            Nk, Mk, jnp.array(0.0), SRTSO4,
            xk, jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(1.0)
        )
        np.testing.assert_allclose(np.array(Nk_out), np.array(Nk), rtol=1e-10)
        np.testing.assert_allclose(np.array(Mk_out), np.array(Mk), rtol=1e-10)


# =========================================================================
# Test: condensation_step_jax
# =========================================================================

class TestCondensationStepJax:
    """Test the pure-JAX condensation step."""

    def test_basic_run(self):
        """condensation_step_jax should run without error."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        Gc = make_gas()

        Nk_out, Mk_out, Gc_out = condensation_step_jax(
            Nk, Mk, Gc, xk,
            jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.5),
            jnp.array(1.0), jnp.array(60.0)
        )
        assert Nk_out.shape == (NBINS,)
        assert Mk_out.shape == (NBINS, ICOMP)
        assert Gc_out.shape == (N_GAS_SPECIES,)

    def test_no_nan(self):
        """Output should contain no NaN."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        Gc = make_gas()

        Nk_out, Mk_out, Gc_out = condensation_step_jax(
            Nk, Mk, Gc, xk,
            jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.5),
            jnp.array(1.0), jnp.array(60.0)
        )
        assert not jnp.any(jnp.isnan(Nk_out))
        assert not jnp.any(jnp.isnan(Mk_out))
        assert not jnp.any(jnp.isnan(Gc_out))

    def test_mass_conservation(self):
        """Aerosol gained should equal gas lost (dry mass)."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        Gc = make_gas(h2so4_molec_cm3=1e8)

        M_aero_before = float(jnp.sum(Mk[:, :SRTH2O]))
        G_before = float(jnp.sum(Gc))

        Nk_out, Mk_out, Gc_out = condensation_step_jax(
            Nk, Mk, Gc, xk,
            jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.5),
            jnp.array(1.0), jnp.array(60.0)
        )

        M_aero_after = float(jnp.sum(Mk_out[:, :SRTH2O]))
        G_after = float(jnp.sum(Gc_out))

        total_before = M_aero_before + G_before
        total_after = M_aero_after + G_after

        rel_err = abs(total_after - total_before) / max(abs(total_before), 1e-30)
        assert rel_err < 1e-6, f"Mass conservation error: {rel_err:.4e}"

    def test_equivalence_with_ppm(self):
        """ppm_jit should produce similar results to ppm method."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        Gc = make_gas()

        # ppm (numpy wrapper)
        Nk_ppm, Mk_ppm, Gc_ppm = condensation_step(
            Nk, Mk, Gc, xk,
            298.0, 101325.0, BOXVOL, 0.5, 1.0, 60.0,
            method='ppm'
        )

        # ppm_jit (pure JAX)
        Nk_jit, Mk_jit, Gc_jit = condensation_step(
            Nk, Mk, Gc, xk,
            298.0, 101325.0, BOXVOL, 0.5, 1.0, 60.0,
            method='ppm_jit'
        )

        # Compare totals (not bin-by-bin, since MNFIX timing differs slightly)
        N_ppm = float(jnp.sum(Nk_ppm))
        N_jit = float(jnp.sum(Nk_jit))
        rel_N = abs(N_jit - N_ppm) / max(N_ppm, 1e-30)
        assert rel_N < 0.01, f"N total differs by {rel_N:.4e}"

        M_ppm = float(jnp.sum(Mk_ppm[:, :SRTH2O]))
        M_jit = float(jnp.sum(Mk_jit[:, :SRTH2O]))
        rel_M = abs(M_jit - M_ppm) / max(M_ppm, 1e-30)
        assert rel_M < 0.01, f"M dry total differs by {rel_M:.4e}"

    def test_jit_compiles(self):
        """condensation_step_jit should compile and run."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        Gc = make_gas()

        Nk_out, Mk_out, Gc_out = condensation_step_jit(
            Nk, Mk, Gc, xk,
            jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.5),
            jnp.array(1.0), jnp.array(60.0)
        )
        assert Nk_out.shape == (NBINS,)


# =========================================================================
# Test: scan-fused loop
# =========================================================================

class TestCondensationScan:
    """Test the scan-fused time loop."""

    def test_scan_runs(self):
        """run_condensation_scan should run without error."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        Gc = make_gas()

        prod_rate = 1e7 * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO

        Nk_f, Mk_f, Gc_f, hist = run_condensation_scan(
            Nk, Mk, Gc, xk,
            jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.5),
            jnp.array(1.0), jnp.array(60.0),
            nsteps=10,
            prod_rate=jnp.array(prod_rate),
        )

        assert Nk_f.shape == (NBINS,)
        assert Mk_f.shape == (NBINS, ICOMP)
        assert hist.shape == (10, 3)
        assert not jnp.any(jnp.isnan(Nk_f))
        assert not jnp.any(jnp.isnan(Mk_f))

    def test_scan_vs_loop(self):
        """Scan-fused loop should match step-by-step loop."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        Gc = make_gas()

        prod_rate = 1e7 * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO
        dt = 60.0
        nsteps = 5

        # Step-by-step loop
        Nk_loop, Mk_loop, Gc_loop = Nk, Mk, Gc
        for _ in range(nsteps):
            Gc_loop = Gc_loop.at[SRTSO4].add(prod_rate * dt)
            Nk_loop, Mk_loop, Gc_loop = condensation_step_jax(
                Nk_loop, Mk_loop, Gc_loop, xk,
                jnp.array(298.0), jnp.array(101325.0),
                jnp.array(BOXVOL), jnp.array(0.5),
                jnp.array(1.0), jnp.array(dt)
            )

        # Scan-fused
        Nk_scan, Mk_scan, Gc_scan, _ = run_condensation_scan(
            Nk, Mk, Gc, xk,
            jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.5),
            jnp.array(1.0), jnp.array(dt),
            nsteps=nsteps,
            prod_rate=jnp.array(prod_rate),
        )

        np.testing.assert_allclose(
            np.array(Nk_scan), np.array(Nk_loop), rtol=1e-10,
            err_msg="Scan vs loop Nk mismatch"
        )
        np.testing.assert_allclose(
            np.array(Mk_scan), np.array(Mk_loop), rtol=1e-10,
            err_msg="Scan vs loop Mk mismatch"
        )
        np.testing.assert_allclose(
            np.array(Gc_scan), np.array(Gc_loop), rtol=1e-10,
            err_msg="Scan vs loop Gc mismatch"
        )

    def test_n_conservation_cond_only(self):
        """N should be approximately conserved for condensation-only over many steps."""
        xk = make_xk()
        Nk, Mk = make_lognormal(N_total=1e4, GMD_um=0.1, xk=xk)
        Gc = make_gas(h2so4_molec_cm3=1e7)

        prod_rate = 1e7 * BOXVOL * (MW_H2SO4 / 1000.0) / AVOGADRO

        N_init = float(jnp.sum(Nk))

        Nk_f, Mk_f, Gc_f, _ = run_condensation_scan(
            Nk, Mk, Gc, xk,
            jnp.array(298.0), jnp.array(101325.0),
            jnp.array(BOXVOL), jnp.array(0.5),
            jnp.array(1.0), jnp.array(60.0),
            nsteps=60,  # 1 hour
            prod_rate=jnp.array(prod_rate),
        )

        N_final = float(jnp.sum(Nk_f))
        ratio = N_final / N_init
        # N should be conserved within a few percent for condensation-only
        assert abs(ratio - 1.0) < 0.05, f"N ratio = {ratio:.4f}, expected ~1.0"


# =========================================================================
# Test: dispatcher routes ppm_jit correctly
# =========================================================================

class TestDispatcher:
    """Test that method='ppm_jit' in condensation_step routes correctly."""

    def test_ppm_jit_method(self):
        """condensation_step with method='ppm_jit' should work."""
        xk = make_xk()
        Nk, Mk = make_lognormal(xk=xk)
        Gc = make_gas()

        Nk_out, Mk_out, Gc_out = condensation_step(
            Nk, Mk, Gc, xk,
            298.0, 101325.0, BOXVOL, 0.5, 1.0, 60.0,
            method='ppm_jit'
        )
        assert Nk_out.shape == (NBINS,)
        assert not jnp.any(jnp.isnan(Nk_out))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
