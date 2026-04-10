"""Lightweight smoke tests that always run (no Fortran data needed).

Exercises the full solver pipeline with synthetic initial conditions.
Should complete in <10 seconds.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO,
    xk_boundaries,
)
from tomas_jax.solvers.diffrax import coag_euler_step
from tomas_jax.solvers.condensation import condensation_step, make_step
from conftest import make_lognormal


BOXVOL = 1e6  # cm^3


class TestCoagulationSmoke:
    """Basic coagulation sanity checks."""

    def test_coag_no_nan(self, xk):
        Nk, Mk, _ = make_lognormal(xk, N_total=1e4, GMD_um=0.1, GSD=1.8)
        Nk_out, Mk_out = coag_euler_step(
            Nk, Mk, xk, 298.0, 101325.0, BOXVOL, dt=60.0,
            icomp_nodiag=ICOMP_NODIAG, n_substeps=3)
        assert jnp.all(jnp.isfinite(Nk_out)), "NaN in Nk after coagulation"
        assert jnp.all(jnp.isfinite(Mk_out)), "NaN in Mk after coagulation"

    def test_coag_number_decreases(self, xk):
        """Coagulation merges particles — total number must decrease."""
        Nk, Mk, _ = make_lognormal(xk, N_total=1e4, GMD_um=0.1, GSD=1.8)
        Nk_out, Mk_out = coag_euler_step(
            Nk, Mk, xk, 298.0, 101325.0, BOXVOL, dt=60.0,
            icomp_nodiag=ICOMP_NODIAG, n_substeps=3)
        assert float(jnp.sum(Nk_out)) < float(jnp.sum(Nk)), \
            "Total number should decrease after coagulation"

    def test_coag_mass_budget_with_overflow(self, xk):
        """M(0) = M(dt) + overflow should close to machine precision."""
        Nk, Mk, _ = make_lognormal(xk, N_total=1e4, GMD_um=0.1, GSD=1.8)
        M_dry_0 = float(jnp.sum(Mk[:, :ICOMP_NODIAG]))

        Nk_out, Mk_out, overflow = coag_euler_step(
            Nk, Mk, xk, 298.0, 101325.0, BOXVOL, dt=60.0,
            icomp_nodiag=ICOMP_NODIAG, n_substeps=3,
            return_overflow=True)

        M_dry_out = float(jnp.sum(Mk_out[:, :ICOMP_NODIAG]))
        M_overflow = float(jnp.sum(overflow[:ICOMP_NODIAG]))

        residual = abs(M_dry_0 - M_dry_out - M_overflow)
        rel_residual = residual / max(M_dry_0, 1e-30)
        assert rel_residual < 1e-8, \
            f"Mass budget not closed: residual={rel_residual:.2e}"


class TestCondensationSmoke:
    """Basic condensation sanity checks."""

    def test_cond_no_nan(self, xk):
        Nk, Mk, Gc = make_lognormal(xk, N_total=1e4, GMD_um=0.1, GSD=1.8,
                                     h2so4_molec_cm3=1e7)
        Nk_out, Mk_out, Gc_out = condensation_step(
            Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 0.5, 1.0, 60.0,
            method='ppm_jit')
        assert jnp.all(jnp.isfinite(Nk_out)), "NaN in Nk after condensation"
        assert jnp.all(jnp.isfinite(Mk_out)), "NaN in Mk after condensation"

    def test_cond_mass_increases(self, xk):
        """H2SO4 condensation adds mass to aerosol."""
        Nk, Mk, Gc = make_lognormal(xk, N_total=1e4, GMD_um=0.1, GSD=1.8,
                                     h2so4_molec_cm3=1e7)
        M_dry_0 = float(jnp.sum(Mk[:, :ICOMP_NODIAG]))

        Nk_out, Mk_out, Gc_out = condensation_step(
            Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 0.5, 1.0, 60.0,
            method='ppm_jit')
        M_dry_out = float(jnp.sum(Mk_out[:, :ICOMP_NODIAG]))
        assert M_dry_out > M_dry_0, \
            "Dry mass should increase after H2SO4 condensation"

    def test_cond_gas_decreases(self, xk):
        """Gas-phase H2SO4 should decrease as it condenses."""
        Nk, Mk, Gc = make_lognormal(xk, N_total=1e4, GMD_um=0.1, GSD=1.8,
                                     h2so4_molec_cm3=1e7)
        Gc_so4_0 = float(Gc[SRTSO4])

        _, _, Gc_out = condensation_step(
            Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 0.5, 1.0, 60.0,
            method='ppm_jit')
        assert float(Gc_out[SRTSO4]) < Gc_so4_0, \
            "Gas-phase H2SO4 should decrease after condensation"


class TestMakeStepSmoke:
    """Test the composable make_step API."""

    def test_make_step_coag_cond(self, xk):
        """make_step(['coagulation', 'condensation']) returns correct shapes."""
        step_fn = make_step(['coagulation', 'condensation'], cond_method='ppm_jit')
        Nk, Mk, Gc = make_lognormal(xk, N_total=1e4, GMD_um=0.1, GSD=1.8,
                                     h2so4_molec_cm3=1e7)

        Nk_out, Mk_out, Gc_out = step_fn(
            Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 0.5, 1.0, 60.0)

        assert Nk_out.shape == Nk.shape
        assert Mk_out.shape == Mk.shape
        assert Gc_out.shape == Gc.shape
        assert jnp.all(jnp.isfinite(Nk_out))
        assert jnp.all(jnp.isfinite(Mk_out))

    def test_make_step_all_processes(self, xk):
        """Full 5-process pipeline runs without error."""
        step_fn = make_step(
            ['so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'],
            cond_method='ppm_jit')
        Nk, Mk, Gc = make_lognormal(xk, N_total=1e4, GMD_um=0.1, GSD=1.8,
                                     h2so4_molec_cm3=1e7)

        Nk_out, Mk_out, Gc_out = step_fn(
            Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 0.5, 1.0, 60.0,
            oh_conc=1e6,
            org_conc=1e7, nh3_conc=1.0, fion=0.0,
            kdil=1e-5,
            Nk_bg=jnp.zeros_like(Nk),
            Mk_bg=jnp.zeros_like(Mk),
            Gc_bg=jnp.zeros_like(Gc),
        )

        assert Nk_out.shape == Nk.shape
        assert jnp.all(jnp.isfinite(Nk_out))
        assert jnp.all(jnp.isfinite(Gc_out))
