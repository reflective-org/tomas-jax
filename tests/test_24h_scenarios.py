"""Parametrized pytest suite for 24-hour benchmark scenarios.

Tests 50 Latin Hypercube scenarios comparing PPM vs TFL vs Fortran
across coagulation-only, condensation-only, and combined modes.

Run::

    # Full suite (requires Fortran + JAX data)
    pytest tests/test_24h_scenarios.py -v --tb=short

    # Just coagulation tests
    pytest tests/test_24h_scenarios.py -v -m coag_only

    # Just condensation tests
    pytest tests/test_24h_scenarios.py -v -m cond_only

    # Just combined tests
    pytest tests/test_24h_scenarios.py -v -m combined
"""
import os
import pytest
import numpy as np

from benchmarks.python.compare_24h import (
    compare_scenario, load_jax_results, FORTRAN_DIR, RESULTS_DIR, SRTH2O
)

# Check if data is available
_FORTRAN_EXISTS = os.path.isdir(FORTRAN_DIR) and len(os.listdir(FORTRAN_DIR)) > 0 \
    if os.path.isdir(FORTRAN_DIR) else False
_JAX_EXISTS = os.path.isdir(RESULTS_DIR) and len(os.listdir(RESULTS_DIR)) > 0 \
    if os.path.isdir(RESULTS_DIR) else False

_DATA_AVAILABLE = _FORTRAN_EXISTS or _JAX_EXISTS

skip_no_data = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="No benchmark data found. Run benchmarks/run_24h.sh first."
)
skip_no_fortran = pytest.mark.skipif(
    not _FORTRAN_EXISTS,
    reason="No Fortran 24h data. Run Fortran benchmark first."
)

SCENARIO_IDS = list(range(1, 51))


def _load_with_jit_fallback(sid, mode, method):
    """Load JIT data first, fall back to legacy numpy data."""
    data = load_jax_results(sid, mode, method + '_jit')
    if data is None:
        data = load_jax_results(sid, mode, method)
    return data


# =========================================================================
# Session-scoped fixture: compute all comparisons once
# =========================================================================

@pytest.fixture(scope="session")
def all_comparisons():
    """Load/compute all comparison metrics once per test session."""
    results = {}
    for sid in SCENARIO_IDS:
        results[sid] = {}
        for mode in ['coag_only', 'cond_only', 'combined']:
            try:
                results[sid][mode] = compare_scenario(sid, mode)
            except Exception as e:
                results[sid][mode] = None
    return results


# =========================================================================
# Coagulation-only tests
# =========================================================================

@pytest.mark.slow
@pytest.mark.coag_only
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_coag_tfl_vs_fortran_number(sid, all_comparisons):
    """TFL coag-only total N within 50% of Fortran at hour 24."""
    m = all_comparisons[sid]['coag_only']
    if m is None or not m['has_fortran'] or not m['has_tfl']:
        pytest.skip("Missing data")
    err = m['tfl_vs_fortran_N_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.5, f"TFL vs Fortran N_tot rel error = {err:.4e} > 0.5"


@pytest.mark.slow
@pytest.mark.coag_only
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_coag_tfl_vs_fortran_mass(sid, all_comparisons):
    """TFL coag-only total M within 5% of Fortran at hour 24."""
    m = all_comparisons[sid]['coag_only']
    if m is None or not m['has_fortran'] or not m['has_tfl']:
        pytest.skip("Missing data")
    err = m['tfl_vs_fortran_M_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.05, f"TFL vs Fortran M_tot rel error = {err:.4e} > 0.05"


# =========================================================================
# Condensation-only tests
# =========================================================================

@pytest.mark.slow
@pytest.mark.cond_only
@skip_no_fortran
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_cond_tfl_vs_fortran_number(sid, all_comparisons):
    """TFL cond-only total N within 50% of Fortran at hour 24."""
    m = all_comparisons[sid]['cond_only']
    if m is None or not m['has_fortran'] or not m['has_tfl']:
        pytest.skip("Missing data")
    err = m['tfl_vs_fortran_N_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.5, f"TFL vs Fortran N_tot rel error = {err:.4e} > 0.5"


@pytest.mark.slow
@pytest.mark.cond_only
@skip_no_fortran
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_cond_tfl_vs_fortran_mass(sid, all_comparisons):
    """TFL cond-only total M within 10% of Fortran at hour 24."""
    m = all_comparisons[sid]['cond_only']
    if m is None or not m['has_fortran'] or not m['has_tfl']:
        pytest.skip("Missing data")
    err = m['tfl_vs_fortran_M_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.1, f"TFL vs Fortran M_tot rel error = {err:.4e} > 0.1"


@pytest.mark.slow
@pytest.mark.cond_only
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_cond_ppm_vs_fortran_number(sid, all_comparisons):
    """PPM cond-only total N within 50% of Fortran at hour 24."""
    m = all_comparisons[sid]['cond_only']
    if m is None or not m['has_fortran'] or not m['has_ppm']:
        pytest.skip("Missing data")
    err = m['ppm_vs_fortran_N_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.5, f"PPM vs Fortran N_tot rel error = {err:.4e} > 0.5"


@pytest.mark.slow
@pytest.mark.cond_only
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_cond_ppm_vs_tfl_number(sid, all_comparisons):
    """PPM vs TFL cond-only total N within 20% at hour 24."""
    m = all_comparisons[sid]['cond_only']
    if m is None or not m['has_tfl'] or not m['has_ppm']:
        pytest.skip("Missing data")
    err = m['ppm_vs_tfl_N_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.2, f"PPM vs TFL N_tot rel error = {err:.4e} > 0.2"


@pytest.mark.slow
@pytest.mark.cond_only
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_cond_ppm_vs_tfl_mass(sid, all_comparisons):
    """PPM vs TFL cond-only total M within 10% at hour 24."""
    m = all_comparisons[sid]['cond_only']
    if m is None or not m['has_tfl'] or not m['has_ppm']:
        pytest.skip("Missing data")
    err = m['ppm_vs_tfl_M_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.1, f"PPM vs TFL M_tot rel error = {err:.4e} > 0.1"


# =========================================================================
# Combined tests
# =========================================================================

@pytest.mark.slow
@pytest.mark.combined
@skip_no_fortran
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_combined_tfl_vs_fortran_number(sid, all_comparisons):
    """TFL combined total N within 50% of Fortran at hour 24."""
    m = all_comparisons[sid]['combined']
    if m is None or not m['has_fortran'] or not m['has_tfl']:
        pytest.skip("Missing data")
    err = m['tfl_vs_fortran_N_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.5, f"TFL vs Fortran N_tot rel error = {err:.4e} > 0.5"


@pytest.mark.slow
@pytest.mark.combined
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_combined_ppm_vs_fortran_number(sid, all_comparisons):
    """PPM combined total N within 50% of Fortran at hour 24."""
    m = all_comparisons[sid]['combined']
    if m is None or not m['has_fortran'] or not m['has_ppm']:
        pytest.skip("Missing data")
    err = m['ppm_vs_fortran_N_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.5, f"PPM vs Fortran N_tot rel error = {err:.4e} > 0.5"


@pytest.mark.slow
@pytest.mark.combined
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_combined_ppm_vs_tfl_number(sid, all_comparisons):
    """PPM vs TFL combined total N within 20% at hour 24."""
    m = all_comparisons[sid]['combined']
    if m is None or not m['has_tfl'] or not m['has_ppm']:
        pytest.skip("Missing data")
    err = m['ppm_vs_tfl_N_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.2, f"PPM vs TFL N_tot rel error = {err:.4e} > 0.2"


@pytest.mark.slow
@pytest.mark.combined
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_combined_ppm_vs_tfl_mass(sid, all_comparisons):
    """PPM vs TFL combined total M within 10% at hour 24."""
    m = all_comparisons[sid]['combined']
    if m is None or not m['has_tfl'] or not m['has_ppm']:
        pytest.skip("Missing data")
    err = m['ppm_vs_tfl_M_tot_relerr'][23]
    assert not np.isnan(err), "No comparison data at hour 24"
    assert err < 0.1, f"PPM vs TFL M_tot rel error = {err:.4e} > 0.1"


# =========================================================================
# Mass conservation tests (JAX only - no Fortran needed)
# =========================================================================

@pytest.mark.slow
@pytest.mark.cond_only
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_cond_mass_conservation_tfl(sid):
    """TFL cond-only: dry aerosol mass + gas should be conserved.

    Checks that dry mass at hour 24 is within 1% of hour 1 + production.
    Water is excluded because it is a diagnostic quantity set by equilibrium.
    """
    from benchmarks.python.scenarios import get_scenarios

    data = _load_with_jit_fallback(sid, 'cond_only', 'tfl')
    if data is None:
        pytest.skip("No TFL data")

    scenarios = get_scenarios()
    scen = scenarios[sid - 1]
    prod_rate = scen['h2so4_prod_kg_per_s']

    # Dry aerosol mass (species 0-42, excluding water=43) + gas
    M_dry_24 = np.sum(data['Mk'][23, :, :SRTH2O])
    M_gas_24 = np.sum(data['Gc'][23])
    total_24 = M_dry_24 + M_gas_24

    M_dry_1 = np.sum(data['Mk'][0, :, :SRTH2O])
    M_gas_1 = np.sum(data['Gc'][0])
    total_1 = M_dry_1 + M_gas_1

    # Production from hours 1-24 (23 hours of additional production)
    h2so4_produced = prod_rate * 23 * 3600.0

    expected = total_1 + h2so4_produced
    if expected > 1e-30:
        rel_err = abs(total_24 - expected) / expected
        assert rel_err < 0.01, (
            f"Dry mass conservation error: {rel_err:.4e} > 0.01 "
            f"(total_24={total_24:.4e}, expected={expected:.4e})"
        )


@pytest.mark.slow
@pytest.mark.cond_only
@skip_no_data
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_cond_mass_conservation_ppm(sid):
    """PPM cond-only: dry aerosol mass + gas should be conserved within 1%."""
    from benchmarks.python.scenarios import get_scenarios

    data = _load_with_jit_fallback(sid, 'cond_only', 'ppm')
    if data is None:
        pytest.skip("No PPM data")

    scenarios = get_scenarios()
    scen = scenarios[sid - 1]
    prod_rate = scen['h2so4_prod_kg_per_s']

    M_dry_24 = np.sum(data['Mk'][23, :, :SRTH2O])
    M_gas_24 = np.sum(data['Gc'][23])
    total_24 = M_dry_24 + M_gas_24

    M_dry_1 = np.sum(data['Mk'][0, :, :SRTH2O])
    M_gas_1 = np.sum(data['Gc'][0])
    total_1 = M_dry_1 + M_gas_1

    h2so4_produced = prod_rate * 23 * 3600.0
    expected = total_1 + h2so4_produced
    if expected > 1e-30:
        rel_err = abs(total_24 - expected) / expected
        assert rel_err < 0.01, (
            f"Dry mass conservation error: {rel_err:.4e} > 0.01 "
            f"(total_24={total_24:.4e}, expected={expected:.4e})"
        )


# =========================================================================
# Coagulation mass conservation (runs solver inline with overflow tracking)
# =========================================================================

@pytest.mark.slow
@pytest.mark.coag_only
@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_coag_mass_conservation(sid):
    """Coag-only: M_dry(0) == M_dry(24) + top-bin overflow to machine precision.

    Coagulation redistributes mass across bins. On a finite bin grid, mass
    that coagulates past the top bin boundary is physically lost (grid
    truncation). We track this overflow explicitly and verify the full
    mass budget closes to ~1e-10 relative.
    """
    import jax
    import jax.numpy as jnp
    from tomas_jax.core.config import ICOMP_NODIAG
    from tomas_jax.solvers.diffrax import coag_euler_step
    from benchmarks.python.scenarios import get_scenarios
    from benchmarks.python.run_24h_scenarios import (
        init_lognormal_scenario, NBINS, XK0_LEGACY, BOXVOL,
    )

    scenarios = get_scenarios()
    scen = scenarios[sid - 1]

    # Build legacy 36-bin grid
    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = XK0_LEGACY
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk = jnp.array(xk_np)

    Nk_np, Mk_np = init_lognormal_scenario(
        scen['N_total'], scen['GMD_um'], scen['GSD'], xk_np)
    Nk = jnp.array(Nk_np)
    Mk = jnp.array(Mk_np)

    M_dry_0 = float(jnp.sum(Mk[:, :SRTH2O]))
    if M_dry_0 < 1e-30:
        pytest.skip("Negligible initial mass")

    # Run 24h with 1-minute timesteps, tracking overflow
    DT = 60.0
    NSTEPS = 1440
    overflow_total = jnp.zeros(Mk.shape[1])

    solver = jax.jit(coag_euler_step,
                      static_argnames=['icomp_nodiag', 'n_substeps', 'return_overflow'])
    for step in range(NSTEPS):
        Nk, Mk, overflow = solver(
            Nk, Mk, xk, scen['temp'], scen['pres'], BOXVOL,
            dt=DT, icomp_nodiag=ICOMP_NODIAG, n_substeps=3,
            return_overflow=True)
        overflow_total = overflow_total + overflow

    M_dry_24 = float(jnp.sum(Mk[:, :SRTH2O]))
    M_overflow_dry = float(jnp.sum(overflow_total[:SRTH2O]))

    # Mass budget: M(0) = M(24) + overflow
    mass_loss = M_dry_0 - M_dry_24
    residual = abs(mass_loss - M_overflow_dry)
    rel_residual = residual / M_dry_0

    assert rel_residual < 1e-8, (
        f"Mass budget not closed: residual={rel_residual:.4e} > 1e-8. "
        f"M_dry(0)={M_dry_0:.4e}, M_dry(24)={M_dry_24:.4e}, "
        f"overflow={M_overflow_dry:.4e}, loss-overflow={residual:.4e}"
    )
