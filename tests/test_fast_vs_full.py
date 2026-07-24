"""Accuracy comparison: fast reduced model vs the full 44-species model.

The full model runs with organics/NH3/ions zeroed (enable_organic=0,
nh3_conc=0, fion=0), so its Dunne nucleation is Jbn-only, matching the
fast branch's rate. Remaining documented deviations (docs/gpu_fast.md):

  - nucleation deposit: full = 90% SO4 + 10% organic-from-nothing with
    1:1 gas depletion; fast = 100% SO4 with 98/96 depletion, analytic
    integration instead of <=20 Euler substeps
  - condensation gas->aerosol MW: full 1:1; fast * 96/98 (S-conserving)
  - water: ISORROPIA NH4HSO4 fit vs Tabazadeh 1997 binary (wet size ->
    coag kernel / condensation sink differ at the few-% level)
  - substep policies (adaptive shared vs fixed per-cell)

so agreement is expected at the few-% level, not machine precision.
Marked slow: runs the full model cell-by-cell.
"""
import numpy as np
import pytest

import jax.numpy as jnp

import tomas_jax.core.config as full_cfg
from tomas_jax.fast import FastState, run_fast
from tomas_jax.fast.config import (
    GH2SO4,
    GSO2,
    NBINS,
    SRTSO4,
    xk_boundaries,
)
from tomas_jax.solvers.condensation import make_step

XK = xk_boundaries()
XMID = jnp.sqrt(XK[:-1] * XK[1:])

pytestmark = pytest.mark.slow


def _scenarios(C=4, seed=11, boxvol=1e6):
    rng = np.random.default_rng(seed)
    dp = np.cbrt(np.asarray(XMID) / 1770.0 * 6 / np.pi)
    cells = []
    for _ in range(C):
        gmd = rng.uniform(30e-9, 120e-9)
        gsd = rng.uniform(1.6, 2.0)
        w = np.exp(-0.5 * ((np.log(dp) - np.log(gmd)) / np.log(gsd)) ** 2)
        Nk = rng.uniform(5e2, 2e4) * boxvol * w / w.sum()
        cells.append({
            "Nk": Nk,
            "temp": rng.uniform(220.0, 290.0),
            "pres": rng.uniform(2e4, 1e5),
            "rh": rng.uniform(0.1, 0.8),
            "h2so4": 1e-13,
            "so2": 5e-10,
        })
    return cells


def _run_full(cell, processes, n_steps, dt, oh_conc):
    """Run the 44-species model on one cell, organics/NH3/ions zeroed."""
    step = make_step(processes, cond_method="ppm_jit")
    Nk = jnp.asarray(cell["Nk"])
    Mk = jnp.zeros((NBINS, full_cfg.ICOMP)).at[:, 0].set(Nk * XMID)
    Gc = jnp.zeros(full_cfg.N_GAS_SPECIES)
    Gc = Gc.at[0].set(cell["h2so4"]).at[full_cfg.SRTSO2].set(cell["so2"])
    kwargs = dict(
        oh_conc=cell.get("oh_conc", oh_conc),
        org_conc=0.0, nh3_conc=0.0, fion=0.0,
        enable_organic=0.0, enable_inorganic=1.0,
    )
    temp = jnp.asarray(cell["temp"]); pres = jnp.asarray(cell["pres"])
    boxvol = jnp.asarray(1e6); rh = jnp.asarray(cell["rh"])
    alpha = jnp.asarray(1.0)
    for _ in range(n_steps):
        Nk, Mk, Gc = step(
            Nk, Mk, Gc, XK, temp, pres, boxvol, rh, alpha, dt, **kwargs
        )
    return np.asarray(Nk), np.asarray(Mk), np.asarray(Gc)


def _run_fast_batch(cells, n_steps, dt, oh_conc):
    C = len(cells)
    Nk = jnp.asarray(np.stack([c["Nk"] for c in cells]))
    Mk = jnp.zeros((C, NBINS, 2)).at[..., SRTSO4].set(Nk * XMID)
    Gc = jnp.zeros((C, 2))
    Gc = Gc.at[:, GH2SO4].set(jnp.asarray([c["h2so4"] for c in cells]))
    Gc = Gc.at[:, GSO2].set(jnp.asarray([c["so2"] for c in cells]))
    state = FastState.create(
        Nk, Mk, Gc,
        temp=np.array([c["temp"] for c in cells]),
        pres=np.array([c["pres"] for c in cells]),
        boxvol=1e6,
        rh=np.array([c["rh"] for c in cells]),
    )
    out, diags = run_fast(state, n_steps=n_steps, dt=dt, oh_conc=oh_conc)
    return out, diags


class TestFastVsFull:
    def test_cond_coag_2h(self):
        """Chemistry + coagulation + condensation (no nucleation):
        deviations are water/density/MW/substeps only -> few-%."""
        cells = _scenarios(C=4, seed=11)
        n_steps, dt, oh = 20, 360.0, 1.5e6
        fast_out, _ = _run_fast_batch(cells, n_steps, dt, oh)
        procs = ["so2_chemistry", "coagulation", "condensation"]
        for c, cell in enumerate(cells):
            Nk_f, Mk_f, Gc_f = _run_full(cell, procs, n_steps, dt, oh)
            n_fast = float(jnp.sum(fast_out.Nk[c]))
            n_full = float(np.sum(Nk_f))
            assert n_fast == pytest.approx(n_full, rel=0.02), f"cell {c} N"
            m_fast = float(jnp.sum(fast_out.Mk[c, :, SRTSO4]))
            m_full = float(np.sum(Mk_f[:, 0]))
            # 96/98 condensation MW conversion accounts for up to ~2%
            assert m_fast == pytest.approx(m_full, rel=0.05), f"cell {c} M"
            # gas trajectories
            assert float(fast_out.Gc[c, GSO2]) == pytest.approx(
                float(Gc_f[full_cfg.SRTSO2]), rel=1e-6
            ), f"cell {c} SO2"

    def test_full_pipeline_2h(self):
        """With nucleation: deposit-convention differences add a bit more
        spread in total number; dry mass stays within a few %."""
        cells = _scenarios(C=4, seed=12)
        for cell in cells:
            cell["h2so4"] = 5e-13  # stronger nucleation forcing
            cell["temp"] = min(cell["temp"], 250.0)
        n_steps, dt, oh = 20, 360.0, 2e6
        fast_out, _ = _run_fast_batch(cells, n_steps, dt, oh)
        procs = ["so2_chemistry", "nucleation", "coagulation", "condensation"]
        for c, cell in enumerate(cells):
            Nk_f, Mk_f, Gc_f = _run_full(cell, procs, n_steps, dt, oh)
            m_fast = float(jnp.sum(fast_out.Mk[c, :, SRTSO4]))
            m_full = float(np.sum(Mk_f[:, :full_cfg.ICOMP_NODIAG].sum()))
            assert m_fast == pytest.approx(m_full, rel=0.10), f"cell {c} M"
            n_fast = float(jnp.sum(fast_out.Nk[c]))
            n_full = float(np.sum(Nk_f))
            assert n_fast == pytest.approx(n_full, rel=0.15), f"cell {c} N"
