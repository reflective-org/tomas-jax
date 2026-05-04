# TOMAS-JAX vs TOMAS-Fortran — Optimization & Parity Audit

**Date**: 2026-04-18 · **Version**: v0.2.0 · **Branch**: `feature/radiative-forcing-jax`

This document answers two questions that come up when presenting TOMAS-JAX to
a new audience:

1. **Is the code fully JAX? Is everything optimized?**
2. **What compromises have been made relative to the Fortran reference — in
   particular, things we compute once and recycle that would be invalid if
   the atmospheric state varied inside a step?**

The numbers in the skill tables refer to the companion statistical benchmark
(49 Latin Hypercube scenarios × 5 process modes). See
`benchmarks/results/24h/stats/` for plots and
`table_ensemble_summary.md` for the full table.

---

## 1. Fortran subroutines → JAX status

All 17 core Fortran subroutines in `tomas_fortran/src/` have pure-JAX
JIT-compilable equivalents. There is no Fortran code reached from any JAX
runtime path.

| Fortran file | JAX module | JIT | Tests |
|---|---|---|---|
| `multicoag.f` (kernel) | `tomas_jax/physics/coagulation_kernel.py` | Yes | via integration |
| `multicoag.f` (rates, time-step) | `tomas_jax/physics/coagulation_rates.py`, `solvers/diffrax.py` | Yes | via integration |
| `multicoag.f` (properties) | `tomas_jax/physics/properties.py` | Yes | 22 |
| `mnfix.f` | `tomas_jax/core/mnfix_jax.py` | Yes | 22 |
| `aerodens.f` | `tomas_jax/physics/density.py` | Yes | 22 |
| `tmcond.f` + `dmdt_int.f` | `tomas_jax/physics/condensation_tfl_jax.py` | Yes | 15 |
| `ezcond.f` | `tomas_jax/physics/ezcond_ppm_jax.py` | Yes | 13 |
| `getCondSink.f` | `tomas_jax/physics/condensation_sink.py` | Yes | 18 |
| `gasdiff.f` | `tomas_jax/physics/gas_properties.py` | Yes | 22 |
| `eznh3eqm.f` | `tomas_jax/physics/nh3_equilibrium.py` | Yes | 20 |
| `ezwatereqm.f` etc. | `tomas_jax/physics/water_equilibrium.py` | Yes | 30 |
| `ricco_nucl.f` | `tomas_jax/physics/nucleation.py` (`ricco_*`) | Yes | 58 |
| `dunne_inorg_nucl.f` | `tomas_jax/physics/nucleation.py` (`dunne_*`) | Yes | 58 |
| `nucleation_driver.f` | `tomas_jax/physics/nucleation.py` (`nucleation_step`) | Yes | 58 |
| `initbounds.f` | `tomas_jax/core/config.py` (`make_grid`) | Yes | — |
| `loginit.f` | `tests/conftest.py` (`make_lognormal`) | Yes | — |

### 1.1 Modules with no Fortran counterpart (JAX-only)

| Module | Purpose | JIT | Tests |
|---|---|---|---|
| `physics/condensation_ppm.py` | PPM Eulerian advection (Colella-Woodward 1984) | Yes | 44 |
| `physics/so2_chemistry.py` | SO₂+OH Troe kinetics (Sun et al. 2022) | Yes | 28 |
| `physics/dilution.py` | First-order relaxation toward background | Yes | 10 |
| `physics/nucleation.py` (zhao2024) | Zhao 2024 11-mechanism NPF | Yes | 58 |
| `physics/bhmie.py` | Bohren-Huffman Mie scattering (`bhmie_jax`, `bhmie_qsca_jax`) | Yes | 63 |
| `physics/radiative_forcing.py` | Direct SW RF (Chylek & Wong 1995) + 32-pt GL upscatter | Yes | 84 |
| `solvers/diffrax.py` | Tsit5 adaptive RK + forward Euler coagulation | Yes | via integration |
| `solvers/condensation.py` | `make_step()` orchestrator + scan-fused loops | Yes | 8 |

**PPM condensation was back-ported TO Fortran** (`tomas_fortran/src_ppm/tmcond_ppm.f`)
for cross-validation — see the 5 PPM columns in the ensemble table.

### 1.2 Legacy non-JIT paths (validation backstops only)

These live in the tree but are not reached from any production path. They
emit `DeprecationWarning`; the JIT benchmarks do not touch them.

| Module | Why kept | Replacement |
|---|---|---|
| `physics/condensation.py` (sequential numpy) | Reference implementation for Fortran divergence debugging | `condensation_tfl_jax.py` |
| `physics/ezcond.py` | Original Fortran port, exact numerical match | `ezcond_ppm_jax.py` |
| `physics/ezcond_ppm.py` | Pure-numpy PPM; 131 s per 24h scenario | `ezcond_ppm_jax.py` |
| `core/mnfix_fortran.py` | Exact Fortran port for debugging | `mnfix_jax.py` |

---

## 2. Compromises & "precomputed-and-recycled" quantities

These are places where we do less work than the Fortran reference by assuming
something is constant over a macrostep. They are accurate whenever that
assumption holds, and must be flagged for users who sweep T/P/RH/composition
inside a step.

| # | Item | File · line | What we assume constant | How to disable | Impact |
|---|---|---|---|---|---|
| 1 | **Coagulation kernel** | `tomas_jax/solvers/diffrax.py:138,162–164` | T, P, RH within a macrostep (kernel computed once, reused across all substeps) | Wrap multiple short calls to `coag_euler_step`; there is no per-substep `recompute_kernel` flag in the default path | Fortran-equivalent. Invalid if T/P vary > ~few K within a timestep. The shorter the macrostep, the less the approximation matters. |
| 2 | **Mie properties precomputation** | `tomas_jax/physics/radiative_forcing.py:231–295` | Particle composition, refractive index, wavelength band — all static at model init | Recall `precompute_mie_properties` whenever composition materially changes | Qsca/Qext/gsca assume pure ammonium sulfate ρ=1770 kg/m³, n=1.4+1e-8j, λ=550 nm. Invalid for evolving composition. |
| 3 | **Upscatter lookup (GL quadrature)** | `tomas_jax/physics/radiative_forcing.py` (GL nodes module-level) | β(g, SZA) precomputed on a grid, then `jnp.interp` lookup at runtime | Resize GL node count (currently 32-pt) | ~1e-4 accuracy vs scipy.quad — verified across 16 (g, SZA) combinations. Byte-identical fig2/fig3; visually identical fig1 (121-byte binary diff). |
| 4 | **Nucleation rate clamping** | `tomas_jax/physics/nucleation.py:493–499, 618–629` | *(Intentional divergence from Fortran)* clamp dN and dM so nucleated particles cannot produce avg mass > bin[0] × 1.1 | Edit/remove the clamp (not recommended) | Without the clamp, organic-dominated nucleation events cause MNFIX to redistribute enormous mass and destabilise the coagulation step that follows. Adds ~5–10 % divergence from Fortran in `full` mode (see table below), but prevents crashes. |
| 5 | **ezcond `moxd = 0`** | `tomas_jax/physics/ezcond_ppm_jax.py` (per CLAUDE.md) | The `c1`/`c2` species-correction terms in `tmcond` are disabled for SO4-only cases | Edit, pass non-zero moxd | Negligible impact for SO₄ runs; documented for completeness. |
| 6 | **Nucleation substep count** | `tomas_jax/physics/nucleation.py:68–78` | J rate is approximately constant across a macrostep when computing adaptive substep count | Tune `max_frac`, `max_substeps` in `compute_nucleation_substeps` | J-rate varies smoothly; impact is low. |

### 2.1 Deliberate divergences from Fortran (physics improvements)

Not bugs — these were chosen for physical correctness, numerical stability,
or because there is no Fortran reference.

- **Cunningham slip factor**: JAX uses the modern formulation; Fortran has a
  typo that shifts the coefficient by a factor ~4 for bins ≤10 nm.
- **Constants**: π, k_B, M_air — JAX uses CODATA 2018; Fortran uses 1980s-era
  truncated values. Sub-ULP impact.
- **PPM condensation**: no Fortran reference existed; back-ported for validation.
- **Organic-mass clamp** (item 4 above): introduced to stabilize the coagulation
  step after large nucleation bursts.

---

## 3. GPU readiness

- **Float64 everywhere** — centralized in `tomas_jax/core/config.py:17`
  (`jax.config.update("jax_enable_x64", True)`), enforced before any JAX
  import. No float32 fallbacks anywhere in the runtime path.
- **`scipy.integrate.quad`** only runs in offline precomputation
  (`radiative_forcing.py::_solar_spectral_weights`, one-time at init) and is
  skipped entirely when callers use the GL-quadrature path added in Phase 5.
- **All runtime JIT paths are GPU-compatible**: no Python loops, no numpy
  array allocations inside `jit`-traced code, no host-side state mutation.
- **Consumer GPU caveat**: float64 throughput on consumer GeForce cards is
  1/32 of float32. Target A100/H100 for production runs; CPU remains fine
  for development.

---

## 4. Out-of-scope / broken / future

| Item | Status | Notes |
|---|---|---|
| VBS / SOA condensation | **Broken** on `vbs-soa` branch | TFL bimodal artifact from `jnp.power()` vs Fortran `**` ULP differences compounding through discrete bin remapping. Needs PPM-only backend rearchitecture. |
| GPU timing | **Not yet measured** | All code is GPU-compatible; benchmarks are CPU-only for now. |
| SO₂ chemistry Fortran parity | **Partial** | Separate benchmark suite (`validate_so2_chemistry.py`); not in the 24 h Fortran vs JAX matrix. |
| Dilution Fortran parity | **Not applicable** | Dilution is JAX-only (no Fortran counterpart). Validated against analytical exponential relaxation. |
| RF Fortran parity | **Not applicable** | Pure-JAX Mie/RF pipeline; no Fortran RF code exists to compare to. |

---

## 5. Ensemble skill snapshot (hour 24 across 49 scenarios)

Source: `benchmarks/results/24h/stats/table_ensemble_summary.md`.
Full statistical plots (Taylor, Q-Q, CDF, per-bin heatmap, error-vs-parameter,
wall-clock, convergence) in `benchmarks/results/24h/stats/`.

| Mode | Method | Nk R² (med) | Nk KGE (med) | \|N_tot relerr\| med | \|N_tot relerr\| p95 |
|---|---|---:|---:|---:|---:|
| coag_only | TFL vs Fortran | 0.9998 | 0.9979 | 1.2e-03 | 9.0e-03 |
| coag_only | PPM vs Fortran | 0.9998 | 0.9979 | 1.2e-03 | 9.0e-03 |
| cond_only | TFL vs Fortran | 0.9986 | 0.9961 | 1.4e-07 | 6.8e-04 |
| cond_only | PPM vs Fortran | 0.9616 | 0.9062 | 1.4e-07 | 7.4e-04 |
| combined  | TFL vs Fortran | 0.9996 | 0.9958 | 1.5e-03 | 4.5e-03 |
| combined  | PPM vs Fortran | 0.9972 | 0.9907 | 2.1e-03 | 3.1e-02 |
| nucl_cond | TFL vs Fortran | 0.9942 | 0.9731 | 1.2e-03 | 1.5e-01 |
| full      | TFL vs Fortran | 0.9928 | 0.8996 | 5.9e-02 | 9.9e-01 |

The `full` mode is where the intentional nucleation-clamping divergence
(compromise #4 above) shows up in the tail: p95 of ~1 means a few scenarios
have completely different total-N at h24. R² stays high because the
*distribution shape* still agrees — it's the overall count that diverges
when a huge nucleation burst triggers different downstream coagulation.

---

## 6. Reproducing the audit

```bash
# 1. Regenerate scenarios (deterministic, seed=42)
python -m benchmarks.python.scenarios

# 2. Fortran reference runs (expensive — hours; skip if tomas_fortran/output/24h exists)
cd tomas_fortran && make benchmark_24h && ./benchmark_24h.exe && cd ..

# 3. JAX runs — all modes, both methods
python -m benchmarks.python.run_24h_scenarios

# 4. Statistical reporting (no simulations — fast)
./benchmarks/run_stats.sh
```

All outputs land in `benchmarks/results/24h/stats/`.

---

## Appendix: figure inventory

| File | Content |
|---|---|
| `fig_A_taylor.png` | Taylor diagrams per mode (σ ratio × correlation × RMSE) |
| `fig_B_qq.png` | Q-Q plot of log₁₀(N_tot_test / N_tot_ref) vs normal quantiles |
| `fig_C_cdf.png` | Empirical CDF of \|N_tot relerr\| and \|M_dry relerr\| |
| `fig_D_perbin_heatmap.png` | Per-bin log-bias, scenarios sorted by GMD |
| `fig_E_error_vs_params.png` | N_tot relerr vs each LHC parameter |
| `fig_F_walltime.png` | Wall-clock: Fortran vs TFL_JIT vs PPM_JIT |
| `fig_G_convergence.png` | JAX-only 36/40/80-bin convergence (low-N, high-N) |
