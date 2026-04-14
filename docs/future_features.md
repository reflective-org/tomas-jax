# Future Features & Improvements

This document tracks features that are not currently needed but would unlock significant capabilities. Each item describes what the change is, what it unlocks, and an estimate of the effort involved.

---

## 1. Reverse-Mode Autodiff Through Condensation

**What:** Replace the `jax.lax.fori_loop` with dynamic bounds in `condensation_ppm.py` (`ppm_condensation_step`) with a fixed-length `jax.lax.scan` + active-step masking. The dynamic bound (`n_sub` from CFL substepping) prevents JAX from building the backward computation graph.

**Change:** ~1 function in `condensation_ppm.py`. Add `MAX_SUBSTEPS` static parameter, use `scan` with `jnp.where(step_idx < n_sub, new_state, old_state)` masking.

**Unlocks:**
- **Gradient-based parameter fitting** — backprop through full 24h simulations to fit accommodation coefficient, nucleation rates, or emission factors to observations
- **Hybrid physics-ML models** — embed a neural network inside a TOMAS timestep (e.g., learned condensation sink) and train end-to-end
- **Optimal control** — gradient descent on emission trajectories to minimize PM2.5 exposure
- **Adjoint sensitivity analysis** — one backward pass gives gradients w.r.t. all parameters simultaneously, vs. N+1 forward runs for finite differences

**Effort:** ~1 day + regression testing

**Prerequisite:** A concrete ML architecture that requires backprop through the physics. If training a standalone surrogate model (supervised learning on forward-simulation data), this is not needed.

---

## 2. GPU/TPU Acceleration

**What:** The current codebase runs on CPU. JAX natively supports GPU/TPU execution with zero code changes for most operations, but some patterns (e.g., `fori_loop` iteration counts, host callbacks) may need adjustment.

**Unlocks:**
- **10-100x speedup** for batch scenario runs (vmap over initial conditions)
- **Massive ensemble generation** — thousands of 24h simulations in parallel for ML training data
- **Real-time or near-real-time** aerosol modeling for operational forecasting

**Effort:** ~2-3 days for profiling, fixing any GPU-incompatible patterns, and benchmarking. Main risks: memory layout for the 44-component mass array, `fori_loop` performance on GPU vs. scan.

---

## 3. Batched Scenario Execution via `jax.vmap`

**What:** Vectorize the full timestep (or `make_step()`) over an ensemble dimension using `jax.vmap`. Currently each scenario runs sequentially.

**Unlocks:**
- **Parallel scenario sweeps** — run 50-1000 scenarios simultaneously on one GPU
- **Monte Carlo uncertainty quantification** — sample parameter distributions and propagate through the model in one batched call
- **Fast ML training data generation** — generate millions of (input, output) pairs per hour

**Effort:** ~2 days. Main work: ensure all functions accept batched inputs cleanly (no Python-level conditionals on array values). The modular `make_step()` API already provides a clean entry point for this.

**Prerequisite:** GPU acceleration (feature #2) to actually benefit from the parallelism.

---

## 4. Multi-Species Condensation with VBS Partitioning

**What:** Extend condensation beyond H2SO4-only to handle semi-volatile organic species using the Volatility Basis Set (VBS) framework. Each organic "bin" has a saturation concentration; partitioning depends on the Kelvin effect and absorptive mass.

**Unlocks:**
- **Realistic SOA formation** — secondary organic aerosol is the dominant fine-particle source in many environments
- **Kelvin effect on nanoparticles** — critical for accurately modeling new particle growth in the 1-10 nm range
- **Multi-component growth rates** — needed for comparison with field measurements (e.g., CLOUD chamber, boreal forests)

**Effort:** ~1-2 weeks. Requires new physics modules (VBS partitioning, Kelvin effect calculation) and extending the condensation pipeline to loop over condensable species. The `_condensation_step_core` architecture makes this straightforward — add a species loop around the existing CS + ezcond pattern.

---

## 5. Learned Surrogate for Coagulation Kernel

**What:** Replace the O(N^2) Brownian coagulation kernel computation with a neural network surrogate trained on forward-simulation data.

**Unlocks:**
- **Dramatic speedup** for the coagulation step — the kernel matrix computation dominates wall time for Tsit5 adaptive solves
- **Amortized cost** — a learned kernel that generalizes across temperature/pressure conditions avoids recomputing the full matrix each step
- **Foundation for online learning** — update the surrogate during simulation based on residuals

**Effort:** ~1-2 weeks for data generation, architecture search, training, and validation against the exact kernel. Risk: ensuring the surrogate preserves mass/number conservation properties.

**Prerequisite:** Feature #3 (batched execution) for efficient training data generation.

---

## 6. Emission and Deposition Modules

**What:** Add size-resolved emission sources (e.g., sea salt, dust, primary organic) and removal processes (dry deposition, wet scavenging, gravitational settling).

**Unlocks:**
- **Closed mass budget** — currently the model only has H2SO4 production; real simulations need sources and sinks for all species
- **Realistic lifetime predictions** — particle lifetime depends critically on size-dependent removal rates
- **Comparison with CTM output** — needed to couple TOMAS-JAX with chemical transport models

**Effort:** ~1 week per process. The `make_step()` API is designed for this — just add `'emission'` and `'deposition'` as valid process names.

---

## 7. Adaptive Timestepping

**What:** Dynamically adjust `dt` based on the fastest process rate (e.g., nucleation burst, rapid condensational growth). Currently `dt` is fixed at 60s.

**Unlocks:**
- **Accuracy during fast events** — nucleation bursts can produce 10^5 particles/cm3/s; 60s steps miss the dynamics
- **Efficiency during slow periods** — overnight hours with low photochemistry could use 300s+ steps
- **Stiff-system handling** — avoid operator-splitting errors when process timescales differ by orders of magnitude

**Effort:** ~3-5 days. Main challenge: maintaining scan-fused XLA compilation with variable step counts. May require the same `scan` + masking pattern as feature #1.

---

## 8. Explicit ELVOC/HOM Gas-Phase Species

**What:** Add ELVOC (Extremely Low Volatility Organic Compounds) and HOM (Highly Oxygenated Molecules) as explicit tracked species in the TOMAS-JAX gas array (`Gc`), distinct from the existing bulk organics (indices 1–41). This would split the organic gas pool into volatility classes (e.g., ULVOC, ELVOC, LVOC) or O:C ratio classes.

**Context / Why this came up:**
The Zhao 2024 11-mechanism NPF scheme (mechanisms 6–7) uses pure-biogenic nucleation from Kirkby et al. 2016 (Nature 533, 521–526), parameterized with [HOM] in units of 10⁶ cm⁻³. The current implementation passes `ulvoc` as an external scalar — it is not part of `Gc` and is not depleted by nucleation or condensation. Additionally, Kirkby 2016 does not state the units of the ion concentration [n±] explicitly (back-calculated as cm⁻³ from Fig. 3 data), and the common 36% ELVOC fraction approximation (Tröstl et al. 2016, Nature 533, 527–531) has large uncertainty across environments.

**Unlocks:**
- Precise mass balance for nucleation and condensation of distinct organic classes
- Sensitivity studies: how does the ELVOC fraction affect NPF rate vs. condensational growth?
- Proper depletion of the ULVOC/ELVOC pool during nucleation events
- Compatibility with HOM-resolving chemistry schemes (e.g., MCM, GECKO-A outputs)

**Action needed:**
- Follow up with Jeff Pierce (Colorado State) and the CLOUD/ACTRIS-NF community on best practice for treating ELVOC/HOM in sectional models
- Review Tröstl et al. 2016 for the ELVOC fraction basis (36%)
- Check whether CAM-TOMAS, GLOMAP-mode, or EMAC/GMXe have adopted explicit HOM tracers
- Extend `config.py` from 43 → N gas species with new ULVOC/ELVOC entries
- Update nucleation and condensation drivers to deplete the appropriate gas pool

**Effort:** ~1–2 weeks. Main work: extending the gas array, updating all nucleation/condensation call sites, and adding test cases.

**Current workaround:** `ulvoc` passed as external scalar; 36% ELVOC fraction applied outside the model if needed. Documented as approximation in `docs/zhao2024_nucleation.md`.

---

## Priority Guide

| Priority | Feature | Reason |
|----------|---------|--------|
| Do first when needed | #1 Reverse-mode AD | Small, localized change; prerequisite for any gradient-based ML |
| Do first when needed | #3 Batched vmap | Immediate payoff for ML data generation |
| High value | #2 GPU acceleration | Multiplier for everything else |
| High value | #4 Multi-species condensation | Scientific impact; needed for realistic SOA |
| Medium | #6 Emissions & deposition | Required for full atmospheric simulations |
| Medium | #7 Adaptive timestepping | Accuracy/efficiency improvement |
| Exploratory | #5 Learned surrogate | Research project; depends on #1 and #3 |
