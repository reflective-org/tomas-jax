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
