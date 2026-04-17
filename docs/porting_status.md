# TOMAS Fortran → JAX Port — Status, Gaps, and Roadmap

*As of v0.2.0 (2026-04-13)*

---

## 1. FORTRAN → JAX PORTING STATUS

### 1.1 Fully Ported & Validated (Fortran parity confirmed)

| Fortran File | JAX Module | JIT | Tests | Validation |
|---|---|---|---|---|
| `multicoag.f` (coag kernel) | `coagulation_kernel.py` | Yes | 1 | Kernel values match Fortran |
| `multicoag.f` (coag rates) | `coagulation_rates.py` | Yes | 1 | TFL vectorized, overflow tracked |
| `multicoag.f` (properties) | `properties.py` | Yes | 22 | STP viscosity, MFP, Dpk, Dk, ck |
| `multicoag.f` (time-step) | `euler.py` | Yes | — | Adaptive dt matches Fortran loop |
| `mnfix.f` | `mnfix_jax.py` | Yes | 22 | Partial-transfer matches Fortran exactly |
| `aerodens.f` | `density.py` | Yes | 22 | Tang 1997 fits, pure SO4/NH4/mixed |
| `tmcond.f` + `dmdt_int.f` | `condensation_tfl_jax.py` | Yes | 15 | Matches sequential TFL to 60 steps |
| `ezcond.f` | `ezcond_ppm_jax.py` | Yes | 13 | 3-path driver, mass correction |
| `getCondSink.f` | `condensation_sink.py` | Yes | 18 | CS + sinkfrac, Fuchs-Sutugin |
| `gasdiff.f` | `gas_properties.py` | Yes | 22 | Fuller-Schettler-Giddings, H2SO4 at 298K |
| `eznh3eqm.f` | `nh3_equilibrium.py` | Yes | 20 | 2:1 NH4:SO4 stoichiometry |
| `ezwatereqm.f` + `waterso4.f` + `waternacl.f` | `water_equilibrium.py` | Yes | 30 | ISORROPIA curve fits |
| `ricco_nucl.f` | `nucleation.py` (ricco) | Yes | 58 | 20 test cases match Fortran <1e-12 |
| `dunne_inorg_nucl.f` | `nucleation.py` (dunne) | Yes | 58 | 4 mechanisms, ion-induced |
| `nucleation_driver.f` | `nucleation.py` (step) | Yes | 58 | Adaptive sub-stepping, bin 0 placement |
| `initbounds.f` | `config.py` (`make_grid`) | Yes | — | Mass-doubling, configurable NBINS |
| `loginit.f` | `conftest.py` (`make_lognormal`) | Yes | — | Bimodal lognormal init |

**Summary: 17/17 core Fortran subroutines ported.** All have JAX JIT-compilable equivalents.

### 1.2 New in JAX (No Fortran Counterpart)

| JAX Module | Purpose | JIT | Tests |
|---|---|---|---|
| `condensation_ppm.py` | PPM Eulerian advection (Colella & Woodward 1984) | Yes | 44 |
| `so2_chemistry.py` | SO2+OH Troe kinetics (Sun et al. 2022) | Yes | 28 |
| `dilution.py` | First-order relaxation toward background | Yes | 10 |
| `nucleation.py` (zhao2024) | Zhao et al. 2024 11-mechanism NPF | Yes | 58 |
| `bhmie.py` | Bohren-Huffman Mie scattering | Yes (`bhmie_jax`, `bhmie_qsca_jax`) | 63 |
| `radiative_forcing.py` | Direct SW RF (Chylek & Wong 1995) | Yes (runtime path) | 63 |
| `diffrax.py` | Tsit5 adaptive ODE solver | Yes | — |
| `condensation.py` (solver) | `make_step()` composable orchestrator + scan loops | Yes | 8 |
| `state.py` | TomasState NamedTuple | Yes | — |

**Note:** PPM condensation was back-ported TO Fortran (`tomas_fortran/src_ppm/tmcond_ppm.f`) for cross-validation.

### 1.3 Legacy / Deprecated (kept for validation only)

| Module | Why Deprecated | Replacement |
|---|---|---|
| `condensation.py` (physics) | Sequential numpy, not JIT | `condensation_tfl_jax.py` |
| `ezcond.py` | Sequential numpy, not JIT | `ezcond_ppm_jax.py` |
| `ezcond_ppm.py` | Numpy wrapper, 131s/scenario | `ezcond_ppm_jax.py` |
| `mnfix_fortran.py` | Exact Fortran port for validation | `mnfix_jax.py` |

---

## 2. WHAT HAS BEEN ACCOMPLISHED

### 2.1 Physics Processes (6 of 6 operator-split processes)

1. **SO2 Chemistry** — Sun et al. 2022 Troe formalism, H2O enhancement, diurnal OH
2. **Nucleation** — 3 schemes: Riccobono 2014, Dunne 2016, Zhao 2024 (11 mechanisms)
3. **Coagulation** — Brownian (Fuchs correction), 2 solvers (Tsit5 adaptive, forward Euler)
4. **Condensation** — 4 methods: TFL numpy, TFL JIT, PPM numpy, PPM JIT
5. **Dilution** — First-order exponential relaxation toward background
6. **Equilibria** — NH3 stoichiometric + water hygroscopic (ISORROPIA fits)

### 2.2 Infrastructure

- **Composable API**: `make_step(['so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'])`
- **Scan-fused loops**: Zero Python dispatch — single XLA program for 1440 steps
- **Configurable grid**: 40-bin default (1.7nm-17.5um), `make_grid()` for arbitrary, 80-bin preset
- **Float64 everywhere**: Centralized in `config.py`, enforced project-wide
- **GPU-ready**: All JIT paths documented in `docs/gpu_deployment.md`

### 2.3 Validation

- **367 tests** across 16 test files (703 pass, 350 skip needing Fortran data)
- **50-scenario Latin Hypercube** 24h benchmarks (Fortran vs TFL vs PPM, 5 modes)
- **Nucleation**: 20 parameterization-level cases match Fortran to <1e-12
- **SO2 chemistry**: Validated against Sun et al. 2022 published figures (7 plots)
- **Mass conservation**: PPM at machine precision (8.6e-16), coagulation <1e-8 with overflow

### 2.4 Performance (per 24h scenario)

| Solver | Time | vs Fortran |
|---|---|---|
| Fortran coag | 0.27s | 1.0x |
| JAX Euler coag | 0.15s | **0.57x** (faster) |
| Fortran TFL cond | 0.08s | 1.0x |
| JAX TFL JIT cond | 0.47s | 6.2x slower |
| JAX PPM JIT cond | 0.26s | 3.4x slower |
| Fortran combined | 0.33s | 1.0x |
| JAX PPM combined | 0.41s | 1.27x slower |

---

## 3. WHAT IS NOT DONE

### 3.1 Missing Physics (from `docs/missing_physics.md`)

**Tier 1 — Essential for realistic box model:**

| Physics | Priority | Effort | Notes |
|---|---|---|---|
| VBS organic condensation | HIGH | 1-2 weeks | Saturation-dependent partitioning. Prototype on `vbs-soa` branch but broken |
| Kelvin effect | HIGH | 2-3 days | Curvature correction exp(4sM/RTrDp). Critical for <10nm growth |
| Dry deposition | MEDIUM | 3-5 days | Size-dependent removal (Stokes settling + diffusion) |
| Equilibrium partitioning | MEDIUM | 1 week | Implicit solver for semi-volatile organics |

**Tier 2 — Full atmospheric model:**

| Physics | Priority | Effort |
|---|---|---|
| Emissions (primary + gases) | HIGH | 1 week |
| Wet scavenging (in-cloud + below-cloud) | MEDIUM | 1-2 weeks |
| DMS chemistry (DMS->SO2->H2SO4) | LOW | 3-5 days |
| Aqueous-phase SO4 production | LOW | 1 week |
| Composition-dependent surface tension | LOW | 2-3 days |
| Variable accommodation coefficient | LOW | 1-2 days |

**Tier 3 — Cloud & radiation coupling:**

| Physics | Priority | Effort |
|---|---|---|
| CCN activation (Kohler theory) | MEDIUM | 1 week |
| Ice nucleation (DeMott) | LOW | 1 week |
| Gravitational settling (>1um) | LOW | 2-3 days |

### 3.2 Untested Modules (4 remaining, down from 8)

| Module | Risk | Priority |
|---|---|---|
| `ezcond_ppm_jax.py` | Low — tested indirectly via integration tests | MEDIUM |
| `ezcond.py` | Low — deprecated numpy path | LOW |
| `ezcond_ppm.py` | Low — deprecated numpy path | LOW |
| `condensation.py` (physics) | Low — deprecated numpy path | LOW |

**Recently tested (Phase A hardening):**
- `nh3_equilibrium.py` — 20 tests (stoichiometry, conservation, edge cases, JIT)
- `water_equilibrium.py` — 30 tests (sulfate/seasalt WR, piecewise continuity, JIT)
- `condensation_sink.py` — 18 tests (CS scaling, sinkfrac, NEPS threshold, JIT)
- `mnfix_jax.py` — 22 tests (3 phases, conservation, multi-bin jump, JIT)

### 3.3 VBS/SOA Work (on `vbs-soa` branch, NOT working)

Prototype exists with 50 tests but has fundamental issues — not merged to v0.2.0:
- `vbs_config.py` — 6-bin VBS (C*=[0.01..1000] ug/m3)
- `kelvin_effect.py` — Curvature correction
- `vbs_driving_force.py` — Raoult + Kelvin saturation
- `soa_condensation.py` — Sequential + coupled solvers
- **BROKEN**: TFL bimodal artifacts — ULP-level `jnp.power()` vs Fortran `**` differences compound through discrete bin remapping, producing oscillatory size distributions at tails. Sequential solver's tmcond is the root cause. Coupled (Jacobi) solver is 12x faster and smoother but diverges from Fortran. Needs fresh approach — likely PPM-only backend.

---

## 4. OPTIMIZATIONS NEEDED

### 4.1 Completed (v0.2.0 + `perf/jax-optimization` branch)

- PPM substep loop-invariant hoisting (u_edges, moment integrals)
- Removed `.copy()` on immutable JAX arrays in TFL
- Eliminated unused `jnp.arange(nsteps)` allocation in scan

### 4.2 Remaining Performance Work

| Optimization | Impact | Effort | Notes |
|---|---|---|---|
| GPU benchmarking | HIGH | 2-3 days | `docs/gpu_deployment.md` written but no actual GPU timing |
| `vmap` ensemble batching | HIGH | 2 days | Run 50+ scenarios in parallel on GPU |
| Reverse-mode autodiff | HIGH | 1 day | Enable gradient-based parameter fitting |
| ~~`bhmie.py` to JAX port~~ | ~~LOW~~ | ~~3-5 days~~ | **DONE** — `bhmie_jax`, `bhmie_qsca_jax`, runtime RF JIT'd, vmap precomputation |
| `static_argnums` to `static_argnames` | LOW | 1 day | Cleaner but low risk of bugs |
| Coagulation kernel caching | LOW | 1 day | `kij` recomputed per substep when `recompute_kernel=True` |

### 4.3 Code Quality

| Item | Priority | Notes |
|---|---|---|
| ~~Unit tests for NH3/water equilibria~~ | ~~HIGH~~ | **DONE** — 50 tests added |
| ~~Unit tests for condensation_sink~~ | ~~HIGH~~ | **DONE** — 18 tests added |
| ~~Unit tests for mnfix_jax~~ | ~~MEDIUM~~ | **DONE** — 22 tests added |
| Delete deprecated numpy paths | LOW | After confirming all benchmarks use JIT paths |

---

## 5. ARCHITECTURE SUMMARY

```
Operator splitting per timestep (canonical order):
  SO2 chemistry -> Nucleation -> Coagulation -> Condensation -> Dilution

Each process is JIT-compiled independently.
make_step() composes them into a single traced function.
jax.lax.scan fuses 1440 timesteps into one XLA program.

Grid: 40 mass-doubling bins, 1.7nm-17.5um (configurable)
Species: 44 components (SO4, 41 organics, NH4, H2O) + SO2 in gas array
State: (Nk[40], Mk[40,44], Gc[44], xk[41], temp, pres, boxvol, rh, alpha)
```

### Fortran Source Mapping (complete)

```
Fortran                    ->  JAX (JIT)                    ->  JAX (legacy, deprecated)
------------------------------------------------------------------------------------------
multicoag.f (kernel)       ->  coagulation_kernel.py
multicoag.f (rates)        ->  coagulation_rates.py
multicoag.f (properties)   ->  properties.py
multicoag.f (time-step)    ->  euler.py
mnfix.f                    ->  mnfix_jax.py                  ->  mnfix_fortran.py
aerodens.f                 ->  density.py
tmcond.f + dmdt_int.f      ->  condensation_tfl_jax.py       ->  condensation.py
ezcond.f                   ->  ezcond_ppm_jax.py             ->  ezcond.py, ezcond_ppm.py
getCondSink.f              ->  condensation_sink.py
gasdiff.f                  ->  gas_properties.py
eznh3eqm.f                ->  nh3_equilibrium.py
ezwatereqm.f/waterso4.f   ->  water_equilibrium.py
ricco_nucl.f               ->  nucleation.py (ricco)
dunne_inorg_nucl.f         ->  nucleation.py (dunne)
nucleation_driver.f        ->  nucleation.py (step)
initbounds.f               ->  config.py (make_grid)
loginit.f                  ->  conftest.py (make_lognormal)
(none)                     ->  condensation_ppm.py            (new: PPM advection)
(none)                     ->  so2_chemistry.py               (new: Sun et al. 2022)
(none)                     ->  dilution.py                    (new: entrainment)
(none)                     ->  nucleation.py (zhao2024)       (new: 11-mechanism)
(none)                     ->  bhmie.py, radiative_forcing.py (new: Mie + RF)
(none)                     ->  diffrax.py                     (new: Tsit5 adaptive ODE)
(none)                     ->  condensation.py (solver)       (new: make_step + scan)
```

---

## 6. RECOMMENDED NEXT STEPS (prioritized)

### Phase A — Hardening (1 week)
1. Add unit tests for `nh3_equilibrium.py`, `water_equilibrium.py`, `condensation_sink.py`
2. Add unit tests for `mnfix_jax.py` (direct coverage)

### Phase B — Missing Essential Physics (2-3 weeks)
3. Implement Kelvin effect for sub-10nm growth
4. Add dry deposition (size-dependent)
5. VBS/SOA organic condensation — rewrite or fix the `vbs-soa` branch (currently broken: TFL bimodal artifacts). Needs fresh approach, possibly PPM-only backend.

### Phase C — GPU & Scalability (1-2 weeks)
6. GPU benchmarking (actual timing on A100/H100)
7. `vmap` ensemble runs (50+ scenarios in parallel)
8. Reverse-mode autodiff validation

### Phase D — Extended Physics (2-4 weeks)
9. Emissions framework (primary particles + gas sources)
10. Wet scavenging (in-cloud + below-cloud)
11. DMS chemistry chain
12. CCN activation (Kohler theory)
