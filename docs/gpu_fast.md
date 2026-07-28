# GPU-Fast Reduced Model (`tomas_jax.fast`)

A natively-batched recomposition of the TOMAS microphysics for global-model
coupling: run O(10⁶) independent grid cells per call on a datacenter GPU,
float64 throughout. Lives on the `gpu-fast` branch; the full 44-species
model is untouched.

**Physics**: 40 bins (same 1.7 nm mass-doubling grid), aerosol = SO4 + H2O
only, gases = H2SO4 + SO2 only. Processes per step: SO2+OH chemistry
(Sun et al. 2022), Dunne 2016 **neutral-binary** nucleation (Jbn channel
only — no ions, NH3, or organics), Brownian coagulation, PPM condensation,
and **Tabazadeh (1997)** H2SO4/H2O water equilibrium. No NH3 equilibrium,
no Riccobono/Zhao nucleation, no dilution, no radiative forcing (handled by
the host model).

## Usage

```python
from tomas_jax.fast import FastState, run_fast

state = FastState.create(
    Nk,            # (C, 40)      [#/cell]
    Mk,            # (C, 40, 2)   [kg/cell]  columns: SO4=0, H2O=1
    Gc,            # (C, 2)       [kg/cell]  H2SO4=0, SO2=1
    temp=temp_C,   # (C,) [K]     (scalars broadcast)
    pres=pres_C,   # (C,) [Pa]
    boxvol=vol_C,  # (C,) [cm^3]
    rh=rh_C,       # (C,) fraction 0-1
)

out, diags = run_fast(
    state, n_steps=60, dt=360.0,          # 6 h
    oh_conc=oh_profile,                    # see forcing shapes below
    so2_prod=so2_src_C,                    # [kg/cell/s], optional
    n_cell_chunks=8,                       # REQUIRED at C ~ 1M (memory)
    sort_by_coag_cost=True,                # see Performance
)
```

Forcings (`oh_conc`, `h2so4_prod`, `so2_prod`) each accept:

| shape | meaning |
|---|---|
| scalar | constant in time, uniform over cells |
| `(C,)` | constant in time, per cell |
| `(n_steps, 1)` | time profile (e.g. diurnal OH), uniform over cells |
| `(n_steps, C)` | time profile, per cell — row `t` applies to step `t` |

Time-varying forcings are fed through the scan. To build a diurnal OH
profile, `physics/so2_chemistry.calc_solar_zenith_angle` +
`calc_oh_concentration` are reusable (pure JAX).

`diags` holds per-step totals (N_tot, dry SO4 mass, gas totals), the
coagulation top-bin overflow (for mass-budget closure), and the substep
cap-hit flags. For single steps (e.g. called from a GCM driver each
transport step) use `fast_step(state, dt, oh_conc=...)` /
`make_fast_step()` — OH is a per-call argument there, so the host model
supplies fresh per-cell OH every step.

Benchmark harness:

```bash
# GPU target: 1M cells x 6 h in < 10 s
python -m benchmarks.python.bench_fast_1m --cells 1000000 --hours 6 \
    --n-cell-chunks 8 --sort-by-coag-cost

# CPU validation (no GPU)
python -m benchmarks.python.bench_fast_1m --cells 1000 --hours 1 --cpu-smoke
```

## Architecture: "vmap the math, batch the control"

Everything carries a leading cell axis `C`. Per-cell math kernels are
reused from the full model **unchanged** via `jax.vmap`
(`calc_coagulation_kernel`, `calc_coagulation_rates`, the PPM
reconstruction/flux/advection functions, `calc_k1_so2_oh` — all already
shape-agnostic). Control decisions that must be scalars (loop trip counts)
are **global reductions over the batch**, which is why the step is
natively batched instead of `vmap(step)`: under vmap, per-cell traced trip
counts become batch-max masked loops with no way to share work.

The time loop is `jit(lax.scan)` segments (default 10 outer steps) with
the state donated between segments; cells are independent, so chunking
over the cell axis is exact.

### The three restructurings that make 1M cells feasible

1. **Vectorized MNFIX** (`fast/mnfix.py`). The sequential
   `core/mnfix_jax.py` runs 3×40 dependent `fori_loop` iterations per
   call and is called ~6+ times per step — at 1M cells that alone would
   cost seconds per step. The fast version runs phases 1-2 (empty-bin
   reset, extreme trim) as exact boolean-mask updates and phase 3
   (partial transfer) as an elementwise shift computation + one-hot
   conservative scatter, swept a fixed 2×. Number and mass conservation
   are exact by construction; on isolated drifted bins it is bit-equal to
   the sequential version (tests/test_fast_mnfix.py). Depth: ~6 fused
   kernels total.

2. **Shared adaptive substep counts.** Per-cell adaptive loops are
   replaced by one shared count = min(batch-max requirement, static cap):
   - *PPM condensation* (`fast/condensation.py`): per-cell CFL count from
     `compute_substeps`, shared `n_glob = min(max, 40)`. Quiet cells run
     at a smaller Courant number — strictly more accurate.
   - *Coagulation* (`fast/coagulation.py`): forward Euler is only
     accurate for `dt_sub * lambda << 1`, where `lambda` is the per-bin
     loss frequency `kij_kk*N_k + sum_{j>k} kij_kj*N_j + 2*K1M_k/xk_k`
     (self-coagulation + scavenging by larger + TFL promotion by the mass
     flux from below). `n_sub = clip(ceil(dt*max(lambda)/c_max), 1, 256)`
     with `c_max = 0.05`. Calibration (see below) shows fixed coarse
     substeps are catastrophically wrong; this criterion keeps
     content-significant bins within ~1% of a fully converged reference.
   - Cap hits are surfaced per step (`cond_cap_hit` / `coag_cap_hit`) —
     they mean the stiffest cells ran coarser than the accuracy target.

3. **Analytic nucleation** (`fast/nucleation.py`). With T fixed over the
   step, gas under Jbn depletion obeys `dG/dt = -A G^3.95`, which has a
   closed form — the converged limit of the full model's ≤20-substep
   Euler loop (validated against a 10⁵-substep reference to 1e-4; a fixed
   4-substep Euler over-nucleates by ~50% in burst scenarios). No loop,
   no per-substep MNFIX, never over-consumes gas.

## Documented physics deviations from the full model

| # | Deviation | Why |
|---|---|---|
| 1 | Nucleation deposits **100% SO4** (full: 90% SO4 + 10% organic created from nothing) | no organics in the reduced model; removes mass creation |
| 2 | Nucleation gas depletion ×98/96; **condensation aerosol deposit ×96/98** (`S_CONSERVING_CONDENSATION`, default True; full model moves kg 1:1 in both) | exact total-sulfur closure; the 1:1 convention creates S at 2% of the flux |
| 3 | Condensed mass deposits only onto bins with particles **after** PPM transport (full: pre-transport sink fractions, stranding mass in drained bins that the next MNFIX destroys — worst case ~2%/step of a cell's mass at dt=360 s) | conserves the condensed mass |
| 4 | Water: Tabazadeh 1997 binary wt%(T, RH) (full: ISORROPIA ammonium-bisulfate fit) — smooth extrapolation above 260 K (~43 wt% at 298 K/RH50, matching isopiestic data); RH clamps → wt ∈ [10, 80] | correct binary H2SO4/H2O hygroscopicity for a sulfate-only model |
| 5 | Density: Tang (1997) pure-H2SO4 branch (exactly the full `aerodens` limit at NH4=org=0) | full `calc_density` reads out-of-range columns at ICOMP=2 |
| 6 | Water equilibrium also runs **before coagulation** (full: kernel sees the previous step's water) | fresh nucleated/dry mass gets correct wet size |
| 7 | Analytic nucleation integration (see above) | more accurate and faster than substepping |
| 8 | Adaptive shared coag substeps (full: fixed 3-10 per 60 s step) | see calibration; fixed counts at dt=360 s lose >10% mass in burst cells via the positivity clamp |

Accuracy vs the full model (organics/NH3/ions zeroed, 2 h, heterogeneous
cells; `tests/test_fast_vs_full.py`): total dry SO4 mass agrees to
≤ 1.5e-4 relative; total number to ≤ 1.9% (dominated by the different
water scheme feeding the coagulation kernel).

## Conservation

Total sulfur `S(SO2) + S(H2SO4) + S(SO4) + S(coag overflow)` closes to
**< 1e-9 relative** over multi-hour heterogeneous runs
(tests/test_fast_step.py). The remaining floor is MNFIX's Fortran-faithful
empty-bin seeding (`NEPS * xk_geo` SO4 whenever coagulation re-empties a
bin — identical in the full model), which accumulates to ~1e-8..1e-7 in
very clean cells (~10/cm³); the benchmark gates at 1e-7.

## Coagulation substep calibration (why adaptive, why c_max=0.05)

Measured on 360 s steps, closure = |ΔM|/M with overflow accounted:

| state | fixed 3×120 s | dt·λ/0.05 adaptive |
|---|---|---|
| jagged high-N (adversarial) | −12% mass, clamp +73% | closure 6e-11, sig-bin error ≤2% |
| nucleation burst (1e6/cm³ in bin 0) | — | closure 3e-8, sig-bin error ≤~1% (bins holding <1e-6 of content converge slower — as in the full model at 6 s substeps) |
| realistic lognormals (1e3–1e5/cm³) | closure ~1e-4..1e-1 | n_sub = 1-15, closure ≤3.5e-8, dist. error ≤0.8% |

Two rejected shortcuts, for the record: masking near-empty bins out of the
λ criterion loses 3% of mass on adversarial states (a bin with negligible
content can still carry the whole promotion flux), and net-rate-based
criteria miss gain-dominated stiffness entirely.

## Performance

Measured (Apple M-series CPU, float64): ~7,200 cell-steps/s at 10k cells —
e.g. 10k cells × 6 h in 84 s, with zero cap hits on realistic heterogeneous
scenarios. No GPU measurements exist yet; run `bench_fast_1m.py` on the
target GPU.

Cost anatomy at large C (per 360 s outer step): the coagulation rate
evaluation reads the per-cell kernel `kij (C,40,40)` (12.8 GB at C=1M) and
runs `n_sub` times; PPM runs `n_glob ≤ 40` cheap substeps; everything else
is (C,40) elementwise. The λ distribution over cells is heavy-tailed
(median n_sub ≈ 3, p99 ≈ 200+ for humid high-N cells), so:

- **`n_cell_chunks` ≥ 4-8 is required at C=1M** (kij + its triangular
  copies exceed 40 GB unchunked), and
- **`sort_by_coag_cost=True`** orders cells by initial stiffness before
  chunking so the batch-max substep count of each chunk matches its own
  cells — otherwise one stiff cell makes all 1M cells run ~200 substeps.

If the first GPU run misses the 10 s target, the next levers in order:
1. **Pallas fused coagulation kernel**: kij per cell is 12.8 KB — compute
   it on the fly in SMEM instead of materializing (C,40,40) in HBM;
   removes the dominant memory traffic (~10× coag speedup).
2. Finer chunk bucketing (more chunks after sorting).
3. `c_max` 0.05 → 0.1 (halves coag work; ~2× coarser accuracy in the
   stiffest bins only).
4. Smaller outer dt does NOT help coagulation (total substeps are
   dt-invariant) but reduces per-step PPM caps if `cond_cap_hit` fires.

## Files

```
tomas_jax/fast/
  config.py        — NBINS=40, ICOMP=2 (SO4, H2O), Gc: (H2SO4, SO2)
  state.py         — FastState (batched NamedTuple)
  mnfix.py         — vectorized 3-phase MNFIX
  density.py       — binary H2SO4/H2O Tang density
  water.py         — Tabazadeh 1997 wt% + equilibrium water (pure JAX)
  nucleation.py    — Dunne Jbn analytic nucleation
  coagulation.py   — adaptive-capped Euler coagulation
  condensation.py  — batched PPM driver, global-max CFL substeps
  step.py          — process composition + SO2 chemistry
  run.py           — jit(scan) segments, donation, chunking, sorting
benchmarks/python/bench_fast_1m.py — GPU benchmark + CPU smoke
tests/test_fast_{mnfix,water,nucleation,step,vs_full}.py
```
