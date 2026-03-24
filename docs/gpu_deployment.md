# GPU Deployment Guide

TOMAS-JAX is designed for GPU execution via JAX's XLA compiler. This document covers hardware considerations, code path selection, and batch processing.

## float64 Requirement

TOMAS-JAX **requires float64 precision**. This is non-negotiable for aerosol microphysics:

- Mass conservation thresholds operate at 1e-15 relative error
- Condensation sink calculations involve exp(-CS*dt) where CS spans 6 orders of magnitude
- Coagulation kernels have N^2 scaling — rounding errors compound catastrophically
- MNFIX partial-transfer algorithm requires exact bin boundary comparisons

### float64 Performance by GPU Type

| GPU | float64 vs float32 | Notes |
|-----|---------------------|-------|
| NVIDIA A100 | 1/2 FP32 throughput | Best GPU choice for float64 workloads |
| NVIDIA H100 | 1/2 FP32 throughput | Excellent float64 support |
| NVIDIA V100 | 1/2 FP32 throughput | Good float64 support |
| NVIDIA consumer (RTX 30xx/40xx) | 1/32 FP32 throughput | **Avoid** — float64 is crippled |
| Google TPU v3/v4 | Native float64 | Excellent, but JAX TPU support varies |
| AMD MI250X/MI300X | 1/2 FP32 throughput | Good float64 support |

**Recommendation**: Use datacenter GPUs (A100, H100, MI250X) or TPUs. Consumer GPUs will be 16x slower than expected due to float64 penalties.

### Enabling float64

float64 is enforced centrally in `tomas_jax/core/config.py` (line 17):
```python
jax.config.update("jax_enable_x64", True)
```

All library modules import from config, inheriting this setting. Entry-point scripts (tests, benchmarks) may also set it for standalone execution.

## Selecting the GPU Platform

```bash
# Use GPU (default if available)
JAX_PLATFORMS=gpu python run_box_model.py

# Force CPU (for debugging)
JAX_PLATFORMS=cpu python run_box_model.py

# Verify platform
python -c "import jax; print(jax.devices())"
```

## Which Code Paths to Use

### GPU-ready (JIT-compiled, pure JAX)

These compile to a single XLA program with zero Python dispatch overhead:

| API | Description |
|-----|-------------|
| `make_step(processes, cond_method='ppm_jit')` | Composable process API (recommended) |
| `run_condensation_scan()` / `run_condensation_scan_tfl()` | Scan-fused condensation loops |
| `run_combined_scan_ppm()` / `run_combined_scan_tfl()` | Scan-fused coag + cond |
| `run_full_scan()` | Scan-fused nucl + coag + cond |
| `condensation_step_jit` / `condensation_step_tfl_jit` | Single condensation steps |
| `coag_euler_step()` | Forward Euler coagulation |
| `diffrax_step()` | Adaptive Tsit5 coagulation |

### Legacy paths to AVOID on GPU

These use numpy arrays and Python loops — they will silently transfer data CPU↔GPU every step:

| API | Replacement |
|-----|-------------|
| `condensation_step(..., method='tfl')` | Use `method='tfl_jit'` |
| `condensation_step(..., method='ppm')` | Use `method='ppm_jit'` |
| `ezcond()` (from `physics/ezcond.py`) | Use `ezcond_tfl_jax()` |
| `ezcond_ppm()` (from `physics/ezcond_ppm.py`) | Use `ezcond_ppm_jax()` |

A `DeprecationWarning` is raised when legacy methods are used.

## Batch Processing with vmap

For ensemble runs (e.g., Monte Carlo, parameter sweeps), use `jax.vmap` to batch across scenarios:

```python
from tomas_jax.solvers.condensation import make_step

step_fn = make_step(
    ['nucleation', 'coagulation', 'condensation'],
    cond_method='ppm_jit',
)

# Batch over different initial conditions
# Nk_batch shape: (n_scenarios, nbins)
# Mk_batch shape: (n_scenarios, nbins, icomp)
batched_step = jax.vmap(step_fn, in_axes=(0, 0, 0, None, None, None, None, None, None, None))

Nk_out, Mk_out, Gc_out = batched_step(
    Nk_batch, Mk_batch, Gc_batch,
    xk, temp, pres, boxvol, rh, alpha, dt,
)
```

`vmap` compiles into a single kernel that processes all scenarios in parallel on the GPU. This is dramatically faster than a Python loop over scenarios.

**Important**: `vmap` requires all batched arrays to have the same leading dimension. Scalar parameters (temp, pres, etc.) are broadcast automatically via `in_axes=None`.

## JIT Stability Tips

1. **Always pass the same set of kwargs** to `make_step` step functions across calls. Changing which kwargs are present triggers recompilation.

2. **Use float masks instead of booleans** for enable/disable flags (e.g., `enable_organic=1.0` not `True`). This avoids JIT recompilation when flags change.

3. **Pass explicit zero arrays** for backgrounds instead of `None`. The dilution step requires explicit `Nk_bg`, `Mk_bg`, `Gc_bg` arrays.

4. **Avoid Python control flow** that depends on array values. Use `jnp.where()` and `jax.lax.cond()` instead.

## Memory Considerations

A single TOMAS state (40 bins, 44 components) is small:
- Nk: 40 floats = 320 bytes
- Mk: 40 x 44 = 1,760 floats = 14 KB
- Gc: 44 floats = 352 bytes
- xk: 41 floats = 328 bytes

The main memory consumer is the coagulation kernel matrix (40 x 40 = 1,600 entries). Even with 1000 vmap'd scenarios, total GPU memory is < 100 MB. Memory is not a constraint.

## Known Limitations

- **No multi-GPU**: TOMAS-JAX does not use `jax.pmap` or sharding. For multi-GPU, use `vmap` on a single GPU and `pmap` across GPUs for independent scenarios.
- **Compilation time**: First call to a JIT function compiles the XLA program (~10-60s depending on complexity). Subsequent calls are fast. Use `make_step` to compile once and reuse.
- **Diffrax on GPU**: The adaptive Tsit5 solver (`diffrax_step`) works on GPU but may be slower than forward Euler (`coag_euler_step`) due to variable-length stepping.
