# SOA/VBS Benchmark: Fortran (top-hat) vs Python (PPM)

Comprehensive comparison of the Fortran `soacond.f` (36-bin, tau+tmcond
top-hat redistribution) against the JAX `soa_condensation.py` (36-bin,
PPM redistribution).

Both codes use the same 36-bin legacy grid (Mo=1.5625e-23 kg), eliminating
grid mismatch as a variable. Only organic VBS condensation is tested —
no H2SO4 condensation. Sulfate particles serve as an inert substrate.

## Algorithmic Differences

| Feature | Fortran (`soacond.f`) | JAX (`soa_condensation.py`) |
|---------|----------------------|---------------------------|
| Mass redistribution | TAU → `tmcond()` (top-hat) | TAU → PPM advection (3rd-order) |
| Sub-timestepping | Adaptive (max 900s, gas depletion limit) | Single-step analytical |
| Gas depletion | Exponential via sK: `Gc(1-exp(-sK*cdt))` | Analytical: `(Gc-Gc_eq)(1-exp(-CS*dt))` |
| Equilibrium correction | fodc: `maddEQ*(1-exp(-fodc*dt))` always | fodc only when kinetic > equilibrium |
| NH3/H2O equilibrium | Called after each VBS species | Not called (SOA-only step) |
| Kelvin effect | `exp(σ*MW*1e-3/(RT*ρ*Dp))` | `exp(4σMW/(RTρDp))` |
| Grid | 36 bins (Mo=1.5625e-23 kg) | 36 bins (same grid) |
| Condensation sink | `Dpk*Nk*Kgkk` with particle-phase diffusion | Standard Fuchs-Sutugin |
| Zero-organic handling | Cannot start from zero (fodc zeros madd when maddEQ=0) | Can condense from zero (fodc bypassed) |
| Number conservation | tmcond preserves total N exactly | PPM preserves total N to ~1e-5 |

## Scenarios

All scenarios use `boxvol=1e6 cm³`, `dt=60s`, `alpha=1.0`, 24-hour simulation.
Particles are seeded with 10% sulfate mass as VBS bin 1 organic to enable
Raoult-law partitioning in Fortran. No H2SO4 gas — sulfate is inert.

| Parameter | A (Pure condensation) | B (Mixed) | C (Warm evaporative) |
|-----------|----------------------|-----------|---------------------|
| Temperature | 288 K | 270 K | 310 K |
| Pressure | 101325 Pa | 80000 Pa | 101325 Pa |
| RH | 0.5 | 0.3 | 0.6 |
| N_total | 1e4 /cm³ | 5e3 /cm³ | 2e4 /cm³ |
| GMD | 50 nm | 80 nm | 30 nm |
| GSD | 1.6 | 1.5 | 1.8 |
| H2SO4 | 0 (none) | 0 (none) | 0 (none) |
| VBS gas (C*=0.01-1) | 10, 10, 5 µg/m³ | 5, 5, 5 µg/m³ | 2, 2, 2 µg/m³ |
| VBS gas (C*=10-1000) | 5, 1, 1 µg/m³ | 5, 5, 5 µg/m³ | 2, 2, 2 µg/m³ |

## Results

### Gas Depletion (24h)

| Scenario | VBS bin | C* [µg/m³] | Fortran depletion | Python depletion | Difference |
|----------|---------|-----------|-------------------|------------------|------------|
| A (288K) | 1 | 0.01 | −100.0% | −100.0% | 0.0% |
| A | 2 | 0.1 | −99.9% | −99.9% | 0.0% |
| A | 3 | 1.0 | −99.4% | −99.4% | 0.0% |
| A | 4 | 10.0 | −93.0% | −93.0% | 0.0% |
| A | 5 | 100.0 | −52.2% | −52.2% | 0.1% |
| A | 6 | 1000.0 | −8.2% | −8.2% | 0.0% |
| B (270K) | 1–4 | 0.01–10 | −100% to −99.6% | −100% to −99.6% | <0.1% |
| B | 5 | 100.0 | −92.6% | −93.1% | 0.5% |
| B | 6 | 1000.0 | −41.0% | −42.8% | 1.8% |
| C (310K) | 1 | 0.01 | −97.7% | −97.5% | 0.3% |
| C | 2 | 0.1 | −85.8% | −84.7% | 1.1% |
| C | 3 | 1.0 | −43.0% | −40.8% | 2.1% |
| C | 5 | 100.0 | −1.2% | +1.0% | 2.2% |
| C | 6 | 1000.0 | −0.1% | +8.0% | 8.1% |

### Size Distribution

PPM produces smoother size distributions than Fortran's top-hat, especially
at later times (12h, 24h). The top-hat redistribution creates characteristic
bin-to-bin jaggedness, while PPM gives a continuous, monotonic distribution.

### Mass Conservation

- Fortran: machine precision (~1e-15 relative error) for scenarios A and B
- Python: ~1e-5 to 1e-4 relative error (PPM positivity clamping)
- Gas phase correction ensures exact gas+particle conservation

### Performance

| Code | Time per scenario | Speedup |
|------|-------------------| --------|
| Python (PPM, 36-bin) | 0.21s | **12× faster** |
| Fortran (top-hat, 36-bin) | 2.5s | baseline |

## Key Findings

1. **Near-perfect gas depletion agreement** (scenario A): All 6 VBS bins
   match within 0.1%, validating the JAX kinetic+equilibrium algorithm.

2. **PPM gives smoother size distributions**: The key advantage over
   Fortran's top-hat. PPM redistribution avoids the staircase artifacts
   inherent in top-hat reconstruction.

3. **Evaporation regime diverges** (scenario C, 310K): Fortran shows slight
   condensation for VBS5-6 while Python shows slight evaporation. Root cause:
   Fortran's fodc correction always replaces kinetic madd with equilibrium-based
   values, while JAX only applies correction when kinetic exceeds equilibrium.

4. **Organic seed mass required**: The Fortran Raoult-law partitioning requires
   existing organic mass in particles (`totphase > 0`). Both codes use 10%
   organic seed.

5. **JAX is 12× faster than Fortran** with PPM redistribution, despite PPM
   being a more sophisticated algorithm than top-hat.

## Files

| File | Description |
|------|-------------|
| `tomas_fortran/src/soacond.f` | Fortran SOA condensation (781 lines) |
| `tomas_fortran/harness/benchmark_soa.f` | Fortran benchmark harness (SOA-only) |
| `benchmarks/python/benchmark_soa.py` | Python benchmark + 8 figures |
| `benchmarks/results/soa_benchmark/` | Output NPZ files + PNG figures |
| `tomas_jax/physics/soa_condensation.py` | JAX SOA driver with PPM |
| `tomas_jax/physics/vbs_config.py` | VBS bin definitions |
| `tomas_jax/physics/condensation_ppm.py` | PPM advection algorithm |

## Figures

1. **Gas-phase evolution**: 18-panel (3 scenarios × 6 VBS bins) showing Gc vs time
2. **Particle-phase VBS distribution**: Bar charts at 0, 6, 12, 24h
3. **Size distribution**: dN/dlogDp at 4 snapshots, Fortran vs Python
4. **Composition**: Size-resolved SO4 vs organic mass
5. **Total mass/number**: N_total, M_dry, M_organic, M_sulfate timeseries
6. **Mass conservation**: Fractional error (gas+particle) per VBS bin
7. **Relative error**: Python vs Fortran for N, M_dry, M_organic
8. **Diagnostics**: C*(T), Kelvin factor, gas depletion ratios

## Usage

```bash
# 1. Build and run Fortran
cd tomas_fortran && make benchmark_soa && ./benchmark_soa.exe && cd ..

# 2. Run Python (36-bin, PPM, SOA-only)
python -m benchmarks.python.benchmark_soa --run

# 3. Regenerate plots
python -m benchmarks.python.benchmark_soa --plot-only
```
