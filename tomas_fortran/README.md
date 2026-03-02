# TOMAS Fortran Source

Self-contained copy of the original TOMAS (TwO-Moment Aerosol Sectional) Fortran source code used for validation benchmarks. Copied from `original-models/TOMAS/backup/src/` to make this repository self-contained.

## Build

Requires `gfortran`.

```bash
# Build the 24-hour TFL benchmark executable
make benchmark_24h

# Build the 24-hour PPM benchmark executable
make benchmark_24h_ppm

# Build both TFL and PPM benchmarks
make benchmark_both

# Run TFL benchmark (reads scenarios.csv, writes to output/24h/)
make run_24h

# Run PPM benchmark (writes ppm_* prefixed files to output/24h/)
make run_24h_ppm

# Build the single-scenario harness
make all

# Clean
make clean
```

## Directory Structure

```
src/              Core TOMAS source files (14 files, TFL condensation)
src_ppm/          PPM condensation source files (2 files)
include/          Common block include files
harness/          Benchmark driver programs
Makefile          Build system
```

## Source Files

### Core Routines (`src/`)

| File | Description |
|------|-------------|
| `multicoag.f` | Multi-component coagulation kernel and solver |
| `initbounds.f` | Initialize size bin boundaries (xk array) |
| `mnfix.f` | Mass-number consistency correction |
| `aerodens.f` | Aerosol density calculation from composition |
| `loginit.f` | Log-normal initialization of size distributions |
| `ezcond.f` | Simple condensation driver (H2SO4 + organics) |
| `tmcond.f` | Top-hat mass condensation (TFL algorithm) |
| `dmdt_int.f` | Mass growth rate integration |
| `getCondSink.f` | Condensation sink calculation |
| `gasdiff.f` | Gas-phase diffusivity (Fuller-Schettler-Giddings) |
| `eznh3eqm.f` | NH3/NH4 stoichiometric equilibrium |
| `ezwatereqm.f` | Water equilibrium driver |
| `waterso4.f` | Water uptake for sulfate aerosol |
| `waternacl.f` | Water uptake for sea salt aerosol |

### Include Files (`include/`)

| File | Description |
|------|-------------|
| `sizecode.COM` | Common block definitions (NBINS, ICOMP, state arrays) |

### PPM Condensation (`src_ppm/`)

| File | Description |
|------|-------------|
| `tmcond_ppm.f` | PPM condensation algorithm (8 subroutines, ported from JAX) |
| `ezcond_ppm.f` | PPM-aware condensation driver (replaces ezcond.f for PPM mode) |

### Benchmark Harness (`harness/`)

| File | Description |
|------|-------------|
| `benchmark_24h.f` | 24-hour multi-scenario benchmark (reads scenarios.csv) |
| `benchmark_24h_ppm.f` | 24-hour PPM benchmark (same as above, uses ezcond_ppm) |
| `benchmark_harness.f` | Single-scenario benchmark harness |

## Compiler Flags

```
gfortran -O0 -fdefault-real-8 -fdefault-double-8 -ffixed-form -std=legacy
```

- `-O0`: No optimization (reproducibility)
- `-fdefault-real-8 -fdefault-double-8`: All reals are 64-bit (matches JAX float64)
- `-ffixed-form`: Fixed-format Fortran (columns 1-72)
- `-std=legacy`: Allow legacy Fortran constructs
