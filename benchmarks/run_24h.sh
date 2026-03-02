#!/bin/bash
# Master script for 24-hour benchmark: PPM vs TFL vs Fortran
#
# Usage:
#   cd benchmarks
#   bash run_24h.sh              # Full pipeline
#   bash run_24h.sh --skip-fortran  # Skip Fortran (JAX only)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_FORTRAN=false
for arg in "$@"; do
    case $arg in
        --skip-fortran) SKIP_FORTRAN=true ;;
    esac
done

echo "========================================"
echo " TOMAS 24-Hour Benchmark Pipeline"
echo "========================================"
echo " Project root: $PROJECT_ROOT"

# Step 1: Generate scenarios
echo ""
echo "=== Step 1: Generate scenarios ==="
cd "$PROJECT_ROOT"
python -m benchmarks.python.scenarios
echo "Done. scenarios.csv written."

# Step 2: Build and run Fortran
if [ "$SKIP_FORTRAN" = false ]; then
    echo ""
    echo "=== Step 2: Build Fortran harness ==="
    if ! command -v gfortran &> /dev/null; then
        echo "WARNING: gfortran not found. Skipping Fortran benchmark."
        SKIP_FORTRAN=true
    else
        cd "$SCRIPT_DIR/fortran"
        make benchmark_24h
        echo "Build successful."

        echo ""
        echo "=== Step 3: Run Fortran simulations ==="
        ./benchmark_24h.exe
        echo "Fortran simulations complete."
    fi
else
    echo ""
    echo "=== Steps 2-3: Skipping Fortran (--skip-fortran) ==="
fi

# Step 4: Run JAX simulations
echo ""
echo "=== Step 4: Run JAX simulations ==="
cd "$PROJECT_ROOT"
python -m benchmarks.python.run_24h_scenarios
echo "JAX simulations complete."

# Step 5: Run comparisons
echo ""
echo "=== Step 5: Run comparisons ==="
cd "$PROJECT_ROOT"
python -m benchmarks.python.compare_24h
echo "Comparisons complete."

# Step 6: Generate plots
echo ""
echo "=== Step 6: Generate plots ==="
cd "$PROJECT_ROOT"
python -m benchmarks.python.plot_24h_summary
echo "Plots generated."

# Step 6b: Generate timing plots
echo ""
echo "=== Step 6b: Generate timing plots ==="
cd "$PROJECT_ROOT"
python -m benchmarks.python.plot_24h_timing
echo "Timing plots generated."

# Step 7: Run pytest
echo ""
echo "=== Step 7: Run pytest ==="
cd "$PROJECT_ROOT"
python -m pytest tests/test_24h_scenarios.py -v --tb=short -m "slow" || true

echo ""
echo "========================================"
echo " 24-Hour Benchmark Pipeline Complete"
echo "========================================"
echo " Results: benchmarks/results/24h/"
echo " Plots:   benchmarks/results/*.png"
echo "========================================"
