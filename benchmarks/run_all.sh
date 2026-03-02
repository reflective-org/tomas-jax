#!/bin/bash
# TOMAS FORTRAN vs JAX Coagulation Benchmark
#
# This script:
# 1. Builds the FORTRAN benchmark harness
# 2. Runs it to generate reference data
# 3. Runs the Python comparison scripts
# 4. Generates a comparison report
#
# Prerequisites:
#   - gfortran (or compatible Fortran compiler)
#   - Python 3.8+ with JAX, numpy
#   - TOMAS FORTRAN sources at ../../../original-models/TOMAS/src/
#
# Usage:
#   cd benchmarks && bash run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo " TOMAS FORTRAN vs JAX Benchmark Suite"
echo "=============================================="
echo ""

# Step 1: Build FORTRAN harness
echo "[Step 1/3] Building FORTRAN benchmark harness..."
cd "$SCRIPT_DIR/fortran"

if command -v gfortran &> /dev/null; then
    make clean 2>/dev/null || true
    make
    echo "  Build successful."
else
    echo "  WARNING: gfortran not found. Skipping FORTRAN build."
    echo "  Install gfortran or set FC in benchmarks/fortran/Makefile"
    echo ""
    echo "  Checking for existing FORTRAN output..."
    if [ -d "output" ] && [ -f "output/level01_xk.csv" ]; then
        echo "  Found existing output. Continuing with Python comparison."
    else
        echo "  No existing output found. Cannot proceed."
        echo "  Install gfortran and re-run."
        exit 1
    fi
fi

# Step 2: Run FORTRAN harness
echo ""
echo "[Step 2/3] Running FORTRAN benchmark harness..."
if [ -f "benchmark.exe" ]; then
    ./benchmark.exe
    echo "  FORTRAN benchmark complete."
else
    echo "  Skipped (no executable)."
fi

# Step 3: Run Python comparisons
echo ""
echo "[Step 3/3] Running Python comparison scripts..."
cd "$REPO_ROOT"

python -m benchmarks.python.compare_all \
    --data-dir "$SCRIPT_DIR/fortran/output" \
    --report "$SCRIPT_DIR/results/comparison_report.md" \
    --verbose

echo ""
echo "=============================================="
echo " Benchmark complete!"
echo " Report: benchmarks/results/comparison_report.md"
echo "=============================================="
