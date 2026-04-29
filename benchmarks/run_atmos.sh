#!/usr/bin/env bash
# End-to-end driver for the 1000-scenario tropo→strato benchmark.
#
# Stages (each idempotent; skip if outputs exist):
#   1. Scenario generation       (~1 s)
#   2. Fortran 1000-run          (~15-30 min, background)
#   3. JAX 1000-run              (~90 min, foreground)
#   4. Metrics export            (~30 s)
#   5. Plot suite                (~2 min)
#   6. Presentation table        (~5 s)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

STAGES=${1:-all}

echo "[1/6] Generating 1000 atmospheric scenarios..."
if [ ! -f benchmarks/fortran/scenarios_atmos.csv ]; then
    python -m benchmarks.python.scenarios_atmos
fi

echo "[2/6] Fortran 1000-run (backgrounded, logs to /tmp/fortran_atmos.log)..."
# Skip if all expected timing entries exist (1000 lines + header)
if [ ! -f tomas_fortran/output/atmos/timing_fortran_ppm_atmos.csv ] \
   || [ "$(wc -l < tomas_fortran/output/atmos/timing_fortran_ppm_atmos.csv)" -lt 1001 ]; then
    cd tomas_fortran
    cp ../benchmarks/fortran/scenarios_atmos.csv ./scenarios_atmos.csv
    mkdir -p output/atmos
    nohup ./benchmark_24h_ppm_atmos.exe > /tmp/fortran_atmos.log 2>&1 &
    FORTRAN_PID=$!
    cd ..
    echo "   Fortran PID: $FORTRAN_PID (tail /tmp/fortran_atmos.log)"
else
    echo "   Fortran outputs already present — skipping"
fi

echo "[3/6] JAX 1000-run (foreground, ~90 min)..."
python -u -m benchmarks.python.run_atmos_scenarios

# Wait for Fortran to finish before metrics
if [ -n "${FORTRAN_PID:-}" ]; then
    echo "   Waiting on Fortran (PID $FORTRAN_PID)..."
    wait $FORTRAN_PID || true
fi

echo "[4/6] Exporting per-scenario metrics..."
python -m benchmarks.python.export_atmos_metrics

echo "[5/6] Rendering comprehensive plot suite..."
python -m benchmarks.python.plot_atmos_suite

echo "[6/6] Building presentation tables..."
python -m benchmarks.python.build_atmos_tables

echo ""
echo "Done. Outputs:"
ls -1 benchmarks/results/atmos/stats/
