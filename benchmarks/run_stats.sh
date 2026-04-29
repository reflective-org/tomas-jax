#!/usr/bin/env bash
# Orchestrate the Fortran-vs-JAX statistical benchmark reporting pipeline.
#
# Requires that simulations are already on disk:
#   benchmarks/results/24h/*.npz                         (JAX)
#   tomas_fortran/output/24h/s*_*_hour*_*.csv            (Fortran)
#
# This script runs zero simulations — only post-processing and plotting.
# Slide-prep day, single entrypoint.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo "Expected virtualenv at $ROOT_DIR/.venv — activate it or adjust this script." >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[1/3] Exporting per-scenario skill metrics → CSV..."
python -m benchmarks.python.export_metrics_csv "$@"

echo "[2/3] Rendering statistical figure suite (A–G)..."
python -m benchmarks.python.plot_statistical_suite

echo "[3/3] Building presentation-ready tables..."
python -m benchmarks.python.build_presentation_tables

echo ""
echo "Outputs: benchmarks/results/24h/stats/"
ls -1 benchmarks/results/24h/stats/
