"""Stratified Latin Hypercube scenario generator for tropo→strato benchmarks.

Generates 1000 scenarios sampled across three atmospheric regimes:

    - Troposphere (N_tropo = 600)  : P ≥ 25 kPa, warm, humid, dense aerosol
    - UTLS         (N_utls  = 250) : 5 ≤ P < 25 kPa, cold, dry, sparse aerosol
    - Stratosphere (N_strato= 150) : 100 Pa ≤ P < 5 kPa, sulfate Junge layer

Within each regime, LHC-sample the 8-parameter cube with regime-realistic
bounds.  Concatenate, assign global scenario_id = 1..1000.

Output:
    benchmarks/fortran/scenarios_atmos.csv      (9-col, Fortran-readable)
    benchmarks/fortran/scenarios_atmos_meta.json (seed + per-regime counts)

Usage::

    python -m benchmarks.python.scenarios_atmos
    python -m benchmarks.python.scenarios_atmos --n 10   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats.qmc import LatinHypercube

from benchmarks.python.scenarios import (
    PARAM_NAMES, CSV_COLUMNS, molec_cm3_to_kg_cell,
)

# Default regime allocations at N=1000
DEFAULT_SPLIT = dict(tropo=600, utls=250, strato=150)

# Per-regime parameter ranges — (min, max, scale)
REGIME_RANGES = {
    'tropo': {
        'N_total':    (1e3,   1e7,   'log'),
        'GMD_um':     (0.01,  1.0,   'log'),
        'GSD':        (1.2,   2.5,   'linear'),
        'temp':       (220.0, 310.0, 'linear'),
        'pres':       (25000.0, 101325.0, 'linear'),
        'RH':         (0.05,  0.95,  'linear'),
        'Gc_SO4_kg':  (1e-16, 1e-10, 'log'),
        'h2so4_prod': (1e5,   1e9,   'log'),
    },
    'utls': {
        'N_total':    (1e1,   1e4,   'log'),
        'GMD_um':     (0.01,  0.3,   'log'),
        'GSD':        (1.2,   2.2,   'linear'),
        'temp':       (195.0, 240.0, 'linear'),
        'pres':       (5000.0, 25000.0, 'log'),
        'RH':         (1e-3,  0.3,   'log'),
        'Gc_SO4_kg':  (1e-16, 1e-11, 'log'),
        'h2so4_prod': (1e4,   1e7,   'log'),
    },
    'strato': {
        'N_total':    (1.0,   1e3,   'log'),
        'GMD_um':     (0.05,  0.3,   'log'),
        'GSD':        (1.4,   2.0,   'linear'),
        'temp':       (195.0, 275.0, 'linear'),
        'pres':       (100.0, 5000.0, 'log'),
        'RH':         (1e-4,  1e-2,  'log'),
        'Gc_SO4_kg':  (1e-16, 1e-12, 'log'),
        'h2so4_prod': (1e4,   1e6,   'log'),
    },
}

REGIMES = ('tropo', 'utls', 'strato')


def _sample_regime(regime: str, n: int, seed: int) -> list[dict]:
    """LHC-sample n scenarios for one regime. Returns list of dicts
    with parameter values + regime tag (no scenario_id yet)."""
    ranges = REGIME_RANGES[regime]
    d = len(PARAM_NAMES)
    sampler = LatinHypercube(d=d, seed=seed)
    samples = sampler.random(n=n)  # (n, 8), uniform in [0, 1]

    out = []
    for i in range(n):
        rec: dict = {'regime': regime}
        for j, pname in enumerate(PARAM_NAMES):
            pmin, pmax, scale = ranges[pname]
            u = samples[i, j]
            if scale == 'log':
                lmin, lmax = np.log10(pmin), np.log10(pmax)
                rec[pname] = float(10.0 ** (lmin + u * (lmax - lmin)))
            else:
                rec[pname] = float(pmin + u * (pmax - pmin))
        rec['h2so4_prod_kg_per_s'] = molec_cm3_to_kg_cell(rec['h2so4_prod'])
        out.append(rec)
    return out


def generate_atmos_scenarios(
    split: dict | None = None,
    seed: int = 42,
) -> list[dict]:
    """Stratified LHC across tropo / UTLS / strato regimes.

    Args:
        split: {regime: n} counts. Default = 600/250/150.
        seed:  Root seed. Each regime uses seed + offset for reproducibility.

    Returns:
        List of dicts sorted by scenario_id (1-based, global).
    """
    if split is None:
        split = DEFAULT_SPLIT
    out: list[dict] = []
    for offset, regime in enumerate(REGIMES):
        n = split.get(regime, 0)
        if n <= 0:
            continue
        rs = _sample_regime(regime, n, seed + offset)
        out.extend(rs)
    # Shuffle deterministically so adjacent scenario_ids aren't all same regime
    rng = np.random.default_rng(seed + 100)
    rng.shuffle(out)
    for i, rec in enumerate(out):
        rec['scenario_id'] = i + 1
    return out


def get_atmos_scenarios(n: int = 1000, seed: int = 42) -> list[dict]:
    """Convenience: scale the default 600/250/150 split to arbitrary n
    by proportion. n < 3 rounds to the nearest regime.
    """
    if n == 1000:
        return generate_atmos_scenarios(seed=seed)
    # Scale proportionally
    base = DEFAULT_SPLIT
    total = sum(base.values())
    split = {r: max(1, round(base[r] * n / total)) for r in REGIMES}
    # Fix rounding drift
    drift = n - sum(split.values())
    split['tropo'] += drift
    return generate_atmos_scenarios(split=split, seed=seed)


def save_atmos_scenarios(
    scenarios: Iterable[dict],
    csv_path: str,
    meta_path: str,
    seed: int,
    split: dict,
) -> str:
    scenarios = list(scenarios)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        # Write same 9-column header as scenarios.csv for Fortran compatibility
        w.writerow(CSV_COLUMNS)
        for s in scenarios:
            w.writerow([
                s['scenario_id'],
                f"{s['temp']:.16e}",
                f"{s['pres']:.16e}",
                f"{s['N_total']:.16e}",
                f"{s['GMD_um']:.16e}",
                f"{s['GSD']:.16e}",
                f"{s['Gc_SO4_kg']:.16e}",
                f"{s['RH']:.16e}",
                f"{s['h2so4_prod_kg_per_s']:.16e}",
            ])

    # Side-car with regime + full-precision params (CSV is Fortran-reduced)
    side_path = csv_path.replace('.csv', '_full.csv')
    with open(side_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'scenario_id', 'regime', 'temp', 'pres', 'N_total', 'GMD_um',
            'GSD', 'Gc_SO4_kg', 'RH', 'h2so4_prod', 'h2so4_prod_kg_per_s',
        ])
        w.writeheader()
        for s in scenarios:
            w.writerow({k: s[k] for k in w.fieldnames})

    with open(meta_path, 'w') as f:
        json.dump({
            'seed': seed,
            'n_total': len(scenarios),
            'split': split,
            'regimes': REGIMES,
            'param_names': PARAM_NAMES,
            'regime_ranges': REGIME_RANGES,
        }, f, indent=2, default=str)
    return csv_path


def _print_summary(scenarios: list[dict]) -> None:
    from collections import Counter
    counts = Counter(s['regime'] for s in scenarios)
    print(f"\nGenerated {len(scenarios)} atmospheric scenarios")
    for r in REGIMES:
        print(f"  {r:8s}: {counts.get(r, 0)}")
    print("\nParameter coverage (pooled):")
    for p in PARAM_NAMES:
        vals = [s[p] for s in scenarios]
        print(f"  {p:15s}: [{min(vals):.3e}, {max(vals):.3e}]")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--n', type=int, default=1000,
                   help='Total scenarios (scales 600/250/150 proportionally). Default 1000.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default=None,
                   help='Output directory. Default: benchmarks/fortran/')
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = str(Path(__file__).resolve().parent.parent / 'fortran')

    scenarios = get_atmos_scenarios(n=args.n, seed=args.seed)

    # Recover split for metadata
    from collections import Counter
    split = dict(Counter(s['regime'] for s in scenarios))

    csv_path = os.path.join(args.out_dir, 'scenarios_atmos.csv')
    meta_path = os.path.join(args.out_dir, 'scenarios_atmos_meta.json')
    save_atmos_scenarios(scenarios, csv_path, meta_path,
                          seed=args.seed, split=split)

    _print_summary(scenarios)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {meta_path}")


if __name__ == '__main__':
    main()
