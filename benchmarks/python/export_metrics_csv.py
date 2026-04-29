"""Export per-scenario skill metrics for Fortran-vs-JAX comparison.

Reads existing JAX NPZs + Fortran hourly CSVs (via compare_24h helpers)
and writes two CSV files with scalar skill metrics per
(scenario × mode × method_pair × hour) row.

Usage::

    # Full export (all 50 scenarios, all modes, all method pairs)
    python -m benchmarks.python.export_metrics_csv

    # Incremental: one scenario for smoke testing
    python -m benchmarks.python.export_metrics_csv --scenarios 1

    # Force regeneration ignoring manifest hash mismatch
    python -m benchmarks.python.export_metrics_csv --force

Outputs (under ``benchmarks/results/24h/stats/``):

- ``per_scenario_metrics.csv``   one row per (scen, mode, pair, hour)
- ``per_scenario_metrics_summary.csv`` aggregated across scenarios
- ``manifest.json`` records md5 of scenarios.csv + file counts
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Iterable

import numpy as np

from benchmarks.python import compare_24h as cmp24
from benchmarks.python import metrics as M
from benchmarks.python.scenarios import get_scenarios, PARAM_NAMES

# --- Paths ------------------------------------------------------------------

_BASE = os.path.join(os.path.dirname(__file__), '..')
SCENARIOS_CSV = os.path.abspath(os.path.join(_BASE, 'fortran', 'scenarios.csv'))
STATS_DIR = os.path.abspath(os.path.join(_BASE, 'results', '24h', 'stats'))
PER_SCENARIO_CSV = os.path.join(STATS_DIR, 'per_scenario_metrics.csv')
SUMMARY_CSV = os.path.join(STATS_DIR, 'per_scenario_metrics_summary.csv')
MANIFEST_PATH = os.path.join(STATS_DIR, 'manifest.json')

# --- Config -----------------------------------------------------------------

REPORT_HOURS = (1, 6, 12, 24)            # 1-based; hour 'all' added later
METHOD_PAIRS = ('tfl_vs_fortran', 'ppm_vs_fortran', 'ppm_vs_tfl')
MODES = ('coag_only', 'cond_only', 'combined', 'nucl_cond', 'full')

DEFAULT_ATOL_NK = 1.0        # per-cell
DEFAULT_ATOL_MK = 1e-25      # kg/cell — 1e-20 safety floor


# --- Helpers ----------------------------------------------------------------

def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load_scenario_params(scenarios) -> dict:
    """Return {scenario_id: {param: value, ...}}."""
    return {s['scenario_id']: {p: s[p] for p in PARAM_NAMES} for s in scenarios}


def _pair_pick(pair: str, fort_data, tfl_data, ppm_data):
    """Return (ref_dict, test_dict) for a method pair."""
    if pair == 'tfl_vs_fortran':
        return fort_data, tfl_data
    if pair == 'ppm_vs_fortran':
        return fort_data, ppm_data
    if pair == 'ppm_vs_tfl':
        return tfl_data, ppm_data
    raise ValueError(pair)


def _hour_index(hour) -> int:
    """1-based input → 0-based index into (24,) arrays."""
    return int(hour) - 1


def _load_hour_fields(scenario_id: int, mode: str, hour: int):
    """Load Fortran + TFL + PPM Nk/Mk/Ntot/Mdry for one hour.

    Returns three dicts (or None if that method's data is missing) with keys:
    'Nk' (NBINS,), 'Mk' (NBINS*ICOMP,), 'N_tot', 'M_dry'.
    Fortran values loaded lazily per-hour from CSV; JAX values indexed from NPZ.
    """
    fort_mode = cmp24.MODE_NAMES[mode]
    h_idx = _hour_index(hour)

    fort_d = None
    if os.path.exists(
        os.path.join(cmp24.FORTRAN_DIR,
                     f"s{scenario_id:02d}_{fort_mode}_hour{hour:02d}_Nk.csv")
    ):
        try:
            Nk_f, Mk_f, _ = cmp24.load_fortran_hourly(scenario_id, fort_mode, hour)
            fort_d = {
                'Nk': Nk_f.ravel(),
                'Mk': Mk_f.ravel(),
                'N_tot': float(np.sum(Nk_f)),
                'M_dry': float(np.sum(Mk_f[:, :cmp24.SRTH2O])),
            }
        except Exception as e:
            print(f"  [warn] Fortran load failed s{scenario_id:02d} "
                  f"{mode} h{hour}: {e}")

    def _pack(data):
        if data is None:
            return None
        return {
            'Nk': data['Nk'][h_idx].ravel(),
            'Mk': data['Mk'][h_idx].ravel(),
            'N_tot': float(data['N_tot'][h_idx]),
            'M_dry': float(data['M_dry'][h_idx]),
        }

    tfl_full = cmp24.load_jax_results(scenario_id, mode, 'tfl_jit') \
        or cmp24.load_jax_results(scenario_id, mode, 'tfl')
    ppm_full = cmp24.load_jax_results(scenario_id, mode, 'ppm_jit') \
        or cmp24.load_jax_results(scenario_id, mode, 'ppm')

    if mode == 'coag_only':  # same solver — aligns with compare_24h
        ppm_full = tfl_full

    return fort_d, _pack(tfl_full), _pack(ppm_full)


def _row(scenario_id, mode, pair, hour_label, ref, test, params):
    """Build a CSV row (dict) of metrics + scenario params."""
    row = {
        'scenario_id': scenario_id,
        'mode': mode,
        'method_pair': pair,
        'hour': hour_label,
    }
    if ref is None or test is None:
        for key in ('Nk', 'Mk'):
            for m in M.ARRAY_METRICS:
                row[f'{key}_{m}'] = float('nan')
        for field in ('N_tot_ref', 'N_tot_test', 'N_tot_bias', 'N_tot_relerr',
                      'M_dry_ref', 'M_dry_test', 'M_dry_bias', 'M_dry_relerr'):
            row[field] = float('nan')
    else:
        nk = M.compute_all(ref['Nk'], test['Nk'], prefix='Nk_')
        mk = M.compute_all(ref['Mk'], test['Mk'], prefix='Mk_')
        row.update(nk)
        row.update(mk)
        row['N_tot_ref']    = ref['N_tot']
        row['N_tot_test']   = test['N_tot']
        row['N_tot_bias']   = test['N_tot'] - ref['N_tot']
        row['N_tot_relerr'] = M.total_rel_error(ref['N_tot'], test['N_tot'])
        row['M_dry_ref']    = ref['M_dry']
        row['M_dry_test']   = test['M_dry']
        row['M_dry_bias']   = test['M_dry'] - ref['M_dry']
        row['M_dry_relerr'] = M.total_rel_error(ref['M_dry'], test['M_dry'])

    for p in PARAM_NAMES:
        row[p] = params.get(p, float('nan'))
    return row


def _aggregate_hour_all(rows_for_scen_mode_pair):
    """Aggregate a per-hour group into one hour='all' row (median across hours)."""
    if not rows_for_scen_mode_pair:
        return None
    template = rows_for_scen_mode_pair[0]
    agg = {k: template[k] for k in ('scenario_id', 'mode', 'method_pair')}
    agg['hour'] = 'all'
    for p in PARAM_NAMES:
        agg[p] = template[p]
    numeric_keys = [k for k in template
                    if k not in ('scenario_id', 'mode', 'method_pair', 'hour')
                    and k not in PARAM_NAMES]
    for k in numeric_keys:
        vals = np.array([r[k] for r in rows_for_scen_mode_pair], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        agg[k] = float(np.median(vals)) if vals.size else float('nan')
    return agg


# --- Main entrypoint --------------------------------------------------------

def build_rows(scenario_ids: Iterable[int], modes: Iterable[str]) -> list[dict]:
    """Compute metric rows for the cartesian product scenarios × modes × pairs × hours."""
    scenarios = get_scenarios()
    params_by_id = load_scenario_params(scenarios)

    rows: list[dict] = []
    for sid in scenario_ids:
        params = params_by_id.get(sid, {})
        for mode in modes:
            # Per-pair per-hour
            pair_buckets: dict[str, list[dict]] = {p: [] for p in METHOD_PAIRS}

            for hour in REPORT_HOURS:
                fort_d, tfl_d, ppm_d = _load_hour_fields(sid, mode, hour)
                for pair in METHOD_PAIRS:
                    ref, test = _pair_pick(pair, fort_d, tfl_d, ppm_d)
                    row = _row(sid, mode, pair, hour, ref, test, params)
                    rows.append(row)
                    pair_buckets[pair].append(row)

            # Aggregated hour='all' rows (median of per-hour)
            for pair, bucket in pair_buckets.items():
                agg = _aggregate_hour_all(bucket)
                if agg is not None:
                    rows.append(agg)

        print(f"  scenario {sid:02d}: {sum(1 for r in rows if r['scenario_id']==sid)} rows")
    return rows


def write_per_scenario_csv(rows: list[dict], out_path: str) -> None:
    if not rows:
        print("  [warn] no rows to write")
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_path}")


def write_summary_csv(rows: list[dict], out_path: str) -> None:
    """Aggregate across scenarios for each (mode, method_pair, hour).
    Reports median, p05, p95, and n of valid (non-NaN) values for each metric.
    """
    # Group
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r['mode'], r['method_pair'], r['hour'])
        groups.setdefault(key, []).append(r)

    numeric_keys = [k for k in rows[0]
                    if k not in ('scenario_id', 'mode', 'method_pair', 'hour')
                    and k not in PARAM_NAMES]

    summary_rows = []
    for (mode, pair, hour), bucket in sorted(groups.items(),
                                             key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        out = {'mode': mode, 'method_pair': pair, 'hour': hour, 'n_scenarios': len(bucket)}
        for k in numeric_keys:
            vals = np.array([b[k] for b in bucket], dtype=np.float64)
            valid = vals[np.isfinite(vals)]
            out[f'{k}_n'] = int(valid.size)
            if valid.size:
                out[f'{k}_median'] = float(np.median(valid))
                out[f'{k}_p05']    = float(np.percentile(valid, 5))
                out[f'{k}_p95']    = float(np.percentile(valid, 95))
            else:
                out[f'{k}_median'] = float('nan')
                out[f'{k}_p05']    = float('nan')
                out[f'{k}_p95']    = float('nan')
        summary_rows.append(out)

    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} summary rows to {out_path}")


def update_manifest(rows: list[dict], force: bool = False) -> None:
    """Write manifest.json recording inputs and row counts. Prints warnings
    if scenarios.csv hash has changed since the last manifest.
    """
    os.makedirs(STATS_DIR, exist_ok=True)
    new_md5 = md5_file(SCENARIOS_CSV)

    if os.path.exists(MANIFEST_PATH) and not force:
        try:
            prev = json.load(open(MANIFEST_PATH))
            if prev.get('scenarios_md5') and prev['scenarios_md5'] != new_md5:
                print(f"  [warn] scenarios.csv md5 changed "
                      f"({prev['scenarios_md5'][:8]} -> {new_md5[:8]}). "
                      f"Use --force to overwrite.")
        except Exception:
            pass

    manifest = {
        'scenarios_md5':   new_md5,
        'n_rows':          len(rows),
        'scenario_ids':    sorted({r['scenario_id'] for r in rows}),
        'modes':           sorted({r['mode'] for r in rows}),
    }
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {MANIFEST_PATH}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scenarios', type=int, nargs='+', default=None,
                   help='1-based scenario IDs. Default: 1..50.')
    p.add_argument('--modes', nargs='+', default=None,
                   choices=MODES, help='Modes to include.')
    p.add_argument('--force', action='store_true',
                   help='Ignore manifest md5 mismatch')
    args = p.parse_args()

    scenario_ids = args.scenarios or list(range(1, 51))
    modes = args.modes or list(MODES)

    if os.path.exists(MANIFEST_PATH) and not args.force:
        try:
            prev = json.load(open(MANIFEST_PATH))
            cur_md5 = md5_file(SCENARIOS_CSV)
            if prev.get('scenarios_md5') and prev['scenarios_md5'] != cur_md5:
                print("scenarios.csv md5 changed since last export. "
                      "Rerun with --force if intentional.", file=sys.stderr)
                sys.exit(2)
        except FileNotFoundError:
            pass

    print(f"Exporting metrics for {len(scenario_ids)} scenarios × "
          f"{len(modes)} modes × {len(METHOD_PAIRS)} pairs × "
          f"{len(REPORT_HOURS)} hours + aggregates...")

    rows = build_rows(scenario_ids, modes)
    write_per_scenario_csv(rows, PER_SCENARIO_CSV)
    write_summary_csv(rows, SUMMARY_CSV)
    update_manifest(rows, force=args.force)


if __name__ == '__main__':
    main()
