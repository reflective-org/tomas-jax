"""Export per-scenario skill metrics for the 1000-scenario atmospheric benchmark.

Reads JAX NPZs from benchmarks/results/atmos/ and Fortran CSVs from
tomas_fortran/output/atmos/. PPM-only comparison. Adds a regime column
(tropo | utls | strato) based on pressure.

Usage::

    python -m benchmarks.python.export_atmos_metrics
    python -m benchmarks.python.export_atmos_metrics --scenarios 1 5 7

Outputs (benchmarks/results/atmos/stats/):
    per_scenario_metrics.csv
    per_scenario_metrics_summary.csv
    failed_scenarios.csv
    manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.python import metrics as M
from benchmarks.python.scenarios_atmos import (
    get_atmos_scenarios, REGIMES,
)

_BASE = Path(__file__).resolve().parent.parent
SCENARIOS_CSV = _BASE / 'fortran' / 'scenarios_atmos.csv'
JAX_DIR = _BASE / 'results' / 'atmos'
FORTRAN_DIR = _BASE / '..' / 'tomas_fortran' / 'output' / 'atmos'
STATS_DIR = JAX_DIR / 'stats'
PER_SCENARIO_CSV = STATS_DIR / 'per_scenario_metrics.csv'
SUMMARY_CSV = STATS_DIR / 'per_scenario_metrics_summary.csv'
FAILED_CSV = STATS_DIR / 'failed_scenarios.csv'
MANIFEST_PATH = STATS_DIR / 'manifest.json'

NBINS = 36
ICOMP = 44
SRTH2O = 43
NHOURS = 24

REPORT_HOURS = (1, 6, 12, 24)
MODES = ('coag_only', 'cond_only', 'combined')
MODE_FORT = {'coag_only': 'coag', 'cond_only': 'cond', 'combined': 'combined'}


def classify_regime(pres_pa: float) -> str:
    if pres_pa >= 25000:
        return 'tropo'
    elif pres_pa >= 5000:
        return 'utls'
    else:
        return 'strato'


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load_fortran_hourly(sid: int, mode: str, hour: int):
    fort_mode = MODE_FORT[mode]
    prefix = f"ppm_s{sid:04d}_{fort_mode}_hour{hour:02d}"
    p_nk = FORTRAN_DIR / f"{prefix}_Nk.csv"
    p_mk = FORTRAN_DIR / f"{prefix}_Mk.csv"
    if not p_nk.exists() or not p_mk.exists():
        return None, None
    try:
        Nk = np.loadtxt(p_nk).reshape(-1)
        Mk = np.loadtxt(p_mk, delimiter=',').reshape(NBINS, ICOMP)
        return Nk, Mk
    except Exception:
        return None, None


def load_jax_npz(sid: int, mode: str):
    p = JAX_DIR / f"s{sid:04d}_{mode}_ppm_jit.npz"
    if not p.exists():
        return None
    try:
        return np.load(p)
    except Exception:
        return None


def _row(sid, mode, regime, hour, ref_Nk, ref_Mk, ref_Ntot, ref_Mdry,
         test_Nk, test_Mk, test_Ntot, test_Mdry, params):
    row = {
        'scenario_id': sid, 'mode': mode, 'regime': regime,
        'hour': hour,
    }
    if ref_Nk is None or test_Nk is None:
        for key in ('Nk', 'Mk'):
            for m in M.ARRAY_METRICS:
                row[f'{key}_{m}'] = float('nan')
        for f in ('N_tot_ref', 'N_tot_test', 'N_tot_bias', 'N_tot_relerr',
                  'M_dry_ref', 'M_dry_test', 'M_dry_bias', 'M_dry_relerr'):
            row[f] = float('nan')
    else:
        row.update(M.compute_all(ref_Nk, test_Nk, prefix='Nk_'))
        row.update(M.compute_all(ref_Mk, test_Mk, prefix='Mk_'))
        row['N_tot_ref']    = float(ref_Ntot)
        row['N_tot_test']   = float(test_Ntot)
        row['N_tot_bias']   = float(test_Ntot - ref_Ntot)
        row['N_tot_relerr'] = M.total_rel_error(ref_Ntot, test_Ntot)
        row['M_dry_ref']    = float(ref_Mdry)
        row['M_dry_test']   = float(test_Mdry)
        row['M_dry_bias']   = float(test_Mdry - ref_Mdry)
        row['M_dry_relerr'] = M.total_rel_error(ref_Mdry, test_Mdry)
    for k, v in params.items():
        row[k] = v
    return row


def build_rows(scenario_ids):
    scens = {s['scenario_id']: s for s in get_atmos_scenarios()}
    rows = []
    failed = []

    for sid in scenario_ids:
        s = scens.get(sid)
        if s is None:
            continue
        regime = s['regime']
        params = {p: s[p] for p in (
            'N_total', 'GMD_um', 'GSD', 'temp', 'pres', 'RH',
            'Gc_SO4_kg', 'h2so4_prod',
        )}

        for mode in MODES:
            jax_data = load_jax_npz(sid, mode)
            if jax_data is None:
                failed.append({'scenario_id': sid, 'mode': mode,
                               'regime': regime, 'reason': 'missing_jax_npz'})
                continue

            # Bucket per-pair hours
            for hour in REPORT_HOURS:
                h_idx = hour - 1
                # Fortran
                Nk_f, Mk_f = load_fortran_hourly(sid, mode, hour)
                if Nk_f is None:
                    if hour == 24:
                        failed.append({'scenario_id': sid, 'mode': mode,
                                       'regime': regime,
                                       'reason': 'missing_fortran_h24'})
                    continue
                Ntot_f = float(np.sum(Nk_f))
                Mdry_f = float(np.sum(Mk_f[:, :SRTH2O]))
                # JAX
                Nk_j = jax_data['Nk'][h_idx]
                Mk_j = jax_data['Mk'][h_idx]
                Ntot_j = float(jax_data['N_tot'][h_idx])
                if 'M_dry' in jax_data.files:
                    Mdry_j = float(jax_data['M_dry'][h_idx])
                else:
                    Mdry_j = float(np.sum(Mk_j[:, :SRTH2O]))

                rows.append(_row(sid, mode, regime, hour,
                                  Nk_f.ravel(), Mk_f.ravel(), Ntot_f, Mdry_f,
                                  Nk_j.ravel(), Mk_j.ravel(), Ntot_j, Mdry_j,
                                  params))

        if sid % 50 == 0:
            print(f"  ..{sid} scenarios processed")

    return rows, failed


def _aggregate(rows, by_keys):
    """Aggregate across (mode, regime, hour) with median/p05/p95/n."""
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in by_keys)
        groups.setdefault(key, []).append(r)

    template = rows[0]
    param_names = ('N_total', 'GMD_um', 'GSD', 'temp', 'pres', 'RH',
                    'Gc_SO4_kg', 'h2so4_prod')
    numeric_keys = [k for k in template
                    if k not in ('scenario_id', 'hour') + by_keys
                    and k not in param_names]

    out = []
    for key, bucket in sorted(groups.items()):
        rec = dict(zip(by_keys, key))
        rec['n_scenarios'] = len(bucket)
        for k in numeric_keys:
            vals = np.array([b[k] for b in bucket], dtype=np.float64)
            valid = vals[np.isfinite(vals)]
            rec[f'{k}_n'] = int(valid.size)
            if valid.size:
                rec[f'{k}_median'] = float(np.median(valid))
                rec[f'{k}_p05']    = float(np.percentile(valid, 5))
                rec[f'{k}_p95']    = float(np.percentile(valid, 95))
            else:
                rec[f'{k}_median'] = float('nan')
                rec[f'{k}_p05']    = float('nan')
                rec[f'{k}_p95']    = float('nan')
        out.append(rec)
    return out


def write_rows(rows, path):
    if not rows:
        print(f"  [warn] nothing to write to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scenarios', nargs='+', type=int, default=None,
                   help='Restrict to these IDs (1-based). Default: all 1000.')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    sids = args.scenarios or list(range(1, 1001))
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building metrics for {len(sids)} scenarios × {len(MODES)} modes "
          f"× {len(REPORT_HOURS)} hours...")
    rows, failed = build_rows(sids)

    write_rows(rows, PER_SCENARIO_CSV)
    if failed:
        write_rows(failed, FAILED_CSV)
        print(f"  [note] {len(failed)} failed scenario/mode pairs; see {FAILED_CSV}")

    summary = _aggregate(rows, ('mode', 'regime', 'hour'))
    write_rows(summary, SUMMARY_CSV)

    manifest = {
        'scenarios_md5': md5_file(SCENARIOS_CSV) if SCENARIOS_CSV.exists() else None,
        'n_rows': len(rows),
        'n_failed': len(failed),
        'scenario_ids': sorted({r['scenario_id'] for r in rows}),
        'modes': list(MODES),
        'regimes': list(REGIMES),
    }
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == '__main__':
    main()
