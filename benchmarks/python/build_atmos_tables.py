"""Regime-stratified ensemble skill tables for the 1000-atmos benchmark.

Reads `benchmarks/results/atmos/stats/per_scenario_metrics_summary.csv`
and emits markdown + LaTeX tables with one row per (regime × mode) at h24.

Usage::

    python -m benchmarks.python.build_atmos_tables
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
STATS_DIR = _BASE / 'results' / 'atmos' / 'stats'
SUMMARY_CSV = STATS_DIR / 'per_scenario_metrics_summary.csv'
MD_PATH = STATS_DIR / 'atmos_table_ensemble_summary.md'
TEX_PATH = STATS_DIR / 'atmos_table_ensemble_summary.tex'

REPORT_HOUR = 24
REGIMES = ('tropo', 'utls', 'strato')
MODES = ('coag_only', 'cond_only', 'combined')


def _fmt(v, digits=3, scientific=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    if scientific:
        return f'{v:.{digits}e}'
    return f'{v:.{digits}f}'


def _pick_row(df, regime, mode, hour=REPORT_HOUR):
    m = ((df['regime'] == regime) & (df['mode'] == mode)
         & (df['hour'] == hour))
    if not m.any():
        return None
    return df[m].iloc[0].to_dict()


def emit_markdown(df, path: Path):
    lines = [
        '# Atmospheric Ensemble Summary — Hour 24',
        '',
        f'1000-scenario LHC spanning troposphere → lower stratosphere. '
        f'Regime classification by pressure: tropo (P ≥ 25 kPa), '
        f'utls (5 ≤ P < 25 kPa), strato (P < 5 kPa). '
        f'JAX PPM_JIT vs Fortran PPM. Source: '
        f'`per_scenario_metrics_summary.csv`.',
        '',
        '| Regime | Mode | n | Nk R² (med) | Nk KGE (med) | '
        '|N_tot relerr| med | |N_tot relerr| p95 | '
        '|M_dry relerr| med | |M_dry relerr| p95 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for regime in REGIMES:
        for mode in MODES:
            r = _pick_row(df, regime, mode)
            if r is None:
                continue
            lines.append(
                f"| {regime} | {mode} | {int(r['n_scenarios'])} "
                f"| {_fmt(r.get('Nk_r2_median'), 4, False)} "
                f"| {_fmt(r.get('Nk_kge_median'), 4, False)} "
                f"| {_fmt(r.get('N_tot_relerr_median'))} "
                f"| {_fmt(r.get('N_tot_relerr_p95'))} "
                f"| {_fmt(r.get('M_dry_relerr_median'))} "
                f"| {_fmt(r.get('M_dry_relerr_p95'))} |"
            )
    lines += [
        '',
        '## Interpretation',
        '',
        '- **Tropospheric** rows should match the existing 49-scenario '
        'benchmark (Nk R² > 0.99).',
        '- **Stratospheric** rows are new territory. Deviation from Fortran '
        'may reflect (a) regime-dependent numerical sensitivity, '
        '(b) constants/coefficients that were tuned for tropospheric use, '
        'or (c) the Fortran code itself being untested at these conditions.',
        '- **p95** values reveal worst-5% tail; use these to decide which '
        'regime needs extra validation.',
    ]
    path.write_text('\n'.join(lines))
    print(f"Wrote {path}")


def emit_latex(df, path: Path):
    header = (
        r"\begin{table}[t]" + "\n"
        r"  \centering" + "\n"
        r"  \caption{TOMAS-JAX PPM\_JIT vs Fortran PPM — 1000-scenario "
        r"atmospheric ensemble, hour 24.}" + "\n"
        r"  \label{tab:atmos_ensemble_summary}" + "\n"
        r"  \begin{tabular}{llrrrrrrr}" + "\n"
        r"    \toprule" + "\n"
        r"    Regime & Mode & $n$ & $R^2$ & KGE & "
        r"$|\Delta N|_{\text{med}}$ & $|\Delta N|_{p95}$ & "
        r"$|\Delta M|_{\text{med}}$ & $|\Delta M|_{p95}$ \\" + "\n"
        r"    \midrule"
    )
    body = []
    for regime in REGIMES:
        for mode in MODES:
            r = _pick_row(df, regime, mode)
            if r is None:
                continue
            body.append(
                f"    {regime} & {mode.replace('_', r' ')} & "
                f"{int(r['n_scenarios'])} & "
                f"{_fmt(r.get('Nk_r2_median'), 4, False)} & "
                f"{_fmt(r.get('Nk_kge_median'), 4, False)} & "
                f"{_fmt(r.get('N_tot_relerr_median'))} & "
                f"{_fmt(r.get('N_tot_relerr_p95'))} & "
                f"{_fmt(r.get('M_dry_relerr_median'))} & "
                f"{_fmt(r.get('M_dry_relerr_p95'))} \\\\"
            )
    footer = (
        r"    \bottomrule" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}"
    )
    path.write_text(header + '\n' + '\n'.join(body) + '\n' + footer + '\n')
    print(f"Wrote {path}")


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_CSV}. Run "
            "`python -m benchmarks.python.export_atmos_metrics` first."
        )
    df = pd.read_csv(SUMMARY_CSV)
    df['hour'] = df['hour'].astype(int)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    emit_markdown(df, MD_PATH)
    emit_latex(df, TEX_PATH)


if __name__ == '__main__':
    main()
