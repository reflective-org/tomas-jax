"""Build presentation-ready tables from the per-scenario summary CSV.

Produces one markdown table (paste into slides / README) and one LaTeX
table (booktabs) covering the ensemble skill at hour 24:

    mode × method_pair: n, Nk R² (median), Nk KGE (median),
    |N_tot relerr| (median, p95), |M_dry relerr| (median, p95)

Usage::

    python -m benchmarks.python.build_presentation_tables

Outputs in ``benchmarks/results/24h/stats/``:
    table_ensemble_summary.md
    table_ensemble_summary.tex
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
STATS_DIR = _BASE / 'results' / '24h' / 'stats'
SUMMARY_CSV = STATS_DIR / 'per_scenario_metrics_summary.csv'
MD_PATH = STATS_DIR / 'table_ensemble_summary.md'
TEX_PATH = STATS_DIR / 'table_ensemble_summary.tex'

REPORT_HOUR = '24'


def _fmt(val, digits=3, scientific=True):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return '—'
    if scientific:
        return f'{val:.{digits}e}'
    return f'{val:.{digits}f}'


def build_rows(df: pd.DataFrame) -> list[dict]:
    sub = df[df['hour'].astype(str) == REPORT_HOUR].copy()
    # Skip ppm_vs_tfl in the presentation table (Fortran is the ground truth)
    sub = sub[sub['method_pair'].isin(('tfl_vs_fortran', 'ppm_vs_fortran'))]
    return sub.to_dict(orient='records')


def emit_markdown(rows: list[dict], path: Path) -> None:
    hdr = ('| Mode | Method | n | Nk R² (med) | Nk KGE (med) | '
           '|N_tot relerr| med | |N_tot relerr| p95 | '
           '|M_dry relerr| med | |M_dry relerr| p95 |')
    sep = ('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    lines = [
        '# Ensemble Summary — Hour 24',
        '',
        f'Aggregated across 49 Latin Hypercube scenarios '
        f'(S50 excluded — Fortran crashes). Source: '
        f'`per_scenario_metrics_summary.csv`.',
        '',
        hdr, sep,
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['method_pair'].replace('_vs_', ' vs ')} "
            f"| {int(r['n_scenarios'])} "
            f"| {_fmt(r.get('Nk_r2_median'), 4, scientific=False)} "
            f"| {_fmt(r.get('Nk_kge_median'), 4, scientific=False)} "
            f"| {_fmt(r.get('N_tot_relerr_median'))} "
            f"| {_fmt(r.get('N_tot_relerr_p95'))} "
            f"| {_fmt(r.get('M_dry_relerr_median'))} "
            f"| {_fmt(r.get('M_dry_relerr_p95'))} |"
        )
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    lines.append('- **Nk R²**: coefficient of determination on the 36-bin size '
                 'distribution at h24. Close to 1.0 ⇒ JAX reproduces Fortran '
                 'per-bin.')
    lines.append('- **KGE**: Kling-Gupta Efficiency — combines correlation, '
                 'variance ratio, and bias ratio. Optimal = 1; climatology ≈ -0.41.')
    lines.append('- **N_tot / M_dry relerr**: scalar relative error on total '
                 'number / dry mass at h24. p95 shows the worst-5% tail.')
    lines.append('')
    path.write_text('\n'.join(lines))
    print(f"Wrote markdown table → {path}")


def emit_latex(rows: list[dict], path: Path) -> None:
    header = (
        r"\begin{table}[t]" + "\n"
        r"  \centering" + "\n"
        r"  \caption{TOMAS-JAX vs Fortran — ensemble skill at hour 24 across "
        r"49 Latin Hypercube scenarios.}" + "\n"
        r"  \label{tab:ensemble_summary}" + "\n"
        r"  \begin{tabular}{llrrrrrrr}" + "\n"
        r"    \toprule" + "\n"
        r"    Mode & Method & $n$ & $R^2$ & KGE & "
        r"$|\Delta N|_{\text{med}}$ & $|\Delta N|_{p95}$ & "
        r"$|\Delta M|_{\text{med}}$ & $|\Delta M|_{p95}$ \\" + "\n"
        r"    \midrule"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"    {r['mode'].replace('_', r'\_')} & "
            f"{r['method_pair'].replace('_vs_', ' vs ').replace('_', r' ')} & "
            f"{int(r['n_scenarios'])} & "
            f"{_fmt(r.get('Nk_r2_median'), 4, scientific=False)} & "
            f"{_fmt(r.get('Nk_kge_median'), 4, scientific=False)} & "
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
    path.write_text(header + '\n' + '\n'.join(body_lines) + '\n' + footer + '\n')
    print(f"Wrote LaTeX table → {path}")


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_CSV}. Run "
            "`python -m benchmarks.python.export_metrics_csv` first."
        )
    df = pd.read_csv(SUMMARY_CSV)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows(df)
    emit_markdown(rows, MD_PATH)
    emit_latex(rows, TEX_PATH)


if __name__ == '__main__':
    main()
