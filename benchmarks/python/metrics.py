"""Skill-metrics for Fortran-vs-JAX ensemble comparison.

Pure numpy. No I/O. Each function accepts two 1-D arrays (ref, test) of
equal shape and returns a scalar. Helpers handle the log-scale Nk case
where bin concentrations span many decades and near-zero bins should be
masked to avoid log(0).

Conventions:
- `ref` is the reference (Fortran); `test` is the candidate (JAX).
- `atol` default 1e-30 protects against divide-by-zero for near-empty bins.
- Functions tolerate NaNs in inputs (propagate nan-safe via nanmean/nansum).
"""
from __future__ import annotations

import numpy as np


_DEFAULT_ATOL = 1e-30


def _as_float(a):
    return np.asarray(a, dtype=np.float64)


def mask_nonzero_bins(arr, atol=_DEFAULT_ATOL):
    """Boolean mask selecting bins with |arr| > atol. Used to drop empty bins
    before computing log-scale stats on aerosol number/mass distributions."""
    return np.abs(_as_float(arr)) > atol


def bias(ref, test):
    """Mean signed error: mean(test - ref). Positive → test overestimates."""
    return float(np.nanmean(_as_float(test) - _as_float(ref)))


def rmse(ref, test):
    """Root-mean-square error."""
    d = _as_float(test) - _as_float(ref)
    return float(np.sqrt(np.nanmean(d * d)))


def nrmse(ref, test):
    """RMSE normalized by the range of ref. Useful when ref spans small values."""
    r = _as_float(ref)
    span = np.nanmax(r) - np.nanmin(r)
    if span < _DEFAULT_ATOL:
        return float('nan')
    return rmse(ref, test) / float(span)


def mape(ref, test, atol=_DEFAULT_ATOL):
    """Mean absolute percentage error with |ref| clamped to atol."""
    r = _as_float(ref)
    t = _as_float(test)
    denom = np.maximum(np.abs(r), atol)
    return float(np.nanmean(np.abs(t - r) / denom))


def r2(ref, test):
    """Coefficient of determination: 1 - SS_res / SS_tot.

    Returns nan if ref is constant (SS_tot ≈ 0). Can be negative when the
    candidate is worse than predicting mean(ref).
    """
    r = _as_float(ref)
    t = _as_float(test)
    ss_res = np.nansum((t - r) ** 2)
    ss_tot = np.nansum((r - np.nanmean(r)) ** 2)
    if ss_tot < _DEFAULT_ATOL:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def pearson_r(ref, test):
    """Pearson correlation coefficient. Returns nan if either series is constant."""
    r = _as_float(ref)
    t = _as_float(test)
    mask = np.isfinite(r) & np.isfinite(t)
    if mask.sum() < 2:
        return float('nan')
    r_m = r[mask]
    t_m = t[mask]
    if np.std(r_m) < _DEFAULT_ATOL or np.std(t_m) < _DEFAULT_ATOL:
        return float('nan')
    return float(np.corrcoef(r_m, t_m)[0, 1])


def kge(ref, test):
    """Kling-Gupta Efficiency (Gupta et al. 2009, J. Hydrol.).

    KGE = 1 - sqrt((r-1)^2 + (sigma_ratio - 1)^2 + (mean_ratio - 1)^2)

    Decomposes skill into correlation, variance ratio, and bias ratio.
    Optimal = 1; climatology = -0.41.
    """
    r = _as_float(ref)
    t = _as_float(test)
    mask = np.isfinite(r) & np.isfinite(t)
    if mask.sum() < 2:
        return float('nan')
    r_m = r[mask]
    t_m = t[mask]

    mu_r = np.mean(r_m)
    mu_t = np.mean(t_m)
    sd_r = np.std(r_m)
    sd_t = np.std(t_m)

    if sd_r < _DEFAULT_ATOL or abs(mu_r) < _DEFAULT_ATOL:
        return float('nan')

    rho = pearson_r(r_m, t_m)
    if not np.isfinite(rho):
        return float('nan')

    alpha = sd_t / sd_r
    beta = mu_t / mu_r
    return float(1.0 - np.sqrt((rho - 1.0) ** 2
                               + (alpha - 1.0) ** 2
                               + (beta - 1.0) ** 2))


def log_bias(ref, test, atol=1e-20):
    """Mean log10 ratio: mean(log10(test) - log10(ref)) over non-zero bins.

    Both arrays floored to `atol` to prevent log(0). Useful for aerosol
    size distributions where Nk spans many decades and a linear bias is
    dominated by the largest bin.
    """
    r = np.maximum(_as_float(ref), atol)
    t = np.maximum(_as_float(test), atol)
    mask = mask_nonzero_bins(_as_float(ref), atol)
    if mask.sum() == 0:
        return float('nan')
    return float(np.nanmean(np.log10(t[mask]) - np.log10(r[mask])))


def max_rel_error(ref, test, atol=_DEFAULT_ATOL):
    """Maximum |test - ref| / max(|ref|, atol). Matches existing
    `_safe_rel_error` in compare_24h.py."""
    r = _as_float(ref)
    t = _as_float(test)
    denom = np.maximum(np.abs(r), atol)
    return float(np.nanmax(np.abs(t - r) / denom))


def total_rel_error(val_ref, val_test, atol=_DEFAULT_ATOL):
    """Relative error between two scalars. Matches existing
    `_total_rel_error` in compare_24h.py."""
    r = float(val_ref)
    t = float(val_test)
    if abs(r) < atol and abs(t) < atol:
        return 0.0
    return abs(t - r) / max(abs(r), atol)


# -------------------------------------------------------------------------
# Bundle helpers
# -------------------------------------------------------------------------

# Metric function names that work on 1-D arrays.
ARRAY_METRICS = ('bias', 'rmse', 'nrmse', 'mape',
                 'r2', 'pearson_r', 'kge', 'log_bias', 'max_relerr')


def compute_all(ref, test, prefix=''):
    """Return {metric_name: value} for all ARRAY_METRICS on flattened inputs.

    NaN-tolerant; returns {f"{prefix}{name}": scalar, ...}.
    """
    r = _as_float(ref).ravel()
    t = _as_float(test).ravel()
    out = {
        f'{prefix}bias':         bias(r, t),
        f'{prefix}rmse':         rmse(r, t),
        f'{prefix}nrmse':        nrmse(r, t),
        f'{prefix}mape':         mape(r, t),
        f'{prefix}r2':           r2(r, t),
        f'{prefix}pearson_r':    pearson_r(r, t),
        f'{prefix}kge':          kge(r, t),
        f'{prefix}log_bias':     log_bias(r, t),
        f'{prefix}max_relerr':   max_rel_error(r, t),
    }
    return out
