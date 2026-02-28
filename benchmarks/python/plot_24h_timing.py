"""Timing comparison plots for 24-hour benchmark.

Loads Fortran (timing_fortran.csv) and JAX (timing_jax.csv) timing data
and generates performance comparison figures.

Usage::

    python -m benchmarks.python.plot_24h_timing
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
FORTRAN_TIMING = os.path.join(_BASE_DIR, 'fortran', 'output', '24h', 'timing_fortran.csv')
JAX_TIMING = os.path.join(_BASE_DIR, 'results', '24h', 'timing_jax.csv')
_RESULTS_OUT = os.path.join(_BASE_DIR, 'results')


def load_fortran_timing():
    """Load Fortran timing CSV.

    Returns:
        dict: {scenario_id: {'coag': s, 'cond': s, 'combined': s}}
    """
    if not os.path.exists(FORTRAN_TIMING):
        return None
    data = {}
    with open(FORTRAN_TIMING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['scenario_id'].strip())
            data[sid] = {
                'coag': float(row['coag_s'].strip()),
                'cond': float(row['cond_s'].strip()),
                'combined': float(row['combined_s'].strip()),
            }
    return data


def load_jax_timing():
    """Load JAX timing CSV.

    Returns:
        dict: {(scenario_id, mode, method): wall_time_s}
    """
    if not os.path.exists(JAX_TIMING):
        return None
    data = {}
    with open(JAX_TIMING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['scenario_id']), row['mode'], row['method'])
            data[key] = float(row['wall_time_s'])
    return data


def plot_timing_bars(output_dir=None):
    """Figure A: Bar chart — per-mode average wall-clock time."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    os.makedirs(output_dir, exist_ok=True)

    fort = load_fortran_timing()
    jax = load_jax_timing()

    if fort is None and jax is None:
        print("No timing data found. Skipping timing bar chart.")
        return

    modes = ['coag_only', 'cond_only', 'combined']
    mode_labels = {'coag_only': 'Coag Only', 'cond_only': 'Cond Only', 'combined': 'Combined'}
    fort_mode_keys = {'coag_only': 'coag', 'cond_only': 'cond', 'combined': 'combined'}

    # Compute averages
    means = {}
    for label in ['Fortran', 'JAX-TFL', 'JAX-PPM']:
        means[label] = []

    for mode in modes:
        # Fortran
        if fort:
            fkey = fort_mode_keys[mode]
            fort_times = [fort[sid][fkey] for sid in fort if fkey in fort[sid]]
            means['Fortran'].append(np.mean(fort_times) if fort_times else 0)
        else:
            means['Fortran'].append(0)

        # JAX-TFL
        if jax:
            tfl_times = [jax[k] for k in jax if k[1] == mode and k[2] == 'tfl']
            means['JAX-TFL'].append(np.mean(tfl_times) if tfl_times else 0)
        else:
            means['JAX-TFL'].append(0)

        # JAX-PPM
        if jax:
            ppm_times = [jax[k] for k in jax if k[1] == mode and k[2] == 'ppm']
            means['JAX-PPM'].append(np.mean(ppm_times) if ppm_times else 0)
        else:
            means['JAX-PPM'].append(0)

    x = np.arange(len(modes))
    width = 0.25
    colors = {'Fortran': '0.5', 'JAX-TFL': 'tab:blue', 'JAX-PPM': 'tab:red'}

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(['Fortran', 'JAX-TFL', 'JAX-PPM']):
        vals = means[label]
        if any(v > 0 for v in vals):
            ax.bar(x + i * width, vals, width, label=label, color=colors[label], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([mode_labels[m] for m in modes])
    ax.set_ylabel('Average Wall-Clock Time [s]')
    ax.set_title('24-Hour Benchmark: Average Timing per Mode')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_timing_bars.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def plot_timing_scatter(output_dir=None):
    """Figure B: Scatter — per-scenario Fortran vs JAX timing."""
    if output_dir is None:
        output_dir = _RESULTS_OUT
    os.makedirs(output_dir, exist_ok=True)

    fort = load_fortran_timing()
    jax = load_jax_timing()

    if fort is None or jax is None:
        print("Need both Fortran and JAX timing for scatter. Skipping.")
        return

    fort_mode_keys = {'coag_only': 'coag', 'cond_only': 'cond', 'combined': 'combined'}
    mode_labels = {'coag_only': 'Coag Only', 'cond_only': 'Cond Only', 'combined': 'Combined'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Per-Scenario Timing: Fortran vs JAX', fontsize=14)

    for idx, mode in enumerate(['coag_only', 'cond_only', 'combined']):
        ax = axes[idx]
        ax.set_title(mode_labels[mode])
        ax.set_xlabel('Fortran [s]')
        ax.set_ylabel('JAX [s]')

        fkey = fort_mode_keys[mode]

        for method, color, marker in [('tfl', 'tab:blue', 'o'), ('ppm', 'tab:red', '^')]:
            if mode == 'coag_only' and method == 'ppm':
                continue
            f_vals, j_vals = [], []
            for sid in sorted(fort.keys()):
                jax_key = (sid, mode, method)
                if jax_key in jax:
                    f_vals.append(fort[sid][fkey])
                    j_vals.append(jax[jax_key])

            if f_vals:
                f_vals = np.array(f_vals)
                j_vals = np.array(j_vals)
                ax.scatter(f_vals, j_vals, alpha=0.6, s=20, color=color,
                           marker=marker, label=method.upper())
                # Median speedup
                speedups = f_vals / np.maximum(j_vals, 1e-6)
                med_speedup = np.median(speedups)
                ax.text(0.05, 0.95 if method == 'tfl' else 0.88,
                        f'{method.upper()} median speedup: {med_speedup:.1f}x',
                        transform=ax.transAxes, fontsize=8, va='top',
                        color=color)

        # 1:1 line
        all_vals = []
        for sid in fort:
            all_vals.append(fort[sid][fkey])
        if all_vals:
            vmin = min(all_vals) * 0.5
            vmax = max(all_vals) * 2.0
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, label='1:1')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    path = os.path.join(output_dir, '24h_timing_scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def print_timing_summary():
    """Print performance summary table to stdout."""
    fort = load_fortran_timing()
    jax = load_jax_timing()

    print("\n" + "=" * 70)
    print("24-Hour Benchmark Timing Summary")
    print("=" * 70)

    modes = ['coag_only', 'cond_only', 'combined']
    fort_mode_keys = {'coag_only': 'coag', 'cond_only': 'cond', 'combined': 'combined'}

    print(f"\n{'Mode':<12s}  {'Fortran':>10s}  {'JAX-TFL':>10s}  "
          f"{'JAX-PPM':>10s}  {'Speedup TFL':>12s}  {'Speedup PPM':>12s}")
    print("-" * 70)

    for mode in modes:
        fkey = fort_mode_keys[mode]
        # Fortran mean
        if fort:
            f_times = [fort[sid][fkey] for sid in fort]
            f_mean = np.mean(f_times)
            f_str = f"{f_mean:.2f}s"
        else:
            f_mean = None
            f_str = "N/A"

        # JAX-TFL mean
        if jax:
            tfl_times = [jax[k] for k in jax if k[1] == mode and k[2] == 'tfl']
            tfl_mean = np.mean(tfl_times) if tfl_times else None
            tfl_str = f"{tfl_mean:.2f}s" if tfl_mean else "N/A"
        else:
            tfl_mean = None
            tfl_str = "N/A"

        # JAX-PPM mean
        if jax:
            ppm_times = [jax[k] for k in jax if k[1] == mode and k[2] == 'ppm']
            ppm_mean = np.mean(ppm_times) if ppm_times else None
            ppm_str = f"{ppm_mean:.2f}s" if ppm_mean else "N/A"
        else:
            ppm_mean = None
            ppm_str = "N/A"

        # Speedups
        if f_mean and tfl_mean and tfl_mean > 0:
            spd_tfl = f"{f_mean / tfl_mean:.1f}x"
        else:
            spd_tfl = "N/A"
        if f_mean and ppm_mean and ppm_mean > 0:
            spd_ppm = f"{f_mean / ppm_mean:.1f}x"
        else:
            spd_ppm = "N/A"

        print(f"{mode:<12s}  {f_str:>10s}  {tfl_str:>10s}  "
              f"{ppm_str:>10s}  {spd_tfl:>12s}  {spd_ppm:>12s}")

    print("=" * 70)


if __name__ == '__main__':
    plot_timing_bars()
    plot_timing_scatter()
    print_timing_summary()
