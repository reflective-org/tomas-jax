"""Run all 13 benchmark levels and generate comparison report.

Usage:
    python -m benchmarks.python.compare_all [--data-dir PATH] [--verbose]
"""
import os
import sys
import argparse

from . import (
    level01_initbounds,
    level02_air_properties,
    level03_density,
    level04_properties,
    level05_kernel,
    level06_tfl,
    level07_rates,
    level08_mnfix,
    level09_euler_step,
    level10_full_coag,
    level11_condsink,
    level12_isolated_cond,
    level13_coagcond,
    level14_ppm_vs_tfl,
    level15_ppm_performance,
)
from .utils import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="TOMAS FORTRAN vs JAX Coagulation Benchmark"
    )
    parser.add_argument(
        '--data-dir',
        default=os.path.join(os.path.dirname(__file__),
                             '..', 'fortran', 'output'),
        help='Path to FORTRAN benchmark output directory'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Print detailed comparison results'
    )
    parser.add_argument(
        '--report',
        default=os.path.join(os.path.dirname(__file__),
                             '..', 'results', 'comparison_report.md'),
        help='Path to write markdown report'
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Run the FORTRAN benchmark harness first:")
        print("  cd benchmarks/fortran && make && make run")
        sys.exit(1)

    print("=" * 60)
    print("TOMAS FORTRAN vs JAX Benchmark")
    print("=" * 60)
    print(f"Data directory: {data_dir}")

    all_results = []

    levels = [
        ("Level 1", level01_initbounds),
        ("Level 2", level02_air_properties),
        ("Level 3", level03_density),
        ("Level 4", level04_properties),
        ("Level 5", level05_kernel),
        ("Level 6", level06_tfl),
        ("Level 7", level07_rates),
        ("Level 8", level08_mnfix),
        ("Level 9", level09_euler_step),
        ("Level 10", level10_full_coag),
        ("Level 11", level11_condsink),
        ("Level 12", level12_isolated_cond),
        ("Level 13", level13_coagcond),
        ("Level 14", level14_ppm_vs_tfl),
        ("Level 15", level15_ppm_performance),
    ]

    for name, module in levels:
        try:
            results = module.run(data_dir, verbose=args.verbose)
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"\n=== {name}: SKIPPED (missing data) ===")
            print(f"  {e}")
        except Exception as e:
            print(f"\n=== {name}: ERROR ===")
            print(f"  {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_pass = sum(1 for r in all_results if r['passed'])
    n_total = len(all_results)
    print(f"\n  {n_pass}/{n_total} comparisons passed\n")

    print(f"  {'Level':<8} {'Component':<30} {'Max Rel Err':<15} {'Status':<8}")
    print(f"  {'-'*8} {'-'*30} {'-'*15} {'-'*8}")
    for r in all_results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  {str(r.get('level', '-')):<8} {r['name']:<30} "
              f"{r['max_rel_error']:<15.4e} {status:<8}")

    # Generate report
    try:
        generate_report(all_results, args.report)
    except Exception as e:
        print(f"\nWarning: Could not generate report: {e}")

    # Exit code
    if n_pass == n_total:
        print(f"\nAll {n_total} comparisons PASSED.")
        return 0
    else:
        print(f"\n{n_total - n_pass} comparisons FAILED.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
