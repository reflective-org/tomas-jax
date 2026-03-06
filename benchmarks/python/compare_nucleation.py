"""Parameterization-level comparison: JAX vs Fortran nucleation rates.

Loads Fortran CSV output from tomas_fortran/output/ and runs the same
test cases through JAX. Prints pass/fail table with relative errors.

Usage::

    python -m benchmarks.python.compare_nucleation
"""
import os
import numpy as np
import jax.numpy as jnp

from tomas_jax.physics.nucleation import (
    ricco_nucleation_rate,
    dunne_nucleation_rate,
)

FORTRAN_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'tomas_fortran', 'output'
)
RTOL = 1e-10


def load_fortran_ricco():
    """Load Fortran Riccobono CSV output."""
    path = os.path.join(FORTRAN_DIR, 'nucleation_ricco.csv')
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    return data  # columns: temp, h2so4, org, fn


def load_fortran_dunne():
    """Load Fortran Dunne CSV output."""
    path = os.path.join(FORTRAN_DIR, 'nucleation_dunne.csv')
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    return data  # columns: temp, fion, h2so4, nh3, Mair, fn, Jbn, Jtn, Jbi, Jti


def compare_ricco(fort_data):
    """Compare Riccobono: JAX vs Fortran."""
    print("\n" + "=" * 70)
    print("Riccobono 2014 Comparison (JAX vs Fortran)")
    print("=" * 70)
    print(f"{'Case':>4s}  {'T':>7s}  {'H2SO4':>10s}  {'org':>10s}  "
          f"{'Fort fn':>12s}  {'JAX fn':>12s}  {'RelErr':>12s}  {'Pass':>4s}")
    print("-" * 70)

    all_pass = True
    for i, row in enumerate(fort_data):
        temp, h2so4, org, fn_fort = row

        fn_jax = float(ricco_nucleation_rate(
            jnp.float64(temp), jnp.float64(h2so4), jnp.float64(org)
        ))

        if abs(fn_fort) > 0:
            rel_err = abs(fn_jax - fn_fort) / abs(fn_fort)
        else:
            rel_err = abs(fn_jax)

        passed = rel_err < RTOL
        if not passed:
            all_pass = False

        print(f"{i+1:4d}  {temp:7.0f}  {h2so4:10.2e}  {org:10.2e}  "
              f"{fn_fort:12.4e}  {fn_jax:12.4e}  {rel_err:12.4e}  "
              f"{'OK' if passed else 'FAIL':>4s}")

    return all_pass


def compare_dunne(fort_data):
    """Compare Dunne: JAX vs Fortran (total rate + 4 mechanisms)."""
    print("\n" + "=" * 90)
    print("Dunne 2016 Comparison (JAX vs Fortran)")
    print("=" * 90)
    print(f"{'Case':>4s}  {'T':>7s}  {'H2SO4':>10s}  {'NH3':>10s}  {'fion':>6s}  "
          f"{'Fort fn':>12s}  {'JAX fn':>12s}  {'RelErr':>12s}  {'Pass':>4s}")
    print("-" * 90)

    all_pass = True
    for i, row in enumerate(fort_data):
        temp, fion, h2so4, nh3, Mair = row[:5]
        fn_fort, Jbn_f, Jtn_f, Jbi_f, Jti_f = row[5:]

        fn_jax, Jbn_j, Jtn_j, Jbi_j, Jti_j = dunne_nucleation_rate(
            jnp.float64(temp), jnp.float64(fion),
            jnp.float64(h2so4), jnp.float64(nh3), jnp.float64(Mair)
        )
        fn_jax = float(fn_jax)
        Jbn_j = float(Jbn_j)
        Jtn_j = float(Jtn_j)
        Jbi_j = float(Jbi_j)
        Jti_j = float(Jti_j)

        if abs(fn_fort) > 0:
            rel_err_fn = abs(fn_jax - fn_fort) / abs(fn_fort)
        else:
            rel_err_fn = abs(fn_jax)

        passed = rel_err_fn < RTOL
        if not passed:
            all_pass = False

        print(f"{i+1:4d}  {temp:7.0f}  {h2so4:10.2e}  {nh3:10.2e}  {fion:6.1f}  "
              f"{fn_fort:12.4e}  {fn_jax:12.4e}  {rel_err_fn:12.4e}  "
              f"{'OK' if passed else 'FAIL':>4s}")

        # Check individual mechanisms
        for name, jval, fval in [('Jbn', Jbn_j, Jbn_f),
                                  ('Jtn', Jtn_j, Jtn_f),
                                  ('Jbi', Jbi_j, Jbi_f),
                                  ('Jti', Jti_j, Jti_f)]:
            if abs(fval) > 0:
                mech_err = abs(jval - fval) / abs(fval)
            else:
                mech_err = abs(jval)

            if mech_err > RTOL:
                all_pass = False
                print(f"      {name}: Fort={fval:12.4e} JAX={jval:12.4e} "
                      f"err={mech_err:12.4e} FAIL")

    return all_pass


def main():
    print("Nucleation Parameterization Benchmark: JAX vs Fortran")
    print(f"Tolerance: rtol = {RTOL:.0e}")

    ricco_data = load_fortran_ricco()
    dunne_data = load_fortran_dunne()

    ricco_pass = compare_ricco(ricco_data)
    dunne_pass = compare_dunne(dunne_data)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Riccobono 2014: {'ALL PASS' if ricco_pass else 'SOME FAILURES'}")
    print(f"  Dunne 2016:     {'ALL PASS' if dunne_pass else 'SOME FAILURES'}")

    if ricco_pass and dunne_pass:
        print("\n  >>> ALL NUCLEATION PARAMETERIZATIONS MATCH TO MACHINE PRECISION <<<")
        return 0
    else:
        print("\n  >>> SOME TESTS FAILED <<<")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
