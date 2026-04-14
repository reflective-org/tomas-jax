"""Numerical solvers for aerosol microphysics.

Modules:
    diffrax       - Coagulation ODE solver (Tsit5 + MNFIX, JIT-compiled)
    condensation  - Condensation operator-split driver
                    method='ppm_jit': Fully JIT-compiled PPM pipeline (default, fastest)
                    method='tfl_jit': Fully JIT-compiled TFL (Fortran-matching)
                    method='tfl': Sequential TFL (Fortran-faithful)
                    method='ppm': Sequential PPM wrapper
                    run_condensation_scan: Scan-fused PPM time loop
                    run_condensation_scan_tfl: Scan-fused TFL time loop
"""
