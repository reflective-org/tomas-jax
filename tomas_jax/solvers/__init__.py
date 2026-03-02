"""Numerical solvers for aerosol microphysics.

Modules:
    diffrax       - Coagulation ODE solver (Tsit5 + MNFIX, JIT-compiled)
    condensation  - Condensation operator-split driver
                    method='tfl': Sequential TFL (Fortran-faithful)
                    method='tfl_jit': Fully JIT-compiled TFL (Fortran-matching, recommended)
                    method='ppm': Sequential PPM wrapper
                    method='ppm_jit': Fully JIT-compiled PPM pipeline
                    run_condensation_scan_tfl: Scan-fused TFL time loop (fastest + accurate)
                    run_condensation_scan: Scan-fused PPM time loop
"""
