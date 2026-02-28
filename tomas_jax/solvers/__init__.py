"""Numerical solvers for aerosol microphysics.

Modules:
    diffrax       - Coagulation ODE solver (Tsit5 + MNFIX, JIT-compiled)
    condensation  - Condensation operator-split driver
                    method='tfl': Sequential TFL (Fortran-faithful)
                    method='ppm': Sequential PPM wrapper
                    method='ppm_jit': Fully JIT-compiled PPM pipeline
                    run_condensation_scan: Scan-fused time loop (fastest)
"""
