"""GPU-fast reduced TOMAS: SO4+H2O aerosol, H2SO4+SO2 gases, 40 bins.

A natively-batched (leading cell axis) re-composition of the TOMAS
microphysics for global-model coupling: SO2+OH chemistry, Dunne 2016
neutral-binary nucleation, coagulation, PPM condensation, and
Tabazadeh (1997) H2SO4/H2O water equilibrium. float64 everywhere.

Design target: 1M cells x 6 simulated hours in <10 s on an A100/H100.
See docs/gpu_fast.md.
"""
from .config import NBINS, ICOMP, ICOMP_NODIAG, SRTSO4, SRTH2O, GH2SO4, GSO2, NGAS
from .state import FastState
from .step import fast_step, make_fast_step
from .run import run_fast
