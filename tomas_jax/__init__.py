"""
TOMAS-JAX: TwO-Moment Aerosol Sectional model in JAX.

A modern, high-performance implementation of aerosol microphysics
with coagulation (JIT-compiled) and condensation (Fortran-faithful).

Modules:
    core        - Configuration, state containers, MNFIX
    physics     - Coagulation, condensation, gas properties, equilibria
    solvers     - ODE integration (diffrax) and condensation driver
    utils       - Plotting and diagnostics

Quick start::

    from tomas_jax import TomasState, CoagulationSolver
    from tomas_jax.solvers.condensation import condensation_step
"""

__version__ = "0.2.0"

from .core.state import TomasState
from .solvers.diffrax import diffrax_step as CoagulationSolver