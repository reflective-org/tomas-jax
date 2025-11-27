"""
TOMAS-JAX: TwO-Moment Aerosol Sectional model in JAX

A modern, high-performance implementation of aerosol microphysics.
"""

__version__ = "0.1.0"

# Expose the solver and state so users can do:
# from tomas_jax import TomasState, CoagulationSolver
from .core.state import TomasState
from .solvers.diffrax import diffrax_step as CoagulationSolver