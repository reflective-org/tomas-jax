"""TOMAS state structures with enforced float64 precision.

This module defines the immutable state container for the simulation.
It strictly enforces JAX float64 types to prevent numerical instability
in aerosol microphysics (values spanning 1e-23 to 1e12).
"""
import jax
# Import config FIRST to trigger the x64 update
from .config import ICOMP, N_GAS_SPECIES
import jax.numpy as jnp
from typing import NamedTuple, Any

class TomasState(NamedTuple):
    """Immutable, JIT-compatible state for TOMAS simulation.

    Attributes:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm³]
    """
    Nk: jnp.ndarray
    Mk: jnp.ndarray
    xk: jnp.ndarray
    temp: jnp.ndarray  # Scalar array
    pres: jnp.ndarray  # Scalar array
    boxvol: jnp.ndarray  # Scalar array
    Gc: jnp.ndarray    # Gas-phase concentrations [kg/grid cell], shape (N_GAS_SPECIES,)
    rh: jnp.ndarray    # Relative humidity [fraction 0-1]
    alpha: jnp.ndarray # Accommodation coefficient (scalar)

    @classmethod
    def create(
        cls,
        Nk: Any,
        Mk: Any,
        xk: Any,
        temp: float,
        pres: float,
        boxvol: float,
        Gc: Any = None,
        rh: float = 0.5,
        alpha: float = 1.0
    ) -> "TomasState":
        """Factory method to create a state with strict type enforcement.

        This is the preferred way to initialize the state. It ensures all
        inputs are cast to jnp.float64 and validates shapes.
        """
        # 1. Cast to JAX Arrays with float64 enforcement
        Nk_arr = jnp.asarray(Nk, dtype=jnp.float64)
        Mk_arr = jnp.asarray(Mk, dtype=jnp.float64)
        xk_arr = jnp.asarray(xk, dtype=jnp.float64)

        # Convert scalars to 0-dim JAX arrays (prevents JIT recompilation on value change)
        t_arr = jnp.asarray(temp, dtype=jnp.float64)
        p_arr = jnp.asarray(pres, dtype=jnp.float64)
        v_arr = jnp.asarray(boxvol, dtype=jnp.float64)

        # Gas-phase concentrations (default: zeros)
        if Gc is None:
            Gc_arr = jnp.zeros(N_GAS_SPECIES, dtype=jnp.float64)
        else:
            Gc_arr = jnp.asarray(Gc, dtype=jnp.float64)

        rh_arr = jnp.asarray(rh, dtype=jnp.float64)
        alpha_arr = jnp.asarray(alpha, dtype=jnp.float64)

        # 2. Shape Integrity Checks
        # Only run these checks outside of JIT compilation (concrete values)
        # Inside JIT, shapes are static knowns, but values are tracers.
        if isinstance(Nk_arr, jnp.ndarray):
            assert Nk_arr.ndim == 1, f"Nk must be 1D, got shape {Nk_arr.shape}"
            assert Mk_arr.shape[0] == Nk_arr.shape[0], (
                f"Mk bins {Mk_arr.shape[0]} != Nk bins {Nk_arr.shape[0]}"
            )

        return cls(
            Nk=Nk_arr,
            Mk=Mk_arr,
            xk=xk_arr,
            temp=t_arr,
            pres=p_arr,
            boxvol=v_arr,
            Gc=Gc_arr,
            rh=rh_arr,
            alpha=alpha_arr
        )

    def update(self, **kwargs) -> "TomasState":
        """Return a new state with updated fields (Functional setter)."""
        # _replace is a standard NamedTuple method
        return self._replace(**kwargs)

    def __repr__(self):
        """Clean string representation for debugging."""
        return (
            f"TomasState(nbins={self.Nk.shape[0]}, "
            f"ncomp={self.Mk.shape[1]}, "
            f"temp={self.temp:.2f}K, "
            f"pres={self.pres:.2f}Pa, "
            f"rh={self.rh:.2f})"
        )