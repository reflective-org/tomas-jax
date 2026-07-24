"""Batched state container for the GPU-fast reduced model.

All per-cell fields carry a leading cell axis C. The bin grid xk is
shared across cells (same 40-bin mass-doubling grid for every cell).
"""
from typing import Any, NamedTuple

from .config import NBINS, ICOMP, NGAS, xk_boundaries
import jax.numpy as jnp


class FastState(NamedTuple):
    """Immutable batched state for the reduced model.

    Attributes:
        Nk: Number concentration [#/grid cell], shape (C, NBINS)
        Mk: Mass concentration [kg/grid cell], shape (C, NBINS, 2) [SO4, H2O]
        Gc: Gas concentrations [kg/grid cell], shape (C, 2) [H2SO4, SO2]
        xk: Bin boundaries [kg], shape (NBINS+1,) — shared across cells
        temp: Temperature [K], shape (C,)
        pres: Pressure [Pa], shape (C,)
        boxvol: Grid cell volume [cm^3], shape (C,)
        rh: Relative humidity [fraction 0-1], shape (C,)
    """
    Nk: jnp.ndarray
    Mk: jnp.ndarray
    Gc: jnp.ndarray
    xk: jnp.ndarray
    temp: jnp.ndarray
    pres: jnp.ndarray
    boxvol: jnp.ndarray
    rh: jnp.ndarray

    @classmethod
    def create(
        cls,
        Nk: Any,
        Mk: Any,
        Gc: Any = None,
        xk: Any = None,
        temp: Any = 273.0,
        pres: Any = 101325.0,
        boxvol: Any = 1.0e6,
        rh: Any = 0.5,
    ) -> "FastState":
        """Create a batched state with float64 enforcement.

        Nk may be (C, NBINS) or (NBINS,) (promoted to C=1). Scalar
        environment fields are broadcast to (C,).
        """
        Nk_arr = jnp.atleast_2d(jnp.asarray(Nk, dtype=jnp.float64))
        C = Nk_arr.shape[0]
        Mk_arr = jnp.asarray(Mk, dtype=jnp.float64).reshape(C, NBINS, ICOMP)
        if Gc is None:
            Gc_arr = jnp.zeros((C, NGAS), dtype=jnp.float64)
        else:
            Gc_arr = jnp.asarray(Gc, dtype=jnp.float64).reshape(C, NGAS)
        xk_arr = xk_boundaries() if xk is None else jnp.asarray(xk, dtype=jnp.float64)

        def _percell(v):
            return jnp.broadcast_to(
                jnp.asarray(v, dtype=jnp.float64), (C,)
            )

        assert Nk_arr.shape == (C, NBINS), f"Nk shape {Nk_arr.shape}"
        assert xk_arr.shape == (NBINS + 1,), f"xk shape {xk_arr.shape}"

        return cls(
            Nk=Nk_arr,
            Mk=Mk_arr,
            Gc=Gc_arr,
            xk=xk_arr,
            temp=_percell(temp),
            pres=_percell(pres),
            boxvol=_percell(boxvol),
            rh=_percell(rh),
        )

    def update(self, **kwargs) -> "FastState":
        """Return a new state with updated fields."""
        return self._replace(**kwargs)

    @property
    def ncells(self) -> int:
        return self.Nk.shape[0]
