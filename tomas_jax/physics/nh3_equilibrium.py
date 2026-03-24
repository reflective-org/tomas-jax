"""NH3/NH4 equilibrium for TOMAS-JAX.

Exact port of eznh3eqm.f (Jeff Pierce, April 2007).

Instantaneous equilibrium partitioning of ammonia between gas and
particle phases. Enforces 2:1 NH4:SO4 molar stoichiometry.

Two cases:
    1. Ammonia-limited (tot_nh3/2 < tot_so4):
       All ammonia goes to particles, distributed proportionally to sulfate.
       No gas-phase NH3 remains.

    2. Ammonia-excess (tot_nh3/2 >= tot_so4):
       Particles filled to 2:1 stoichiometry (fully neutralized).
       Excess NH3 remains in gas phase.

Note:
    Gas-phase ammonia is stored as NH3 (MW=17 g/mol) in Gc[SRTNH4].
    Particle-phase ammonium is stored as NH4 (MW=18 g/mol) in Mk[:, SRTNH4].
    The molecular weight difference is accounted for in the conversions.

References:
    - eznh3eqm.f (Jeff Pierce, April 2007)
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Tuple

from ..core.config import SRTSO4, SRTNH4, MW_NH3, MW_NH4, MW_SO4


def eznh3eqm(
    Gc: jnp.ndarray,
    Mk: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """NH3/NH4 equilibrium between gas and particle phases.

    Port of eznh3eqm.f.

    Args:
        Gc: Gas-phase concentrations [kg/grid cell], shape (N_GAS_SPECIES,)
            Gc[SRTNH4] = gas-phase NH3 mass (stored as NH3, MW=17)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
            Mk[:, SRTNH4] = particle-phase NH4 mass (as NH4, MW=18)

    Returns:
        Gc_new: Updated gas concentrations
        Mk_new: Updated mass concentrations
    """
    nbins = Mk.shape[0]

    # Total kmoles of NH3 (gas as NH3/17 + particle as NH4/18)
    tot_nh3 = Gc[SRTNH4] / MW_NH3 + jnp.sum(Mk[:, SRTNH4]) / MW_NH4

    # Total kmoles of SO4
    tot_so4 = jnp.sum(Mk[:, SRTSO4]) / MW_SO4

    # Sulfate fraction per bin (for distributing NH4)
    so4_total_mass = jnp.sum(Mk[:, SRTSO4])
    sfrac = jnp.where(
        so4_total_mass > 1e-30,
        Mk[:, SRTSO4] / so4_total_mass,
        jnp.zeros(nbins)
    )

    # Case 1: No free ammonia (tot_nh3/2 < tot_so4)
    # Distribute all NH3 proportionally to sulfate
    Gc_no_free = Gc.at[SRTNH4].set(0.0)
    Mk_no_free = Mk.at[:, SRTNH4].set(sfrac * tot_nh3 * MW_NH4)

    # Case 2: Free ammonia (tot_nh3/2 >= tot_so4)
    # Fill particles to 2:1 stoichiometry, excess stays gas phase
    Mk_free = Mk.at[:, SRTNH4].set(Mk[:, SRTSO4] / MW_SO4 * 2.0 * MW_NH4)
    Gc_free = Gc.at[SRTNH4].set((tot_nh3 - tot_so4 * 2.0) * MW_NH3)

    # Select based on ammonia availability
    no_free = (tot_nh3 / 2.0) < tot_so4

    Gc_new = jnp.where(no_free, Gc_no_free, Gc_free)
    Mk_new = jnp.where(no_free, Mk_no_free, Mk_free)

    return Gc_new, Mk_new
