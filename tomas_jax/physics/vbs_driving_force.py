"""VBS driving force calculations for SOA condensation in TOMAS-JAX.

Computes the thermodynamic driving force for condensation/evaporation
of each VBS species onto each size bin:

    dp(k) = pamb − psat(k)

where:
    - pamb = ambient vapor pressure (from gas concentration)
    - psat = equilibrium vapor pressure (Raoult + Kelvin)
    - dp > 0: condensation
    - dp < 0: evaporation

References:
    - soacond.f lines 324–377 (Jeff Pierce, September 2010)
    - Donahue et al. (2006), Environ. Sci. Technol., 40, 2635–2643
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ..core.config import R_GAS, KB, AVOGADRO, MOLAR_MASS_AIR


def calc_boxmass(temp, pres, boxvol):
    """Mass of air in grid cell [kg].

    boxmass = pres × boxvol_m3 × Mair / (R × T)

    Args:
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm³]

    Returns:
        boxmass: Air mass [kg]
    """
    boxvol_m3 = boxvol * 1.0e-6  # cm³ → m³
    return pres * boxvol_m3 * MOLAR_MASS_AIR / (R_GAS * temp)


def calc_ambient_pressure(Gc_species, mw, temp, pres, boxvol):
    """Ambient vapor pressure from gas-phase concentration.

    Converts gas mass in grid cell to partial pressure via ideal gas law:
        pamb = (Gc / MW) / (boxmass / MW_air) × P

    Port of soacond.f line 337.

    Args:
        Gc_species: Gas concentration of this species [kg/grid cell]
        mw: Molecular weight [g/mol]
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm³]

    Returns:
        pamb: Ambient vapor pressure [Pa]
    """
    mw_kg = mw * 1.0e-3
    boxmass = calc_boxmass(temp, pres, boxvol)
    n_gas = Gc_species / mw_kg            # moles of gas
    n_air = boxmass / MOLAR_MASS_AIR      # moles of air
    pamb = jnp.where(n_air > 1.0e-30, (n_gas / n_air) * pres, 0.0)
    return jnp.maximum(pamb, 0.0)


def calc_equilibrium_pressure(cstar_Pa, Mk_species, Mtot_org, kelvin):
    """Equilibrium vapor pressure per bin (Raoult + Kelvin).

    psat(k) = C* × (Mk_j(k) / Mtot_org(k)) × Ke(k)

    Port of soacond.f line 341.

    Args:
        cstar_Pa: Temperature-corrected C* [Pa] (scalar)
        Mk_species: Mass of this species per bin [kg], shape (nbins,)
        Mtot_org: Total organic mass per bin [kg], shape (nbins,)
        kelvin: Kelvin correction factors per bin, shape (nbins,)

    Returns:
        psat: Equilibrium vapor pressure per bin [Pa], shape (nbins,)
    """
    safe_Mtot = jnp.maximum(Mtot_org, 1.0e-30)
    psat = jnp.where(
        Mtot_org > 1.0e-30,
        cstar_Pa * (Mk_species / safe_Mtot) * kelvin,
        0.0,
    )
    return psat


def calc_driving_force(pamb, psat):
    """Condensation/evaporation driving force per bin.

    dp(k) = pamb − psat(k)

    Args:
        pamb: Ambient vapor pressure [Pa] (scalar)
        psat: Equilibrium vapor pressure per bin [Pa], shape (nbins,)

    Returns:
        dp: Driving force [Pa], shape (nbins,)
            Positive = condensation, negative = evaporation.
    """
    return pamb - psat


def calc_equilibrium_mass(pamb, cstar_Pa, Mtot_org, kelvin, Mk_species, Nk,
                          Q=None):
    """Per-particle equilibrium mass change for each bin.

    From soacond.f lines 364–371:
        masseqm = pamb / (C* / Mtot × Ke)
        maddEQ(k) = (masseqm − Mk_j(k)) / Nk(k)

    When Q is provided (Zaveri Approximation 1, fast reactions), the
    effective saturation pressure is psat/Q, so the equilibrium mass is:
        masseqm = pamb / (C* × Ke / (Mtot × Q))
                = pamb × Mtot × Q / (C* × Ke)

    Args:
        pamb: Ambient vapor pressure [Pa]
        cstar_Pa: C* at current temperature [Pa]
        Mtot_org: Total organic mass per bin [kg], shape (nbins,)
        kelvin: Kelvin correction per bin, shape (nbins,)
        Mk_species: Mass of this species per bin [kg], shape (nbins,)
        Nk: Number per bin [#], shape (nbins,)
        Q: Quasi-steady-state parameter per bin, shape (nbins,), or None.
           When provided, equilibrium uses effective psat/Q (Zaveri Approx 1).

    Returns:
        maddEQ: Per-particle equilibrium mass change [kg], shape (nbins,)
    """
    safe_Mtot = jnp.maximum(Mtot_org, 1.0e-30)
    safe_Nk = jnp.maximum(Nk, 1.0e-30)

    # Equilibrium total mass of this species in each bin
    # masseqm = pamb / (C* × Ke / Mtot)  [soacond.f line 369]
    # With Q correction: masseqm = pamb × Mtot × Q / (C* × Ke)
    Q_factor = jnp.maximum(Q, 1.0e-10) if Q is not None else 1.0
    masseqm = jnp.where(
        (Mtot_org > 1.0e-30) & (cstar_Pa > 1.0e-30),
        pamb * safe_Mtot * Q_factor / (cstar_Pa * kelvin),
        0.0,
    )

    # Per-particle change needed
    maddEQ = jnp.where(
        Nk > 1.0e-10,
        (masseqm - Mk_species) / safe_Nk,
        0.0,
    )

    return maddEQ
