"""Aerosol density calculations with float64 precision.

Port of aerodens.f (lines 1-209).
Optimized for JAX with Horner's Method and direct Mk array integration.

Reference:
    Tang, I. N. (1997). Thermodynamic and optical properties of mixed-salt
    aerosols of atmospheric importance. JGR, 102, 1883-1893.
"""
import jax
# Enforce float64 for thermodynamics
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import NamedTuple

# Import indices from state (assuming standard TOMAS configuration)
# You can also pass these as arguments if they vary dynamically
from ..core.config import SRTSO4, SRTNH4, SRTH2O #, SRTNACL

# Molecular Weights [g/mol] -> used as ratios, so units cancel or matter relatively
MWSO4 = 96.0
MWNO3 = 62.0
MWNH4 = 18.0
MWNACL = 58.45
MWH2O = 18.0

MWAN = MWNH4 + MWNO3       # 80.0
MWS0 = MWSO4 + 2.0         # 98.0 (H2SO4)
MWS1 = MWSO4 + 1.0 + MWNH4 # 115.0 (NH4HSO4)
MWS2 = 2.0 * MWNH4 + MWSO4 # 132.0 ((NH4)2SO4)

def calc_density(Mk: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate density of mixed aerosol particles for all bins.

    Args:
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)

    Returns:
        Density [kg/m³], shape (ibins,)
    """
    # 1. Extract Species Mass
    # Slicing Mk is efficient (view, not copy)
    mso4 = Mk[:, SRTSO4]
    mnh4 = Mk[:, SRTNH4]
    mh2o = Mk[:, SRTH2O]
    mnacl = 0.0
    
    # NOTE: TOMAS usually lumps Nitrate into "Mass of H2O" or separate. 
    # If NO3 is tracked explicitly, add its index. 
    # For this port, assuming 0.0 if not tracked, or passed in specific slot.
    # Here we assume standard TOMAS where NO3 might be 0.0 for now.
    mno3 = jnp.zeros_like(mso4) 

    # 2. Total Mass & Early Exit Check
    mtot = mso4 + mno3 + mnh4 + mnacl + mh2o
    
    # 3. Molar Calculations
    nso4 = mso4 / MWSO4
    nno3 = mno3 / MWNO3
    nnh4 = mnh4 / MWNH4
    nnacl = mnacl / MWNACL
    # nh2o not needed in moles for stoichiometry, only mass for density

    # 4. Handle "Excess Nitrate" Logic (Fortran Port)
    # If nno3 > nnh4, the code converts excess nitrate to pseudo-sulfate behavior
    excess_no3 = jnp.maximum(nno3 - nnh4, 0.0)
    nso4 = nso4 + excess_no3
    nno3 = jnp.minimum(nno3, nnh4)

    # 5. Determine Sulfate Speciation (Ammonium-Sulfate Ratio)
    # ASR = moles NH4 / moles SO4 (after neutralizing NO3)
    nso4_safe = jnp.maximum(nso4, 1e-30)
    
    # Calculate available ammonium for sulfate
    # (Subtract ammonium used by nitrate)
    nh4_avail = jnp.maximum(nnh4 - nno3, 0.0)
    
    asr = nh4_avail / nso4_safe
    asr = jnp.clip(asr, 0.0, 2.0)

    # Distribute Sulfate:
    # ASR < 1: H2SO4 (s0) + NH4HSO4 (s1)
    # ASR >= 1: NH4HSO4 (s1) + (NH4)2SO4 (s2)
    
    mask_high = asr >= 1.0
    
    # Moles of species
    ns1 = jnp.where(mask_high, nso4 * (2.0 - asr), nso4 * asr)
    ns2 = jnp.where(mask_high, nso4 * (asr - 1.0), 0.0)
    ns0 = jnp.where(mask_high, 0.0, nso4 * (1.0 - asr))
    nan = nno3 # Ammonium Nitrate

    # 6. Calculate Weight Percents (x)
    # We need total mass of the SOLUTION (solutes + water + etc)
    # Re-sum mtot based on species to be perfectly consistent with moles
    mtot_sol = (
        nan * MWAN + 
        ns0 * MWS0 + 
        ns1 * MWS1 + 
        ns2 * MWS2 + 
        nnacl * MWNACL + 
        mh2o
    )
    mtot_safe = jnp.maximum(mtot_sol, 1e-30)
    
    # Percentages (0-100 scale)
    factor = 100.0 / mtot_safe
    xan = (nan * MWAN) * factor
    xs0 = (ns0 * MWS0) * factor
    xs1 = (ns1 * MWS1) * factor
    xs2 = (ns2 * MWS2) * factor
    xnacl = (nnacl * MWNACL) * factor
    
    xtot = xan + xs0 + xs1 + xs2 + xnacl

    # 7. Binary Densities (Tang 1997) via Horner's Method
    # Horner's method reduces numerical error and operation count.
    # Form: a + x(b + x(c + x(d...)))

    # Ammonium Nitrate
    # 0.9971 + 4.05e-3*x + 9.0e-6*x^2
    dan = 0.9971 + xtot * (4.05e-3 + xtot * 9.0e-6)

    # Sulfuric Acid (s0)
    # 0.9971 + 7.367e-3*x - 4.934e-5*x^2 + 1.754e-6*x^3 - 1.104e-8*x^4
    ds0 = 0.9971 + xtot * (7.367e-3 + xtot * (-4.934e-5 + xtot * (1.754e-6 + xtot * -1.104e-8)))

    # Ammonium Bisulfate (s1)
    # 0.9971 + 5.87e-3*x - 1.89e-6*x^2 + 1.763e-7*x^3
    ds1 = 0.9971 + xtot * (5.87e-3 + xtot * (-1.89e-6 + xtot * 1.763e-7))

    # Ammonium Sulfate (s2)
    # 0.9971 + 5.92e-3*x - 5.036e-6*x^2 + 1.024e-8*x^3
    ds2 = 0.9971 + xtot * (5.92e-3 + xtot * (-5.036e-6 + xtot * 1.024e-8))

    # Sea Salt (NaCl)
    # 0.9971 + 7.41e-3*x - 3.741e-5*x^2 + 2.252e-6*x^3 - 2.06e-8*x^4
    dss = 0.9971 + xtot * (7.41e-3 + xtot * (-3.741e-5 + xtot * (2.252e-6 + xtot * -2.06e-8)))

    # 8. Mixture Density (Mixing Rule)
    # Normalize fractions (0.0 to 1.0) for the mixing rule
    xtot_frac = jnp.maximum(xtot, 1e-30)
    f_an = xan / xtot_frac
    f_s0 = xs0 / xtot_frac
    f_s1 = xs1 / xtot_frac
    f_s2 = xs2 / xtot_frac
    f_nacl = xnacl / xtot_frac

    # Tang Eq 10: 1 / sum(fraction / density)
    inv_d = (f_an / dan) + (f_s0 / ds0) + (f_s1 / ds1) + (f_s2 / ds2) + (f_nacl / dss)
    
    # Specific gravity (g/cm³)
    sg = 1.0 / inv_d

    # Convert to kg/m³ (x1000)
    density = sg * 1000.0

    # 9. Handle Dry/Empty Limit
    # If total mass is vanishingly small, default to water density (1000 kg/m³)
    # or a standard particle density (e.g. 1770) depending on preference.
    # The original code defaulted to water.
    is_empty = mtot < 1e-20
    density = jnp.where(is_empty, 1000.0, density)

    return density