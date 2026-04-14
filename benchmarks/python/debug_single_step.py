"""Debug: single-step TFL vs Fortran for S20 to pinpoint where divergence starts."""
import os
import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS_LEGACY as NBINS, XK0_LEGACY,
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, CS_EPS
)
from tomas_jax.physics.condensation_sink import calc_condensation_sink
from tomas_jax.physics.condensation import tmcond
from tomas_jax.physics.condensation_tfl_jax import tmcond_jax, ezcond_tfl_jax
from tomas_jax.physics.ezcond import ezcond
from tomas_jax.core.mnfix_jax import mnfix_jax
from benchmarks.python.scenarios import get_scenarios

BOXVOL = 1.0e6
DENS_INIT = 1770.0
PI_F = 3.141592654
DT = 60.0


def init_lognormal(N_total, GMD_um, GSD, xk_np):
    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))
    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI_F)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI_F)) ** 0.3333
        Dk = np.sqrt(Dl * Dh)
        n = ((N_total * BOXVOL)
             / (np.sqrt(2.0 * PI_F) * Dk * np.log(GSD))
             * np.exp(-(np.log(Dk / GMD_um) ** 2 / (2.0 * np.log(GSD) ** 2)))
             * (Dh - Dl))
        Nk[k] = n
        Mk[k, SRTSO4] = n * np.sqrt(xk_np[k]) * np.sqrt(xk_np[k + 1])
    for k in range(NBINS):
        if Nk[k] < 1e-3:
            Nk[k] = 1e-3
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = 1e-3 * 1.4 * xk_np[k]
    return Nk, Mk


def main():
    scenarios = get_scenarios(n=50, seed=42)
    sc = scenarios[19]  # S20 (0-indexed = 19)
    sid = 20

    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = XK0_LEGACY
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk_j = jnp.array(xk_np)

    print(f"SCENARIO S{sid}: N={sc['N_total']:.0f}, GMD={sc['GMD_um']:.4f}um, GSD={sc['GSD']:.2f}")
    print(f"  Gc_SO4={sc['Gc_SO4_kg']:.4e}, prod={sc['h2so4_prod_kg_per_s']:.4e}")
    print(f"  temp={sc['temp']:.1f}K, pres={sc['pres']:.0f}Pa, RH={sc['RH']:.2f}")

    Nk0, Mk0 = init_lognormal(sc['N_total'], sc['GMD_um'], sc['GSD'], xk_np)

    # Show initial state
    print(f"\nInitial Nk (bins 5-15):")
    for k in range(5, 16):
        print(f"  bin {k:2d}: Nk={Nk0[k]:.6e}, Mk_so4={Mk0[k, SRTSO4]:.6e}, avg={Mk0[k, SRTSO4]/max(Nk0[k],1e-30):.6e}, xk=[{xk_np[k]:.4e}, {xk_np[k+1]:.4e}]")

    # Step 1: Add gas production
    Gc0 = np.zeros(N_GAS_SPECIES)
    Gc0[SRTSO4] = sc['Gc_SO4_kg']
    Gc0[SRTSO4] += sc['h2so4_prod_kg_per_s'] * DT

    # Step 2: MNFIX
    Nk1, Mk1 = mnfix_jax(jnp.array(Nk0), jnp.array(Mk0), xk_j, ICOMP_NODIAG)
    Nk1_np = np.array(Nk1)
    Mk1_np = np.array(Mk1)

    # Step 3: Condensation sink
    CS, sinkfrac = calc_condensation_sink(
        Nk1, jnp.array(Mk1), sc['temp'], sc['pres'], BOXVOL,
        MW_H2SO4, 42.88e-6, 1.0
    )
    CS_val = float(CS)
    sinkfrac_np = np.array(sinkfrac)
    totsinkfrac = np.sum(sinkfrac_np)

    print(f"\nCS = {CS_val:.6e}")
    print(f"totsinkfrac = {totsinkfrac:.6e}")

    # Step 4: Compute mcond
    mcond = Gc0[SRTSO4] * (1.0 - np.exp(-CS_val * DT))
    print(f"mcond = {mcond:.6e}")

    # Step 5: Compute TAU (matching ezcond.f)
    tdt = 2.0 / 3.0
    tau = np.zeros(NBINS)
    for k in range(NBINS):
        mpo = np.sum(Mk1_np[k, :ICOMP_NODIAG])
        mpw = np.sum(Mk1_np[k, :])
        WR = mpw / mpo if mpo > 0 else 1.0
        if Nk1_np[k] > 0 and totsinkfrac > 0:
            maddp = mcond * sinkfrac_np[k] / totsinkfrac / Nk1_np[k]
            mpw_pp = mpw / Nk1_np[k]
            tau[k] = 1.5 * ((mpw_pp + maddp * WR) ** tdt - mpw_pp ** tdt)
        else:
            tau[k] = 0.0

    print(f"\nTAU (bins 5-15):")
    for k in range(5, 16):
        print(f"  bin {k:2d}: tau={tau[k]:.6e}, sinkfrac={sinkfrac_np[k]:.6e}")

    # Step 6: Call sequential tmcond
    print(f"\n--- Sequential tmcond ---")
    Nk_seq, Mk_seq = tmcond(tau, xk_np, Mk1_np, Nk1_np, SRTSO4, np.zeros(NBINS))

    # Step 7: Call sequential ezcond (full pipeline)
    print(f"\n--- Sequential ezcond (full pipeline) ---")
    Nk_ez, Mk_ez = ezcond(Nk0, Mk0, mcond, SRTSO4, xk_np, sc['temp'], sc['pres'], BOXVOL, 1.0)

    print(f"\nAfter 1 step (bins 5-15):")
    print(f"{'Bin':>4} {'Nk_init':>14} {'Nk_tmcond':>14} {'Nk_ezcond':>14} {'tmcond/init':>14}")
    for k in range(5, 16):
        r = Nk_seq[k] / max(Nk1_np[k], 1e-30)
        print(f"{k:4d} {Nk1_np[k]:14.6e} {Nk_seq[k]:14.6e} {Nk_ez[k]:14.6e} {r:14.6f}")

    # Check if any bins were emptied by tmcond
    print(f"\nBins emptied by tmcond (Nk_tmcond < 1e-10 where Nk_init > 1e3):")
    for k in range(NBINS):
        if Nk1_np[k] > 1e3 and Nk_seq[k] < 1e-10:
            print(f"  bin {k}: init={Nk1_np[k]:.6e} → tmcond={Nk_seq[k]:.6e}")

    # Step 8: Check what happens after multiple steps
    print(f"\n--- Multi-step accumulation ---")
    Nk_acc = np.array(Nk0)
    Mk_acc = np.array(Mk0)
    Gc_acc = np.array(Gc0)

    for step in range(10):
        if step > 0:
            Gc_acc[SRTSO4] += sc['h2so4_prod_kg_per_s'] * DT

        # Compute CS
        CS_j, sf_j = calc_condensation_sink(
            jnp.array(Nk_acc), jnp.array(Mk_acc), sc['temp'], sc['pres'], BOXVOL,
            MW_H2SO4, 42.88e-6, 1.0)
        cs = float(CS_j)
        sf = np.array(sf_j)
        tsf = np.sum(sf)

        if cs > CS_EPS and Gc_acc[SRTSO4] > 0:
            mc = Gc_acc[SRTSO4] * (1.0 - np.exp(-cs * DT))
            Gc_acc[SRTSO4] -= mc
            Nk_acc, Mk_acc = ezcond(Nk_acc, Mk_acc, mc, SRTSO4, xk_np,
                                     sc['temp'], sc['pres'], BOXVOL, 1.0)

        # Equilibria
        from tomas_jax.physics.nh3_equilibrium import eznh3eqm
        from tomas_jax.physics.water_equilibrium import calc_equilibrium_water
        Gc_j, Mk_j = eznh3eqm(jnp.array(Gc_acc), jnp.array(Mk_acc))
        Mk_j = calc_equilibrium_water(Mk_j, sc['RH'])
        Nk_j, Mk_j = mnfix_jax(jnp.array(Nk_acc), Mk_j, xk_j, ICOMP_NODIAG)
        Nk_acc = np.array(Nk_j)
        Mk_acc = np.array(Mk_j)

        if step in [0, 1, 4, 9]:
            print(f"\n  After step {step+1}: Nk bins 7-12:")
            for k in range(7, 13):
                print(f"    bin {k}: {Nk_acc[k]:.6e}")


if __name__ == '__main__':
    main()
