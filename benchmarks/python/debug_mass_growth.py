"""Debug: track mass AND number evolution step-by-step for S20."""
import os
import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS_LEGACY as NBINS, XK0_LEGACY,
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, CS_EPS
)
from tomas_jax.solvers.condensation import condensation_step
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
    sc = scenarios[19]  # S20

    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = XK0_LEGACY
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk_j = jnp.array(xk_np)

    print(f"S20: N={sc['N_total']:.0f}, GMD={sc['GMD_um']:.4f}um, GSD={sc['GSD']:.2f}")

    Nk0, Mk0 = init_lognormal(sc['N_total'], sc['GMD_um'], sc['GSD'], xk_np)
    Gc0 = np.zeros(N_GAS_SPECIES)
    Gc0[SRTSO4] = sc['Gc_SO4_kg']

    # Focus on bins 7-12
    bins = list(range(7, 13))

    print(f"\nStep-by-step evolution (bins 7-12):")
    print(f"{'Step':>5} | {'Nk[7]':>12} {'Nk[8]':>12} {'Nk[9]':>12} {'Nk[10]':>12} {'Nk[11]':>12} {'Nk[12]':>12} | {'avg[9]':>12} {'avg[10]':>12}")

    Nk = jnp.array(Nk0)
    Mk = jnp.array(Mk0)
    Gc = jnp.array(Gc0)

    def print_state(step, Nk_np, Mk_np):
        avg9 = np.sum(Mk_np[9, :ICOMP_NODIAG]) / max(Nk_np[9], 1e-30)
        avg10 = np.sum(Mk_np[10, :ICOMP_NODIAG]) / max(Nk_np[10], 1e-30)
        nk_str = " ".join(f"{Nk_np[k]:12.4e}" for k in bins)
        print(f"{step:5d} | {nk_str} | {avg9:12.4e} {avg10:12.4e}")

    Nk_np = np.array(Nk)
    Mk_np = np.array(Mk)
    print_state(0, Nk_np, Mk_np)

    for step in range(60):
        Gc = Gc.at[SRTSO4].add(sc['h2so4_prod_kg_per_s'] * DT)
        Nk, Mk, Gc = condensation_step(
            Nk, Mk, Gc, xk_j,
            sc['temp'], sc['pres'], BOXVOL, sc['RH'], 1.0, DT,
            method='tfl')

        Nk_np = np.array(Nk)
        Mk_np = np.array(Mk)

        if step < 5 or step % 5 == 4 or step in [9, 19, 29, 39, 49, 59]:
            print_state(step + 1, Nk_np, Mk_np)

    # Print bin boundaries for reference
    print(f"\nBin boundaries:")
    for k in bins:
        print(f"  bin {k}: xk=[{xk_np[k]:.4e}, {xk_np[k+1]:.4e}], ratio upper/lower = {xk_np[k+1]/xk_np[k]:.1f}")


if __name__ == '__main__':
    main()
