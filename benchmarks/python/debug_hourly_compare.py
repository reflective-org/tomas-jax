"""Debug: compare hour-by-hour Fortran vs JAX for S20 to find divergence onset."""
import os
import numpy as np
import jax.numpy as jnp

from tomas_jax.core.config import NBINS_LEGACY as NBINS, XK0_LEGACY, ICOMP, ICOMP_NODIAG, N_GAS_SPECIES, SRTSO4, SRTH2O
from tomas_jax.solvers.condensation import condensation_step
from benchmarks.python.scenarios import get_scenarios

BOXVOL = 1.0e6
DENS_INIT = 1770.0
PI_F = 3.141592654
DT = 60.0
FORTRAN_DIR = os.path.join(os.path.dirname(__file__), '..', 'fortran', 'output', '24h')


def init_lognormal(N_total, GMD_um, GSD, xk_np):
    Nk = np.zeros(NBINS)
    Mk = np.zeros((NBINS, ICOMP))
    for k in range(NBINS):
        Dl = 1e6 * ((6.0 * xk_np[k]) / (DENS_INIT * PI_F)) ** 0.3333
        Dh = 1e6 * ((6.0 * xk_np[k + 1]) / (DENS_INIT * PI_F)) ** 0.3333
        Dk = np.sqrt(Dl * Dh)
        n = ((N_total * BOXVOL) / (np.sqrt(2.0 * PI_F) * Dk * np.log(GSD))
             * np.exp(-(np.log(Dk / GMD_um) ** 2 / (2.0 * np.log(GSD) ** 2))) * (Dh - Dl))
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

    Nk0, Mk0 = init_lognormal(sc['N_total'], sc['GMD_um'], sc['GSD'], xk_np)
    Gc0 = np.zeros(N_GAS_SPECIES)
    Gc0[SRTSO4] = sc['Gc_SO4_kg']

    # Run JAX 24 hours
    Nk = jnp.array(Nk0)
    Mk = jnp.array(Mk0)
    Gc = jnp.array(Gc0)

    jax_hourly = {}
    for step in range(1440):
        Gc = Gc.at[SRTSO4].add(sc['h2so4_prod_kg_per_s'] * DT)
        Nk, Mk, Gc = condensation_step(
            Nk, Mk, Gc, xk_j,
            sc['temp'], sc['pres'], BOXVOL, sc['RH'], 1.0, DT,
            method='tfl')
        if (step + 1) % 60 == 0:
            h = (step + 1) // 60
            jax_hourly[h] = {'Nk': np.array(Nk), 'Mk': np.array(Mk)}

    # Load Fortran hourly
    bins = list(range(6, 16))
    print(f"S20 Hour-by-hour Nk comparison (bins 6-15):")
    print(f"{'Hour':>5} | " + " ".join(f"{'Fort['+str(b)+']':>12}" for b in bins))
    print(f"      | " + " ".join(f"{'JAX['+str(b)+']':>12}" for b in bins))
    print("-" * (6 + 13 * len(bins)))

    for h in range(1, 25):
        f_path = os.path.join(FORTRAN_DIR, f"s20_cond_hour{h:02d}_Nk.csv")
        if not os.path.exists(f_path):
            continue
        Nk_f = np.loadtxt(f_path)
        Nk_j = jax_hourly.get(h, {}).get('Nk', np.zeros(NBINS))

        f_str = " ".join(f"{Nk_f[b]:12.4e}" for b in bins)
        j_str = " ".join(f"{Nk_j[b]:12.4e}" for b in bins)
        print(f"{h:5d} | {f_str}")
        print(f"      | {j_str}")

        # Max relative error
        peak = max(np.max(Nk_f[bins[0]:bins[-1]+1]), 1e-30)
        max_err = np.max(np.abs(Nk_f[bins[0]:bins[-1]+1] - Nk_j[bins[0]:bins[-1]+1])) / peak
        print(f"      | max_rel_err = {max_err:.4e}")
        print()


if __name__ == '__main__':
    main()
