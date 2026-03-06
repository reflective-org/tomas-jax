"""Fast combined (coag+cond) benchmark using direct JIT calls.

Runs 49 scenarios x 2 methods (tfl_jit, ppm_jit) for combined mode,
saving NPZ files compatible with the plotting scripts.

Usage::

    .venv/bin/python benchmarks/python/run_combined_jit.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, N_GAS_SPECIES,
    SRTSO4, SRTH2O
)
from tomas_jax.solvers.condensation import (
    condensation_step_jit, condensation_step_tfl_jit,
)
from tomas_jax.solvers.coagulation_jax import coagulation_step_jit
from benchmarks.python.scenarios import get_scenarios

# Constants
BOXVOL = 1.0e6
DENS_INIT = 1770.0
PI_F = 3.141592654
DT = 60.0
NSTEPS = 1440
NHOURS = 24
NEPS = 1e-3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', '24h')


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
        if Nk[k] < NEPS:
            Nk[k] = NEPS
            Mk[k, :] = 0.0
            Mk[k, SRTSO4] = NEPS * 1.4 * xk_np[k]
    return Nk, Mk


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    scenarios = get_scenarios(n=50, seed=42)
    scenarios = [s for s in scenarios if s['scenario_id'] <= 49]

    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = 1.6033e-23
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk_j = jnp.array(xk_np)

    # JIT warmup
    print("JIT warmup...")
    sc0 = scenarios[0]
    Nk0, Mk0 = init_lognormal(sc0['N_total'], sc0['GMD_um'], sc0['GSD'], xk_np)
    Gc0 = np.zeros(N_GAS_SPECIES)
    Gc0[SRTSO4] = sc0['Gc_SO4_kg']

    Nk_j = jnp.array(Nk0)
    Mk_j = jnp.array(Mk0)
    Gc_j = jnp.array(Gc0)
    temp_j = jnp.array(sc0['temp'])
    pres_j = jnp.array(sc0['pres'])
    boxvol_j = jnp.array(BOXVOL)
    rh_j = jnp.array(sc0['RH'])
    alpha_j = jnp.array(1.0)
    dt_j = jnp.array(DT)

    # Warmup coag
    Nk_w, Mk_w = coagulation_step_jit(Nk_j, Mk_j, xk_j, temp_j, pres_j, boxvol_j, dt_j)
    jax.block_until_ready(Nk_w)
    print("  Coag warmup done")

    # Warmup TFL cond
    _ = condensation_step_tfl_jit(Nk_j, Mk_j, Gc_j, xk_j, temp_j, pres_j,
                                   boxvol_j, rh_j, alpha_j, dt_j)
    jax.block_until_ready(_[0])
    print("  TFL cond warmup done")

    # Warmup PPM cond
    _ = condensation_step_jit(Nk_j, Mk_j, Gc_j, xk_j, temp_j, pres_j,
                               boxvol_j, rh_j, alpha_j, dt_j)
    jax.block_until_ready(_[0])
    print("  PPM cond warmup done")

    methods = [
        ('tfl_jit', condensation_step_tfl_jit),
        ('ppm_jit', condensation_step_jit),
    ]

    total = len(scenarios) * len(methods)
    count = 0

    for sc in scenarios:
        sid = sc['scenario_id']
        Nk0, Mk0 = init_lognormal(sc['N_total'], sc['GMD_um'], sc['GSD'], xk_np)
        Gc0 = np.zeros(N_GAS_SPECIES)
        Gc0[SRTSO4] = sc['Gc_SO4_kg']

        for method_name, cond_fn in methods:
            count += 1
            label = f"s{sid:02d}_combined_{method_name}"
            npz_path = os.path.join(RESULTS_DIR, f"{label}.npz")

            if os.path.exists(npz_path):
                print(f"[{count}/{total}] {label} — exists, skipping")
                continue

            Nk = jnp.array(Nk0)
            Mk = jnp.array(Mk0)
            Gc = jnp.array(Gc0)
            temp_j = jnp.array(sc['temp'])
            pres_j = jnp.array(sc['pres'])
            boxvol_j = jnp.array(BOXVOL)
            rh_j = jnp.array(sc['RH'])
            alpha_j = jnp.array(1.0)
            dt_j = jnp.array(DT)
            prod = sc['h2so4_prod_kg_per_s']

            Nk_hourly = np.zeros((NHOURS, NBINS))
            Mk_hourly = np.zeros((NHOURS, NBINS, ICOMP))
            Gc_hourly = np.zeros((NHOURS, N_GAS_SPECIES))
            N_tot = np.zeros(NHOURS)
            M_tot = np.zeros(NHOURS)
            M_dry = np.zeros(NHOURS)

            t0 = time.perf_counter()
            for i in range(NSTEPS):
                # 1. Coagulation
                Nk, Mk = coagulation_step_jit(Nk, Mk, xk_j, temp_j, pres_j, boxvol_j, dt_j)

                # 2. H2SO4 production + condensation
                Gc = Gc.at[SRTSO4].add(prod * DT)
                Nk, Mk, Gc = cond_fn(Nk, Mk, Gc, xk_j, temp_j, pres_j,
                                      boxvol_j, rh_j, alpha_j, dt_j)

                if (i + 1) % 60 == 0:
                    h = (i + 1) // 60 - 1
                    Nk.block_until_ready()
                    Nk_hourly[h] = np.array(Nk)
                    Mk_hourly[h] = np.array(Mk)
                    Gc_hourly[h] = np.array(Gc)
                    N_tot[h] = float(jnp.sum(Nk))
                    M_tot[h] = float(jnp.sum(Mk))
                    M_dry[h] = float(jnp.sum(Mk[:, :SRTH2O]))

            Nk.block_until_ready()
            wall = time.perf_counter() - t0

            np.savez_compressed(npz_path,
                                Nk=Nk_hourly, Mk=Mk_hourly,
                                Gc=Gc_hourly, N_tot=N_tot,
                                M_tot=M_tot, M_dry=M_dry,
                                wall_time_s=np.array(wall))

            print(f"[{count}/{total}] {label}: {wall:.2f}s  N24={N_tot[-1]:.3e}")

    print(f"\nDone! Files saved to {os.path.abspath(RESULTS_DIR)}")


if __name__ == '__main__':
    main()
