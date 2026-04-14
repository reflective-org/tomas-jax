"""Debug: compare Fortran vs sequential TFL vs TFL_JIT bin-by-bin for a problematic scenario."""
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS_LEGACY as NBINS, XK0_LEGACY,
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES,
    SRTSO4, SRTH2O, MW_H2SO4, AVOGADRO
)
from tomas_jax.solvers.condensation import condensation_step, condensation_step_tfl_jit
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

    # Test a few problematic scenarios
    test_sids = [10, 20, 39]

    xk_np = np.zeros(NBINS + 1)
    xk_np[0] = XK0_LEGACY
    for k in range(NBINS):
        xk_np[k + 1] = 2.0 * xk_np[k]
    xk_j = jnp.array(xk_np)

    # JIT warmup
    sc0 = scenarios[0]
    Nk0, Mk0 = init_lognormal(sc0['N_total'], sc0['GMD_um'], sc0['GSD'], xk_np)
    Gc0 = np.zeros(N_GAS_SPECIES)
    Gc0[SRTSO4] = sc0['Gc_SO4_kg']
    _ = condensation_step_tfl_jit(
        jnp.array(Nk0), jnp.array(Mk0), jnp.array(Gc0), xk_j,
        jnp.array(sc0['temp']), jnp.array(sc0['pres']),
        jnp.array(BOXVOL), jnp.array(sc0['RH']),
        jnp.array(1.0), jnp.array(DT))

    for sid in test_sids:
        sc = scenarios[sid - 1]  # 0-indexed
        print(f"\n{'='*70}")
        print(f"SCENARIO S{sid:02d}: N={sc['N_total']:.0f}, GMD={sc['GMD_um']:.4f}um, "
              f"GSD={sc['GSD']:.2f}")
        print(f"{'='*70}")

        Nk0, Mk0 = init_lognormal(sc['N_total'], sc['GMD_um'], sc['GSD'], xk_np)
        Gc0 = np.zeros(N_GAS_SPECIES)
        Gc0[SRTSO4] = sc['Gc_SO4_kg']

        # Run 60 steps (1 hour) with each method
        nsteps = 60

        # --- Sequential TFL ---
        Nk_seq = jnp.array(Nk0)
        Mk_seq = jnp.array(Mk0)
        Gc_seq = jnp.array(Gc0)
        for i in range(nsteps):
            Gc_seq = Gc_seq.at[SRTSO4].add(sc['h2so4_prod_kg_per_s'] * DT)
            Nk_seq, Mk_seq, Gc_seq = condensation_step(
                Nk_seq, Mk_seq, Gc_seq, xk_j,
                sc['temp'], sc['pres'], BOXVOL, sc['RH'], 1.0, DT,
                method='tfl')

        # --- TFL JIT ---
        Nk_jit = jnp.array(Nk0)
        Mk_jit = jnp.array(Mk0)
        Gc_jit = jnp.array(Gc0)
        for i in range(nsteps):
            Gc_jit = Gc_jit.at[SRTSO4].add(sc['h2so4_prod_kg_per_s'] * DT)
            Nk_jit, Mk_jit, Gc_jit = condensation_step_tfl_jit(
                Nk_jit, Mk_jit, Gc_jit, xk_j,
                jnp.array(sc['temp']), jnp.array(sc['pres']),
                jnp.array(BOXVOL), jnp.array(sc['RH']),
                jnp.array(1.0), jnp.array(DT))

        # --- Fortran hour 1 ---
        f_nk_path = os.path.join(FORTRAN_DIR, f"s{sid:02d}_cond_hour01_Nk.csv")
        f_mk_path = os.path.join(FORTRAN_DIR, f"s{sid:02d}_cond_hour01_Mk.csv")
        Nk_fort = np.loadtxt(f_nk_path) if os.path.exists(f_nk_path) else None
        Mk_fort = np.loadtxt(f_mk_path, delimiter=',') if os.path.exists(f_mk_path) else None

        Nk_seq_np = np.array(Nk_seq)
        Nk_jit_np = np.array(Nk_jit)

        # Find peak bin
        if Nk_fort is not None:
            peak_bin = np.argmax(Nk_fort)
        else:
            peak_bin = np.argmax(Nk_seq_np)

        lo = max(0, peak_bin - 5)
        hi = min(NBINS, peak_bin + 6)

        print(f"\nBin-by-bin Nk comparison (bins {lo}-{hi-1}, peak at bin {peak_bin}):")
        print(f"{'Bin':>4} {'Fortran':>14} {'Seq TFL':>14} {'TFL JIT':>14} {'Seq/Fort':>12} {'JIT/Fort':>12}")
        print("-" * 75)
        for k in range(lo, hi):
            nf = Nk_fort[k] if Nk_fort is not None else 0.0
            ns = Nk_seq_np[k]
            nj = Nk_jit_np[k]
            r_seq = ns / nf if nf > 1e-3 else float('nan')
            r_jit = nj / nf if nf > 1e-3 else float('nan')
            print(f"{k:4d} {nf:14.6e} {ns:14.6e} {nj:14.6e} {r_seq:12.6f} {r_jit:12.6f}")

        # Check totals
        Ntot_fort = np.sum(Nk_fort) if Nk_fort is not None else 0
        Ntot_seq = np.sum(Nk_seq_np)
        Ntot_jit = np.sum(Nk_jit_np)
        print(f"\nN_tot: Fort={Ntot_fort:.6e}, Seq={Ntot_seq:.6e}, JIT={Ntot_jit:.6e}")
        print(f"  Seq vs Fort: {abs(Ntot_seq - Ntot_fort)/max(Ntot_fort, 1e-30):.2e}")
        print(f"  JIT vs Fort: {abs(Ntot_jit - Ntot_fort)/max(Ntot_fort, 1e-30):.2e}")
        print(f"  JIT vs Seq:  {abs(Ntot_jit - Ntot_seq)/max(Ntot_seq, 1e-30):.2e}")

        # Check oscillation metric: ratio of adjacent bins
        print(f"\nOscillation check (Nk[k+1]/Nk[k] ratios, bins {lo}-{hi-2}):")
        print(f"{'Bin':>4} {'Fort ratio':>12} {'Seq ratio':>12} {'JIT ratio':>12}")
        for k in range(lo, hi - 1):
            nf0 = Nk_fort[k] if Nk_fort is not None else 1e-30
            nf1 = Nk_fort[k+1] if Nk_fort is not None else 1e-30
            print(f"{k:4d} {nf1/max(nf0,1e-30):12.4f} "
                  f"{Nk_seq_np[k+1]/max(Nk_seq_np[k],1e-30):12.4f} "
                  f"{Nk_jit_np[k+1]/max(Nk_jit_np[k],1e-30):12.4f}")

        # Also check Mk_dry per bin
        if Mk_fort is not None:
            Mk_seq_np = np.array(Mk_seq)
            Mk_jit_np = np.array(Mk_jit)
            Mdry_fort = np.sum(Mk_fort[:, :SRTH2O], axis=1)
            Mdry_seq = np.sum(Mk_seq_np[:, :SRTH2O], axis=1)
            Mdry_jit = np.sum(Mk_jit_np[:, :SRTH2O], axis=1)

            print(f"\nBin-by-bin M_dry comparison (bins {lo}-{hi-1}):")
            print(f"{'Bin':>4} {'Fortran':>14} {'Seq TFL':>14} {'TFL JIT':>14}")
            for k in range(lo, hi):
                print(f"{k:4d} {Mdry_fort[k]:14.6e} {Mdry_seq[k]:14.6e} {Mdry_jit[k]:14.6e}")


if __name__ == '__main__':
    main()
