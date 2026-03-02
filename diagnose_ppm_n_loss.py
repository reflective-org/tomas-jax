"""Diagnose where PPM loses number concentration.

Runs a single condensation step on S01 (small GMD) and tracks N_tot
at every stage of the PPM pipeline.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tomas_jax.core.config import (
    NBINS, ICOMP, ICOMP_NODIAG, SRTSO4, SRTH2O, N_GAS_SPECIES
)
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics.condensation_sink import calc_condensation_sink
from tomas_jax.physics.condensation_ppm import (
    ppm_condensation_step, compute_wr, compute_substeps,
    compute_edge_velocity, ppm_reconstruct, ppm_flux,
    advect_totals, passive_flux, DELTA_XI, EPSN
)
from benchmarks.python.scenarios import get_scenarios
from benchmarks.python.run_24h_scenarios import init_lognormal_scenario

# Load scenario 1 (small GMD)
scenarios = get_scenarios()
s = scenarios[0]
print(f"Scenario 1: GMD={s['GMD_um']:.4f} um, N={s['N_total']:.2e}, "
      f"Gc={s['Gc_SO4_kg']:.2e}, prod={s['h2so4_prod']:.2e}")

# Initialize state (matching Fortran exactly)
xk_np = np.zeros(NBINS + 1)
xk_np[0] = 1.6033e-23
for k in range(NBINS):
    xk_np[k + 1] = 2.0 * xk_np[k]
Nk_np, Mk_np = init_lognormal_scenario(s['N_total'], s['GMD_um'], s['GSD'], xk_np)

# Gas phase
Gc_np = np.zeros(N_GAS_SPECIES)
Gc_np[SRTSO4] = s['Gc_SO4_kg']

Nk = jnp.array(Nk_np)
Mk = jnp.array(Mk_np)
xk = jnp.array(xk_np)
Gc = jnp.array(Gc_np)

dt = 60.0  # 1 minute timestep

print(f"\n{'='*70}")
print("INITIAL STATE")
print(f"{'='*70}")
print(f"N_tot = {float(jnp.sum(Nk)):.6e}")
print(f"M_dry = {float(jnp.sum(Mk[:, :SRTH2O])):.6e}")
print(f"Gc[SO4] = {float(Gc[SRTSO4]):.6e}")
print(f"Nk range: [{float(jnp.min(Nk)):.2e}, {float(jnp.max(Nk)):.2e}]")

# Step 1: Condensation sink
BOXVOL = 1.0e6
CS, sinkfrac = calc_condensation_sink(Nk, Mk, s['temp'], s['pres'], BOXVOL)
CS_val = float(CS)
print(f"\nCS = {CS_val:.6e} s^-1")

# Step 2: Gas depletion
mcond = float(Gc[SRTSO4]) * (1.0 - np.exp(-CS_val * dt))
print(f"mcond = {mcond:.6e} kg")

# Step 3: ezcond_ppm pipeline (manual decomposition)
Nk_np = np.array(Nk)
Mk_np = np.array(Mk)
xk_np = np.array(xk)

# MNFIX #1 (input)
Nk_m1, Mk_m1 = mnfix_jax(jnp.array(Nk_np), jnp.array(Mk_np), jnp.array(xk_np), ICOMP_NODIAG)
N_after_mnfix1 = float(jnp.sum(Nk_m1))
print(f"\n--- MNFIX #1 (input) ---")
print(f"N_tot = {N_after_mnfix1:.6e} (delta = {N_after_mnfix1 - float(jnp.sum(Nk)):.6e})")

# Compute TAU
Nk1 = np.array(Nk_m1)
Mk1 = np.array(Mk_m1)
sinkfrac_np = np.array(sinkfrac)
totsinkfrac = np.sum(sinkfrac_np)

tdt = 2.0 / 3.0
tau = np.zeros(NBINS)
for k in range(NBINS):
    mpo = np.sum(Mk1[k, :ICOMP_NODIAG])
    mpw = np.sum(Mk1[k, :])
    WR = mpw / mpo if mpo > 0 else 1.0
    if Nk1[k] > 0 and totsinkfrac > 0:
        maddp_k = mcond * sinkfrac_np[k] / totsinkfrac / Nk1[k]
        mpw_pp = mpw / Nk1[k]
        tau[k] = 1.5 * ((mpw_pp + maddp_k * WR) ** tdt - mpw_pp ** tdt)
    else:
        tau[k] = 0.0

print(f"\nTAU range: [{np.min(tau):.6e}, {np.max(tau):.6e}]")
print(f"TAU > 0: {np.sum(tau > 0)} bins")

# MNFIX #2 (before PPM)
Nk_m2, Mk_m2 = mnfix_jax(jnp.array(Nk1), jnp.array(Mk1), jnp.array(xk_np), ICOMP_NODIAG)
N_after_mnfix2 = float(jnp.sum(Nk_m2))
print(f"\n--- MNFIX #2 (before PPM) ---")
print(f"N_tot = {N_after_mnfix2:.6e} (delta = {N_after_mnfix2 - N_after_mnfix1:.6e})")

# Now run PPM step-by-step
print(f"\n{'='*70}")
print("PPM CONDENSATION STEP (manual)")
print(f"{'='*70}")

Nk_ppm = Nk_m2
Mk_ppm = Mk_m2
TAU_jax = jnp.array(tau)

# Freeze WR
WR = compute_wr(Mk_ppm, ICOMP_NODIAG)
print(f"WR range: [{float(jnp.min(WR)):.4f}, {float(jnp.max(WR)):.4f}]")

# Substep count
n_sub = compute_substeps(jnp.array(xk_np), TAU_jax, WR, 1.0, 0.8)
n_sub_int = int(n_sub)
print(f"n_sub = {n_sub_int}")

dt_sub = 1.0 / n_sub_int
TAU_sub = TAU_jax / n_sub_int

# Run ONE substep manually
u_edges = compute_edge_velocity(jnp.array(xk_np), TAU_sub, WR, dt_sub)
print(f"\nu_edges range: [{float(jnp.min(u_edges)):.6e}, {float(jnp.max(u_edges)):.6e}]")
print(f"u_edges sign: {int(jnp.sum(u_edges > 0))} positive, "
      f"{int(jnp.sum(u_edges < 0))} negative, "
      f"{int(jnp.sum(u_edges == 0))} zero")

# Courant numbers
C = u_edges * dt_sub / DELTA_XI
print(f"Courant range: [{float(jnp.min(C)):.6f}, {float(jnp.max(C)):.6f}]")
print(f"CFL violations (|C| > 1): {int(jnp.sum(jnp.abs(C) > 1.0))}")

# PPM reconstruct number
n_bar = Nk_ppm / DELTA_XI
n_L, n_R, n_6 = ppm_reconstruct(n_bar)
print(f"\nn_bar range: [{float(jnp.min(n_bar)):.6e}, {float(jnp.max(n_bar)):.6e}]")
print(f"n_L min: {float(jnp.min(n_L)):.6e}")
print(f"n_R min: {float(jnp.min(n_R)):.6e}")

# Check if parabola goes negative at any point inside cells
# q(eta) = q_L + eta*(dq + q_6*(1-eta))
# Check at eta = 0, 0.25, 0.5, 0.75, 1.0
eta_check = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
dq = n_R - n_L
for eta in eta_check:
    q_val = n_L + eta * (dq + n_6 * (1.0 - eta))
    n_neg = int(jnp.sum(q_val < 0))
    if n_neg > 0:
        print(f"  NEGATIVE at eta={float(eta):.2f}: {n_neg} bins, "
              f"min={float(jnp.min(q_val)):.6e}")

# Compute flux
F_N = ppm_flux(n_L, n_R, n_6, u_edges, dt_sub, DELTA_XI)
print(f"\nF_N range: [{float(jnp.min(F_N)):.6e}, {float(jnp.max(F_N)):.6e}]")
print(f"F_N < 0: {int(jnp.sum(F_N < 0))} edges")

# Check flux telescoping (should sum to zero for conservation)
div_F = F_N[1:] - F_N[:-1]
print(f"sum(div_F) = {float(jnp.sum(div_F)):.6e} (should be ~0)")

# Advect number
Nk_adv = advect_totals(Nk_ppm, F_N, dt_sub)
N_after_advect = float(jnp.sum(Nk_adv))
print(f"\n--- After number advection ---")
print(f"N_tot = {N_after_advect:.6e} (delta = {N_after_advect - N_after_mnfix2:.6e})")
print(f"Nk_adv min = {float(jnp.min(Nk_adv)):.6e}")
n_negative = int(jnp.sum(Nk_adv < 0))
print(f"NEGATIVE bins: {n_negative}")
if n_negative > 0:
    neg_idx = jnp.where(Nk_adv < 0)[0]
    for idx in neg_idx:
        print(f"  bin {int(idx)}: Nk={float(Nk_adv[idx]):.6e}, "
              f"was {float(Nk_ppm[idx]):.6e}, "
              f"F_out={float(F_N[int(idx)+1]):.6e}, "
              f"F_in={float(F_N[int(idx)]):.6e}")

# After positivity clamp
Nk_clamped = jnp.maximum(Nk_adv, 0.0)
N_after_clamp = float(jnp.sum(Nk_clamped))
print(f"\n--- After positivity clamp ---")
print(f"N_tot = {N_after_clamp:.6e} (delta = {N_after_clamp - N_after_advect:.6e})")
print(f"N created by clamp: {N_after_clamp - N_after_advect:.6e}")

# Advect ALL species (including SO4) with upwind flux
Mk_adv = Mk_ppm.copy()
for j in range(ICOMP):
    F_M_j = passive_flux(F_N, Mk_ppm[:, j], Nk_ppm, u_edges)
    Mk_adv = Mk_adv.at[:, j].set(advect_totals(Mk_ppm[:, j], F_M_j, dt_sub))

# Clamp mass
Mk_adv = jnp.maximum(Mk_adv, 0.0)

print(f"\n--- After transport (all species) ---")
M_so4_post = float(jnp.sum(Mk_adv[:, SRTSO4]))
M_so4_pre = float(jnp.sum(Mk_ppm[:, SRTSO4]))
print(f"M_SO4 before = {M_so4_pre:.6e}")
print(f"M_SO4 after  = {M_so4_post:.6e}")
print(f"M_SO4 delta  = {M_so4_post - M_so4_pre:.6e} (should be ~0)")
M_dry_post = float(jnp.sum(Mk_adv[:, :SRTH2O]))
print(f"M_dry = {M_dry_post:.6e}")

# Add condensed mass (as ezcond_ppm now does)
sinkfrac_np = np.array(sinkfrac)
totsinkfrac = np.sum(sinkfrac_np)
Mk_with_cond = np.array(Mk_adv)
if totsinkfrac > 0:
    for k in range(NBINS):
        Mk_with_cond[k, SRTSO4] += mcond * sinkfrac_np[k] / totsinkfrac
Mk_with_cond = jnp.array(Mk_with_cond)
M_so4_after_add = float(jnp.sum(Mk_with_cond[:, SRTSO4]))
print(f"\n--- After adding mcond ---")
print(f"M_SO4 = {M_so4_after_add:.6e} (gained {M_so4_after_add - M_so4_pre:.6e}, mcond={mcond:.6e})")

# Now run full PPM step and compare
print(f"\n{'='*70}")
print("FULL PPM STEP (via ppm_condensation_step)")
print(f"{'='*70}")
Nk_full, Mk_full = ppm_condensation_step(
    Nk_m2, Mk_m2, jnp.array(xk_np), TAU_jax, SRTSO4, 1.0, ICOMP_NODIAG
)
N_after_ppm = float(jnp.sum(Nk_full))
print(f"N_tot = {N_after_ppm:.6e} (delta from input: {N_after_ppm - N_after_mnfix2:.6e})")
print(f"Relative N change: {(N_after_ppm - N_after_mnfix2) / N_after_mnfix2:.6e}")

# Simulate new ezcond_ppm: add condensed mass + final MNFIX
Nk2 = np.array(Nk_full)
Mk2 = np.array(Mk_full)

tot_i = np.sum(Mk1[:, SRTSO4])
print(f"\n--- Adding condensed mass (new approach) ---")
print(f"M_SO4 before PPM = {tot_i:.6e}")
print(f"M_SO4 after PPM  = {np.sum(Mk2[:, SRTSO4]):.6e}")
if totsinkfrac > 0:
    for k in range(NBINS):
        Mk2[k, SRTSO4] += mcond * sinkfrac_np[k] / totsinkfrac
print(f"M_SO4 after add  = {np.sum(Mk2[:, SRTSO4]):.6e}")
print(f"mcond = {mcond:.6e}, actual gain = {np.sum(Mk2[:, SRTSO4]) - tot_i:.6e}")

# Final MNFIX
Nk_final, Mk_final = mnfix_jax(
    jnp.array(Nk2), jnp.array(Mk2), jnp.array(xk_np), ICOMP_NODIAG
)
N_final = float(jnp.sum(Nk_final))
print(f"\n--- Final MNFIX ---")
print(f"N_tot = {N_final:.6e}")
print(f"N change from PPM output: {N_final - N_after_ppm:.6e}")

print(f"\n{'='*70}")
print("SUMMARY (one timestep)")
print(f"{'='*70}")
N_init = float(jnp.sum(Nk))
print(f"N initial:           {N_init:.6e}")
print(f"N after MNFIX #1:    {N_after_mnfix1:.6e} ({(N_after_mnfix1-N_init)/N_init:+.4e})")
print(f"N after MNFIX #2:    {N_after_mnfix2:.6e} ({(N_after_mnfix2-N_init)/N_init:+.4e})")
print(f"N after PPM advect:  {N_after_ppm:.6e} ({(N_after_ppm-N_init)/N_init:+.4e})")
print(f"N final (MNFIX out): {N_final:.6e} ({(N_final-N_init)/N_init:+.4e})")
print(f"\nTotal N loss per step: {(N_final-N_init)/N_init:.4e}")
print(f"Projected 24h loss: {(1 + (N_final-N_init)/N_init)**1440 - 1:.4f} ({((1 + (N_final-N_init)/N_init)**1440 - 1)*100:.1f}%)")
