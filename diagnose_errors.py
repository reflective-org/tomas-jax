"""Diagnostic script: compare all 36 bins for Level 10a (Euler) and 10b (Tsit5).

Shows that large relative errors only appear in near-empty bins with
tiny Nk values (near Neps = 1e-3 .. 1e-5), while the physically
meaningful bins agree to < 1% relative error.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────────
DATA = os.path.join(os.path.dirname(__file__),
                    "benchmarks", "fortran", "output")

def load(name):
    return np.loadtxt(os.path.join(DATA, name), delimiter=",",
                      dtype=np.float64, comments="#")

# ── load FORTRAN reference ─────────────────────────────────────────
Nk_init    = load("initial_Nk.csv").flatten()
Mk_init    = load("initial_Mk.csv")
Nk_fort    = load("level10_final_Nk.csv").flatten()
Mk_fort    = load("level10_final_Mk.csv")
xk         = load("level01_xk.csv").flatten()
params     = load("params.csv").flatten()
temp, pres, boxvol = params[0], params[1], params[2]
dt = 60.0

# ── run JAX Euler ──────────────────────────────────────────────────
from tomas_jax.solvers.euler import euler_step
from tomas_jax.solvers.diffrax import diffrax_step
from tomas_jax.core.config import ICOMP_NODIAG

Nk_j = jnp.array(Nk_init)
Mk_j = jnp.array(Mk_init)
xk_j = jnp.array(xk)

print("Running JAX Euler (use_fortran_mnfix=True, recompute_kernel=True) ...")
Nk_euler, Mk_euler = euler_step(
    Nk_j, Mk_j, xk_j, temp, pres, boxvol, dt,
    icomp_nodiag=ICOMP_NODIAG,
    use_fortran_mnfix=True,
    recompute_kernel=True,
)
Nk_euler = np.array(Nk_euler)

print("Running JAX Tsit5 ...")
Nk_tsit5, Mk_tsit5 = diffrax_step(
    Nk_j, Mk_j, xk_j, temp, pres, boxvol, dt,
    icomp_nodiag=ICOMP_NODIAG,
)
Nk_tsit5 = np.array(Nk_tsit5)

# ── compute totals ─────────────────────────────────────────────────
N_total_init  = np.sum(Nk_init)
N_total_fort  = np.sum(Nk_fort)
N_total_euler = np.sum(Nk_euler)
N_total_tsit5 = np.sum(Nk_tsit5)

# ── per-bin comparison ─────────────────────────────────────────────
header = (f"{'Bin':>3s}  {'FORTRAN Nk':>14s}  {'JAX Euler Nk':>14s}  "
          f"{'Euler relErr':>12s}  {'JAX Tsit5 Nk':>14s}  "
          f"{'Tsit5 relErr':>12s}  {'Comment':s}")

sep = "-" * len(header)

print("\n" + "=" * len(header))
print("Level 10a/b — Full coagulation dt=60 s, Nk per bin")
print("=" * len(header))
print(header)
print(sep)

for k in range(36):
    f_val = Nk_fort[k]
    e_val = Nk_euler[k]
    t_val = Nk_tsit5[k]

    # relative error  (avoid /0  — use max(|ref|, 1e-30))
    denom = max(abs(f_val), 1e-30)
    rel_euler = abs(e_val - f_val) / denom
    rel_tsit5 = abs(t_val - f_val) / denom

    # classify
    if f_val < 1.0:
        comment = "<-- NEAR-EMPTY (Nk < 1)"
    elif f_val < 1e3:
        comment = "<-- small Nk"
    else:
        comment = ""

    print(f"{k:3d}  {f_val:14.6e}  {e_val:14.6e}  {rel_euler:12.4e}  "
          f"{t_val:14.6e}  {rel_tsit5:12.4e}  {comment}")

print(sep)

# ── summary ────────────────────────────────────────────────────────
# bins where Nk > 1e6 (physically active region)
active = Nk_fort > 1e6
rel_euler_active = np.max(np.abs(Nk_euler[active] - Nk_fort[active])
                          / Nk_fort[active])
rel_tsit5_active = np.max(np.abs(Nk_tsit5[active] - Nk_fort[active])
                          / Nk_fort[active])

print(f"\n--- Summary ---")
print(f"Total N  init:    {N_total_init:.8e}")
print(f"Total N  FORTRAN: {N_total_fort:.8e}")
print(f"Total N  Euler:   {N_total_euler:.8e}  (rel diff {abs(N_total_euler-N_total_fort)/N_total_fort:.4e})")
print(f"Total N  Tsit5:   {N_total_tsit5:.8e}  (rel diff {abs(N_total_tsit5-N_total_fort)/N_total_fort:.4e})")
print()
print(f"Max relative error  (ALL 36 bins):")
print(f"  Euler:  {np.max(np.abs(Nk_euler - Nk_fort) / np.maximum(np.abs(Nk_fort), 1e-30)):.4e}")
print(f"  Tsit5:  {np.max(np.abs(Nk_tsit5 - Nk_fort) / np.maximum(np.abs(Nk_fort), 1e-30)):.4e}")
print()
print(f"Max relative error  (active bins with Nk > 1e6):")
print(f"  Euler:  {rel_euler_active:.4e}")
print(f"  Tsit5:  {rel_tsit5_active:.4e}")
print()

# Identify which bins dominate the worst-case error
worst_euler_idx = np.argmax(np.abs(Nk_euler - Nk_fort)
                            / np.maximum(np.abs(Nk_fort), 1e-30))
worst_tsit5_idx = np.argmax(np.abs(Nk_tsit5 - Nk_fort)
                            / np.maximum(np.abs(Nk_fort), 1e-30))
print(f"Worst Euler bin:  index={worst_euler_idx}, "
      f"FORTRAN={Nk_fort[worst_euler_idx]:.6e}, "
      f"JAX={Nk_euler[worst_euler_idx]:.6e}")
print(f"Worst Tsit5 bin:  index={worst_tsit5_idx}, "
      f"FORTRAN={Nk_fort[worst_tsit5_idx]:.6e}, "
      f"JAX={Nk_tsit5[worst_tsit5_idx]:.6e}")

# ── Level 8 quick analysis ─────────────────────────────────────────
print("\n" + "=" * 70)
print("Level 8 — MNFIX comparison (pre/post on artificially drifted state)")
print("=" * 70)

Nk_pre      = load("level08_pre_mnfix_Nk.csv").flatten()
Nk_post_f   = load("level08_post_mnfix_Nk.csv").flatten()
Mk_pre      = load("level08_pre_mnfix_Mk.csv")

from tomas_jax.core.mnfix_fortran import mnfix_fortran

Nk_post_jax, _ = mnfix_fortran(
    jnp.array(Nk_pre), jnp.array(Mk_pre), xk_j, ICOMP_NODIAG
)
Nk_post_jax = np.array(Nk_post_jax)

header8 = (f"{'Bin':>3s}  {'Pre-MNFIX Nk':>14s}  {'FORTRAN post':>14s}  "
           f"{'JAX post':>14s}  {'relErr':>12s}  {'Comment':s}")
sep8 = "-" * len(header8)
print(header8)
print(sep8)

for k in range(36):
    pre  = Nk_pre[k]
    fval = Nk_post_f[k]
    jval = Nk_post_jax[k]
    denom = max(abs(fval), 1e-30)
    rel = abs(jval - fval) / denom

    # Detect bins where MNFIX actually moved particles
    moved = abs(pre - fval) / max(abs(pre), 1e-30) > 0.01
    if moved:
        comment = "<-- MNFIX REDISTRIBUTED"
    elif fval < 1.0:
        comment = "<-- near-empty"
    else:
        comment = ""

    print(f"{k:3d}  {pre:14.6e}  {fval:14.6e}  {jval:14.6e}  "
          f"{rel:12.4e}  {comment}")

print(sep8)

# Worst bin
worst8 = np.argmax(np.abs(Nk_post_jax - Nk_post_f)
                   / np.maximum(np.abs(Nk_post_f), 1e-30))
print(f"\nWorst bin: index={worst8}, "
      f"FORTRAN={Nk_post_f[worst8]:.6e}, "
      f"JAX={Nk_post_jax[worst8]:.6e}, "
      f"rel_err={abs(Nk_post_jax[worst8]-Nk_post_f[worst8])/max(abs(Nk_post_f[worst8]),1e-30):.4e}")
print(f"Pre-MNFIX value at that bin: {Nk_pre[worst8]:.6e}")
print()

# Show total N conservation
print(f"Level 8 total N  pre-MNFIX:   {np.sum(Nk_pre):.8e}")
print(f"Level 8 total N  FORTRAN:     {np.sum(Nk_post_f):.8e}")
print(f"Level 8 total N  JAX port:    {np.sum(Nk_post_jax):.8e}")
