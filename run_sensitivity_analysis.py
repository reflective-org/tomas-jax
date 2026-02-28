"""
TOMAS-JAX Sensitivity Analysis.

This script demonstrates JAX's Automatic Differentiation (AD) capabilities.
It calculates the gradient (sensitivity) of the FINAL simulation state
with respect to the INITIAL inputs (Concentration or Temperature).

Key Concepts:
1. Differentiable Simulation Loop (jax.lax.scan)
2. Loss Function definition
3. jax.grad / jax.value_and_grad
"""

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import TOMAS-JAX modules
import tomas_jax.core.config as config
from tomas_jax.core.config import NBINS, ICOMP, ICOMP_NODIAG, SRTSO4
from tomas_jax.solvers.diffrax import diffrax_step

# =========================================================================
# 1. Differentiable Setup
# =========================================================================

def create_dist_differentiable(
    n_total_conc: float, 
    nbins: int,
    gmd: float = 1.0e-7,
    gsd: float = 1.6
):
    """
    A simplified, differentiable version of the lognormal initialization.
    Note: 'nbins' is static, but 'n_total_conc' is the variable we differentiate.
    """
    # Fixed Grid
    mass_min, mass_max = 1e-23, 1e-12
    xk = jnp.logspace(jnp.log10(mass_min), jnp.log10(mass_max), nbins + 1)
    
    m_mid = jnp.sqrt(xk[:-1] * xk[1:])
    rho_init = 1770.0
    dp_mid = jnp.cbrt(m_mid / rho_init * (6.0 / jnp.pi))

    # Log-Normal Math
    log_dp = jnp.log(dp_mid)
    log_gmd = jnp.log(gmd)
    log_gsd = jnp.log(gsd)
    
    # n_total_conc is our dynamic variable here
    dNdlogDp = (n_total_conc / (jnp.sqrt(2 * jnp.pi) * log_gsd)) * \
               jnp.exp(-(log_dp - log_gmd)**2 / (2 * log_gsd**2))
    
    dlogDp = jnp.log(xk[1:] / xk[:-1]) / 3.0 
    Nk_conc = dNdlogDp * dlogDp
    
    Mk_conc = jnp.zeros((nbins, ICOMP))
    Mk_conc = Mk_conc.at[:, SRTSO4].set(Nk_conc * m_mid)
    
    return Nk_conc, Mk_conc, xk

# =========================================================================
# 2. Differentiable Simulation Loop
# =========================================================================

def run_simulation(n_total_input):
    """
    The function we want to differentiate.
    Input: Initial Number Concentration [#/cm3]
    Output: Final Total Number of Particles [#/cell]
    """
    # 1. Setup (Tracing inputs)
    boxvol = 1.0e6
    temp = 298.0
    pres = 101325.0
    
    # Generate initial state based on the input variable
    Nk_conc, Mk_conc, xk = create_dist_differentiable(n_total_input, NBINS)
    
    Nk = Nk_conc * boxvol
    Mk = Mk_conc * boxvol
    
    # 2. Define Time Loop
    # We use jax.lax.scan instead of a Python 'while' loop.
    # scan is differentiable; python loops are not easily differentiated.
    total_time = 3600.0 * 6  # 6 hours
    dt_model = 60.0          # 1 minute steps
    num_steps = int(total_time / dt_model)

    def step_fn(carry, _):
        curr_Nk, curr_Mk = carry
        
        # Run one physics step
        new_Nk, new_Mk = diffrax_step(
            curr_Nk, curr_Mk, xk, 
            temp, pres, boxvol, 
            dt=dt_model, 
            icomp_nodiag=ICOMP_NODIAG
        )
        return (new_Nk, new_Mk), None

    # Run the loop (Forward Pass)
    (final_Nk, final_Mk), _ = jax.lax.scan(
        step_fn, (Nk, Mk), None, length=num_steps
    )
    
    # 3. Return a scalar "Loss" or "Metric"
    # Let's track Final Total Number
    return jnp.sum(final_Nk)

# =========================================================================
# 3. Execution
# =========================================================================

def main():
    print("="*60)
    print("🤖 TOMAS-JAX Sensitivity Analysis (Automatic Differentiation)")
    print("="*60)
    
    # Define the baseline parameter
    # We want to know: If we perturb this initial concentration, how much does the result change?
    n_init_baseline = 100000.0 # 1e5 particles/cm3
    
    print(f"Baseline Initial Conc: {n_init_baseline:.1f} #/cm3")

    # --- 1. JIT Compile the Gradient Function ---
    print("\n[System] Compiling Gradient Function (Forward + Backward pass)...")
    start_comp = time.time()
    
    # jax.value_and_grad returns both the Output (Forward) and Gradient (Backward)
    grad_fn = jax.jit(jax.value_and_grad(run_simulation))
    
    # Warmup call
    _ = grad_fn(100.0)
    print(f"[System] Compilation finished in {time.time() - start_comp:.2f}s")
    
    # --- 2. Calculate Sensitivity ---
    print("\n[Analysis] Calculating Sensitivity...")
    start_run = time.time()
    
    final_N, dN_final_dN_init = grad_fn(n_init_baseline)
    
    duration = time.time() - start_run
    
    # --- 3. Interpret Results ---
    print(f"Calculation time:      {duration*1000:.2f} ms")
    print("-" * 30)
    print(f"Final Particle Count:  {final_N:.4e}")
    print(f"Gradient (dN_f/dN_i):  {dN_final_dN_init:.4f}")
    
    # Sensitivity Calculation
    # If Gradient < 1.0, it means the system is "damped" (Coagulation destroys memory of initial state).
    # If Gradient = 1.0, changes in initial state propagate perfectly (No coagulation).
    
    print("-" * 30)
    print("INTERPRETATION:")
    print(f"If you add 1 particle initially, {dN_final_dN_init:.4f} particles remain at the end.")
    loss_pct = (1.0 - dN_final_dN_init) * 100
    print(f"Coagulation 'efficiency' at this concentration: {loss_pct:.2f}% of added particles are lost.")

    # --- 4. Plot Sensitivity Curve (Optional) ---
    # Let's run this for a range of concentrations to see how non-linear it is
    print("\n[Plotting] Generating sensitivity curve over range 1e3 to 1e6...")
    
    concs = jnp.logspace(3, 6, 20) # 1k to 1M
    
    # Map the compiled function over the array (Vectorized execution)
    results, grads = jax.vmap(grad_fn)(concs)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Initial Concentration [#/cm³]')
    ax1.set_ylabel('Final Total Number [#]', color=color)
    ax1.loglog(concs, results, color=color, lw=2, label='Final N')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Sensitivity (Gradient)', color=color)
    ax2.semilogx(concs, grads, color=color, lw=2, linestyle='--', label='Gradient')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Non-Linearity of Coagulation")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
