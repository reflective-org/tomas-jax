"""
TOMAS-JAX Sensitivity Analysis: Mode Diameter (GMD).

Calculates d(N_final) / d(GMD) using Automatic Differentiation.
This answers: "Does starting with slightly larger particles help them survive better?"
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
# 1. Differentiable Initialization
# =========================================================================

def get_initial_state_with_gmd(gmd_input):
    """
    Create size distribution dependent on the TRACED variable 'gmd_input'.
    JAX will differentiate through this function!
    """
    # Fixed Grid
    xk = jnp.logspace(jnp.log10(1e-23), jnp.log10(1e-12), NBINS + 1)
    
    # Calculate diameters
    m_mid = jnp.sqrt(xk[:-1] * xk[1:])
    rho = 1770.0
    dp = jnp.cbrt(m_mid / rho * (6.0 / jnp.pi))
    
    # Fixed distribution params
    n_total_conc = 10000.0 # 1e4 / cm3
    gsd = 1.6
    
    # Log-Normal Math (Differentiable w.r.t gmd_input)
    term1 = n_total_conc / (jnp.sqrt(2 * jnp.pi) * jnp.log(gsd))
    
    # This is the key line where gradients flow:
    log_diff = jnp.log(dp) - jnp.log(gmd_input) 
    
    term2 = jnp.exp(-(log_diff)**2 / (2 * jnp.log(gsd)**2))
    dNdlogDp = term1 * term2
    
    dlogDp = jnp.log(xk[1:] / xk[:-1]) / 3.0 
    Nk_conc = dNdlogDp * dlogDp
    
    Mk_conc = jnp.zeros((NBINS, ICOMP))
    Mk_conc = Mk_conc.at[:, SRTSO4].set(Nk_conc * m_mid)
    
    return Nk_conc, Mk_conc, xk

# =========================================================================
# 2. Differentiable Simulation Loop
# =========================================================================

def run_simulation_gmd(gmd_input):
    """
    Input: Geometric Mean Diameter [m]
    Output: Final Total Number of Particles [#/cell]
    """
    # 1. Constants
    boxvol = 1.0e6
    temp = 298.0
    pres = 101325.0
    
    # 2. Generate Initial State from GMD
    Nk_conc, Mk_conc, xk = get_initial_state_with_gmd(gmd_input)
    
    Nk = Nk_conc * boxvol
    Mk = Mk_conc * boxvol
    
    # 3. Time Loop
    total_time = 3600.0  # 1 hour
    dt_model = 60.0      # 1 minute
    num_steps = int(total_time / dt_model)

    def step_fn(carry, _):
        curr_Nk, curr_Mk = carry
        
        new_Nk, new_Mk = diffrax_step(
            curr_Nk, curr_Mk, xk, 
            temp, pres, boxvol, 
            dt=dt_model, 
            icomp_nodiag=ICOMP_NODIAG
        )
        return (new_Nk, new_Mk), None

    # Run Loop
    (final_Nk, final_Mk), _ = jax.lax.scan(
        step_fn, (Nk, Mk), None, length=num_steps
    )
    
    return jnp.sum(final_Nk)

# =========================================================================
# 3. Execution
# =========================================================================

def main():
    print("="*60)
    print("📏  Mode Diameter (GMD) Sensitivity Analysis")
    print("="*60)

    # 1. Compile
    print("[System] Compiling Gradient Function...")
    grad_fn = jax.jit(jax.value_and_grad(run_simulation_gmd))
    
    # Warmup
    _ = grad_fn(1.0e-7)
    print("[System] Ready.")
    
    # 2. Sweep Diameters: 10nm to 200nm
    gmds = jnp.linspace(10e-9, 200e-9, 30) 
    
    print("\n[Analysis] Computing gradients across size range...")
    results, grads = jax.vmap(grad_fn)(gmds)
    
    # 3. Visualization
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    gmd_nm = gmds * 1e9 # Convert to nm for plotting
    
    # Plot Final Number
    color = 'tab:blue'
    ax1.set_xlabel('Geometric Mean Diameter [nm]')
    ax1.set_ylabel('Final Total Number [#/cell]', color=color)
    ax1.plot(gmd_nm, results, color=color, lw=2, label='Final N')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot Sensitivity
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Sensitivity (dN_final / dGMD)', color=color)
    ax2.plot(gmd_nm, grads, color=color, lw=2, linestyle='--', label='Gradient')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)

    plt.title("Effect of Initial Size on Survival")
    plt.tight_layout()
    plt.show()
    
    # Interpretation
    print("-" * 60)
    print("INTERPRETATION:")
    print("Positive Gradient means: Increasing initial size -> MORE particles survive.")
    print("This makes sense: Larger particles diffuse slower -> Coagulate slower -> Survive longer.")
    print("-" * 60)

if __name__ == "__main__":
    main()