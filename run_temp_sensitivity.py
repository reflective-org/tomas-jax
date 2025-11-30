"""
TOMAS-JAX Temperature Sensitivity Analysis.

Calculates d(N_final) / d(Temperature) using Automatic Differentiation.
This reveals how environmental temperature drives the coagulation rate.
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
# 1. Setup Fixed Initial State
# =========================================================================

def get_fixed_initial_state(n_total_conc=10000.0):
    """Create a standard fixed size distribution."""
    # Standard Grid
    xk = jnp.logspace(jnp.log10(1e-23), jnp.log10(1e-12), NBINS + 1)
    
    # Log-Normal Params
    m_mid = jnp.sqrt(xk[:-1] * xk[1:])
    rho = 1770.0
    dp = jnp.cbrt(m_mid / rho * (6.0 / jnp.pi))
    
    gmd = 1.0e-7 # 100 nm
    gsd = 1.6
    
    # Distribution
    term1 = n_total_conc / (jnp.sqrt(2 * jnp.pi) * jnp.log(gsd))
    term2 = jnp.exp(-(jnp.log(dp) - jnp.log(gmd))**2 / (2 * jnp.log(gsd)**2))
    dNdlogDp = term1 * term2
    
    dlogDp = jnp.log(xk[1:] / xk[:-1]) / 3.0 
    Nk_conc = dNdlogDp * dlogDp
    
    Mk_conc = jnp.zeros((NBINS, ICOMP))
    Mk_conc = Mk_conc.at[:, SRTSO4].set(Nk_conc * m_mid)
    
    return Nk_conc, Mk_conc, xk

# =========================================================================
# 2. Differentiable Simulation Loop (W.R.T Temperature)
# =========================================================================

def run_simulation_temp(temp_input):
    """
    Input: Temperature [K]
    Output: Final Total Number of Particles [#/cell]
    """
    # 1. Fixed Constants
    boxvol = 1.0e6
    pres = 101325.0
    
    # 2. Initial State (Fixed)
    Nk_conc, Mk_conc, xk = get_fixed_initial_state()
    Nk = Nk_conc * boxvol
    Mk = Mk_conc * boxvol
    
    # 3. Time Loop
    total_time = 3600.0  # 1 hour
    dt_model = 60.0      # 1 minute
    num_steps = int(total_time / dt_model)

    def step_fn(carry, _):
        curr_Nk, curr_Mk = carry
        
        # Pass the TRACED 'temp_input' into the solver
        # JAX will trace how T affects Viscosity -> Diffusivity -> Kernel -> Rates
        new_Nk, new_Mk = diffrax_step(
            curr_Nk, curr_Mk, xk, 
            temp_input, pres, boxvol,  # <--- Differentiating this!
            dt=dt_model, 
            icomp_nodiag=ICOMP_NODIAG
        )
        return (new_Nk, new_Mk), None

    # Run Loop
    (final_Nk, final_Mk), _ = jax.lax.scan(
        step_fn, (Nk, Mk), None, length=num_steps
    )
    
    # Return Metric (Total Number)
    return jnp.sum(final_Nk)

# =========================================================================
# 3. Execution
# =========================================================================

def main():
    print("="*60)
    print("🌡️  Temperature Sensitivity Analysis")
    print("="*60)

    # 1. Compile Gradient Function
    print("[System] Compiling Gradient Function...")
    grad_fn = jax.jit(jax.value_and_grad(run_simulation_temp))
    
    # Warmup
    _ = grad_fn(298.0)
    print("[System] Ready.")
    
    # 2. Run for a Range of Temperatures
    temps = jnp.linspace(250.0, 320.0, 20) # 250K to 320K
    
    print("\n[Analysis] computing gradients across temperature range...")
    results, grads = jax.vmap(grad_fn)(temps)
    
    # 3. Visualization
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Plot Final Number vs Temp
    color = 'tab:blue'
    ax1.set_xlabel('Temperature [K]')
    ax1.set_ylabel('Final Total Number [#/cell]', color=color)
    ax1.plot(temps, results, color=color, lw=2, label='Final N')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot Sensitivity Gradient
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sensitivity (dN_final / dT)', color=color)
    ax2.plot(temps, grads, color=color, lw=2, linestyle='--', label='Gradient')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add a zero line to check sign of gradient
    ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)

    plt.title(f"Does Heat Increase Coagulation?")
    plt.tight_layout()
    plt.show()
    
    # Interpretation
    mean_grad = jnp.mean(grads)
    print("-" * 60)
    print(f"Mean Gradient: {mean_grad:.2e} particles / K")
    if mean_grad < 0:
        print("CONCLUSION: Gradient is NEGATIVE.")
        print("Higher Temp -> Faster Coagulation -> Fewer Particles Remaining.")
    else:
        print("CONCLUSION: Gradient is POSITIVE.")
        print("Higher Temp -> Slower Coagulation -> More Particles Remaining.")
    print("-" * 60)

if __name__ == "__main__":
    main()
