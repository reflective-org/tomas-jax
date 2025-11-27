import sys
import os
import jax
import jax.numpy as jnp

# Patch sys.path to allow 'import tomas_jax' when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel, calc_single_kernel_element

def test_kernel_logic():
    print("--- Testing Coagulation Kernel ---")
    
    # 1. Dummy Data (3 bins)
    Dpk = jnp.array([1e-8, 1e-7, 1e-6], dtype=jnp.float64)  # 10nm, 100nm, 1um
    Dk = jnp.array([5e-6, 4e-7, 3e-8], dtype=jnp.float64)   # Diffusivity (made up)
    ck = jnp.array([200.0, 50.0, 10.0], dtype=jnp.float64)  # Thermal speed
    boxvol = 1000.0  # cm3

    # 2. Run Vectorized
    kij_matrix = calc_coagulation_kernel(Dpk, Dk, ck, boxvol)

    # 3. Run Scalar Check on corner (0, 2)
    kij_scalar_02 = calc_single_kernel_element(
        Dpk[0], Dpk[2], 
        Dk[0], Dk[2], 
        ck[0], ck[2], 
        boxvol
    )
    
    # 4. Assert
    print(f"Matrix[0,2]: {kij_matrix[0,2]:.6e}")
    print(f"Scalar Check: {kij_scalar_02:.6e}")
    
    assert jnp.isclose(kij_matrix[0,2], kij_scalar_02), "Mismatch in Kernel Calculation!"
    
    # Check Symmetry
    assert jnp.allclose(kij_matrix, kij_matrix.T), "Kernel is not symmetric!"
    
    print("✅ Kernel Logic Matches & is Symmetric")

if __name__ == "__main__":
    test_kernel_logic()