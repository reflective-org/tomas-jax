"""Global configuration for TOMAS-JAX.

CRITICAL: Import this module at the top of every file to ensure float64 precision.
"""
import jax

# Enable 64-bit floating point precision
# MUST be called before any jax.numpy imports
jax.config.update("jax_enable_x64", True)

# Optionally, verify it's enabled
import jax.numpy as jnp
_test_array = jnp.array(1.0)
if _test_array.dtype != jnp.float64:
    raise RuntimeError(
        "Failed to enable float64 in JAX! "
        "Make sure to import tomas_jax.config before any JAX operations."
    )

# Clean up test
del _test_array

print("✓ TOMAS-JAX: Float64 precision enabled")
