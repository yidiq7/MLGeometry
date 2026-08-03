import jax
import jax.numpy as jnp
import numpy as np

# Global dtypes (default to 32-bit)
real_dtype = jnp.float32
complex_dtype = jnp.complex64
np_real_dtype = np.float32
np_complex_dtype = np.complex64

def set_precision(bits: int):
    """
    Sets the floating point precision for the package.
    
    Args:
        bits: 32 or 64.
    """
    global real_dtype, complex_dtype, np_real_dtype, np_complex_dtype
    
    if bits == 64:
        jax.config.update("jax_enable_x64", True)
        real_dtype = jnp.float64
        complex_dtype = jnp.complex128
        np_real_dtype = np.float64
        np_complex_dtype = np.complex128
        print("MLGeometry: Precision set to FP64 (x64 enabled).")
    elif bits == 32:
        jax.config.update("jax_enable_x64", False)
        real_dtype = jnp.float32
        complex_dtype = jnp.complex64
        np_real_dtype = np.float32
        np_complex_dtype = np.complex64
        print("MLGeometry: Precision set to FP32.")
    else:
        raise ValueError("Precision bits must be 32 or 64.")
