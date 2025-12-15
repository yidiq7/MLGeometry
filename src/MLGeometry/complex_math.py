"""
Complex mathematical utilities for JAX.
"""

from typing import Callable, Union, Tuple
import jax
import jax.numpy as jnp

__all__ = ['complex_hessian']


def complex_hessian(func: Callable[[jnp.ndarray], float], z: jnp.ndarray) -> jnp.ndarray:
    """Computes the complex Hessian (dz dzbar) of a real-valued function.

    The complex Hessian is defined as:
        H_{j, k} = d^2 f / (dz_j d\\bar{z}_k)

    Args:
        func: A function that takes a complex vector `z` and returns a real scalar.
        z: A complex vector of shape (n,).

    Returns:
        A complex matrix of shape (n, n) representing the Hessian.
    """
    
    # Wrap the function to treat z as two real vectors (x, y)
    def func_real(x: jnp.ndarray, y: jnp.ndarray) -> float:
        z_complex = x + 1j * y
        return func(z_complex)

    x = jnp.real(z)
    y = jnp.imag(z)

    # Compute the Hessian w.r.t real and imaginary parts
    # hessian_fn returns a tuple of tuples: ((H_xx, H_xy), (H_yx, H_yy))
    hessian_fn = jax.hessian(func_real, argnums=(0, 1))
    (h_xx, h_xy), (h_yx, h_yy) = hessian_fn(x, y)
    
    # Wirtinger derivative formula for mixed complex derivative:
    # d^2 f / dz d\bar{z} = 1/4 * (d^2/dx^2 + d^2/dy^2 + i*(d^2/dxdy - d^2/dydx))
    # Note: Since mixed partials of smooth real functions commute, h_xy = h_yx^T.
    # For scalar f, h_xy is symmetric if f is smooth enough, but let's use the full formula.
    
    # 0.25 * (H_xx + H_yy + 1j * (H_xy - H_yx))
    hessian_complex = 0.25 * (h_xx + h_yy + 1j * (h_xy - h_yx))
    
    return hessian_complex
