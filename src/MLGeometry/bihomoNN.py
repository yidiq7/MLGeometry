"""
Bihomogeneous Neural Network layers using Flax.

These layers are designed to approximate sections of line bundles on complex manifolds,
enforcing bihomogeneous properties required for the Kähler potential.
"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
from . import config

__all__ = [
    'Bihomogeneous',
    'Bihomogeneous_k2',
    'Bihomogeneous_k3',
    'Bihomogeneous_k4',
    'SquareDense',
    'WidthOneDense'
]


class Bihomogeneous(nn.Module):
    """
    Transforms input vector z to the set of bihomogeneous monomials z_i * z_j_bar.
    
    Output effectively represents the upper triangular part of the Hermitian matrix z @ z^H.
    """
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            inputs: Complex array of shape (..., d).
            
        Returns:
            Real array of shape (..., d*(d+1)).
            Contains [Real(triu), Imag(triu)].
        """
        # Outer product: z_i * conj(z_j)
        # Shape: (..., d, d)
        zzbar = jnp.einsum('...i,...j->...ij', inputs, jnp.conj(inputs))
        
        # Extract upper triangular indices (including diagonal)
        rows, cols = jnp.triu_indices(self.d)
        
        # Flattened upper triangular part
        # Shape: (..., n_triu)
        zzbar_flat = zzbar[..., rows, cols] 
        
        # Concatenate Real and Imaginary parts
        # Note: Diagonal elements are real, so imag part is 0, but kept for structural consistency
        return jnp.concatenate([jnp.real(zzbar_flat), jnp.imag(zzbar_flat)], axis=-1)


class Bihomogeneous_k2(nn.Module):
    """
    Transforms z to symmetrized degree-2 monomials (z_i * z_j), then computes
    the bihomogeneous product with its conjugate.
    """
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # 1. Compute symmetrized z_i * z_j
        zz = jnp.einsum('...i,...j->...ij', inputs, inputs)
        rows, cols = jnp.triu_indices(self.d)
        zz_vec = zz[..., rows, cols] 
        
        # 2. Compute hermitian product of the result
        n_dim_2 = zz_vec.shape[-1] 
        zzbar = jnp.einsum('...i,...j->...ij', zz_vec, jnp.conj(zz_vec))
        
        rows2, cols2 = jnp.triu_indices(n_dim_2)
        zzbar_flat = zzbar[..., rows2, cols2]
        
        return jnp.concatenate([jnp.real(zzbar_flat), jnp.imag(zzbar_flat)], axis=-1)


class Bihomogeneous_k3(nn.Module):
    """
    Transforms z to symmetrized degree-3 monomials (z_i * z_j * z_k), then computes
    the bihomogeneous product.
    """
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Generate indices for unique symmetric degree-3 tensors (i <= j <= k)
        # We pre-calculate indices to avoid dynamic shape issues in JIT
        indices = []
        for i in range(self.d):
            for j in range(i, self.d):
                for k in range(j, self.d):
                    indices.append((i, j, k))
        
        indices = tuple(jnp.array(x) for x in zip(*indices))
        
        # Compute full tensor then extract unique elements
        zz = jnp.einsum('...i,...j,...k->...ijk', inputs, inputs, inputs)
        zz_vec = zz[..., indices[0], indices[1], indices[2]]
        
        n_dim_3 = zz_vec.shape[-1]
        zzbar = jnp.einsum('...i,...j->...ij', zz_vec, jnp.conj(zz_vec))
        
        rows3, cols3 = jnp.triu_indices(n_dim_3)
        zzbar_flat = zzbar[..., rows3, cols3]
        
        return jnp.concatenate([jnp.real(zzbar_flat), jnp.imag(zzbar_flat)], axis=-1)


class Bihomogeneous_k4(nn.Module):
    """
    Transforms z to symmetrized degree-4 monomials, then computes the bihomogeneous product.
    """
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        indices = []
        for i in range(self.d):
            for j in range(i, self.d):
                for k in range(j, self.d):
                    for l in range(k, self.d):
                        indices.append((i, j, k, l))
        
        indices = tuple(jnp.array(x) for x in zip(*indices))
        
        zz = jnp.einsum('...i,...j,...k,...l->...ijkl', inputs, inputs, inputs, inputs)
        zz_vec = zz[..., indices[0], indices[1], indices[2], indices[3]]
        
        n_dim_4 = zz_vec.shape[-1]
        zzbar = jnp.einsum('...i,...j->...ij', zz_vec, jnp.conj(zz_vec))
        
        rows4, cols4 = jnp.triu_indices(n_dim_4)
        zzbar_flat = zzbar[..., rows4, cols4]
        
        return jnp.concatenate([jnp.real(zzbar_flat), jnp.imag(zzbar_flat)], axis=-1)


class SquareDense(nn.Module):
    """
    A dense layer with a specific initialization scheme (absolute value of random normal).
    Commonly used with square activation for positivity.
    """
    features: int
    activation: Callable = lambda x: x**2
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        def abs_normal_init(key, shape, dtype=config.real_dtype):
            return jnp.abs(jax.random.normal(key, shape, dtype) * 0.05)
        
        kernel = self.param('kernel', abs_normal_init, (inputs.shape[-1], self.features))
        
        y = jnp.dot(inputs, kernel)
        if self.activation:
            y = self.activation(y)
        return y


class WidthOneDense(nn.Module):
    """
    A specialized final layer that acts as a reduction sum, initialized to mimic
    the Fubini-Study metric (Identity matrix in implicit basis).
    """
    features: int = 1
    activation: Optional[Callable] = None
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        input_dim = inputs.shape[-1]
        
        n_unique = input_dim // 2
        # n_unique = dim * (dim + 1) / 2
        # dim^2 + dim - 2*n_unique = 0
        # dim = (-1 + sqrt(1 + 8*n_unique)) / 2
        dim = int((-1 + (1 + 8 * n_unique)**0.5) / 2)
        
        def width_one_init(key, shape, dtype=config.real_dtype):
            rows, cols = jnp.triu_indices(dim)
            is_diag = (rows == cols) # Boolean mask of length n_unique
            
            real_w = jnp.where(is_diag, 1.0, 0.0)
            imag_w = jnp.zeros_like(real_w)
            
            w_full = jnp.concatenate([real_w, imag_w])
            return w_full.reshape(shape)

        kernel = self.param('kernel', width_one_init, (input_dim, self.features))
        
        y = jnp.dot(inputs, kernel)
        if self.activation:
            y = self.activation(y)
        return y
