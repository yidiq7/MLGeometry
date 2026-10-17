"""
Bihomogeneous Neural Network layers using Flax.

These layers are designed to approximate sections of line bundles on complex manifolds,
enforcing bihomogeneous properties required for the Kähler potential.
"""

from typing import Callable, Optional
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from . import config

__all__ = [
    'Bihomogeneous',
    'Spectral',
    'Bihomogeneous_k2',
    'Bihomogeneous_k3',
    'Bihomogeneous_k4',
    'Dense',
    'SirenDense',
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


class Spectral(nn.Module):
    """
    Bihomogeneous monomials divided by the Fubini-Study weight: z_i * z_j_bar / |z|^2.

    Each output has degree zero, so a network on these features is a function on CP^(d-1).
    """
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            inputs: Complex array of shape (..., d).

        Returns:
            Real array of shape (..., d*(d+1)).
            Contains [Real(triu), Imag(triu)] / |z|^2.
        """
        # z * conj(z) is smooth in the real and imaginary parts, also where a coordinate is zero.
        norm_sq = jnp.sum(jnp.real(inputs * jnp.conj(inputs)), axis=-1, keepdims=True)
        return Bihomogeneous(d=self.d)(inputs) / norm_sq


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


class Dense(nn.Module):
    """
    A dense layer with Xavier (Glorot) normal initialization of the kernel.

    It is a separate class from nn.Dense, so that kfac-jax does not register it
    automatically as a dense block.
    """
    features: int
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.xavier_normal()
    bias_init: Callable = nn.initializers.zeros
    activation: Optional[Callable] = None

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            inputs: Real array of shape (..., n_in).

        Returns:
            Real array of shape (..., features).
        """
        kernel = self.param('kernel', self.kernel_init,
                            (inputs.shape[-1], self.features), config.real_dtype)
        y = jnp.dot(inputs, kernel)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), config.real_dtype)
            y = y + bias
        if self.activation:
            y = self.activation(y)
        return y


class SirenDense(nn.Module):
    """
    A dense layer with the sine activation and the SIREN initialization.

    Reference: Sitzmann et al., 'Implicit Neural Representations with Periodic
    Activation Functions' (2020). The output is sin(omega_0 * (x @ W + b)).
    """
    features: int
    omega_0: float = 30.0
    is_first_layer: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            inputs: Real array of shape (..., n_in).

        Returns:
            Real array of shape (..., features).
        """
        n_input = inputs.shape[-1]

        # First layer: uniform(-1/n, 1/n). Other layers: uniform(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0).
        if self.is_first_layer:
            bound = 1.0 / n_input
        else:
            bound = math.sqrt(6.0 / n_input) / self.omega_0

        def siren_uniform_init(key, shape, dtype=config.real_dtype):
            return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

        kernel = self.param('kernel', siren_uniform_init, (n_input, self.features))
        bias = self.param('bias', siren_uniform_init, (self.features,))

        return jnp.sin(self.omega_0 * (jnp.dot(inputs, kernel) + bias))


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
