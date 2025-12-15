"""
Optimization utilities, including L-BFGS wrapper and loss closure generation.
"""

from typing import Callable, Any, Tuple
import jax
import jax.numpy as jnp
import jaxopt
from . import complex_math

__all__ = ['create_loss_fn', 'LBFGS']


def create_loss_fn(model: Any, 
                   dataset: dict, 
                   loss_metric: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> Callable:
    """
    Creates a JIT-compatible loss function closing over the dataset and model structure.
    
    The returned function takes `params` and returns a scalar loss.
    Internally, it computes the Calabi-Yau volume form (Monge-Ampere equation).
    """
    
    # Pre-extract and cast data to JAX arrays to avoid overhead in the loop
    points = jnp.array(dataset['points'], dtype=jnp.complex64)
    omega_omegabar = jnp.array(dataset['Omega_Omegabar'], dtype=jnp.float32)
    mass = jnp.array(dataset['mass'], dtype=jnp.float32)
    restriction = jnp.array(dataset['restriction'], dtype=jnp.complex64)
    
    # Pre-calculate restriction adjoint for efficiency
    # shape: (N, n_embed, n_manifold)
    restriction_dag = jnp.swapaxes(jnp.conj(restriction), -1, -2)

    def scalar_potential(params: Any, z: jnp.ndarray) -> float:
        """Evaluates the neural network potential at a single point z."""
        # z shape: (d,) -> add batch dim (1, d)
        z_batch = z[None, :]
        # Output shape (1, 1), extract scalar
        out = model.apply(params, z_batch)
        return jnp.real(out[0, 0])

    def volume_form_batch(params: Any) -> jnp.ndarray:
        """Computes the numerical volume form det(g) for the entire batch."""
        
        # 1. Compute Hessians of the potential
        # We need a function of z with params fixed for autodiff
        pot_fn = lambda z: scalar_potential(params, z)
        
        # vmap over points to get batch of Hessians
        # complex_hessian returns (d, d) matrix
        # hessians shape: (N, d, d)
        hessians = jax.vmap(lambda z: complex_math.complex_hessian(pot_fn, z))(points)
        
        # 2. Restrict metric to the hypersurface tangent bundle
        # g_restricted = R @ g_ambient @ R^H
        # (N, n_man, n_amb) @ (N, n_amb, n_amb) -> (N, n_man, n_amb)
        temp = jnp.matmul(restriction, hessians)
        # (N, n_man, n_amb) @ (N, n_amb, n_man) -> (N, n_man, n_man)
        metric_restricted = jnp.matmul(temp, restriction_dag)
        
        # 3. Compute determinant
        det_vol = jax.vmap(jnp.linalg.det)(metric_restricted)
        det_vol = jnp.real(det_vol)
        
        # 4. Normalize
        # The integration of the volume form should equal the integration of Omega_Omegabar (Volume).
        # We enforce this by a normalization factor calculated batch-wise (or global if full batch).
        weights = mass / jnp.sum(mass)
        
        # We want Int(det_vol) = Int(Omega_Omegabar)
        # Factor C = Int(det_vol) / Int(Omega_Omegabar) is roughly computed here?
        # Original logic: factor = sum(weights * det_g / Omega_Omegabar)
        # This calculates the mean ratio weighted by mass.
        factor = jnp.sum(weights * det_vol / omega_omegabar)
        
        return det_vol / factor

    def loss_fn(params: Any) -> jnp.ndarray:
        det_omega = volume_form_batch(params)
        return loss_metric(omega_omegabar, det_omega, mass)
        
    return loss_fn


class LBFGS:
    """Wrapper for JAXopt L-BFGS solver."""
    
    def __init__(self, loss_fn: Callable, max_iter: int = 1000, tol: float = 1e-5):
        self.loss_fn = loss_fn
        self.max_iter = max_iter
        self.tol = tol
        self.solver = jaxopt.LBFGS(fun=loss_fn, maxiter=max_iter, tol=tol)

    def run(self, init_params: Any) -> Tuple[Any, Any]:
        """Runs the optimization."""
        res = self.solver.run(init_params)
        return res.params, res.state
