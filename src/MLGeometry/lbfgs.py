"""
Optimization utilities, including L-BFGS wrapper and loss closure generation.
"""

from typing import Callable, Any, Tuple, Optional
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from . import complex_math

__all__ = ['compute_loss', 'create_loss_fn', 'LBFGS']


def compute_loss(params: Any, 
                 batch: dict, 
                 model: Any, 
                 loss_metric: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Computes the loss for a given batch of data. 
    This function is pure and can be JIT-compiled.
    
    This performs a self-contained calculation: it normalizes the volume form
    based ONLY on the current batch. This is appropriate for SGD/Adam mini-batches,
    but implies the normalization constant fluctuates per batch.
    """
    
    points = batch['points']
    omega_omegabar = batch['Omega_Omegabar']
    mass = batch['mass']
    restriction = batch['restriction']
    restriction_dag = jnp.swapaxes(jnp.conj(restriction), -1, -2)

    def scalar_potential(p: Any, z: jnp.ndarray) -> float:
        z_batch = z[None, :]
        out = model.apply(p, z_batch)
        return jnp.real(out[0, 0])

    # 1. Compute Hessians
    pot_fn = lambda z: scalar_potential(params, z)
    hessians = jax.vmap(lambda z: complex_math.complex_hessian(pot_fn, z))(points)
    
    # 2. Restrict metric
    temp = jnp.matmul(restriction, hessians)
    metric_restricted = jnp.matmul(temp, restriction_dag)
    
    # 3. Compute determinant (Volume Form)
    det_vol = jnp.real(jax.vmap(jnp.linalg.det)(metric_restricted))
    
    # 4. Normalize (Local Batch Normalization)
    weights = mass / jnp.sum(mass)
    factor = jnp.sum(weights * det_vol / omega_omegabar)
    det_omega = det_vol / factor
    
    return loss_metric(omega_omegabar, det_omega, mass)


def _compute_unnormalized_volumes(params, batch, model):
    """
    Computes only the unnormalized volume determinants for a batch.
    Helper for memory-efficient full-batch evaluation.
    """
    points = batch['points']
    restriction = batch['restriction']
    restriction_dag = jnp.swapaxes(jnp.conj(restriction), -1, -2)

    def scalar_potential(p, z):
        z_batch = z[None, :]
        out = model.apply(p, z_batch)
        return jnp.real(out[0, 0])

    pot_fn = lambda z: scalar_potential(params, z)
    hessians = jax.vmap(lambda z: complex_math.complex_hessian(pot_fn, z))(points)
    
    metric_restricted = jnp.matmul(jnp.matmul(restriction, hessians), restriction_dag)
    det_vol = jnp.real(jax.vmap(jnp.linalg.det)(metric_restricted))
    return det_vol


def create_loss_fn(model: Any, 
                   dataset: dict, 
                   loss_metric: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   batch_size: Optional[int] = None) -> Callable:
    """
    Creates a closure that takes `params` and returns a scalar loss over the entire `dataset`.
    
    Args:
        model: The Flax model instance.
        dataset: Dictionary of numpy/jax arrays.
        loss_metric: Loss function (e.g. weighted_MAPE).
        batch_size: If None, computes loss on the full dataset at once (fastest, high memory).
                    If set, splits dataset into chunks and computes gradients sequentially
                    using `jax.checkpoint` to save memory (Gradient Accumulation).
    """
    
    # 1. Prepare Data
    n_points = dataset['points'].shape[0]
    
    if batch_size is None:
        # --- Standard Full Batch Mode ---
        dataset_jax = {k: jnp.array(v) for k, v in dataset.items()}
        # Ensure complex types
        dataset_jax['points'] = dataset_jax['points'].astype(jnp.complex64)
        dataset_jax['restriction'] = dataset_jax['restriction'].astype(jnp.complex64)
        
        def loss_fn(params):
            return compute_loss(params, dataset_jax, model, loss_metric)
        return loss_fn

    else:
        # --- Memory Efficient Batched Mode ---
        # 1. Pad data to be multiple of batch_size
        remainder = n_points % batch_size
        pad_len = (batch_size - remainder) if remainder > 0 else 0
        total_len = n_points + pad_len
        n_batches = total_len // batch_size
        
        def pad_array(arr):
            # Pad with zeros (or first element to be safe, masked out later)
            if pad_len > 0:
                padding = jnp.zeros((pad_len,) + arr.shape[1:], dtype=arr.dtype)
                return jnp.concatenate([arr, padding], axis=0)
            return arr

        dataset_padded = {k: pad_array(jnp.array(v)) for k, v in dataset.items()}
        
        # Ensure types
        dataset_padded['points'] = dataset_padded['points'].astype(jnp.complex64)
        dataset_padded['restriction'] = dataset_padded['restriction'].astype(jnp.complex64)
        
        # Create mask for valid points (1.0 for valid, 0.0 for padding)
        valid_mask = jnp.concatenate([jnp.ones(n_points), jnp.zeros(pad_len)])
        
        # Reshape into (n_batches, batch_size, ...)
        batched_data = {}
        for k, v in dataset_padded.items():
            batched_data[k] = v.reshape((n_batches, batch_size) + v.shape[1:])
        
        # Mask doesn't need to be batched for final calculation, but helpful for intermediate structure?
        # Actually we can just flatten results at end.
        
        # Define the scan-compatible function
        # We use jax.checkpoint (remat) here! 
        # This tells JAX not to store activations for the backward pass, 
        # but to recompute them chunk-by-chunk. This is the key to memory savings.
        @jax.checkpoint
        def scan_step(params, batch_chunk):
            vols = _compute_unnormalized_volumes(params, batch_chunk, model)
            return vols # We don't carry any state, just map input->output

        def loss_fn_accumulated(params):
            # 1. Compute unnormalized volumes for all chunks sequentially
            # jax.lax.map saves memory compared to vmap by processing sequentially
            # batched_vols shape: (n_batches, batch_size)
            batched_vols = jax.lax.map(lambda b: scan_step(params, b), batched_data)
            
            # 2. Flatten back to (N_padded,)
            all_vols = batched_vols.reshape(-1)
            
            # 3. Apply Mask (zero out padding garbage)
            all_vols = all_vols * valid_mask
            
            # 4. Global Normalization
            # Int(Vol) = Int(Omega). 
            # We assume Omega_Omegabar is correct target measure.
            # Mass was calculated such that Sum(Mass) ~ Volume.
            # But strictly: factor = Sum(weights * det_vol / Omega).
            
            omega_flat = dataset_padded['Omega_Omegabar']
            mass_flat = dataset_padded['mass']
            
            # Zero out padding in mass/omega to be safe? 
            # Weights should handle it.
            # weight = mass / sum(mass). If mass is 0 for padding, weight is 0.
            # But we must ensure mass_flat has 0s at padding.
            mass_flat = mass_flat * valid_mask
            
            # sum(mass) should be sum over valid points
            total_mass = jnp.sum(mass_flat)
            weights = mass_flat / total_mass
            
            # Factor computation
            # Avoid division by zero in Omega if padding exists (though masked out weights help)
            # We add epsilon to Omega just in case
            safe_omega = omega_flat + (1.0 - valid_mask) # make padding 1.0
            
            factor = jnp.sum(weights * all_vols / safe_omega)
            
            # Normalized volume form
            det_omega_norm = all_vols / factor
            
            # 5. Compute Loss
            # loss_metric(y_true, y_pred, mass)
            # Use masked mass so padding contributes 0 to loss
            return loss_metric(omega_flat, det_omega_norm, mass_flat)

        return loss_fn_accumulated


class LBFGS:
    """Wrapper for JAXopt L-BFGS solver."""
    
    def __init__(self, loss_fn: Callable, max_iter: int = 1000, tol: float = 1e-5):
        self.loss_fn = loss_fn
        self.max_iter = max_iter
        self.tol = tol
        # implicit_diff=False is generally faster for exact derivatives if not doing hyperparam opt
        self.solver = jaxopt.LBFGS(fun=loss_fn, maxiter=max_iter, tol=tol)

    def run(self, init_params: Any) -> Tuple[Any, Any]:
        """Runs the optimization."""
        res = self.solver.run(init_params)
        return res.params, res.state
