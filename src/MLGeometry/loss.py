"""
Loss functions and objective factories for Calabi-Yau metric learning.
"""

from typing import Callable, Any, Optional
import jax
import jax.numpy as jnp
from . import complex_math

__all__ = [
    'weighted_MAPE', 
    'weighted_MSE', 
    'max_error', 
    'MAPE_plus_max_error',
    'compute_loss',
    'compute_cy_metric',
    'make_full_dataset_loss_fn'
]


# --- Metrics ---

def weighted_MAPE(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Weighted Mean Absolute Percentage Error."""
    weights = mass / jnp.sum(mass)
    return jnp.sum(jnp.abs(y_true - y_pred) / y_true * weights)


def weighted_MSE(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Weighted Mean Squared Error (of the ratio - 1)."""
    weights = mass / jnp.sum(mass)
    return jnp.sum(jnp.square(y_pred / y_true - 1) * weights)


def max_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Maximum relative error."""
    return jnp.max(jnp.abs(y_true - y_pred) / y_true)


def MAPE_plus_max_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Combination of MAPE and Max Error."""
    return max_error(y_true, y_pred, mass) + weighted_MAPE(y_true, y_pred, mass)


def eta_array(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    return y_pred / y_true


# --- Core Logic ---

def compute_cy_metric(model: Any, params: Any, batch: dict) -> jnp.ndarray:
    """
    Computes the Calabi-Yau metric tensor g_{i j_bar} for a batch of points.
    
    Args:
        params: Model parameters.
        batch: Dictionary containing 'points' and 'restriction'.
        model: Flax model instance.
        
    Returns:
        Complex array of shape (N, n_man, n_man) representing the metric on the hypersurface.
    """
    points = batch['points']
    restriction = batch['restriction']
    
    def scalar_potential(z):
        # z shape: (d,) -> add batch dim (1, d)
        z_batch = z[None, :]
        out = model.apply(params, z_batch)
        return jnp.real(out[0, 0])

    # 1. Compute Hessians (N, d_amb, d_amb)
    hessians = jax.vmap(lambda z: complex_math.complex_hessian(scalar_potential, z))(points)
    
    # 2. Restrict metric to hypersurface (N, d_man, d_man)
    # G_restricted = R @ G_ambient @ R^H
    restriction_dag = jnp.swapaxes(jnp.conj(restriction), -1, -2)
    temp = jnp.matmul(restriction, hessians)
    metric_restricted = jnp.matmul(temp, restriction_dag)
    
    return metric_restricted


def compute_loss(model: Any,
                 params: Any, 
                 batch: dict, 
                 loss_metric: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Computes the loss for a given batch of data (Local Normalization).
    """
    omega_omegabar = batch['Omega_Omegabar']
    mass = batch['mass']
    
    # Reuse the metric computation logic
    metric_restricted = compute_cy_metric(model, params, batch)
    
    # Compute determinant (Volume Form)
    det_vol = jnp.real(jax.vmap(jnp.linalg.det)(metric_restricted))
    
    # Normalize (Local Batch Normalization)
    weights = mass / jnp.sum(mass)
    factor = jnp.sum(weights * det_vol / omega_omegabar)
    det_omega = det_vol / factor
    
    return loss_metric(omega_omegabar, det_omega, mass)


def _compute_unnormalized_volumes(params, batch, model):
    """
    Computes unnormalized volume determinants. Helper for accumulated gradients.
    """
    metric_restricted = compute_cy_metric(model, params, batch)
    return jnp.real(jax.vmap(jnp.linalg.det)(metric_restricted))


def make_full_dataset_loss_fn(model: Any, 
                   dataset: dict, 
                   loss_metric: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                   batch_size: Optional[int] = None) -> Callable:
    """
    Creates a closure that takes `params` and returns scalar loss.
    Supports memory-efficient gradient accumulation if batch_size is set.
    """
    n_points = dataset['points'].shape[0]
    
    if batch_size is None:
        # --- Full Batch Mode ---
        def loss_fn(params):
            return compute_loss(model, params, dataset, loss_metric)
        return loss_fn

    else:
        # --- Memory Efficient Batched Mode ---
        remainder = n_points % batch_size
        pad_len = (batch_size - remainder) if remainder > 0 else 0
        total_len = n_points + pad_len
        n_batches = total_len // batch_size
        
        def pad_array(arr):
            if pad_len > 0:
                padding = jnp.zeros((pad_len,) + arr.shape[1:], dtype=arr.dtype)
                return jnp.concatenate([arr, padding], axis=0)
            return arr

        dataset_padded = {k: pad_array(jnp.array(v)) for k, v in dataset.items()}
        dataset_padded['points'] = dataset_padded['points'].astype(jnp.complex64)
        dataset_padded['restriction'] = dataset_padded['restriction'].astype(jnp.complex64)
        
        valid_mask = jnp.concatenate([jnp.ones(n_points), jnp.zeros(pad_len)])
        
        batched_data = {}
        for k, v in dataset_padded.items():
            batched_data[k] = v.reshape((n_batches, batch_size) + v.shape[1:])
        
        @jax.checkpoint
        def scan_step(params, batch_chunk):
            return _compute_unnormalized_volumes(params, batch_chunk, model)

        def loss_fn_accumulated(params):
            batched_vols = jax.lax.map(lambda b: scan_step(params, b), batched_data)
            all_vols = batched_vols.reshape(-1) * valid_mask
            
            omega_flat = dataset_padded['Omega_Omegabar']
            mass_flat = dataset_padded['mass'] * valid_mask
            
            total_mass = jnp.sum(mass_flat)
            weights = mass_flat / total_mass
            
            safe_omega = omega_flat + (1.0 - valid_mask)
            factor = jnp.sum(weights * all_vols / safe_omega)
            
            return loss_metric(omega_flat, all_vols / factor, mass_flat)

        return loss_fn_accumulated

def evaluate_dataset(model: Any, current_params: Any, dataset: dict, metric_func: Callable, batch_size: Optional[int] = None) -> jnp.ndarray:
    loss_fn_wrapped = make_full_dataset_loss_fn(model, dataset, metric_func, batch_size=batch_size)

    # JIT the returned function
    #jitted_loss_fn = jax.jit(loss_fn_wrapped)
    return loss_fn_wrapped(current_params)
