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


# --- Core Loss Logic ---

def compute_loss(params: Any, 
                 batch: dict, 
                 model: Any, 
                 loss_metric: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Computes the loss for a given batch of data (Local Normalization).
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
    Computes unnormalized volume determinants. Helper for accumulated gradients.
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
        dataset_jax = {k: jnp.array(v) for k, v in dataset.items()}
        dataset_jax['points'] = dataset_jax['points'].astype(jnp.complex64)
        dataset_jax['restriction'] = dataset_jax['restriction'].astype(jnp.complex64)
        
        def loss_fn(params):
            return compute_loss(params, dataset_jax, model, loss_metric)
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