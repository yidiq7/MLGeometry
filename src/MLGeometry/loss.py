"""
Loss functions and objective factories for Calabi-Yau metric learning.
"""

from typing import Callable, Any, Optional
import jax
import jax.numpy as jnp
from . import complex_math
from . import config

__all__ = [
    'weighted_MAPE',
    'weighted_MSPE',
    'weighted_RMSE',
    'max_error',
    'max_abs_error',
    'MAPE_plus_max_error',
    'weighted_MAPE_amp_scaled',
    'weighted_MSPE_amp_scaled',
    'compute_loss',
    'compute_cy_metric',
    'compute_cy_metric_batched',
    'make_full_dataset_loss_fn',
    'compute_stage_loss',
    'make_stage_loss_fn',
    'evaluate_stage_dataset'
]


# --- Metrics ---

def weighted_MAPE(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Weighted Mean Absolute Percentage Error."""
    weights = mass / jnp.sum(mass)
    return jnp.sum(jnp.abs(y_true - y_pred) / y_true * weights)


def weighted_MSPE(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Weighted Mean Squared Percentage Error."""
    weights = mass / jnp.sum(mass)
    return jnp.sum(jnp.square(y_pred / y_true - 1) * weights)


def weighted_RMSE(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Weighted Root Mean Squared Error."""
    weights = mass / jnp.sum(mass)
    return jnp.sqrt(jnp.sum(jnp.square(y_pred - y_true) * weights))


def max_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Maximum Percentage Error."""
    return jnp.max(jnp.abs(y_true - y_pred) / y_true)


def max_abs_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Maximum Absolute Error."""
    return jnp.max(jnp.abs(y_true - y_pred))


def MAPE_plus_max_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Combination of MAPE and Max Error."""
    return max_error(y_true, y_pred, mass) + weighted_MAPE(y_true, y_pred, mass)


def eta_array(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    return y_pred / y_true


def weighted_MAPE_amp_scaled(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray,
                             amp: Any) -> jnp.ndarray:
    """Weighted Mean Absolute Percentage Error divided by the amplitude amp.

    The functions of a stage take a metric with three arguments, so give amp with
    functools.partial(weighted_MAPE_amp_scaled, amp=amp).
    """
    weights = mass / jnp.sum(mass)
    return jnp.sum(jnp.abs(y_true - y_pred) / y_true * weights) / amp


def weighted_MSPE_amp_scaled(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray,
                             amp: Any) -> jnp.ndarray:
    """Weighted Mean Squared Percentage Error divided by the square of the amplitude amp.

    The functions of a stage take a metric with three arguments, so give amp with
    functools.partial(weighted_MSPE_amp_scaled, amp=amp).
    """
    weights = mass / jnp.sum(mass)
    return jnp.sum(jnp.square(y_pred / y_true - 1) * weights) / (amp**2)


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


def compute_cy_metric_batched(model: Any, params: Any, dataset: dict, batch_size: int = 1000) -> jnp.ndarray:
    """
    Computes the Calabi-Yau metric in batches to avoid OOM on GPU.
    """
    n_points = dataset['points'].shape[0]
    metrics = []

    # JIT the step function
    step_fn = jax.jit(lambda b: compute_cy_metric(model, params, b))

    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        # Slice the batch
        batch = jax.tree_util.tree_map(lambda x: x[i:end], dataset)
        m = step_fn(batch)
        metrics.append(m)

    return jnp.concatenate(metrics, axis=0)


def compute_loss(model: Any,
                 params: Any, 
                 batch: dict, 
                 loss_metric: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Computes the loss for a given batch of data (Local Normalization).
    """
    omega_omegabar = batch['Omega_Omegabar']
    mass = batch['mass']
   
    det_vol = _compute_unnormalized_volumes(model, params, batch) 
    
    # Normalize (Local Batch Normalization)
    weights = mass / jnp.sum(mass)
    factor = jnp.sum(weights * det_vol / omega_omegabar)
    
    # Treat normalization factor as constant during backprop
    factor = jax.lax.stop_gradient(factor)
    
    det_omega = det_vol / factor
    
    return loss_metric(omega_omegabar, det_omega, mass)


def _compute_unnormalized_volumes(model, params, batch):
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
            # The model evaluates padded rows too, so the filler must be a real
            # point (a zero vector gives log(0) -> NaN). Sliced off below.
            if pad_len > 0:
                padding = jnp.repeat(arr[-1:], pad_len, axis=0)
                return jnp.concatenate([arr, padding], axis=0)
            return arr

        dataset_padded = {k: pad_array(jnp.array(v)) for k, v in dataset.items()}
        dataset_padded['points'] = dataset_padded['points'].astype(config.complex_dtype)
        dataset_padded['restriction'] = dataset_padded['restriction'].astype(config.complex_dtype)

        # Must match the volume slice below, or the filler row skews the weights.
        omega_flat = dataset_padded['Omega_Omegabar'][:n_points]
        mass_flat = dataset_padded['mass'][:n_points]

        batched_data = {}
        for k, v in dataset_padded.items():
            batched_data[k] = v.reshape((n_batches, batch_size) + v.shape[1:])
        
        @jax.checkpoint
        def scan_step(params, batch_chunk):
            return _compute_unnormalized_volumes(model, params, batch_chunk)

        def loss_fn_accumulated(params):
            batched_vols = jax.lax.map(lambda b: scan_step(params, b), batched_data)
            # Padding sits at the tail; n_points is static, so this is jit-safe.
            all_vols = batched_vols.reshape(-1)[:n_points]

            total_mass = jnp.sum(mass_flat)
            weights = mass_flat / total_mass

            factor = jnp.sum(weights * all_vols / omega_flat)

            # Treat normalization factor as constant during backprop
            factor = jax.lax.stop_gradient(factor)
            
            return loss_metric(omega_flat, all_vols / factor, mass_flat)

        return loss_fn_accumulated

def evaluate_dataset(model: Any, current_params: Any, dataset: dict, metric_func: Callable, batch_size: Optional[int] = None) -> jnp.ndarray:
    loss_fn_wrapped = make_full_dataset_loss_fn(model, dataset, metric_func, batch_size=batch_size)

    return loss_fn_wrapped(current_params)


# --- Stages ---

def compute_stage_loss(model: Any,
                       params: Any,
                       batch: dict,
                       loss_metric: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """
    Computes the loss of one stage on top of the stored metric of the earlier stages.

    The restricted metric is linear in the Kahler potential. So the metric of the full
    potential is batch['cymetric'] plus the metric of the model. The volume form is
    normalized over the batch (local normalization), as in compute_loss.

    Args:
        model: Flax model of the correction potential of this stage.
        params: Model parameters.
        batch: Dictionary with 'points', 'Omega_Omegabar', 'mass', 'restriction' and
            'cymetric', the restricted metric of the earlier stages at each point.
        loss_metric: Metric function (y_true, y_pred, mass) -> scalar, for example
            functools.partial(weighted_MAPE_amp_scaled, amp=amp).

    Returns:
        Scalar loss.
    """
    omega_omegabar = batch['Omega_Omegabar']
    mass = batch['mass']

    det_vol = _compute_stage_volumes(model, params, batch)

    weights = mass / jnp.sum(mass)
    factor = jnp.sum(weights * det_vol / omega_omegabar)

    # The gradient does not go through the normalization factor.
    factor = jax.lax.stop_gradient(factor)

    return loss_metric(omega_omegabar, det_vol / factor, mass)


def _compute_stage_volumes(model, params, batch):
    """
    Computes the unnormalized volume determinants of the stored metric plus the metric of the model.
    """
    metric = batch['cymetric'] + compute_cy_metric(model, params, batch)
    return jnp.real(jax.vmap(jnp.linalg.det)(metric))


def make_stage_loss_fn(model: Any,
                       dataset: dict,
                       loss_metric: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
                       batch_size: Optional[int] = None) -> Callable:
    """
    Creates a closure that takes `params` and returns the loss of one stage on the full dataset.

    The volume form is normalized over the full dataset. If batch_size is set, the
    volumes come in batches of this size under jax.checkpoint, so the memory of the
    gradient does not grow with the dataset.

    Args:
        model: Flax model of the correction potential of this stage.
        dataset: Dictionary with the keys of compute_stage_loss, 'cymetric' included.
        loss_metric: Metric function (y_true, y_pred, mass) -> scalar.
        batch_size: Number of points in each batch, or None for one full batch.

    Returns:
        Function params -> scalar loss.
    """
    n_points = dataset['points'].shape[0]

    if batch_size is None:
        def loss_fn(params):
            return compute_stage_loss(model, params, dataset, loss_metric)
        return loss_fn

    remainder = n_points % batch_size
    pad_len = (batch_size - remainder) if remainder > 0 else 0
    n_batches = (n_points + pad_len) // batch_size

    def pad_array(arr):
        # The model evaluates the filler rows too, so they repeat a real point. The loss cuts them off.
        if pad_len > 0:
            return jnp.concatenate([arr, jnp.repeat(arr[-1:], pad_len, axis=0)], axis=0)
        return arr

    # The arrays keep their dtypes, so a dataset of low precision is not hidden by a cast.
    dataset_padded = {k: pad_array(jnp.asarray(v)) for k, v in dataset.items()}

    omega_flat = dataset_padded['Omega_Omegabar'][:n_points]
    mass_flat = dataset_padded['mass'][:n_points]

    batched_data = {k: v.reshape((n_batches, batch_size) + v.shape[1:])
                    for k, v in dataset_padded.items()}

    @jax.checkpoint
    def batch_volumes(params, batch_chunk):
        return _compute_stage_volumes(model, params, batch_chunk)

    def loss_fn_accumulated(params):
        batched_vols = jax.lax.map(lambda b: batch_volumes(params, b), batched_data)
        # The filler rows are at the tail, and n_points is static, so this slice is jit-safe.
        all_vols = batched_vols.reshape(-1)[:n_points]

        weights = mass_flat / jnp.sum(mass_flat)
        factor = jnp.sum(weights * all_vols / omega_flat)

        # The gradient does not go through the normalization factor.
        factor = jax.lax.stop_gradient(factor)

        return loss_metric(omega_flat, all_vols / factor, mass_flat)

    return loss_fn_accumulated


def evaluate_stage_dataset(model: Any,
                           current_params: Any,
                           dataset: dict,
                           metric_func: Callable,
                           batch_size: Optional[int] = None) -> jnp.ndarray:
    """
    Evaluates a metric of the full potential, the stored metric plus the model, on a dataset.

    Args:
        model: Flax model of the correction potential of this stage.
        current_params: Model parameters.
        dataset: Dictionary with the keys of compute_stage_loss, 'cymetric' included.
        metric_func: Metric function (y_true, y_pred, mass) -> scalar, e.g. weighted_MAPE.
        batch_size: Number of points in each batch, or None for one full batch.

    Returns:
        Scalar value of metric_func, with the normalization over the full dataset.
    """
    loss_fn_wrapped = make_stage_loss_fn(model, dataset, metric_func, batch_size=batch_size)
    return loss_fn_wrapped(current_params)
