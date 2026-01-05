"""
High-level training utilities for MLGeometry.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Sequence
import time
import jax
import jax.numpy as jnp
import optax
import jaxopt
import kfac_jax
import numpy as np
from . import loss as mlg_loss
from . import config

__all__ = ['train_optax', 'train_lbfgs', 'train_kfac', 'init_params']


def init_params(model: Any, input_shape: Sequence[int], seed: int = 42) -> Any:
    """
    Initializes Flax model parameters.
    
    Args:
        model: Flax model instance.
        input_shape: Shape of the input (excluding batch dimension), e.g. (5,).
        seed: Random seed.
        
    Returns:
        Initialized parameters (PyTree).
    """
    rng = jax.random.PRNGKey(seed)
    # Add batch dimension (1, ...) for initialization
    dummy_input = jnp.ones((1,) + tuple(input_shape), dtype=config.complex_dtype)
    params = model.init(rng, dummy_input)
    return params


def train_optax(model: Any,
                dataset: Dict[str, jnp.ndarray],
                optimizer: optax.GradientTransformation,
                epochs: int,
                batch_size: int,
                loss_metric: Callable,
                params: Optional[Any] = None,
                residue_amp: Optional[config.real_dtype] = None,
                seed: int = 42,
                verbose: bool = True,
                history: Optional[list] = None) -> Tuple[Any, float]:
    """
    Runs a training loop using an Optax optimizer (e.g., Adam, SGD) with mini-batching.
    
    Args:
        model: Flax model instance.
        dataset: Dictionary of JAX arrays (points, Omega_Omegabar, mass, restriction).
        optimizer: Optax optimizer instance (e.g. optax.adam(lr)).
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        loss_metric: Metric function (e.g. mlg.loss.weighted_MAPE).
        params: Initial model parameters. If None, initialized automatically from dataset shape.
        seed: Random seed for shuffling.
        verbose: Whether to print progress.
        history: Optional list to append log messages to.
        
    Returns:
        Tuple of (trained_params, final_avg_loss).
    """
    
    # Auto-initialize parameters if not provided
    if params is None:
        input_dim = dataset['points'].shape[-1]
        if verbose:
            msg = f"Initializing parameters for input dimension {input_dim}..."
            print(msg)
            if history is not None: history.append(msg)
        params = init_params(model, (input_dim,), seed)

    rng = jax.random.PRNGKey(seed)
    opt_state = optimizer.init(params)
    
    num_points = dataset['points'].shape[0]
    num_batches = int(np.ceil(num_points / batch_size))
    
    # Define JIT-compiled step function
    @jax.jit
    def step(current_params, current_opt_state, batch_data):
        if residue_amp is None:
            loss_val, grads = jax.value_and_grad(
                lambda p: mlg_loss.compute_loss(model, p, batch_data, loss_metric)
            )(current_params)
        else:
            loss_val, grads = jax.value_and_grad(
                lambda p: mlg_loss.compute_residual_loss(model, p, batch_data, residue_amp, loss_metric)
            )(current_params)
             
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        return new_params, new_opt_state, loss_val

    if verbose:
        msg = f"Starting training with {epochs} epochs, {num_batches} batches/epoch..."
        print(msg)
        if history is not None: history.append(msg)
    start_time = time.time()
    
    avg_loss = 0.0
    for epoch in range(1, epochs + 1):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, num_points)
        
        # Shuffle data (pytree of arrays)
        shuffled_data = jax.tree_util.tree_map(lambda x: x[perm], dataset)
        
        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            
            # Slice batch
            batch = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], shuffled_data)
            
            params, opt_state, loss_val = step(params, opt_state, batch)
            epoch_loss += loss_val.item()
            
        avg_loss = epoch_loss / num_batches
        
        if verbose and (epoch % 10 == 0 or epoch == 1):
            msg = f"Epoch {epoch}: Avg Loss = {avg_loss:.5f}"
            print(msg)
            if history is not None: history.append(msg)
            
    total_time = time.time() - start_time
    if verbose:
        msg = f"Training finished in {total_time:.2f}s. Final Loss: {avg_loss:.5f}"
        print(msg)
        if history is not None: history.append(msg)
        
    return params, avg_loss


def train_kfac(model: Any,
               dataset: Dict[str, jnp.ndarray],
               epochs: int,
               batch_size: int,
               loss_metric: Callable,
               params: Optional[Any] = None,
               residue_amp: Optional[config.real_dtype] = None,
               seed: int = 42,
               verbose: bool = True,
               history: Optional[list] = None) -> Tuple[Any, float]:
    """
    Runs a training loop using the K-FAC (Kronecker-Factored Approximate Curvature) optimizer.
    
    Args:
        model: Flax model.
        dataset: Data dictionary.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        loss_metric: Metric function.
        params: Initial parameters.
        residue_amp: Optional scaling factor for residual loss.
        seed: Random seed.
        verbose: Print progress.
        history: Optional list to append log messages.
        
    Returns:
        (trained_params, final_loss)
    """
    # Auto-initialize parameters if not provided
    if params is None:
        input_dim = dataset['points'].shape[-1]
        if verbose:
            msg = f"Initializing parameters for input dimension {input_dim}..."
            print(msg)
            if history is not None: history.append(msg)
        params = init_params(model, (input_dim,), seed)

    rng = jax.random.PRNGKey(seed)
    
    # Define internal loss function for K-FAC with registration
    def loss_func_kfac(p, batch):
        omega_omegabar = batch['Omega_Omegabar']
        mass = batch['mass']
        
        # Determine if we use residual physics
        metric0 = batch.get('cymetric', None) if residue_amp is not None else None
        
        # Forward pass
        metric1 = mlg_loss.compute_cy_metric(model, p, batch)
        metric = metric1 + metric0 if metric0 is not None else metric1
        det_vol = jnp.real(jax.vmap(jnp.linalg.det)(metric))
        
        weights = mass / jnp.sum(mass)
        factor = jnp.sum(weights * det_vol / omega_omegabar)
        factor = jax.lax.stop_gradient(factor)
        det_omega = det_vol / factor
        
        # Registration for K-FAC curvature approximation
        # We register the ratio det_omega/omega_omegabar against 1.0
        ratio = det_omega / omega_omegabar
        target = jnp.ones_like(ratio)
        w_normalized = jnp.sqrt(mass)
        
        kfac_jax.register_squared_error_loss(
            ratio[:, None] * w_normalized[:, None], 
            target[:, None] * w_normalized[:, None]
        )
        
        # Compute the actual loss to return
        if residue_amp is not None and 'amp_scaled' in loss_metric.__name__:
            loss_val = loss_metric(omega_omegabar, det_omega, mass, residue_amp)
        else:
            loss_val = loss_metric(omega_omegabar, det_omega, mass)
            
        return loss_val, {}

    # Initialize K-FAC Optimizer
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_func_kfac, has_aux=True),
        l2_reg=0.0,
        value_func_has_aux=True,
        value_func_has_rng=False,
        use_adaptive_learning_rate=True,
        use_adaptive_momentum=True,
        use_adaptive_damping=True,
        initial_damping=1.0,
        multi_device=False,
        num_burnin_steps=0
    )

    # Initialize optimizer state
    dummy_batch = jax.tree_util.tree_map(lambda x: x[:batch_size], dataset)
    rng, init_rng = jax.random.split(rng)
    opt_state = optimizer.init(params, init_rng, dummy_batch)

    num_points = dataset['points'].shape[0]
    num_batches = int(np.ceil(num_points / batch_size))

    if verbose:
        msg = f"Starting K-FAC training with {epochs} epochs, {num_batches} batches/epoch..."
        print(msg)
        if history is not None: history.append(msg)
        
    start_time = time.time()
    global_step = 0
    avg_loss = 0.0

    for epoch in range(1, epochs + 1):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, num_points)
        shuffled_data = jax.tree_util.tree_map(lambda x: x[perm], dataset)
        
        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            batch = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], shuffled_data)
            
            rng, step_rng = jax.random.split(rng)
            
            # Step without top-level JIT to allow K-FAC internal control flow
            params, opt_state, stats = optimizer.step(
                params, opt_state, step_rng, batch=batch, global_step_int=global_step
            )
            
            epoch_loss += stats['loss']
            global_step += 1
            
        avg_loss = epoch_loss / num_batches
        if verbose and (epoch % 10 == 0 or epoch == 1):
            msg = f"Epoch {epoch}: Avg Loss = {avg_loss:.5f}"
            print(msg)
            if history is not None: history.append(msg)
            
    total_time = time.time() - start_time
    if verbose:
        msg = f"K-FAC finished in {total_time:.2f}s. Final Loss: {avg_loss:.5f}"
        print(msg)
        if history is not None: history.append(msg)
        
    return params, avg_loss


def train_lbfgs(model: Any,
                dataset: Dict[str, jnp.ndarray],
                max_iter: int,
                loss_metric: Callable,
                params: Optional[Any] = None,
                batch_size: Optional[int] = None,
                residue_amp: Optional[config.real_dtype] = None,
                seed: int = 42,
                verbose: bool = True,
                history: Optional[list] = None) -> Tuple[Any, float]:
    """
    Runs L-BFGS training. Supports memory-efficient gradient accumulation if batch_size is provided.
    
    Args:
        model: Flax model.
        dataset: Data dictionary.
        max_iter: Maximum L-BFGS iterations.
        loss_metric: Metric function.
        params: Initial parameters. If None, initialized automatically.
        batch_size: If provided, uses gradient accumulation to handle large datasets.
        seed: Random seed for initialization (if params is None).
        verbose: Print status.
        history: Optional list to append log messages to.
        
    Returns:
        (trained_params, final_loss)
    """
    # Auto-initialize parameters if not provided
    if params is None:
        input_dim = dataset['points'].shape[-1]
        if verbose:
            msg = f"Initializing parameters for input dimension {input_dim}..."
            print(msg)
            if history is not None: history.append(msg)
        params = init_params(model, (input_dim,), seed)

    loss_fn = mlg_loss.make_full_dataset_loss_fn(
        model, dataset, loss_metric, batch_size=batch_size, residue_amp=residue_amp
    )

    if verbose:
        mode = "Accumulated Gradients" if batch_size else "Full Batch"
        msg = f"Starting L-BFGS training ({mode})..."
        print(msg)
        if history is not None: history.append(msg)

    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=max_iter, tol=1e-5)
    start_time = time.time()
    
    if verbose:
        state = solver.init_state(params)
        
        @jax.jit
        def step(p, s):
            return solver.update(p, s)
            
        msg = f"Initial Loss: {state.value:.5f}"
        print(msg)
        if history is not None: history.append(msg)
        
        for i in range(1, max_iter + 1):
            params, state = step(params, state)
            msg = f"Iteration {i}: Loss = {state.value:.5f}"
            print(msg)
            if history is not None: history.append(msg)
            
            if state.error < 1e-5:
                msg = f"Converged at iteration {i}"
                print(msg)
                if history is not None: history.append(msg)
                break
        
        final_loss = state.value
        msg = f"L-BFGS finished in {time.time() - start_time:.2f}s. Final Loss: {final_loss:.5f}"
        print(msg)
        if history is not None: history.append(msg)

    else:
        res = solver.run(params)
        params = res.params
        final_loss = loss_fn(params)
        
    return params, final_loss
