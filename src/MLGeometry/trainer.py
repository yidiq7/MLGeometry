"""
High-level training utilities for MLGeometry.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Sequence
import time
import jax
import jax.numpy as jnp
import optax
import jaxopt
import numpy as np
from . import loss as mlg_loss
from . import config

__all__ = ['train_optax', 'train_lbfgs', 'init_params']


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
        loss_val, grads = jax.value_and_grad(
            lambda p: mlg_loss.compute_loss(model, p, batch_data, loss_metric)
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


def train_lbfgs(model: Any,
                dataset: Dict[str, jnp.ndarray],
                max_iter: int,
                loss_metric: Callable,
                params: Optional[Any] = None,
                batch_size: Optional[int] = None,
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
        model, dataset, loss_metric, batch_size=batch_size
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
