"""
High-level training utilities for MLGeometry.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Sequence
import time
import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
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
            msg = f"Epoch {epoch}: Avg Loss = {avg_loss:.5e}"
            print(msg)
            if history is not None: history.append(msg)
            
    total_time = time.time() - start_time
    if verbose:
        msg = f"Training finished in {total_time:.2f}s. Final Loss: {avg_loss:.5e}"
        print(msg)
        if history is not None: history.append(msg)
        
    return params, avg_loss


def train_lbfgs(model: Any,
                dataset: Dict[str, jnp.ndarray],
                epochs: int,
                loss_metric: Callable,
                params: Optional[Any] = None,
                batch_size: Optional[int] = None,
                tolerence: Any = 1e-8,
                memory_size: int = 10,
                max_linesearch_steps: int = 30,
                seed: int = 42,
                verbose: bool = True,
                history: Optional[list] = None) -> Tuple[Any, float]:
    """
    Runs L-BFGS training. Supports memory-efficient gradient accumulation if batch_size is provided.

    Args:
        model: Flax model.
        dataset: Data dictionary.
        epochs: Maximum L-BFGS iterations.
        loss_metric: Metric function.
        params: Initial parameters. If None, initialized automatically.
        batch_size: If provided, uses gradient accumulation to handle large datasets.
        tolerence: Convergence tolerance on the gradient norm.
        memory_size: Number of past updates L-BFGS keeps to build its inverse
            Hessian approximation.
        max_linesearch_steps: Cap on zoom line search trials per iteration. Raising
            this above the optax default of 20 matters when starting far from the
            optimum, where it is worth several orders of magnitude on the
            non-smooth losses such as weighted_MAPE; it makes little difference
            when fine-tuning a model that Adam has already brought close.
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

    solver = optax.lbfgs(
        memory_size=memory_size,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps
        ),
    )
    # Reuses the value and gradient the line search already computed, so each
    # iteration costs one function evaluation rather than two.
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def step(p, state):
        value, grad = value_and_grad(p, state=state)
        updates, new_state = solver.update(
            grad, state, p, value=value, grad=grad, value_fn=loss_fn
        )
        new_params = optax.apply_updates(p, updates)
        # Both are evaluated at new_params, so they describe the iterate returned.
        return (new_params, new_state,
                otu.tree_get(new_state, 'value'),
                otu.tree_norm(otu.tree_get(new_state, 'grad')))

    state = solver.init(params)
    final_loss = loss_fn(params)
    start_time = time.time()

    if verbose:
        msg = f"Initial Loss: {final_loss:.5e}"
        print(msg)
        if history is not None: history.append(msg)

    for i in range(1, epochs + 1):
        params, state, final_loss, grad_norm = step(params, state)

        if verbose:
            msg = f"Iteration {i}: Loss = {final_loss:.5e}"
            print(msg)
            if history is not None: history.append(msg)

        if grad_norm < tolerence:
            if verbose:
                msg = f"Converged at iteration {i}"
                print(msg)
                if history is not None: history.append(msg)
            break

    if verbose:
        msg = f"L-BFGS finished in {time.time() - start_time:.2f}s. Final Loss: {final_loss:.5e}"
        print(msg)
        if history is not None: history.append(msg)

    return params, final_loss
