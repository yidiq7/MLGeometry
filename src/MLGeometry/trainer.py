"""
High-level training utilities for MLGeometry.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Sequence
import time
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax.flatten_util import ravel_pytree
import optax
import optax.tree_utils as otu
import numpy as np
from . import loss as mlg_loss
from . import complex_math
from . import config

__all__ = ['train_optax', 'train_lbfgs', 'train_stage_optax', 'train_stage_lbfgs',
           'solve_stage_output', 'train_stage_lm', 'init_params']


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


# --- Stages ---

class _Plateau:
    """Counts the steps since the loss last went below (1 - min_delta) times its best value."""

    def __init__(self, patience: Optional[int], min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.count = 0

    def update(self, value: float) -> bool:
        """Records one loss value. Returns True when the count reaches the patience."""
        if value < self.best * (1.0 - self.min_delta):
            self.best = value
            self.count = 0
        else:
            self.count += 1
        return self.patience is not None and self.count >= self.patience


def _make_validation_fn(model: Any, val_dataset: Optional[dict],
                        val_batch_size: Optional[int]) -> Optional[Callable]:
    """Sigma (weighted MAPE) of the full potential on the validation set, or None."""
    if val_dataset is None:
        return None
    return jax.jit(mlg_loss.make_stage_loss_fn(
        model, val_dataset, mlg_loss.weighted_MAPE, batch_size=val_batch_size))


def train_stage_optax(model: Any,
                      dataset: Dict[str, jnp.ndarray],
                      optimizer: optax.GradientTransformation,
                      epochs: int,
                      batch_size: int,
                      loss_metric: Callable,
                      params: Optional[Any] = None,
                      patience: Optional[int] = None,
                      min_delta: float = 0.0,
                      loss_record: Optional[list] = None,
                      val_dataset: Optional[Dict[str, jnp.ndarray]] = None,
                      val_batch_size: Optional[int] = 10000,
                      val_patience: Optional[int] = None,
                      val_min_delta: float = 0.0,
                      val_record: Optional[list] = None,
                      seed: int = 42,
                      verbose: bool = True,
                      history: Optional[list] = None) -> Tuple[Any, float]:
    """
    Runs the mini-batch loop of train_optax for one stage on top of the stored metric.

    The loss of each batch is mlg.loss.compute_stage_loss. So the dataset must hold
    'cymetric', the restricted metric of the earlier stages at each point.

    Args:
        model: Flax model of the correction potential of this stage.
        dataset: Dictionary of JAX arrays (points, Omega_Omegabar, mass, restriction, cymetric).
        optimizer: Optax optimizer instance (e.g. optax.adam(lr)).
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        loss_metric: Metric function (y_true, y_pred, mass) -> scalar.
        params: Initial model parameters. If None, initialized automatically from dataset shape.
        patience: If set, stop when the average loss of an epoch has not gone below
            (1 - min_delta) times its best earlier value for this number of epochs.
        min_delta: Relative decrease of the loss that resets the patience count.
        loss_record: Optional list. The average loss of each epoch is appended to it.
        val_dataset: Optional validation set with the keys of dataset. If set, sigma on it
            is computed after each epoch, and the function returns the parameters with the
            lowest sigma (the initial parameters included). It is never used for the fit.
        val_batch_size: Batch size of the metric on the validation set.
        val_patience: If set, also stop when the validation sigma has not gone below
            (1 - val_min_delta) times its best earlier value for this number of epochs.
        val_min_delta: Relative decrease of the validation sigma that resets val_patience.
        val_record: Optional list. The validation sigma after each epoch is appended to it.
        seed: Random seed for the initialization and the shuffle.
        verbose: Whether to print progress.
        history: Optional list to append log messages to.

    Returns:
        Tuple of (trained_params, final_avg_loss). With val_dataset: (the parameters with
        the lowest validation sigma, that sigma).
    """
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

    @jax.jit
    def step(current_params, current_opt_state, batch_data):
        loss_val, grads = jax.value_and_grad(
            lambda p: mlg_loss.compute_stage_loss(model, p, batch_data, loss_metric)
        )(current_params)
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        return new_params, new_opt_state, loss_val

    if verbose:
        msg = f"Starting training with {epochs} epochs, {num_batches} batches/epoch..."
        print(msg)
        if history is not None: history.append(msg)
    start_time = time.time()

    val_fn = _make_validation_fn(model, val_dataset, val_batch_size)
    if val_fn is not None:
        best_val, best_params, best_epoch = float(val_fn(params)), params, 0
        val_plateau = _Plateau(val_patience, val_min_delta)
        val_plateau.update(best_val)

    plateau = _Plateau(patience, min_delta)
    avg_loss = 0.0
    for epoch in range(1, epochs + 1):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, num_points)
        shuffled_data = jax.tree_util.tree_map(lambda x: x[perm], dataset)

        # The sum stays on the device, so the loop waits for the device once in each epoch.
        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            batch = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], shuffled_data)
            params, opt_state, loss_val = step(params, opt_state, batch)
            epoch_loss = epoch_loss + loss_val

        avg_loss = float(epoch_loss) / num_batches
        if loss_record is not None: loss_record.append(avg_loss)

        val_stop = False
        if val_fn is not None:
            val_sigma = float(val_fn(params))
            if val_record is not None: val_record.append(val_sigma)
            if val_sigma < best_val:
                best_val, best_params, best_epoch = val_sigma, params, epoch
            val_stop = val_plateau.update(val_sigma)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            msg = f"Epoch {epoch}: Avg Loss = {avg_loss:.5e}"
            if val_fn is not None:
                msg += f", Val Sigma = {val_sigma:.5e}"
            print(msg)
            if history is not None: history.append(msg)

        if plateau.update(avg_loss):
            if verbose:
                msg = (f"Stopped at epoch {epoch}: no relative decrease of {min_delta:g} "
                       f"in {patience} epochs. Avg Loss = {avg_loss:.5e}")
                print(msg)
                if history is not None: history.append(msg)
            break

        if val_stop:
            if verbose:
                msg = (f"Stopped at epoch {epoch}: no relative decrease of the validation sigma "
                       f"of {val_min_delta:g} in {val_patience} epochs")
                print(msg)
                if history is not None: history.append(msg)
            break

    total_time = time.time() - start_time
    if verbose:
        msg = f"Training finished in {total_time:.2f}s. Final Loss: {avg_loss:.5e}"
        if val_fn is not None:
            msg += f". Best Val Sigma = {best_val:.5e} at epoch {best_epoch}"
        print(msg)
        if history is not None: history.append(msg)

    if val_fn is not None:
        return best_params, best_val
    return params, avg_loss


def train_stage_lbfgs(model: Any,
                      dataset: Dict[str, jnp.ndarray],
                      epochs: int,
                      loss_metric: Callable,
                      params: Optional[Any] = None,
                      batch_size: Optional[int] = None,
                      tolerence: Any = 1e-8,
                      memory_size: int = 10,
                      max_linesearch_steps: int = 30,
                      patience: Optional[int] = None,
                      min_delta: float = 0.0,
                      loss_record: Optional[list] = None,
                      val_dataset: Optional[Dict[str, jnp.ndarray]] = None,
                      val_batch_size: Optional[int] = 10000,
                      val_patience: Optional[int] = None,
                      val_min_delta: float = 0.0,
                      val_record: Optional[list] = None,
                      seed: int = 42,
                      verbose: bool = True,
                      history: Optional[list] = None) -> Tuple[Any, float]:
    """
    Runs the L-BFGS loop of train_lbfgs for one stage on top of the stored metric.

    The loss is mlg.loss.make_stage_loss_fn on the full dataset. So the dataset must
    hold 'cymetric', the restricted metric of the earlier stages at each point.

    Args:
        model: Flax model of the correction potential of this stage.
        dataset: Data dictionary, 'cymetric' included.
        epochs: Maximum L-BFGS iterations.
        loss_metric: Metric function (y_true, y_pred, mass) -> scalar.
        params: Initial parameters. If None, initialized automatically.
        batch_size: If provided, the gradient is accumulated over batches of this size.
        tolerence: Convergence tolerance on the gradient norm.
        memory_size: Number of past updates L-BFGS keeps for its inverse Hessian approximation.
        max_linesearch_steps: Maximum number of zoom line search trials in each iteration.
        patience: If set, stop when the loss has not gone below (1 - min_delta) times
            its best earlier value for this number of iterations.
        min_delta: Relative decrease of the loss that resets the patience count.
        loss_record: Optional list. The loss after each iteration is appended to it.
        val_dataset: Optional validation set with the keys of dataset. If set, sigma on it
            is computed after each iteration, and the function returns the parameters with
            the lowest sigma (the initial parameters included). It is never used for the fit.
        val_batch_size: Batch size of the metric on the validation set.
        val_patience: If set, also stop when the validation sigma has not gone below
            (1 - val_min_delta) times its best earlier value for this number of iterations.
        val_min_delta: Relative decrease of the validation sigma that resets val_patience.
        val_record: Optional list. The validation sigma after each iteration is appended to it.
        seed: Random seed for initialization (if params is None).
        verbose: Print status.
        history: Optional list to append log messages to.

    Returns:
        (params with the lowest loss, that loss), or with val_dataset (params with the
        lowest validation sigma, that sigma). A failed line search can make the last
        iterate worse than an earlier one. Each parameter keeps its dtype: the L-BFGS
        state is in config.real_dtype, and each iterate is rounded to the dtype of each
        parameter, so float32 parameters stay on the float32 grid.
    """
    if params is None:
        input_dim = dataset['points'].shape[-1]
        if verbose:
            msg = f"Initializing parameters for input dimension {input_dim}..."
            print(msg)
            if history is not None: history.append(msg)
        params = init_params(model, (input_dim,), seed)

    # The line search of optax needs one dtype for all leaves, so it works on wide copies.
    param_dtypes = jax.tree_util.tree_map(lambda x: jnp.asarray(x).dtype, params)
    narrow = any(dt != jnp.dtype(config.real_dtype)
                 for dt in jax.tree_util.tree_leaves(param_dtypes))
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=config.real_dtype), params)

    def round_to_param_dtypes(p):
        return jax.tree_util.tree_map(
            lambda x, dt: x.astype(dt).astype(config.real_dtype), p, param_dtypes)

    loss_fn = mlg_loss.make_stage_loss_fn(
        model, dataset, loss_metric, batch_size=batch_size
    )
    jit_loss_fn = jax.jit(loss_fn)

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
    # The line search computed the value and the gradient already, so each iteration reuses them.
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def step(p, state):
        value, grad = value_and_grad(p, state=state)
        updates, new_state = solver.update(
            grad, state, p, value=value, grad=grad, value_fn=loss_fn
        )
        new_params = optax.apply_updates(p, updates)
        if narrow:
            new_params = round_to_param_dtypes(new_params)
        # The line search evaluated both at new_params, before any rounding to the parameter dtypes.
        return (new_params, new_state,
                otu.tree_get(new_state, 'value'),
                otu.tree_norm(otu.tree_get(new_state, 'grad')))

    state = solver.init(params)
    final_loss = float(jit_loss_fn(params))
    start_time = time.time()

    if verbose:
        msg = f"Initial Loss: {final_loss:.5e}"
        print(msg)
        if history is not None: history.append(msg)

    best_params, best_loss = params, final_loss
    plateau = _Plateau(patience, min_delta)
    plateau.update(final_loss)

    val_fn = _make_validation_fn(model, val_dataset, val_batch_size)
    if val_fn is not None:
        best_val, best_val_params, best_iter = float(val_fn(params)), params, 0
        val_plateau = _Plateau(val_patience, val_min_delta)
        val_plateau.update(best_val)

    for i in range(1, epochs + 1):
        params, state, loss_val, grad_norm = step(params, state)
        # After a rounding, the value of the line search is not the value at the iterate.
        final_loss = float(jit_loss_fn(params)) if narrow else float(loss_val)
        if loss_record is not None: loss_record.append(final_loss)
        if final_loss < best_loss:
            best_params, best_loss = params, final_loss

        val_stop = False
        if val_fn is not None:
            val_sigma = float(val_fn(params))
            if val_record is not None: val_record.append(val_sigma)
            if val_sigma < best_val:
                best_val, best_val_params, best_iter = val_sigma, params, i
            val_stop = val_plateau.update(val_sigma)

        if verbose:
            msg = f"Iteration {i}: Loss = {final_loss:.5e}"
            if val_fn is not None:
                msg += f", Val Sigma = {val_sigma:.5e}"
            print(msg)
            if history is not None: history.append(msg)

        if grad_norm < tolerence:
            if verbose:
                msg = f"Converged at iteration {i}"
                print(msg)
                if history is not None: history.append(msg)
            break

        if plateau.update(final_loss):
            if verbose:
                msg = (f"Stopped at iteration {i}: no relative decrease of {min_delta:g} "
                       f"in {patience} iterations")
                print(msg)
                if history is not None: history.append(msg)
            break

        if val_stop:
            if verbose:
                msg = (f"Stopped at iteration {i}: no relative decrease of the validation sigma "
                       f"of {val_min_delta:g} in {val_patience} iterations")
                print(msg)
                if history is not None: history.append(msg)
            break

    if verbose:
        msg = f"L-BFGS finished in {time.time() - start_time:.2f}s. Final Loss: {best_loss:.5e}"
        if val_fn is not None:
            msg += f". Best Val Sigma = {best_val:.5e} at iteration {best_iter}"
        print(msg)
        if history is not None: history.append(msg)

    if val_fn is not None:
        best_params, best_loss = best_val_params, best_val
    best_params = jax.tree_util.tree_map(lambda x, dt: x.astype(dt), best_params, param_dtypes)
    return best_params, best_loss


# --- Exact solve and Levenberg-Marquardt for one stage ---

def _batched(arrays: Dict[str, jnp.ndarray], batch_size: int) -> Tuple[Dict[str, jnp.ndarray], int]:
    """Pads each array to a multiple of batch_size and splits it into batches.

    The filler rows repeat the last row, so the model sees only real points.

    Returns:
        (the arrays with the shape (n_batches, batch_size, ...), the number of real rows).
    """
    n_points = next(iter(arrays.values())).shape[0]
    pad_len = (-n_points) % batch_size

    def pad(arr):
        arr = jnp.asarray(arr)
        if pad_len > 0:
            arr = jnp.concatenate([arr, jnp.repeat(arr[-1:], pad_len, axis=0)], axis=0)
        return arr.reshape((-1, batch_size) + arr.shape[1:])
    return {k: pad(v) for k, v in arrays.items()}, n_points


def _pad_rows(x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Pads a vector with zeros to a multiple of batch_size and splits it into batches."""
    pad_len = (-x.shape[0]) % batch_size
    return jnp.concatenate([x, jnp.zeros(pad_len, x.dtype)]).reshape(-1, batch_size)


def _stage_arrays(dataset: dict, batch_size: int) -> Tuple[Dict[str, jnp.ndarray], int]:
    """The batched points, restrictions and stored metrics of a dataset."""
    return _batched({k: dataset[k] for k in ('points', 'restriction', 'cymetric')}, batch_size)


def _mass_weights(dataset: dict) -> jnp.ndarray:
    """The weights w_i = mass_i / sum(mass)."""
    mass = jnp.asarray(dataset['mass'])
    return mass / jnp.sum(mass)


def _make_stage_residual_fn(model: Any, dataset: dict, to_params: Callable,
                            batch_size: int) -> Callable:
    """The function x -> f, with f_i = eta_i - 1 of the full potential at each point.

    eta_i = rho_i / rho_bar, with rho_i = det g_i / Omega_Omegabar_i, g_i the stored metric
    plus the metric of the model with the parameters to_params(x), and rho_bar the mean of
    rho with the weights w_i.
    """
    batched, n_points = _stage_arrays(dataset, batch_size)
    omega = jnp.asarray(dataset['Omega_Omegabar'])
    weights = _mass_weights(dataset)

    @jax.jit
    def residual(x, batched, omega, weights):
        params = to_params(x)
        vols = jax.lax.map(lambda b: mlg_loss._compute_stage_volumes(model, params, b), batched)
        rho = vols.reshape(-1)[:n_points] / omega
        return rho / jnp.sum(weights * rho) - 1.0

    return lambda x: residual(x, batched, omega, weights)


def _make_log_volume_grad(model: Any, to_params: Callable) -> Callable:
    """The gradient a_i of log det g_i with respect to x at each point of a batch.

    g_i is the stored metric plus the metric of the model with the parameters to_params(x).
    The function maps (x, points, restriction, cymetric) to an array (batch, x.size).
    """
    def log_volume(x, point, restriction, cymetric):
        params = to_params(x)

        def potential(z):
            return jnp.real(model.apply(params, z[None, :])[0, 0])
        hessian = complex_math.complex_hessian(potential, point)
        metric = cymetric + restriction @ hessian @ jnp.conj(restriction).T
        return jnp.log(jnp.real(jnp.linalg.det(metric)))

    return jax.vmap(jax.grad(log_volume), in_axes=(None, 0, 0, 0))


def _make_stage_rows_fn(model: Any, dataset: dict, to_params: Callable, index: jnp.ndarray,
                        batch_size: int) -> Callable:
    """The function x -> the entries `index` of a_i at each point, an array (n_points, index.size)."""
    batched, n_points = _stage_arrays(dataset, batch_size)
    grad_fn = _make_log_volume_grad(model, to_params)

    @jax.jit
    def rows(x, batched):
        a = jax.lax.map(lambda b: grad_fn(x, b['points'], b['restriction'], b['cymetric'])[:, index],
                        batched)
        return a.reshape(-1, a.shape[-1])[:n_points]

    return lambda x: rows(x, batched)


def _make_stage_normal_fn(model: Any, dataset: dict, to_params: Callable,
                          batch_size: int) -> Callable:
    """The function (x, f) -> (J^T J, J^T r) of the residual r_i = sqrt(w_i) f_i.

    The row of J at point i is sqrt(w_i) (1 + f_i) (a_i - a_bar), with a_bar =
    sum_j w_j (1 + f_j) a_j the gradient of log rho_bar. The sums run over batches, so J
    itself is never stored.
    """
    batched, _ = _stage_arrays(dataset, batch_size)
    weights = _pad_rows(_mass_weights(dataset), batch_size)
    grad_fn = _make_log_volume_grad(model, to_params)

    @jax.jit
    def normal(x, f, batched, weights):
        # The filler rows have the weight zero, so they add nothing to the sums.
        f = _pad_rows(f, batch_size)
        n = x.shape[0]
        zeros = jnp.zeros(n, x.dtype)

        def body(carry, inputs):
            s, u, a_bar, v = carry
            batch, w, fb = inputs
            a = grad_fn(x, batch['points'], batch['restriction'], batch['cymetric'])
            w1 = w * (1.0 + fb)
            w2 = w1 * (1.0 + fb)
            return (s + a.T @ (w2[:, None] * a), u + w2 @ a, a_bar + w1 @ a, v + (w1 * fb) @ a), None

        (s, u, a_bar, v), _ = jax.lax.scan(body, (jnp.zeros((n, n), x.dtype), zeros, zeros, zeros),
                                           (batched, weights, f))
        c = jnp.sum(weights * (1.0 + f)**2)
        q = jnp.sum(weights * (1.0 + f) * f)
        jtj = s - jnp.outer(u, a_bar) - jnp.outer(a_bar, u) + c * jnp.outer(a_bar, a_bar)
        return 0.5 * (jtj + jtj.T), v - q * a_bar

    return lambda x, f: normal(x, f, batched, weights)


@jax.jit
def _lm_delta(jtj: jnp.ndarray, jtr: jnp.ndarray, damping: Any) -> jnp.ndarray:
    """The solution of (J^T J + damping diag(J^T J)) delta = -J^T r by a Cholesky factorization."""
    idx = jnp.arange(jtj.shape[0])
    matrix = jtj.at[idx, idx].add(damping * jnp.diag(jtj))
    chol = jnp.linalg.cholesky(matrix)
    return -jax.scipy.linalg.cho_solve((chol, True), jtr)


def _leaf_index(params: Any, name: str) -> jnp.ndarray:
    """The positions in ravel_pytree(params) of the leaves with `name` in their path."""
    def mark(path, x):
        names = [str(getattr(k, 'key', k)) for k in path]
        return jnp.full(jnp.shape(x), 1.0 if name in names else 0.0)
    mask, _ = ravel_pytree(jax.tree_util.tree_map_with_path(mark, params))
    return jnp.nonzero(mask > 0.5)[0]


def solve_stage_output(model: Any,
                       dataset: Dict[str, jnp.ndarray],
                       params: Any,
                       solves: int = 3,
                       output_name: str = 'output',
                       batch_size: int = 2000,
                       val_dataset: Optional[Dict[str, jnp.ndarray]] = None,
                       val_batch_size: Optional[int] = 10000,
                       step_record: Optional[list] = None,
                       verbose: bool = True,
                       history: Optional[list] = None) -> Tuple[Any, float]:
    """
    Solves the linear output layer of one stage exactly, with the hidden layers fixed.

    To first order in the change dc of the output kernel c, f_i(c + dc) = f_i + J_i dc, with
    J_i = (1 + f_i) (a_i - a_bar), a_i the gradient of log det g_i with respect to c, that is
    tr(g_i^-1 d dbar psi_m) for the output units, and a_bar the gradient of log rho_bar, the
    change of the mean of the volume form. Each solve minimizes sum_i w_i (f_i + J_i dc)^2
    with a QR factorization of the rows sqrt(w_i) J_i, and the next solve starts from the new f.
    The dataset must hold 'cymetric', the restricted metric of the earlier stages.

    Args:
        model: Flax model of the correction potential, linear in the kernel of `output_name`.
        dataset: Data dictionary, 'cymetric' included.
        params: Parameters of the model. The output kernel is the start of the first solve.
        solves: Number of solves.
        output_name: Name of the output layer in the parameter tree.
        batch_size: Number of points in each batch of the metric and of the gradients.
        val_dataset: Optional validation set with the keys of dataset. If set, the function
            returns the parameters with the lowest sigma on it, the start included.
        val_batch_size: Batch size of the metric on the validation set.
        step_record: Optional list. One dict for each solve is appended to it: the MSPE and
            sigma on the dataset, the validation sigma, the condition number of the rows and
            the time.
        verbose: Whether to print progress.
        history: Optional list to append log messages to.

    Returns:
        (params after the last solve, their MSPE on the dataset), or with val_dataset (the
        parameters with the lowest validation sigma, that sigma).
    """
    theta, unravel = ravel_pytree(params)
    index = _leaf_index(params, output_name)
    if index.size == 0:
        raise ValueError(f"The parameters have no layer with the name '{output_name}'.")
    residual_fn = _make_stage_residual_fn(model, dataset, unravel, batch_size)
    rows_fn = _make_stage_rows_fn(model, dataset, unravel, index, batch_size)
    weights = _mass_weights(dataset)
    sqrt_w = jnp.sqrt(weights)
    val_fn = _make_validation_fn(model, val_dataset, val_batch_size)

    def report(k, f, cond, elapsed):
        entry = {'solve': k, 'mspe': float(jnp.sum(weights * f**2)),
                 'sigma': float(jnp.sum(weights * jnp.abs(f))), 'cond': cond, 'time_s': elapsed}
        if val_fn is not None:
            entry['sigma_val'] = float(val_fn(unravel(theta)))
        if step_record is not None: step_record.append(entry)
        if verbose:
            msg = (f"Solve {k}: MSPE = {entry['mspe']:.5e}, Sigma = {entry['sigma']:.5e}"
                   + (f", Val Sigma = {entry['sigma_val']:.5e}" if val_fn is not None else "")
                   + (f", cond = {cond:.3e}" if cond is not None else ""))
            print(msg)
            if history is not None: history.append(msg)
        return entry

    start_time = time.time()
    f = residual_fn(theta)
    entry = report(0, f, None, 0.0)
    best_params, best_val = theta, entry.get('sigma_val')
    for k in range(1, solves + 1):
        t0 = time.time()
        a = rows_fn(theta)
        a_bar = (weights * (1.0 + f)) @ a
        rows = (sqrt_w * (1.0 + f))[:, None] * (a - a_bar)
        q, r = jnp.linalg.qr(rows)
        dc = -jax.scipy.linalg.solve_triangular(r, q.T @ (sqrt_w * f))
        sv = jnp.linalg.svd(r, compute_uv=False)
        theta = theta.at[index].add(dc)
        f = residual_fn(theta)
        entry = report(k, f, float(sv[0] / sv[-1]), time.time() - t0)
        if val_fn is not None and entry['sigma_val'] < best_val:
            best_params, best_val = theta, entry['sigma_val']

    if verbose:
        msg = f"Solves finished in {time.time() - start_time:.2f}s."
        print(msg)
        if history is not None: history.append(msg)

    if val_fn is not None:
        return unravel(best_params), best_val
    return unravel(theta), float(jnp.sum(weights * f**2))


def train_stage_lm(model: Any,
                   dataset: Dict[str, jnp.ndarray],
                   epochs: int,
                   params: Any,
                   batch_size: int = 2000,
                   damping: float = 1e-3,
                   patience: Optional[int] = None,
                   min_delta: float = 0.0,
                   val_dataset: Optional[Dict[str, jnp.ndarray]] = None,
                   val_batch_size: Optional[int] = 10000,
                   step_record: Optional[list] = None,
                   verbose: bool = True,
                   history: Optional[list] = None) -> Tuple[Any, float]:
    """
    Runs Levenberg-Marquardt for one stage on the MSPE of the full dataset, on all parameters.

    The residual is r_i = sqrt(w_i) f_i, with w_i = mass_i / sum(mass) and f_i = eta_i - 1
    for the volume form normalized by its mean over the dataset. Each step sums J^T J and
    J^T r over batches, with the change of the mean in J, and solves
    (J^T J + damping diag(J^T J)) delta = -J^T r with a Cholesky factorization. If the
    factorization fails, the damping grows by 10. A step is accepted only if the MSPE falls.
    The damping follows the gain ratio rho, the actual over the predicted decrease of the
    MSPE: an accepted step multiplies it by max(1/3, 1 - (2 rho - 1)^3), and a rejected step
    by nu, which then doubles. A rejected step reuses J^T J. The dataset must hold
    'cymetric', the restricted metric of the earlier stages.

    Args:
        model: Flax model of the correction potential of this stage.
        dataset: Data dictionary, 'cymetric' included.
        epochs: Maximum number of LM steps, accepted or rejected.
        params: Initial parameters.
        batch_size: Number of points in each batch of the sums.
        damping: Initial damping lambda.
        patience: If set, stop when the validation sigma (with no validation set, the MSPE)
            has not gone below (1 - min_delta) times its best earlier value for this number
            of steps.
        min_delta: Relative decrease that resets the patience count.
        val_dataset: Optional validation set with the keys of dataset. If set, sigma on it
            is computed after each accepted step, and the function returns the parameters
            with the lowest sigma (the initial parameters included).
        val_batch_size: Batch size of the metric on the validation set.
        step_record: Optional list. One dict for each step is appended to it: the damping,
            the acceptance, the gain ratio, the MSPE and sigma on the dataset, the validation
            sigma, the ratio of the largest to the smallest diagonal entry of J^T J, and the
            times.
        verbose: Whether to print progress.
        history: Optional list to append log messages to.

    Returns:
        (params with the lowest validation sigma, that sigma), or with no validation set
        (the last accepted params, their MSPE).
    """
    theta, unravel = ravel_pytree(params)
    residual_fn = _make_stage_residual_fn(model, dataset, unravel, batch_size)
    normal_fn = _make_stage_normal_fn(model, dataset, unravel, batch_size)
    weights = _mass_weights(dataset)
    val_fn = _make_validation_fn(model, val_dataset, val_batch_size)

    if verbose:
        msg = f"Starting LM training (Full Batch, {theta.size} parameters, {epochs} steps)..."
        print(msg)
        if history is not None: history.append(msg)
    start_time = time.time()

    f = residual_fn(theta)
    mspe = float(jnp.sum(weights * f**2))
    sigma = float(jnp.sum(weights * jnp.abs(f)))
    val_sigma = float(val_fn(unravel(theta))) if val_fn is not None else None
    best_theta, best_val = theta, val_sigma
    plateau = _Plateau(patience, min_delta)
    plateau.update(val_sigma if val_fn is not None else mspe)
    if verbose:
        msg = f"Initial MSPE = {mspe:.5e}, Sigma = {sigma:.5e}" + (
            f", Val Sigma = {val_sigma:.5e}" if val_fn is not None else "")
        print(msg)
        if history is not None: history.append(msg)

    jtj = None
    nu = 2.0
    for step in range(1, epochs + 1):
        t0 = time.time()
        time_normal = 0.0
        if jtj is None:
            jtj, jtr = normal_fn(theta, f)
            diag = jnp.diag(jtj)
            diag_ratio = float(jnp.max(diag) / jnp.min(diag))
            time_normal = time.time() - t0

        used_damping, retries = damping, 0
        delta = _lm_delta(jtj, jtr, used_damping)
        while not bool(jnp.all(jnp.isfinite(delta))) and retries < 30:
            used_damping, retries = used_damping * 10.0, retries + 1
            delta = _lm_delta(jtj, jtr, used_damping)
        predicted = float(-2.0 * (delta @ jtr) - delta @ (jtj @ delta))

        f_new = residual_fn(theta + delta)
        mspe_new = float(jnp.sum(weights * f_new**2))
        gain = (mspe - mspe_new) / predicted if predicted > 0 else -np.inf
        accepted = bool(np.isfinite(mspe_new) and mspe_new < mspe)
        if accepted:
            theta, f, mspe = theta + delta, f_new, mspe_new
            sigma = float(jnp.sum(weights * jnp.abs(f)))
            jtj = None
            # Nielsen's rule; a gain ratio that is not finite keeps the damping.
            factor = max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0)**3) if np.isfinite(gain) else 1.0
            damping = used_damping * factor
            nu = 2.0
            if val_fn is not None:
                val_sigma = float(val_fn(unravel(theta)))
                if val_sigma < best_val:
                    best_theta, best_val = theta, val_sigma
        else:
            damping = used_damping * nu
            nu *= 2.0

        entry = {'step': step, 'damping': used_damping, 'cholesky_retries': retries,
                 'accepted': accepted, 'gain_ratio': float(gain), 'mspe_fit': mspe,
                 'sigma_fit': sigma, 'sigma_val': val_sigma, 'diag_ratio': diag_ratio,
                 'time_normal_s': time_normal, 'time_s': time.time() - t0}
        if step_record is not None: step_record.append(entry)
        if verbose:
            msg = (f"LM step {step}: MSPE = {mspe:.5e}, Sigma = {sigma:.5e}"
                   + (f", Val Sigma = {val_sigma:.5e}" if val_fn is not None else "")
                   + f", lambda = {used_damping:.3e}, gain = {gain:.3e}, "
                   + ("accepted" if accepted else "rejected") + f", {entry['time_s']:.1f}s")
            print(msg)
            if history is not None: history.append(msg)

        if plateau.update(val_sigma if val_fn is not None else mspe):
            if verbose:
                msg = (f"Stopped at step {step}: no relative decrease of {min_delta:g} "
                       f"in {patience} steps")
                print(msg)
                if history is not None: history.append(msg)
            break

    if verbose:
        msg = f"LM finished in {time.time() - start_time:.2f}s. Final MSPE: {mspe:.5e}"
        if val_fn is not None:
            msg += f". Best Val Sigma = {best_val:.5e}"
        print(msg)
        if history is not None: history.append(msg)

    if val_fn is not None:
        return unravel(best_theta), best_val
    return unravel(theta), mspe
