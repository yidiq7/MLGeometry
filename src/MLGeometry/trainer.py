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
           'solve_stage_output', 'train_stage_lm', 'stage_normal_equations', 'solve_stage_units',
           'train_stage_varpro', 'init_params']


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
    """The function (x, aux) -> f, with f_i = eta_i - 1 of the full potential at each point.

    eta_i = rho_i / rho_bar, with rho_i = det g_i / Omega_Omegabar_i, g_i the stored metric
    plus the metric of the model with the parameters to_params(x, aux), and rho_bar the mean
    of rho with the weights w_i. aux holds the parameters that stay fixed, or None.
    """
    batched, n_points = _stage_arrays(dataset, batch_size)
    omega = jnp.asarray(dataset['Omega_Omegabar'])
    weights = _mass_weights(dataset)

    @jax.jit
    def residual(x, aux, batched, omega, weights):
        params = to_params(x, aux)
        vols = jax.lax.map(lambda b: mlg_loss._compute_stage_volumes(model, params, b), batched)
        rho = vols.reshape(-1)[:n_points] / omega
        return rho / jnp.sum(weights * rho) - 1.0

    return lambda x, aux=None: residual(x, aux, batched, omega, weights)


def _make_log_volume_grad(model: Any, to_params: Callable) -> Callable:
    """The gradient a_i of log det g_i with respect to x at each point of a batch.

    g_i is the stored metric plus the metric of the model with the parameters to_params(x, aux).
    The function maps (x, aux, points, restriction, cymetric) to an array (batch, x.size).
    """
    def log_volume(x, aux, point, restriction, cymetric):
        params = to_params(x, aux)

        def potential(z):
            return jnp.real(model.apply(params, z[None, :])[0, 0])
        hessian = complex_math.complex_hessian(potential, point)
        metric = cymetric + restriction @ hessian @ jnp.conj(restriction).T
        return jnp.log(jnp.real(jnp.linalg.det(metric)))

    return jax.vmap(jax.grad(log_volume), in_axes=(None, None, 0, 0, 0))


def _make_stage_rows_fn(model: Any, dataset: dict, to_params: Callable, index: jnp.ndarray,
                        batch_size: int) -> Callable:
    """The function (x, aux) -> the entries `index` of a_i at each point, an array (n_points, index.size)."""
    batched, n_points = _stage_arrays(dataset, batch_size)
    grad_fn = _make_log_volume_grad(model, to_params)

    @jax.jit
    def rows(x, aux, batched):
        a = jax.lax.map(lambda b: grad_fn(x, aux, b['points'], b['restriction'], b['cymetric'])[:, index],
                        batched)
        return a.reshape(-1, a.shape[-1])[:n_points]

    return lambda x, aux=None: rows(x, aux, batched)


def _make_stage_normal_fn(model: Any, dataset: dict, to_params: Callable,
                          batch_size: int) -> Callable:
    """The function (x, f, aux) -> (J^T J, J^T r) of the residual r_i = sqrt(w_i) f_i.

    The row of J at point i is sqrt(w_i) (1 + f_i) (a_i - a_bar), with a_bar =
    sum_j w_j (1 + f_j) a_j the gradient of log rho_bar. The sums run over batches, so J
    itself is never stored.
    """
    batched, _ = _stage_arrays(dataset, batch_size)
    weights = _pad_rows(_mass_weights(dataset), batch_size)
    grad_fn = _make_log_volume_grad(model, to_params)

    @jax.jit
    def normal(x, f, aux, batched, weights):
        # The filler rows have the weight zero, so they add nothing to the sums.
        f = _pad_rows(f, batch_size)
        n = x.shape[0]
        zeros = jnp.zeros(n, x.dtype)

        def body(carry, inputs):
            s, u, a_bar, v = carry
            batch, w, fb = inputs
            a = grad_fn(x, aux, batch['points'], batch['restriction'], batch['cymetric'])
            return _accumulate_normal((s, u, a_bar, v), a, w, fb), None

        (s, u, a_bar, v), _ = jax.lax.scan(body, (jnp.zeros((n, n), x.dtype), zeros, zeros, zeros),
                                           (batched, weights, f))
        return _finish_normal(s, u, a_bar, v, weights, f)

    return lambda x, f, aux=None: normal(x, f, aux, batched, weights)


def _accumulate_normal(carry: tuple, a: jnp.ndarray, w: jnp.ndarray, f: jnp.ndarray) -> tuple:
    """Adds the rows a of a batch, with the weights w and the residuals f, to the normal sums."""
    s, u, a_bar, v = carry
    w1 = w * (1.0 + f)
    w2 = w1 * (1.0 + f)
    return s + a.T @ (w2[:, None] * a), u + w2 @ a, a_bar + w1 @ a, v + (w1 * f) @ a


def _finish_normal(s, u, a_bar, v, weights, f) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """J^T J and J^T r from the sums of _accumulate_normal, with the change of the mean in J."""
    c = jnp.sum(weights * (1.0 + f)**2)
    q = jnp.sum(weights * (1.0 + f) * f)
    jtj = s - jnp.outer(u, a_bar) - jnp.outer(a_bar, u) + c * jnp.outer(a_bar, a_bar)
    return 0.5 * (jtj + jtj.T), v - q * a_bar


@jax.jit
def _lm_delta(jtj: jnp.ndarray, jtr: jnp.ndarray, damping: Any) -> jnp.ndarray:
    """The solution of (J^T J + damping diag(J^T J)) delta = -J^T r by a Cholesky factorization.

    A zero entry of diag(J^T J) belongs to a parameter with no effect on f. It takes the
    damping alone, so its step is zero.
    """
    idx = jnp.arange(jtj.shape[0])
    diag = jnp.diag(jtj)
    matrix = jtj.at[idx, idx].add(damping * jnp.where(diag > 0, diag, 1.0))
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
        val_dataset: Optional validation set with the keys of dataset. If set, sigma on it is
            recorded after each solve. It does not select the result.
        val_batch_size: Batch size of the metric on the validation set.
        step_record: Optional list. One dict for each solve is appended to it: the MSPE and
            sigma on the dataset, the validation sigma, the condition number of the rows and
            the time.
        verbose: Whether to print progress.
        history: Optional list to append log messages to.

    Returns:
        (params after the last solve, their MSPE on the dataset).
    """
    theta, unravel = ravel_pytree(params)
    index = _leaf_index(params, output_name)
    if index.size == 0:
        raise ValueError(f"The parameters have no layer with the name '{output_name}'.")
    residual_fn = _make_stage_residual_fn(model, dataset, lambda x, aux: unravel(x), batch_size)
    rows_fn = _make_stage_rows_fn(model, dataset, lambda x, aux: unravel(x), index, batch_size)
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
    report(0, f, None, 0.0)
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
        report(k, f, float(sv[0] / sv[-1]), time.time() - t0)

    if verbose:
        msg = f"Solves finished in {time.time() - start_time:.2f}s."
        print(msg)
        if history is not None: history.append(msg)

    return unravel(theta), float(jnp.sum(weights * f**2))


def train_stage_lm(model: Any,
                   dataset: Dict[str, jnp.ndarray],
                   epochs: int,
                   params: Any,
                   batch_size: int = 2000,
                   damping: float = 1e-3,
                   max_trials: int = 20,
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
    J^T r over batches once, with the change of the mean in J. Then it solves
    (J^T J + damping diag(J^T J)) delta = -J^T r with a Cholesky factorization, and it tries
    the step. If the MSPE does not fall, the damping is multiplied by nu, nu doubles, and
    the step tries again, up to max_trials times. If the factorization fails, the damping
    grows by 10. After an accepted trial with the gain ratio rho, the actual over the
    predicted decrease of the MSPE, the damping is multiplied by max(1/3, 1 - (2 rho - 1)^3)
    and nu is 2. The dataset must hold 'cymetric', the restricted metric of the earlier stages.

    Args:
        model: Flax model of the correction potential of this stage.
        dataset: Data dictionary, 'cymetric' included.
        epochs: Maximum number of LM steps. Each step computes J^T J once.
        params: Initial parameters.
        batch_size: Number of points in each batch of the sums.
        damping: Initial damping lambda.
        max_trials: Maximum number of trials in a step. If no trial lowers the MSPE, LM stops.
        patience: If set, stop when the validation sigma (with no validation set, the MSPE)
            has not gone below (1 - min_delta) times its best earlier value for this number
            of steps.
        min_delta: Relative decrease that resets the patience count.
        val_dataset: Optional validation set with the keys of dataset. If set, sigma on it
            is computed after each step, and the function returns the parameters with the
            lowest sigma (the initial parameters included).
        val_batch_size: Batch size of the metric on the validation set.
        step_record: Optional list. One dict for each step is appended to it: the damping of
            the accepted trial, the number of trials, the gain ratio, the MSPE and sigma on
            the dataset, the validation sigma, the ratio of the largest to the smallest
            positive diagonal entry of J^T J, the number of its zero diagonal entries, and
            the times.
        verbose: Whether to print progress.
        history: Optional list to append log messages to.

    Returns:
        (params with the lowest validation sigma, that sigma), or with no validation set
        (the last params, their MSPE).
    """
    theta, unravel = ravel_pytree(params)
    residual_fn = _make_stage_residual_fn(model, dataset, lambda x, aux: unravel(x), batch_size)
    normal_fn = _make_stage_normal_fn(model, dataset, lambda x, aux: unravel(x), batch_size)
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

    nu = 2.0
    for step in range(1, epochs + 1):
        t0 = time.time()
        jtj, jtr = normal_fn(theta, f)
        diag = jnp.diag(jtj)
        n_zero = int(jnp.sum(diag <= 0))
        diag_ratio = float(jnp.max(diag) / jnp.min(jnp.where(diag > 0, diag, jnp.inf)))
        time_normal = time.time() - t0

        # The trials of the step: a larger damping after each trial that does not lower the MSPE.
        accepted, trials, retries = False, 0, 0
        while not accepted and trials < max_trials:
            trials += 1
            delta = _lm_delta(jtj, jtr, damping)
            while not bool(jnp.all(jnp.isfinite(delta))) and retries < 30:
                damping, retries = damping * 10.0, retries + 1
                delta = _lm_delta(jtj, jtr, damping)
            predicted = float(-2.0 * (delta @ jtr) - delta @ (jtj @ delta))
            f_new = residual_fn(theta + delta)
            mspe_new = float(jnp.sum(weights * f_new**2))
            gain = (mspe - mspe_new) / predicted if predicted > 0 else -np.inf
            accepted = bool(np.isfinite(mspe_new) and mspe_new < mspe)
            used_damping = damping
            if accepted:
                theta, f, mspe = theta + delta, f_new, mspe_new
                sigma = float(jnp.sum(weights * jnp.abs(f)))
                # The rule of Nielsen; a gain ratio that is not finite keeps the damping.
                factor = max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0)**3) if np.isfinite(gain) else 1.0
                damping, nu = damping * factor, 2.0
            else:
                damping, nu = damping * nu, nu * 2.0

        if accepted and val_fn is not None:
            val_sigma = float(val_fn(unravel(theta)))
            if val_sigma < best_val:
                best_theta, best_val = theta, val_sigma

        entry = {'step': step, 'damping': used_damping, 'trials': trials, 'cholesky_retries': retries,
                 'accepted': accepted, 'gain_ratio': float(gain), 'mspe_fit': mspe,
                 'sigma_fit': sigma, 'sigma_val': val_sigma, 'diag_ratio': diag_ratio,
                 'n_zero_diag': n_zero, 'time_normal_s': time_normal, 'time_s': time.time() - t0}
        if step_record is not None: step_record.append(entry)
        if verbose:
            msg = (f"LM step {step}: MSPE = {mspe:.5e}, Sigma = {sigma:.5e}"
                   + (f", Val Sigma = {val_sigma:.5e}" if val_fn is not None else "")
                   + f", lambda = {used_damping:.3e}, gain = {gain:.3e}, {trials} trials, "
                   + f"{entry['time_s']:.1f}s")
            print(msg)
            if history is not None: history.append(msg)

        if not accepted:
            if verbose:
                msg = f"Stopped at step {step}: no trial of {max_trials} lowers the MSPE"
                print(msg)
                if history is not None: history.append(msg)
            break

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


def stage_normal_equations(model: Any,
                           dataset: Dict[str, jnp.ndarray],
                           params: Any,
                           batch_size: int = 2000) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes J^T J and J^T r of the MSPE of one stage at params, for all parameters.

    The residual and the rows of J are those of train_stage_lm: r_i = sqrt(w_i) f_i, and the
    row of J at point i is sqrt(w_i) (1 + f_i) (a_i - a_bar), with a_i the gradient of
    log det g_i and a_bar the change of the mean of the volume form. The eigenvalues of
    J^T J give cond(J). The dataset must hold 'cymetric', the restricted metric of the
    earlier stages.

    Args:
        model: Flax model of the correction potential of this stage.
        dataset: Data dictionary, 'cymetric' included.
        params: Parameters of the model.
        batch_size: Number of points in each batch of the sums.

    Returns:
        (J^T J, J^T r, f), with f_i = eta_i - 1 at each point of the dataset.
    """
    theta, unravel = ravel_pytree(params)

    def to_params(x, aux):
        return unravel(x)

    f = _make_stage_residual_fn(model, dataset, to_params, batch_size)(theta)
    jtj, jtr = _make_stage_normal_fn(model, dataset, to_params, batch_size)(theta, f)
    return jtj, jtr, f


# --- Potentials linear in fixed units ---

class _UnitsSolver:
    """The jitted parts of the exact solve of c for a potential psi = units(z) @ c with fixed units.

    model.apply(params, z, method=method) gives the units, an array (..., M), and
    params['params'][output_name] is c, an array (M,).
    """

    def __init__(self, model: Any, dataset: dict, method: str, output_name: str, batch_size: int):
        self.batched, self.n_points = _stage_arrays(dataset, batch_size)
        self.omega = jnp.asarray(dataset['Omega_Omegabar'])
        self.weights = _mass_weights(dataset)
        self.w_batched = _pad_rows(self.weights, batch_size)
        self.output_name = output_name
        n_points = self.n_points

        def units_at(params, z):
            return model.apply(params, z[None, :], method=method)[0]

        def second_derivative(params, z, d):
            # The second derivative of the units at z + t d with respect to t at t = 0.
            def first(s):
                return jax.jvp(lambda t: units_at(params, z + t * d), (s,), (jnp.ones_like(s),))[1]
            zero = jnp.zeros((), jnp.real(z).dtype)
            return jax.jvp(first, (zero,), (jnp.ones_like(zero),))[1]

        def rows_point(params, z, restriction, metric):
            # tr(g^-1 R H_m R^H) = sum_k (m_k H_m m_k^H) for the rows m_k of L^-1 R, with g = L L^H.
            # m H m^H is a quarter of the sum of the second derivatives along m and along i m.
            chol = jnp.linalg.cholesky(metric)
            m = jax.scipy.linalg.solve_triangular(chol, restriction, lower=True)
            total = 0.0
            for k in range(m.shape[0]):
                total = total + second_derivative(params, z, m[k]) + second_derivative(params, z, 1j * m[k])
            return 0.25 * total

        rows_fn = jax.vmap(rows_point, in_axes=(None, 0, 0, 0))

        @jax.jit
        def metrics(params, batched):
            return jax.lax.map(lambda b: b['cymetric'] + mlg_loss.compute_cy_metric(model, params, b), batched)

        @jax.jit
        def residual(metric_b, omega, weights):
            rho = jnp.real(jnp.linalg.det(metric_b)).reshape(-1)[:n_points] / omega
            return rho / jnp.sum(weights * rho) - 1.0

        @jax.jit
        def normal(params, metric_b, f, batched, w_batched):
            fb = _pad_rows(f, batch_size)
            n = params['params'][output_name].shape[0]
            zeros = jnp.zeros(n, config.real_dtype)

            def body(carry, inputs):
                batch, metric, w, fbb = inputs
                a = rows_fn(params, batch['points'], batch['restriction'], metric)
                return _accumulate_normal(carry, a, w, fbb), None

            init = (jnp.zeros((n, n), config.real_dtype), zeros, zeros, zeros)
            sums, _ = jax.lax.scan(body, init, (batched, metric_b, w_batched, fb))
            return _finish_normal(*sums, w_batched, fb)

        self._metrics, self._residual, self._normal, self._rows = metrics, residual, normal, rows_fn

    def residual(self, params: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """The batched full metric and f at all points of the dataset."""
        metric_b = self._metrics(params, self.batched)
        return metric_b, self._residual(metric_b, self.omega, self.weights)

    def solve(self, params: Any, solves: int, val_fn: Optional[Callable] = None,
              step_record: Optional[list] = None, verbose: bool = True,
              history: Optional[list] = None, rtol: Optional[float] = None) -> Tuple[Any, float]:
        """Runs the solves from params. Returns (params after the last solve, their MSPE).

        With rtol, the solves stop after the first one that changes the MSPE by not more than
        rtol times its value before that solve.
        """
        theta, unravel = ravel_pytree(params)
        index = _leaf_index(params, self.output_name)
        metric_b, f = self.residual(params)

        def report(k, cond, dropped, elapsed):
            entry = {'solve': k, 'mspe': float(jnp.sum(self.weights * f**2)),
                     'sigma': float(jnp.sum(self.weights * jnp.abs(f))), 'cond': cond,
                     'dropped': dropped, 'time_s': elapsed}
            if val_fn is not None:
                entry['sigma_val'] = float(val_fn(params))
            if step_record is not None: step_record.append(entry)
            if verbose:
                msg = (f"Solve {k}: MSPE = {entry['mspe']:.5e}, Sigma = {entry['sigma']:.5e}"
                       + (f", Val Sigma = {entry['sigma_val']:.5e}" if val_fn is not None else "")
                       + (f", cond = {cond:.3e}, dropped {dropped}" if cond is not None else ""))
                print(msg)
                if history is not None: history.append(msg)

        report(0, None, None, 0.0)
        for k in range(1, solves + 1):
            t0 = time.time()
            mspe_before = float(jnp.sum(self.weights * f**2))
            jtj, jtr = self._normal(params, metric_b, f, self.batched, self.w_batched)
            # The eigenvalues of J^T J give cond(J); directions below 1e-15 of the largest are dropped.
            evals, evecs = jnp.linalg.eigh(jtj)
            cond = float(jnp.sqrt(evals[-1] / evals[0])) if float(evals[0]) > 0 else float('inf')
            keep = evals > 1e-15 * evals[-1]
            inv = jnp.where(keep, 1.0 / jnp.where(keep, evals, 1.0), 0.0)
            dc = -evecs @ (inv * (evecs.T @ jtr))
            theta = theta.at[index].add(dc)
            params = unravel(theta)
            metric_b, f = self.residual(params)
            report(k, cond, int(jnp.sum(~keep)), time.time() - t0)
            mspe_after = float(jnp.sum(self.weights * f**2))
            if rtol is not None and abs(mspe_before - mspe_after) <= rtol * mspe_before:
                break
        return params, float(jnp.sum(self.weights * f**2))


def solve_stage_units(model: Any,
                      dataset: Dict[str, jnp.ndarray],
                      params: Any,
                      solves: int = 3,
                      units_method: str = 'units',
                      output_name: str = 'c',
                      batch_size: int = 1000,
                      val_dataset: Optional[Dict[str, jnp.ndarray]] = None,
                      val_batch_size: Optional[int] = 10000,
                      step_record: Optional[list] = None,
                      verbose: bool = True,
                      history: Optional[list] = None,
                      rtol: Optional[float] = None) -> Tuple[Any, float]:
    """
    Solves the coefficients c of a potential psi = units(z) @ c exactly, with the units fixed.

    The row of J at point i is (1 + f_i) (a_i - a_bar), with a_im = tr(g_i^-1 dd^bar u_m) on
    the hypersurface for the current full metric g_i, and a_bar the gradient of the log of
    the mean of the volume form. The second derivatives come along the three rows of
    L_i^-1 R_i, with g_i = L_i L_i^H and R_i the restriction, so no Hessian of all units is
    stored. Each solve sums J^T J and J^T r over batches and solves the normal equations with
    an eigendecomposition. cond(J) is the square root of the ratio of the largest to the
    smallest eigenvalue. The dataset must hold 'cymetric', the restricted metric of the
    earlier stages.

    Args:
        model: Flax model of the potential, with model.apply(params, z) = units(z) @ c.
        dataset: Data dictionary, 'cymetric' included.
        params: Parameters of the model. Their c is the start of the first solve.
        solves: Number of solves.
        units_method: Name of the method of the model that gives the units, (..., M).
        output_name: Name of c in params['params'].
        batch_size: Number of points in each batch of the metric and of the sums.
        val_dataset: Optional validation set. If set, sigma on it is recorded after each solve.
        val_batch_size: Batch size of the metric on the validation set.
        step_record: Optional list. One dict for each solve is appended to it: the MSPE and
            sigma on the dataset, the validation sigma, cond(J), the number of dropped
            directions and the time.
        verbose: Whether to print progress.
        history: Optional list to append log messages to.
        rtol: If set, stop after the first solve that changes the MSPE on the dataset by not
            more than rtol times its value before that solve. Then `solves` is the maximum.

    Returns:
        (params after the last solve, their MSPE on the dataset).
    """
    start_time = time.time()
    solver = _UnitsSolver(model, dataset, units_method, output_name, batch_size)
    val_fn = _make_validation_fn(model, val_dataset, val_batch_size)
    result = solver.solve(params, solves, val_fn, step_record, verbose, history, rtol=rtol)
    if verbose:
        msg = f"Solves finished in {time.time() - start_time:.2f}s."
        print(msg)
        if history is not None: history.append(msg)
    return result


def train_stage_varpro(model: Any,
                       dataset: Dict[str, jnp.ndarray],
                       epochs: int,
                       params: Any,
                       recalibrate: Optional[Callable] = None,
                       solves: int = 2,
                       units_method: str = 'units',
                       output_name: str = 'c',
                       batch_size: int = 1000,
                       damping: float = 1e-3,
                       max_trials: int = 20,
                       patience: Optional[int] = None,
                       min_delta: float = 0.0,
                       val_dataset: Optional[Dict[str, jnp.ndarray]] = None,
                       val_batch_size: Optional[int] = 10000,
                       step_record: Optional[list] = None,
                       verbose: bool = True,
                       history: Optional[list] = None) -> Tuple[Any, float]:
    """
    Runs LM on the parameters of a potential psi = units(z) @ c other than c (variable projection).

    The trained parameters are the leaves of params['params'] other than output_name. Each
    step sums J^T J and J^T r of them with c and the other collections of params fixed (for
    UnitRecurrence, the orthonormalization), and tries damped steps as train_stage_lm does,
    with the MSPE at the fixed c. After an accepted trial, recalibrate(params) computes the
    fixed collections again, and `solves` exact solves of c follow, as in solve_stage_units.
    The dataset must hold 'cymetric', the restricted metric of the earlier stages.

    Args:
        model: Flax model of the potential, with model.apply(params, z) = units(z) @ c.
        dataset: Data dictionary, 'cymetric' included.
        epochs: Maximum number of LM steps. Each step computes J^T J once.
        params: Initial parameters, a dict with the collection 'params'. Solve c first.
        recalibrate: Optional function params -> params that computes the fixed collections
            again after an accepted step. It must keep the shapes of the trained parameters.
        solves: Number of solves of c after each accepted step.
        units_method: Name of the method of the model that gives the units, (..., M).
        output_name: Name of c in params['params'].
        batch_size: Number of points in each batch of the sums.
        damping: Initial damping lambda.
        max_trials: Maximum number of trials in a step. If no trial lowers the MSPE, LM stops.
        patience: If set, stop when the validation sigma (with no validation set, the MSPE)
            has not gone below (1 - min_delta) times its best earlier value for this number
            of steps.
        min_delta: Relative decrease that resets the patience count.
        val_dataset: Optional validation set. If set, sigma on it is computed after each step,
            and the function returns the parameters with the lowest sigma (the start included).
        val_batch_size: Batch size of the metric on the validation set.
        step_record: Optional list. One dict for each step is appended to it: the damping of
            the accepted trial, the number of trials, the gain ratio, the MSPE of the trial and
            after the solves, sigma on the dataset, the validation sigma, the ratio of the
            largest to the smallest positive diagonal entry of J^T J, the number of its zero
            diagonal entries, and the times.
        verbose: Whether to print progress.
        history: Optional list to append log messages to.

    Returns:
        (params with the lowest validation sigma, that sigma), or with no validation set
        (the last params, their MSPE).
    """
    solver = _UnitsSolver(model, dataset, units_method, output_name, batch_size)
    weights = _mass_weights(dataset)
    val_fn = _make_validation_fn(model, val_dataset, val_batch_size)

    def split(p):
        trained = {k: v for k, v in p['params'].items() if k != output_name}
        aux = {'c': p['params'][output_name], 'rest': {k: v for k, v in p.items() if k != 'params'}}
        return trained, aux

    trained, aux = split(params)
    theta, unravel = ravel_pytree(trained)

    def to_params(x, aux):
        return {**aux['rest'], 'params': {**unravel(x), output_name: aux['c']}}

    residual_fn = _make_stage_residual_fn(model, dataset, to_params, batch_size)
    normal_fn = _make_stage_normal_fn(model, dataset, to_params, batch_size)

    if verbose:
        msg = f"Starting VarPro LM training ({theta.size} trained parameters, {epochs} steps)..."
        print(msg)
        if history is not None: history.append(msg)
    start_time = time.time()

    f = residual_fn(theta, aux)
    mspe = float(jnp.sum(weights * f**2))
    sigma = float(jnp.sum(weights * jnp.abs(f)))
    val_sigma = float(val_fn(params)) if val_fn is not None else None
    best_params, best_val = params, val_sigma
    plateau = _Plateau(patience, min_delta)
    plateau.update(val_sigma if val_fn is not None else mspe)
    if verbose:
        msg = f"Initial MSPE = {mspe:.5e}, Sigma = {sigma:.5e}" + (
            f", Val Sigma = {val_sigma:.5e}" if val_fn is not None else "")
        print(msg)
        if history is not None: history.append(msg)

    nu = 2.0
    for step in range(1, epochs + 1):
        t0 = time.time()
        jtj, jtr = normal_fn(theta, f, aux)
        diag = jnp.diag(jtj)
        n_zero = int(jnp.sum(diag <= 0))
        diag_ratio = float(jnp.max(diag) / jnp.min(jnp.where(diag > 0, diag, jnp.inf)))
        time_normal = time.time() - t0

        # The trials of the step, with c and the fixed collections held.
        accepted, trials, retries = False, 0, 0
        while not accepted and trials < max_trials:
            trials += 1
            delta = _lm_delta(jtj, jtr, damping)
            while not bool(jnp.all(jnp.isfinite(delta))) and retries < 30:
                damping, retries = damping * 10.0, retries + 1
                delta = _lm_delta(jtj, jtr, damping)
            predicted = float(-2.0 * (delta @ jtr) - delta @ (jtj @ delta))
            mspe_trial = float(jnp.sum(weights * residual_fn(theta + delta, aux)**2))
            gain = (mspe - mspe_trial) / predicted if predicted > 0 else -np.inf
            accepted = bool(np.isfinite(mspe_trial) and mspe_trial < mspe)
            used_damping = damping
            if accepted:
                factor = max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0)**3) if np.isfinite(gain) else 1.0
                damping, nu = damping * factor, 2.0
            else:
                damping, nu = damping * nu, nu * 2.0

        time_recal = time_solve = 0.0
        if accepted:
            # The fixed collections again, then c again, at the new parameters.
            params = to_params(theta + delta, aux)
            t1 = time.time()
            if recalibrate is not None:
                params = recalibrate(params)
            time_recal = time.time() - t1
            params, _ = solver.solve(params, solves, verbose=False)
            time_solve = time.time() - t1 - time_recal
            trained, aux = split(params)
            theta, _ = ravel_pytree(trained)
            f = residual_fn(theta, aux)
            mspe = float(jnp.sum(weights * f**2))
            sigma = float(jnp.sum(weights * jnp.abs(f)))
            if val_fn is not None:
                val_sigma = float(val_fn(params))
                if val_sigma < best_val:
                    best_params, best_val = params, val_sigma

        entry = {'step': step, 'damping': used_damping, 'trials': trials, 'cholesky_retries': retries,
                 'accepted': accepted, 'gain_ratio': float(gain), 'mspe_trial': mspe_trial,
                 'mspe_fit': mspe, 'sigma_fit': sigma, 'sigma_val': val_sigma, 'diag_ratio': diag_ratio,
                 'n_zero_diag': n_zero, 'time_normal_s': time_normal, 'time_recalibrate_s': time_recal,
                 'time_solve_s': time_solve, 'time_s': time.time() - t0}
        if step_record is not None: step_record.append(entry)
        if verbose:
            msg = (f"VarPro step {step}: MSPE = {mspe:.5e}, Sigma = {sigma:.5e}"
                   + (f", Val Sigma = {val_sigma:.5e}" if val_fn is not None else "")
                   + f", lambda = {used_damping:.3e}, gain = {gain:.3e}, {trials} trials, "
                   + f"{entry['time_s']:.1f}s")
            print(msg)
            if history is not None: history.append(msg)

        if not accepted:
            if verbose:
                msg = f"Stopped at step {step}: no trial of {max_trials} lowers the MSPE"
                print(msg)
                if history is not None: history.append(msg)
            break

        if plateau.update(val_sigma if val_fn is not None else mspe):
            if verbose:
                msg = (f"Stopped at step {step}: no relative decrease of {min_delta:g} "
                       f"in {patience} steps")
                print(msg)
                if history is not None: history.append(msg)
            break

    if verbose:
        msg = f"VarPro LM finished in {time.time() - start_time:.2f}s. Final MSPE: {mspe:.5e}"
        if val_fn is not None:
            msg += f". Best Val Sigma = {best_val:.5e}"
        print(msg)
        if history is not None: history.append(msg)

    if val_fn is not None:
        return best_params, best_val
    return params, mspe
