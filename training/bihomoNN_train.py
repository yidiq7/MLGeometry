import os
import sys
import argparse
import time
import math
import pickle

import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import jaxopt

import MLGeometry as mlg
from MLGeometry import bihomoNN as bnn
# Import model architectures from local models.py
from models import (
    zerolayer, onelayer, twolayers, threelayers, fourlayers, fivelayers,
    OuterProductNN_k2, OuterProductNN_k3, OuterProductNN_k4,
    k2_twolayers, k2_threelayers, k4_onelayer, k4_twolayers
)


# --- Argument Parsing ---
def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train a bihomogeneous neural network for Calabi-Yau metrics.")

    # Data generation
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--n_pairs', type=int, default=10000, help='Number of random pairs for hypersurface point generation.')
    parser.add_argument('--batch_size', type=int, default=None, help='Mini-batch size for Adam/SGD. If None, uses full batch.')
    parser.add_argument('--function', type=str, default='f0', choices=['f0', 'f1', 'f2'], help='Type of Calabi-Yau function to use.')
    parser.add_argument('--psi', type=float, default=0.5, help='Parameter psi for function f0.')
    parser.add_argument('--phi', type=float, default=0.0, help='Parameter phi for function f1.')
    parser.add_argument('--alpha', type=float, default=0.0, help='Parameter alpha for function f2.')

    # Network
    parser.add_argument('--OuterProductNN_k', type=int, default=None, help='Use OuterProductNN for specific k (2,3,4).')
    parser.add_argument('--layers', type=str, default='70_100', help='Hidden layer sizes, e.g., "70_100" for two layers.')
    parser.add_argument('--k2_as_first_layer', action='store_true', help='Use k=2 bihomogeneous layer as first layer.')
    parser.add_argument('--k4_as_first_layer', action='store_true', help='Use k=4 bihomogeneous layer as first layer.')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load pre-trained model parameters.')
    parser.add_argument('--save_dir', type=str, default='trained_model', help='Directory to save trained model.')
    parser.add_argument('--save_name', type=str, default='default_model', help='Name for saving the model.')

    # Training
    parser.add_argument('--precision', type=int, default=32, choices=[32, 64], help='Floating point precision (32 or 64 bits).')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs for training.')
    parser.add_argument('--loss_func', type=str, default='weighted_MAPE', choices=['weighted_MAPE', 'weighted_MSPE', 'weighted_RMSE', 'max_error', 'max_abs_error', 'MAPE_plus_max_error'], help='Loss function to use.')
    parser.add_argument('--clip_threshold', type=float, default=None, help='Gradient clipping threshold. If None, no clipping.')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'LBFGS'], help='Optimizer to use.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam/SGD.')
    parser.add_argument('--decay_rate', type=float, default=1.0, help='Learning rate decay rate for Adam.')
    parser.add_argument('--num_correction_pairs', type=int, default=10, help='Number of correction pairs for L-BFGS.')

    return parser.parse_args()


# --- Main Training Function ---
def main():
    """Main function to setup and run the training process."""
    args = parse_args()
    
    # Set global precision before any JAX operations
    mlg.set_precision(args.precision)
    
    print(f"--- Processing Model: {args.save_name} ---")

    # Set random seeds
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # --- Define Hypersurface ---
    z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
    Z = [z0, z1, z2, z3, z4]

    # Construct function 'f' based on arguments
    f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + args.psi * z0 * z1 * z2 * z3 * z4
    if args.function == 'f1':
        f += args.phi * (z3 * z4**4 + z3**2 * z4**3 + z3**3 * z4**2 + z3**4 * z4)
    elif args.function == 'f2':
        f += args.alpha * (z2 * z0**4 + z0 * z4 * z1**3 + z0 * z2 * z3 * z4**2 + z3**2 * z1**3 + z4 * z1**2 * z2**2 + z0 * z1 * z2 * z3**2 +
                           z2 * z4 * z3**3 + z0 * z1**4 + z0 * z4**2 * z2**2 + z4**3 * z1**2 + z0 * z2 * z3**3 + z3 * z4 * z0**3 + z1**3 * z4**2 +
                           z0 * z2 * z4 * z1**2 + z1**2 * z3**3 + z1 * z4**4 + z1 * z2 * z0**3 + z2**2 * z4**3 + z4 * z2**4 + z1 * z3**4)

    print("\n--- Generating Hypersurface Points ---")
    HS_train = mlg.hypersurface.Hypersurface(Z, f, args.n_pairs)
    HS_test = mlg.hypersurface.Hypersurface(Z, f, args.n_pairs)
    print(f"Generated {HS_train.n_points} training points.")
    print(f"Generated {HS_test.n_points} test points.")

    # --- Generate Datasets ---
    print("\n--- Generating Datasets for Training/Testing ---")
    train_set = mlg.dataset.generate_dataset(HS_train)
    test_set = mlg.dataset.generate_dataset(HS_test)

    # --- Network Model Selection ---
    print("\n--- Selecting Network Architecture ---")
    n_units = []
    if args.layers:
        n_units = [int(u) for u in args.layers.split('_')]
    n_hidden = len(n_units) - 1 if n_units else -1 # -1 if zero layer, 0 if one layer

    if args.OuterProductNN_k is not None:
        model_list_OuterProductNN = {2: OuterProductNN_k2, 3: OuterProductNN_k3, 4: OuterProductNN_k4}
        model_cls = model_list_OuterProductNN.get(args.OuterProductNN_k)
        if model_cls is None:
            raise ValueError(f"OuterProductNN_k only supports 2, 3, 4. Got {args.OuterProductNN_k}")
        model = model_cls()
    elif args.k2_as_first_layer:
        model_list_k2_as_first_layer = {1: k2_twolayers, 2: k2_threelayers} # n_hidden: 1 -> 2 layers, 2 -> 3 layers
        model_cls = model_list_k2_as_first_layer.get(n_hidden)
        if model_cls is None:
             raise ValueError(f"k2_as_first_layer only supports n_hidden=1 (2layers) or n_hidden=2 (3layers). Got {n_hidden}")
        model = model_cls(n_units)
    elif args.k4_as_first_layer:
        model_list_k4_as_first_layer = {0: k4_onelayer, 1: k4_twolayers} # n_hidden: 0 -> 1 layer, 1 -> 2 layers
        model_cls = model_list_k4_as_first_layer.get(n_hidden)
        if model_cls is None:
            raise ValueError(f"k4_as_first_layer only supports n_hidden=0 (1layer) or n_hidden=1 (2layers). Got {n_hidden}")
        model = model_cls(n_units)
    else:
        model_list_general = {0: zerolayer, 1: onelayer, 2: twolayers, 3: threelayers, 4: fourlayers, 5: fivelayers}
        model_cls = model_list_general.get(n_hidden)
        if model_cls is None:
            raise ValueError(f"General layers supports n_hidden from 0 to 5. Got {n_hidden}")
        model = model_cls(n_units)

    # Load pre-trained parameters if specified
    params = None
    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Loading model parameters from {args.load_model}")
            with open(args.load_model, 'rb') as pkl:
                params = pickle.load(pkl)
        else:
            print(f"Warning: Model file not found at {args.load_model}. Starting with random initialization.")

    # --- Loss Function Setup ---
    loss_func_map = {
        "weighted_MAPE": mlg.loss.weighted_MAPE,
        "weighted_MSPE": mlg.loss.weighted_MSPE,
        "max_error": mlg.loss.max_error,
        "max_abs_error": mlg.loss.max_abs_error,
        "MAPE_plus_max_error": mlg.loss.MAPE_plus_max_error
    }
    loss_metric = loss_func_map[args.loss_func]

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Training ---
    print("\n--- Starting Training ---")
    train_start_time = time.time()
    training_history = []

    if args.optimizer.lower() == 'lbfgs':
        params, final_loss = mlg.trainer.train_lbfgs(
            model=model,
            dataset=train_set,
            max_iter=args.max_epochs,
            loss_metric=loss_metric,
            params=params,
            batch_size=args.batch_size or 2048,
            verbose=True,
            history=training_history
        )
    else:
        train_batch_size = args.batch_size if args.batch_size is not None else train_set['points'].shape[0]
        steps_per_epoch = int(np.ceil(train_set['points'].shape[0] / train_batch_size))
        
        schedule = optax.exponential_decay(
            init_value=args.learning_rate,
            transition_steps=steps_per_epoch, 
            decay_rate=args.decay_rate
        )
        
        if args.optimizer.lower() == 'sgd':
            optimizer = optax.sgd(schedule)
        else: # Adam
            optimizer = optax.adam(schedule)
        
        if args.clip_threshold is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(args.clip_threshold),
                optimizer
            )
            
        params, final_loss = mlg.trainer.train_optax(
            model=model,
            dataset=train_set,
            optimizer=optimizer,
            epochs=args.max_epochs,
            batch_size=train_batch_size,
            loss_metric=loss_metric,
            params=params,
            verbose=True,
            history=training_history
        )

    train_time = time.time() - train_start_time
    print(f"\nTraining finished in {train_time:.2f} seconds.")

    # --- Save Trained Model Parameters ---
    model_save_path = os.path.join(args.save_dir, args.save_name + '.pkl')
    with open(model_save_path, 'wb') as pkl:
        pickle.dump(params, pkl)
    print(f"Trained model parameters saved to {model_save_path}")

    # --- Final Evaluation ---
    print("\n--- Final Evaluation ---")
    sigma_train = mlg.loss.evaluate_dataset(model, params, train_set, mlg.loss.weighted_MAPE, args.batch_size)
    sigma_test = mlg.loss.evaluate_dataset(model, params, test_set, mlg.loss.weighted_MAPE, args.batch_size*10)
    E_train = mlg.loss.evaluate_dataset(model, params, train_set, mlg.loss.weighted_MSPE, args.batch_size)
    E_test = mlg.loss.evaluate_dataset(model, params, test_set, mlg.loss.weighted_MSPE, args.batch_size*10)
    sigma_max_train = mlg.loss.evaluate_dataset(model, params, train_set, mlg.loss.max_error, args.batch_size)
    sigma_max_test = mlg.loss.evaluate_dataset(model, params, test_set, mlg.loss.max_error, args.batch_size*10)

    # Delta Sigma calculation
    def delta_sigma_square_metric_maker(sigma_val: jnp.ndarray):
        def metric_func(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
            weights = mass / jnp.sum(mass)
            return jnp.sum((jnp.abs(y_true - y_pred) / y_true - sigma_val)**2 * weights)
        return metric_func

    delta_sigma_train_sq = mlg.loss.evaluate_dataset(model, params, train_set, delta_sigma_square_metric_maker(sigma_train), args.batch_size)
    delta_sigma_test_sq = mlg.loss.evaluate_dataset(model, params, test_set, delta_sigma_square_metric_maker(sigma_test), args.batch_size*10)
    delta_sigma_train = math.sqrt(delta_sigma_train_sq.item() / HS_train.n_points)
    delta_sigma_test = math.sqrt(delta_sigma_test_sq.item() / HS_test.n_points)

    # Delta E calculation
    def delta_E_square_metric_maker(E_val: jnp.ndarray):
        def metric_func(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
            weights = mass / jnp.sum(mass)
            return jnp.sum(((y_pred / y_true - 1)**2 - E_val)**2 * weights)
        return metric_func
    
    delta_E_train_sq = mlg.loss.evaluate_dataset(model, params, train_set, delta_E_square_metric_maker(E_train), args.batch_size)
    delta_E_test_sq = mlg.loss.evaluate_dataset(model, params, test_set, delta_E_square_metric_maker(E_test), args.batch_size*10)
    delta_E_train = math.sqrt(delta_E_train_sq.item() / HS_train.n_points)
    delta_E_test = math.sqrt(delta_E_test_sq.item() / HS_test.n_points)

    print(f"Sigma Train: {sigma_train:.5f}, Test: {sigma_test:.5f}")
    print(f"E Train: {E_train:.5f}, Test: {E_test:.5f}")
    print(f"Delta Sigma Train: {delta_sigma_train:.5f}, Test: {delta_sigma_test:.5f}")
    print(f"Delta E Train: {delta_E_train:.5f}, Test: {delta_E_test:.5f}")
    print(f"Sigma Max Train: {sigma_max_train:.5f}, Test: {sigma_max_test:.5f}")

    # --- Write Results to File ---
    results_path = os.path.join(args.save_dir, args.save_name + ".txt")
    with open(results_path, "w") as f:
        f.write('[Results] \n')
        f.write(f'model_name = {args.save_name} \n')
        f.write(f'seed = {args.seed} \n')
        f.write(f'n_pairs = {args.n_pairs} \n')
        f.write(f'n_points = {HS_train.n_points} \n')
        f.write(f'batch_size = {args.batch_size} \n')
        f.write(f'precision = {args.precision} \n')
        f.write(f'function = {args.function} \n')
        f.write(f'psi = {args.psi} \n')
        if args.function == 'f1':
            f.write(f'phi = {args.phi} \n')
        elif args.function == 'f2':
            f.write(f'alpha = {args.alpha} \n') 
        f.write(f'n_parameters = {sum(x.size for x in jax.tree_util.tree_leaves(params))} \n') 
        f.write(f'loss function = {loss_metric.__name__} \n')
        f.write(f'optimizer = {args.optimizer} \n')
        if args.clip_threshold is not None:
            f.write(f'clip_threshold = {args.clip_threshold} \n')
        f.write('\n')
        f.write(f'n_epochs = {args.max_epochs} \n')
        f.write(f'train_time = {train_time:.6g} \n')
        f.write(f'sigma_train = {sigma_train:.6g} \n')
        f.write(f'sigma_test = {sigma_test:.6g} \n')
        f.write(f'delta_sigma_train = {delta_sigma_train:.6g} \n')
        f.write(f'delta_sigma_test = {delta_sigma_test:.6g} \n')
        f.write(f'E_train = {E_train:.6g} \n')
        f.write(f'E_test = {E_test:.6g} \n')
        f.write(f'delta_E_train = {delta_E_train:.6g} \n')
        f.write(f'delta_E_test = {delta_E_test:.6g} \n')
        f.write(f'sigma_max_train = {sigma_max_train:.6g} \n')
        f.write(f'sigma_max_test = {sigma_max_test:.6g} \n')
        
        f.write('\n[Training History]\n')
        for line in training_history:
            f.write(line + '\n')
    print(f"Results saved to {results_path}")

    summary_path = os.path.join(args.save_dir, "summary.txt")
    with open(summary_path, "a") as f:
        # Append to summary file
        summary_line = f'{args.save_name} {args.function} {args.psi} '
        if args.function == 'f1':
            summary_line += f'{args.phi} '
        elif args.function == 'f2': 
            summary_line += f'{args.alpha} '
        summary_line += (f'{train_time:.6g} {sigma_train:.6g} {sigma_test:.6g} '
                         f'{E_train:.6g} {E_test:.6g} {sigma_max_train:.6g} {sigma_max_test:.6g}\n')
        f.write(summary_line)
    print(f"Summary appended to {summary_path}")

if __name__ == '__main__':
    main()
