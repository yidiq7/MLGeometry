"""
Fits the stages of a multi-stage potential one after the other on stored datasets.

The base (stage 0) is the Fubini-Study potential log |z|^2, or nothing (--base none).
Each stage j adds a correction network. Its amp_j is the largest absolute error of the
volume form of the potential before stage j on the fit set (max_abs_error). A stage runs
Adam on mini-batches, and then L-BFGS with accumulated gradients. The restricted metric
of the earlier stages is stored at each point (dataset['cymetric']), so a stage
differentiates only its own network.

The networks of --network:
    tanh   SpectralNetwork: tanh layers on z_i z_j_bar / |z|^2, output amp * sinh(x).
    siren  SpectralSirenNetwork: SirenDense layers in place of the tanh layers.
    tanh_linear
           SpectralLinearNetwork: the tanh layers of tanh and the linear output amp * (h w),
           with no sinh and no bias.
    wanglai
           SpectralWangLaiNetwork: the first hidden layer is sin(kappa x W + b) (--kappa),
           the others are tanh layers, and the output amp * (h w) is linear with no bias.
    poly   PolynomialCorrection: x^2 layers on z_i z_j_bar / |z|^2, a linear output and
           amp, as a sum of the branches of --branches. --weight base gives each branch
           of degree L the base weight exp(-(L/k0) (Phi_prev - k0 log|z|^2)) of the frozen
           earlier stages; --weight fs gives the weight 1.
    bihom  Kahler_potential: log of a bihomogeneous network, a full potential with no FS
           base (use --base none). Its class is k0 H with k0 = 2^len(layers).

A loaded stage (--load_dir) can also be the HN potential of training/full_basis.py --task hn
(network 'hn' in its stage1.json, with the base 'none').

After Adam, --output_solves solves the linear output layer of tanh_linear exactly, with the
hidden layers fixed, and --lm_steps then runs Levenberg-Marquardt (LM) on all parameters,
full batch on MSPE. Each loss in --lbfgs_losses gives one L-BFGS run from the result of the
last phase. The first one continues the chain to the next stage. --init_params gives the
initial parameters of the stage --start_stage, for example the Adam result of another run.
With --validation, each phase returns the parameters with the lowest sigma on the
validation set; the validation set is never used for a fit. --lr_search picks the Adam
learning rate by the validation sigma of short runs.

With --siren_float32, the SirenDense parameters of the stages to fit are float32, as the
branch FP64 created them. Their initial values are the float64 ones rounded to float32.

Example (from the root folder of the repository):
    PYTHONPATH=src python training/spectral_stages.py --fit data/fit.npz \
        --holdout data/holdout.npz --save_dir runs/spectral --stages 3
"""

import argparse
import functools
import json
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

import MLGeometry as mlg
from models import (SpectralNetwork, SpectralLinearNetwork, SpectralSirenNetwork, SpectralWangLaiNetwork,
                    PolynomialCorrection, Kahler_potential, zerolayer, calibrate_polynomial,
                    reduced_potential_mean, HNPotential, load_sections)


LOSSES = {
    'weighted_MAPE_amp_scaled': mlg.loss.weighted_MAPE_amp_scaled,
    'weighted_MSPE_amp_scaled': mlg.loss.weighted_MSPE_amp_scaled,
}

DATA_KEYS = ['points', 'Omega_Omegabar', 'mass', 'restriction']


# --- Argument Parsing ---
def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Fit the stages of a multi-stage potential.")

    # Data and output
    parser.add_argument('--fit', type=str, required=True, help='The .npz file of the fit set.')
    parser.add_argument('--holdout', type=str, required=True, help='The .npz file of the hold-out set.')
    parser.add_argument('--validation', type=str, default=None,
                        help='The .npz file of the validation set. Each phase returns the parameters '
                             'with the lowest sigma on it.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Folder for the parameters, the stage files and summary.json.')

    # Stages
    parser.add_argument('--stages', type=int, default=3, help='The last stage to fit.')
    parser.add_argument('--start_stage', type=int, default=1,
                        help='The first stage to fit. The stages before it come from --load_dir.')
    parser.add_argument('--load_dir', type=str, default=None,
                        help='The save_dir of an earlier run with the stages before --start_stage.')
    parser.add_argument('--base', type=str, default='fs', choices=['fs', 'none'],
                        help='Stage 0: the Fubini-Study potential, or nothing (for --network bihom).')
    parser.add_argument('--network', type=str, default='tanh',
                        choices=['tanh', 'tanh_linear', 'siren', 'wanglai', 'poly', 'bihom'],
                        help='The network of the stages to fit.')
    parser.add_argument('--kappa', type=float, default=1.0,
                        help='Scale of the first-layer kernel of --network wanglai: sin(kappa x W + b).')
    parser.add_argument('--layers', type=str, default='64_64_64', help='Hidden widths, e.g. "64_64_64".')
    parser.add_argument('--branches', type=str, default=None,
                        help='Branches of --network poly: widths with "_", branches with "+", e.g. '
                             '"34_34+34_34_34". Default: one branch with --layers.')
    parser.add_argument('--weight', type=str, default='fs', choices=['fs', 'base'],
                        help='Weight of --network poly: 1 (fs), or the base weight of the frozen earlier stages.')
    parser.add_argument('--out_std', type=float, default=None,
                        help='Standard deviation of the initial output weights of tanh, tanh_linear, wanglai '
                             'and poly. Default: Xavier for tanh, 1e-3 for the others.')
    parser.add_argument('--calib_points', type=int, default=10000,
                        help='Number of fit points for the fixed scales and the weight shift of poly.')
    parser.add_argument('--siren_float32', action='store_true',
                        help='Keep the SirenDense parameters of the stages to fit in float32.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed of the initialization and the shuffle of stage 1. Stage j uses seed + j - 1.')
    parser.add_argument('--init_params', type=str, default=None,
                        help='Pickle of the initial parameters of the stage --start_stage, in place of the '
                             'initialization from the seed.')
    parser.add_argument('--amp', type=float, default=None,
                        help='amp of the stage --start_stage. Default: max_abs_error of the potential '
                             'before it on the fit set.')

    # Adam
    parser.add_argument('--adam_loss', type=str, default='weighted_MAPE_amp_scaled', choices=list(LOSSES))
    parser.add_argument('--adam_epochs', type=int, default=2000, help='Maximum number of Adam epochs.')
    parser.add_argument('--adam_patience', type=int, default=50,
                        help='Stop Adam after this number of epochs with no relative decrease of --adam_min_delta.')
    parser.add_argument('--adam_min_delta', type=float, default=0.01,
                        help='Relative decrease on the scale of sigma and rms(f). An MSPE loss uses 2 times it.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Mini-batch size of Adam.')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_search', type=str, default=None,
                        help='Comma list of Adam learning rates. A short run with each picks the one with '
                             'the lowest validation sigma. Needs --validation.')
    parser.add_argument('--lr_search_epochs', type=int, default=50, help='Epochs of each short run.')
    parser.add_argument('--lr_search_only', action='store_true',
                        help='Stop after the learning rate search of the first stage to fit.')
    parser.add_argument('--val_patience', type=int, default=None,
                        help='Also stop Adam after this number of epochs with no relative decrease of '
                             '--val_min_delta in the validation sigma.')
    parser.add_argument('--val_min_delta', type=float, default=0.01)

    # Exact solve of the output layer and LM
    parser.add_argument('--output_solves', type=int, default=0,
                        help='Number of exact solves of the linear output layer after Adam (tanh_linear).')
    parser.add_argument('--lm_steps', type=int, default=0,
                        help='Maximum number of LM steps after the solves. 0 skips LM.')
    parser.add_argument('--lm_patience', type=int, default=None,
                        help='Stop LM after this number of steps with no relative decrease of --lm_min_delta '
                             'in the validation sigma.')
    parser.add_argument('--lm_min_delta', type=float, default=0.001)
    parser.add_argument('--lm_damping', type=float, default=1e-3, help='Initial damping of LM.')
    parser.add_argument('--lm_batch_size', type=int, default=2000,
                        help='Batch size of the sums of J^T J and of the solves.')

    # L-BFGS
    parser.add_argument('--lbfgs_losses', type=str, nargs='+', default=['weighted_MAPE_amp_scaled'],
                        choices=list(LOSSES),
                        help='One L-BFGS run for each loss, from the same Adam result. The first continues the chain.')
    parser.add_argument('--lbfgs_epochs', type=int, default=1000,
                        help='Maximum number of L-BFGS iterations. 0 skips L-BFGS.')
    parser.add_argument('--lbfgs_patience', type=int, default=50,
                        help='Stop L-BFGS after this number of iterations with no relative decrease of --lbfgs_min_delta.')
    parser.add_argument('--lbfgs_min_delta', type=float, default=1e-4,
                        help='Relative decrease on the scale of sigma and rms(f). An MSPE loss uses 2 times it.')
    parser.add_argument('--lbfgs_batch_size', type=int, default=1000,
                        help='Batch size of the accumulated gradients of L-BFGS.')

    parser.add_argument('--eval_batch_size', type=int, default=10000,
                        help='Batch size of the metric in the evaluation.')

    return parser.parse_args()


# --- Data ---
def load_dataset(path: str) -> tuple:
    """Loads a dataset of generate_data.py. The arrays keep their stored dtypes."""
    with np.load(path) as data:
        stored = {k: data[k] for k in DATA_KEYS}
        info = {'path': path, 'n_pairs': int(data['n_pairs']), 'seed': int(data['seed'])}
    dataset = {k: jnp.asarray(v) for k, v in stored.items()}
    info['n_points'] = int(dataset['points'].shape[0])
    info['stored_dtypes'] = {k: str(v.dtype) for k, v in stored.items()}
    return dataset, info


def dtypes_of(dataset: dict) -> dict:
    """The dtype of each array of a dataset."""
    return {k: str(v.dtype) for k, v in dataset.items()}


def print_dtypes(name: str, dataset: dict):
    """Prints the shape and the dtype of each array of a dataset."""
    print(f"Dtypes of the {name}:")
    for key, val in dataset.items():
        print(f"  {key}: shape {val.shape}, dtype {val.dtype}")


# --- Evaluation ---
def total_metric(model, params, dataset: dict, batch_size: int) -> jnp.ndarray:
    """The stored metric of the earlier stages plus the metric of the model."""
    return dataset['cymetric'] + mlg.loss.compute_cy_metric_batched(model, params, dataset, batch_size)


def metric_stats(metric: jnp.ndarray, dataset: dict) -> dict:
    """Sigma (MAPE), rms(f), max abs(f), max_abs_error and the smallest eigenvalue of the metric.

    f = eta - 1, with eta the volume form of the metric over Omega_Omegabar. The
    normalization is the weighted mean over the dataset, as in evaluate_dataset.
    """
    omega = dataset['Omega_Omegabar']
    mass = dataset['mass']
    det_vol = jnp.real(jax.vmap(jnp.linalg.det)(metric))
    weights = mass / jnp.sum(mass)
    factor = jnp.sum(weights * det_vol / omega)
    det_omega = det_vol / factor
    eig_min = jnp.linalg.eigvalsh(metric)[:, 0]
    return {
        'mape': float(mlg.loss.weighted_MAPE(omega, det_omega, mass)),
        'rms_f': float(jnp.sqrt(mlg.loss.weighted_MSPE(omega, det_omega, mass))),
        'max_abs_f': float(mlg.loss.max_error(omega, det_omega, mass)),
        'max_abs_error': float(mlg.loss.max_abs_error(omega, det_omega, mass)),
        'eig_min': float(jnp.min(eig_min)),
        'n_eig_nonpositive': int(jnp.sum(eig_min <= 0)),
        'dtype_metric': str(metric.dtype),
        'dtype_det_omega': str(det_omega.dtype),
    }


def evaluate(metrics: dict, sets: dict, label: str) -> dict:
    """Computes metric_stats on each dataset and prints sigma and rms(f)."""
    result = {name: metric_stats(metrics[name], sets[name]) for name in sets}
    line = ', '.join(f"{name}: sigma {r['mape']:.5e}, rms(f) {r['rms_f']:.5e}, "
                     f"max abs(f) {r['max_abs_f']:.3e}, eig_min {r['eig_min']:.3e}"
                     for name, r in result.items())
    print(f"[{label}] {line}")
    return result


def n_params_of(params) -> int:
    """Number of trainable parameters."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


# --- Models ---
def parse_branches(branches: str, layers: list) -> list:
    """The branches of --network poly as lists of widths."""
    if branches is None:
        return [layers]
    return [[int(u) for u in b.split('_')] for b in branches.split('+')]


def build_model(meta: dict, d: int, phi_prev=None):
    """The model of one stage from its description (see the stage<j>.json files)."""
    network = meta['network']
    if network == 'siren':
        return SpectralSirenNetwork(layers=meta['layers'], amp=meta['amp'], d=d)
    if network == 'tanh_linear':
        return SpectralLinearNetwork(layers=meta['layers'], amp=meta['amp'], out_std=meta['out_std'], d=d)
    if network == 'wanglai':
        return SpectralWangLaiNetwork(layers=meta['layers'], kappa=meta['kappa'], amp=meta['amp'],
                                      out_std=meta['out_std'], d=d)
    if network == 'poly':
        return PolynomialCorrection(branches=meta['branches'], amp=meta['amp'], out_std=meta['out_std'],
                                    scales=meta.get('scales'), phi_prev=phi_prev, k0=meta['k0'],
                                    phi_shift=meta.get('phi_shift', 0.0), d=d)
    if network == 'bihom':
        return Kahler_potential(layers=meta['layers'], d=d)
    if network == 'hn':
        sections, n_sections = load_sections(meta['sections'], meta['degree'])
        return HNPotential(sections=sections, n_sections=n_sections, degree=meta['degree'], k0=meta['k0'])
    return SpectralNetwork(layers=meta['layers'], amp=meta['amp'], out_std=meta.get('out_std'), d=d)


def make_phi_prev(base: str, frozen: list):
    """The frozen potential of the earlier stages as a function of z, for the base weight."""
    # A copy, so that the stages appended later do not enter this potential.
    frozen = list(frozen)

    def phi_prev(z):
        total = 0.0
        if base == 'fs':
            total = jnp.log(jnp.sum(jnp.real(z * jnp.conj(z)), axis=-1))
        for model, params in frozen:
            total = total + model.apply(params, z)[..., 0]
        return total
    return phi_prev


def siren_to_float32(params):
    """Casts the parameters of the SirenDense layers to float32, the dtype of the branch FP64."""
    def cast(path, x):
        names = [str(getattr(k, 'key', k)) for k in path]
        return x.astype(jnp.float32) if any(n.startswith('SirenDense') for n in names) else x
    return jax.tree_util.tree_map_with_path(cast, params)


def stop_delta(loss_name: str, delta: float) -> float:
    """Relative decrease of the stop rule. MSPE is the square of rms(f), so it takes 2 * delta."""
    return 2.0 * delta if 'MSPE' in loss_name else delta


def save_pickle(obj, path: str):
    with open(path, 'wb') as pkl:
        pickle.dump(obj, pkl)


def load_pickle(path: str):
    with open(path, 'rb') as pkl:
        return pickle.load(pkl)


# --- Main ---
def main():
    """Fits the stages and writes the parameters and summary.json."""
    args = parse_args()
    run_start = time.time()

    # The dtypes of the parameters and of all computed arrays come from this call.
    mlg.set_precision(64)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"JAX {jax.__version__}, devices: {jax.devices()}, default float dtype: {jnp.ones(1).dtype}")

    fit_set, fit_info = load_dataset(args.fit)
    holdout_set, holdout_info = load_dataset(args.holdout)
    sets = {'fit': fit_set}
    data_info = {'fit': fit_info}
    if args.validation is not None:
        sets['validation'], data_info['validation'] = load_dataset(args.validation)
    sets['holdout'], data_info['holdout'] = holdout_set, holdout_info
    val_set = sets.get('validation')
    print(', '.join(f"{name}: {info['n_points']} points (seed {info['seed']})" for name, info in data_info.items()))
    if args.lr_search is not None and val_set is None:
        raise ValueError("--lr_search needs --validation.")

    layers = [int(u) for u in args.layers.split('_')]
    branches = parse_branches(args.branches, layers)
    d = int(fit_set['points'].shape[-1])

    # The calibration points of poly: an even stride over the fit set, which is ordered by patch.
    stride = max(1, fit_info['n_points'] // args.calib_points)
    calib_points = fit_set['points'][::stride][:args.calib_points]

    summary = {
        'args': vars(args),
        'jax': jax.__version__,
        'devices': [str(dev) for dev in jax.devices()],
        'data': data_info,
        'stages': [],
    }
    summary_path = os.path.join(args.save_dir, 'summary.json')

    def write_summary():
        summary['wall_time_s'] = time.time() - run_start
        with open(summary_path, 'w') as fp:
            json.dump(summary, fp, indent=1)

    # The base of a loaded run comes from its stage 1.
    base = args.base
    if args.start_stage > 1:
        with open(os.path.join(args.load_dir, 'stage1.json')) as fp:
            base = json.load(fp).get('base', 'fs')

    # --- Stage 0: the Fubini-Study base log |z|^2, or nothing ---
    fs_model = zerolayer(n_units=[])
    fs_params = mlg.trainer.init_params(fs_model, (d,))
    metrics = {name: mlg.loss.compute_cy_metric_batched(fs_model, fs_params, ds, args.eval_batch_size)
               for name, ds in sets.items()}
    for name, ds in sets.items():
        ds['cymetric'] = metrics[name] if base == 'fs' else jnp.zeros_like(metrics[name])
        print_dtypes(f"{name} set with the stored metric of stage 0", ds)
    summary['dtypes'] = {name: dtypes_of(ds) for name, ds in sets.items()}
    # With --base none, the FS numbers serve only as a reference and as the loss scale of stage 1.
    summary['stages'].append({'stage': 0, 'model': 'Fubini-Study log |z|^2', 'base': base, 'n_params': 0,
                              'eval': evaluate(metrics, sets, 'stage 0, FS')})
    write_summary()

    # --- The stages before start_stage, from load_dir ---
    frozen = []
    k0 = 1
    for j in range(1, args.start_stage):
        with open(os.path.join(args.load_dir, f'stage{j}.json')) as fp:
            meta = json.load(fp)
        meta.setdefault('network', 'tanh')
        k0 = meta.get('k0', 1)
        phi_prev = make_phi_prev(base, frozen) if meta.get('weight') == 'base' else None
        model = build_model(meta, d, phi_prev)
        params = load_pickle(os.path.join(args.load_dir, meta['chain_params']))
        metrics = {name: total_metric(model, params, ds, args.eval_batch_size) for name, ds in sets.items()}
        for name, ds in sets.items():
            ds['cymetric'] = metrics[name]
        frozen.append((model, params))
        summary['stages'].append({'stage': j, 'loaded_from': os.path.join(args.load_dir, meta['chain_params']),
                                  'network': meta['network'], 'amp': meta['amp'], 'n_params': n_params_of(params),
                                  'eval': evaluate(metrics, sets, f'stage {j}, loaded')})
        write_summary()

    # --- The stages to fit ---
    for j in range(args.start_stage, args.stages + 1):
        stage_seed = args.seed + j - 1

        # amp of this stage: the largest absolute error of the volume form before it, on the fit set.
        amp = summary['stages'][-1]['eval']['fit']['max_abs_error']
        if j == args.start_stage and args.amp is not None:
            amp = args.amp

        # The class of the potential: k0 = 1 on the FS base, 2^len(layers) for a bihomogeneous network.
        if args.network == 'bihom':
            if base != 'none' or j != 1:
                raise ValueError("--network bihom is a full potential: use it at stage 1 with --base none.")
            k0 = 2 ** len(layers)

        meta = {'stage': j, 'network': args.network, 'base': base, 'layers': layers, 'amp': amp, 'k0': k0,
                'siren_float32': args.siren_float32}
        phi_prev = None
        if args.network == 'poly':
            meta.update({'branches': branches, 'weight': args.weight,
                         'out_std': 1e-3 if args.out_std is None else args.out_std})
            if args.weight == 'base':
                phi_prev = make_phi_prev(base, frozen)
                meta['phi_shift'] = reduced_potential_mean(phi_prev, k0, calib_points)
        elif args.network == 'tanh':
            meta['out_std'] = args.out_std
        elif args.network == 'wanglai':
            meta.update({'kappa': args.kappa, 'out_std': 1e-3 if args.out_std is None else args.out_std})
        elif args.network == 'tanh_linear':
            meta['out_std'] = 1e-3 if args.out_std is None else args.out_std

        # The initial parameters come from the stage seed. poly then gets its fixed layer scales.
        model = build_model(meta, d, phi_prev)
        params_init = mlg.trainer.init_params(model, (d,), stage_seed)
        if args.network == 'poly':
            meta['scales'], raw_rms = calibrate_polynomial(model, params_init, calib_points)
            model = build_model(meta, d, phi_prev)
            meta['raw_rms'] = raw_rms
            print(f"poly: raw RMS of the x^2 layers {raw_rms}, fixed scales {meta['scales']}")
        if args.siren_float32:
            params_init = siren_to_float32(params_init)
        if args.init_params is not None and j == args.start_stage:
            loaded = load_pickle(args.init_params)
            if jax.tree_util.tree_structure(loaded) != jax.tree_util.tree_structure(params_init):
                raise ValueError(f"{args.init_params} does not match the parameters of the model.")
            params_init = loaded
            meta['init_params'] = args.init_params

        stage = dict(meta, seed=stage_seed, n_params=n_params_of(params_init),
                     n_points_fit=fit_info['n_points'], n_points_holdout=holdout_info['n_points'])
        summary['stages'].append(stage)
        print(f"\n--- Stage {j}: {args.network}, amp = {amp:.6e}, seed = {stage_seed}, "
              f"{stage['n_params']} parameters ---")
        adam_metric = functools.partial(LOSSES[args.adam_loss], amp=amp)

        # The learning rate of Adam: the lowest validation sigma of short runs from the same start.
        learning_rate = args.learning_rate
        if args.lr_search is not None:
            stage['lr_search'] = []
            for lr in [float(u) for u in args.lr_search.split(',')]:
                val_record = []
                start = time.time()
                _, best_val = mlg.trainer.train_stage_optax(
                    model=model, dataset=fit_set, optimizer=optax.adam(lr), epochs=args.lr_search_epochs,
                    batch_size=args.batch_size, loss_metric=adam_metric, params=params_init,
                    val_dataset=val_set, val_batch_size=args.eval_batch_size, val_record=val_record,
                    seed=stage_seed, verbose=False)
                stage['lr_search'].append({'lr': lr, 'best_val_sigma': best_val, 'time_s': time.time() - start,
                                           'val_record': val_record})
                print(f"lr search: lr {lr:g}: best validation sigma {best_val:.5e} "
                      f"in {args.lr_search_epochs} epochs ({time.time() - start:.0f}s)")
            learning_rate = min(stage['lr_search'], key=lambda r: r['best_val_sigma'])['lr']
            print(f"lr search: chose lr {learning_rate:g}")
            write_summary()
            if args.lr_search_only:
                print(f"\nRun finished in {time.time() - run_start:.1f}s. Summary: {summary_path}")
                return
        stage['learning_rate'] = learning_rate

        # Adam on mini-batches.
        adam_record, adam_val_record = [], []
        start = time.time()
        params_adam, adam_loss = mlg.trainer.train_stage_optax(
            model=model,
            dataset=fit_set,
            optimizer=optax.adam(learning_rate),
            epochs=args.adam_epochs,
            batch_size=args.batch_size,
            loss_metric=adam_metric,
            params=params_init,
            patience=args.adam_patience,
            min_delta=stop_delta(args.adam_loss, args.adam_min_delta),
            loss_record=adam_record,
            val_dataset=val_set,
            val_batch_size=args.eval_batch_size,
            val_patience=args.val_patience,
            val_min_delta=args.val_min_delta,
            val_record=adam_val_record,
            seed=stage_seed,
            verbose=True,
        )
        adam_time = time.time() - start
        stage['dtypes_params'] = sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(params_adam)})
        save_pickle(params_adam, os.path.join(args.save_dir, f'stage{j}_adam.pkl'))
        metrics_adam = {name: total_metric(model, params_adam, ds, args.eval_batch_size)
                        for name, ds in sets.items()}
        stage['adam'] = {'loss': args.adam_loss, 'epochs': len(adam_record), 'time_s': adam_time,
                         'stopped_early': len(adam_record) < args.adam_epochs,
                         'min_delta': stop_delta(args.adam_loss, args.adam_min_delta),
                         'final_loss': adam_loss, 'loss_record': adam_record, 'val_record': adam_val_record,
                         'eval': evaluate(metrics_adam, sets, f'stage {j}, Adam')}
        print(f"Stage {j}: {stage['n_params']} parameters, dtypes {stage['dtypes_params']}")
        write_summary()

        chain_params, chain_metrics, chain_file = params_adam, metrics_adam, f'stage{j}_adam.pkl'

        # The exact solves of the linear output layer, with the hidden layers fixed.
        if args.output_solves > 0:
            print(f"\n--- Stage {j}: {args.output_solves} exact solves of the output layer ---")
            solve_record = []
            start = time.time()
            chain_params, _ = mlg.trainer.solve_stage_output(
                model=model, dataset=fit_set, params=chain_params, solves=args.output_solves,
                batch_size=args.lm_batch_size, val_dataset=val_set, val_batch_size=args.eval_batch_size,
                step_record=solve_record, verbose=True)
            chain_file = f'stage{j}_solve.pkl'
            save_pickle(chain_params, os.path.join(args.save_dir, chain_file))
            chain_metrics = {name: total_metric(model, chain_params, ds, args.eval_batch_size)
                             for name, ds in sets.items()}
            stage['solve'] = {'solves': args.output_solves, 'time_s': time.time() - start, 'record': solve_record,
                              'eval': evaluate(chain_metrics, sets, f'stage {j}, exact solve')}
            write_summary()

        # LM on all parameters, full batch on MSPE.
        if args.lm_steps > 0:
            print(f"\n--- Stage {j}: LM ---")
            lm_record = []
            start = time.time()
            chain_params, _ = mlg.trainer.train_stage_lm(
                model=model, dataset=fit_set, epochs=args.lm_steps, params=chain_params,
                batch_size=args.lm_batch_size, damping=args.lm_damping, patience=args.lm_patience,
                min_delta=args.lm_min_delta, val_dataset=val_set, val_batch_size=args.eval_batch_size,
                step_record=lm_record, verbose=True)
            chain_file = f'stage{j}_lm.pkl'
            save_pickle(chain_params, os.path.join(args.save_dir, chain_file))
            chain_metrics = {name: total_metric(model, chain_params, ds, args.eval_batch_size)
                             for name, ds in sets.items()}
            stage['lm'] = {'steps': len(lm_record), 'time_s': time.time() - start, 'record': lm_record,
                           'eval': evaluate(chain_metrics, sets, f'stage {j}, LM')}
            write_summary()

        # L-BFGS from the result of the last phase, once for each loss.
        params_start = chain_params
        stage['lbfgs'] = []
        lbfgs_losses = args.lbfgs_losses if args.lbfgs_epochs > 0 else []
        for k, loss_name in enumerate(lbfgs_losses):
            print(f"\n--- Stage {j}: L-BFGS on {loss_name} ---")
            lbfgs_record, lbfgs_val_record = [], []
            start = time.time()
            params_lbfgs, lbfgs_loss = mlg.trainer.train_stage_lbfgs(
                model=model,
                dataset=fit_set,
                epochs=args.lbfgs_epochs,
                loss_metric=functools.partial(LOSSES[loss_name], amp=amp),
                params=params_start,
                batch_size=args.lbfgs_batch_size,
                patience=args.lbfgs_patience,
                min_delta=stop_delta(loss_name, args.lbfgs_min_delta),
                loss_record=lbfgs_record,
                val_dataset=val_set,
                val_batch_size=args.eval_batch_size,
                val_record=lbfgs_val_record,
                seed=stage_seed,
                verbose=True,
            )
            lbfgs_time = time.time() - start
            file_name = f'stage{j}_lbfgs_{loss_name}.pkl'
            save_pickle(params_lbfgs, os.path.join(args.save_dir, file_name))
            metrics_lbfgs = {name: total_metric(model, params_lbfgs, ds, args.eval_batch_size)
                             for name, ds in sets.items()}
            stage['lbfgs'].append({'loss': loss_name, 'iterations': len(lbfgs_record),
                                   'time_s': lbfgs_time, 'final_loss': lbfgs_loss,
                                   'stopped_early': len(lbfgs_record) < args.lbfgs_epochs,
                                   'min_delta': stop_delta(loss_name, args.lbfgs_min_delta),
                                   'dtypes_params': sorted({str(x.dtype) for x in
                                                            jax.tree_util.tree_leaves(params_lbfgs)}),
                                   'loss_record': lbfgs_record, 'val_record': lbfgs_val_record,
                                   'eval': evaluate(metrics_lbfgs, sets, f'stage {j}, L-BFGS {loss_name}')})
            if k == 0:
                chain_params, chain_metrics, chain_file = params_lbfgs, metrics_lbfgs, file_name
            write_summary()

        # The loss curves of the stage: Adam by epoch, and each L-BFGS run by iteration.
        curves = {'adam_' + args.adam_loss: np.asarray(stage['adam']['loss_record']),
                  'adam_val_sigma': np.asarray(stage['adam']['val_record'])}
        for run in stage['lbfgs']:
            curves['lbfgs_' + run['loss']] = np.asarray(run['loss_record'])
            curves['lbfgs_val_sigma_' + run['loss']] = np.asarray(run['val_record'])
        np.savez(os.path.join(args.save_dir, f'stage{j}_curves.npz'), amp=amp, **curves)

        # The chain continues with the first L-BFGS result.
        for name, ds in sets.items():
            ds['cymetric'] = chain_metrics[name]
        frozen.append((model, chain_params))
        print_dtypes(f"fit set with the stored metric of stages 0 to {j}", fit_set)
        stage['dtypes_stored_metric'] = {name: str(ds['cymetric'].dtype) for name, ds in sets.items()}
        last_phase = next(stage[k] for k in ('lm', 'solve', 'adam') if k in stage)
        stage['eval'] = stage['lbfgs'][0]['eval'] if stage['lbfgs'] else last_phase['eval']
        stage['chain_params'] = chain_file
        meta['chain_params'] = chain_file
        with open(os.path.join(args.save_dir, f'stage{j}.json'), 'w') as fp:
            json.dump(meta, fp, indent=1)
        write_summary()

    print(f"\nRun finished in {time.time() - run_start:.1f}s. Summary: {summary_path}")


if __name__ == '__main__':
    main()
