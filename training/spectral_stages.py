"""
Fits the stages of the spectral scheme one after the other on stored datasets.

The base (stage 0) is the Fubini-Study potential log |z|^2. Each stage j adds a
SpectralNetwork with the output amp_j * sinh(x). amp_j is the largest absolute error
of the volume form of the potential before stage j on the fit set (max_abs_error).
A stage runs Adam on mini-batches, and then L-BFGS with accumulated gradients. The
restricted metric of the earlier stages is stored at each point (dataset['cymetric']),
so a stage differentiates only its own network.

Each loss in --lbfgs_losses gives one L-BFGS run from the same Adam result. The first
one continues the chain to the next stage.

--network siren puts SirenDense hidden layers in place of the tanh layers. With
--siren_float32, the SirenDense parameters of the stages to fit are float32, as the
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
from models import SpectralNetwork, SpectralSirenNetwork, zerolayer


LOSSES = {
    'weighted_MAPE_amp_scaled': mlg.loss.weighted_MAPE_amp_scaled,
    'weighted_MSPE_amp_scaled': mlg.loss.weighted_MSPE_amp_scaled,
}

DATA_KEYS = ['points', 'Omega_Omegabar', 'mass', 'restriction']


# --- Argument Parsing ---
def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Fit the stages of the spectral scheme.")

    # Data and output
    parser.add_argument('--fit', type=str, required=True, help='The .npz file of the fit set.')
    parser.add_argument('--holdout', type=str, required=True, help='The .npz file of the hold-out set.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Folder for the parameters, the stage files and summary.json.')

    # Stages
    parser.add_argument('--stages', type=int, default=3, help='The last stage to fit.')
    parser.add_argument('--start_stage', type=int, default=1,
                        help='The first stage to fit. The stages before it come from --load_dir.')
    parser.add_argument('--load_dir', type=str, default=None,
                        help='The save_dir of an earlier run with the stages before --start_stage.')
    parser.add_argument('--layers', type=str, default='64_64_64', help='Hidden widths, e.g. "64_64_64".')
    parser.add_argument('--network', type=str, default='tanh', choices=['tanh', 'siren'],
                        help='Hidden layers of the stages to fit: tanh Dense or SirenDense.')
    parser.add_argument('--siren_float32', action='store_true',
                        help='Keep the SirenDense parameters of the stages to fit in float32.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed of the initialization and the shuffle of stage 1. Stage j uses seed + j - 1.')
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


def build_model(network: str, layers, amp: float, d: int):
    """The correction model of one stage: SpectralNetwork (tanh) or SpectralSirenNetwork."""
    if network == 'siren':
        return SpectralSirenNetwork(layers=layers, amp=amp, d=d)
    return SpectralNetwork(layers=layers, amp=amp, d=d)


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
    sets = {'fit': fit_set, 'holdout': holdout_set}
    print(f"Fit set: {fit_info['n_points']} points (seed {fit_info['seed']}), "
          f"hold-out: {holdout_info['n_points']} points (seed {holdout_info['seed']})")

    layers = [int(u) for u in args.layers.split('_')]
    d = int(fit_set['points'].shape[-1])

    summary = {
        'args': vars(args),
        'jax': jax.__version__,
        'devices': [str(dev) for dev in jax.devices()],
        'data': {'fit': fit_info, 'holdout': holdout_info},
        'stages': [],
    }
    summary_path = os.path.join(args.save_dir, 'summary.json')

    def write_summary():
        summary['wall_time_s'] = time.time() - run_start
        with open(summary_path, 'w') as fp:
            json.dump(summary, fp, indent=1)

    # --- Stage 0: the Fubini-Study base log |z|^2 ---
    base = zerolayer(n_units=[])
    base_params = mlg.trainer.init_params(base, (d,))
    metrics = {name: mlg.loss.compute_cy_metric_batched(base, base_params, ds, args.eval_batch_size)
               for name, ds in sets.items()}
    for name, ds in sets.items():
        ds['cymetric'] = metrics[name]
        print_dtypes(f"{name} set with the stored metric of stage 0", ds)
    summary['dtypes'] = {name: dtypes_of(ds) for name, ds in sets.items()}
    summary['stages'].append({'stage': 0, 'model': 'Fubini-Study log |z|^2', 'n_params': 0,
                              'eval': evaluate(metrics, sets, 'stage 0, FS')})
    write_summary()

    # --- The stages before start_stage, from load_dir ---
    for j in range(1, args.start_stage):
        with open(os.path.join(args.load_dir, f'stage{j}.json')) as fp:
            meta = json.load(fp)
        model = build_model(meta.get('network', 'tanh'), meta['layers'], meta['amp'], d)
        params = load_pickle(os.path.join(args.load_dir, meta['chain_params']))
        metrics = {name: total_metric(model, params, ds, args.eval_batch_size) for name, ds in sets.items()}
        for name, ds in sets.items():
            ds['cymetric'] = metrics[name]
        summary['stages'].append({'stage': j, 'loaded_from': os.path.join(args.load_dir, meta['chain_params']),
                                  'amp': meta['amp'], 'n_params': n_params_of(params),
                                  'eval': evaluate(metrics, sets, f'stage {j}, loaded')})
        write_summary()

    # --- The stages to fit ---
    for j in range(args.start_stage, args.stages + 1):
        stage_seed = args.seed + j - 1

        # amp of this stage: the largest absolute error of the volume form before it, on the fit set.
        amp = summary['stages'][-1]['eval']['fit']['max_abs_error']
        if j == args.start_stage and args.amp is not None:
            amp = args.amp
        model = build_model(args.network, layers, amp, d)
        stage = {'stage': j, 'seed': stage_seed, 'amp': amp, 'layers': layers, 'network': args.network,
                 'siren_float32': args.siren_float32,
                 'n_points_fit': fit_info['n_points'], 'n_points_holdout': holdout_info['n_points']}
        summary['stages'].append(stage)
        print(f"\n--- Stage {j}: amp = {amp:.6e}, seed = {stage_seed} ---")

        # The initial parameters come from the stage seed. The float32 option rounds the SirenDense ones.
        params_init = mlg.trainer.init_params(model, (d,), stage_seed)
        if args.siren_float32:
            params_init = siren_to_float32(params_init)

        # Adam on mini-batches.
        adam_record, adam_history = [], []
        start = time.time()
        params_adam, adam_loss = mlg.trainer.train_stage_optax(
            model=model,
            dataset=fit_set,
            optimizer=optax.adam(args.learning_rate),
            epochs=args.adam_epochs,
            batch_size=args.batch_size,
            loss_metric=functools.partial(LOSSES[args.adam_loss], amp=amp),
            params=params_init,
            patience=args.adam_patience,
            min_delta=stop_delta(args.adam_loss, args.adam_min_delta),
            loss_record=adam_record,
            seed=stage_seed,
            verbose=True,
            history=adam_history,
        )
        adam_time = time.time() - start
        stage['n_params'] = n_params_of(params_adam)
        stage['dtypes_params'] = sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(params_adam)})
        save_pickle(params_adam, os.path.join(args.save_dir, f'stage{j}_adam.pkl'))
        metrics_adam = {name: total_metric(model, params_adam, ds, args.eval_batch_size)
                        for name, ds in sets.items()}
        stage['adam'] = {'loss': args.adam_loss, 'epochs': len(adam_record), 'time_s': adam_time,
                         'stopped_early': len(adam_record) < args.adam_epochs,
                         'min_delta': stop_delta(args.adam_loss, args.adam_min_delta),
                         'final_loss': adam_loss, 'loss_record': adam_record,
                         'eval': evaluate(metrics_adam, sets, f'stage {j}, Adam')}
        print(f"Stage {j}: {stage['n_params']} parameters, dtypes {stage['dtypes_params']}")
        write_summary()

        # L-BFGS from the same Adam result, once for each loss.
        chain_metrics, chain_file = metrics_adam, f'stage{j}_adam.pkl'
        stage['lbfgs'] = []
        lbfgs_losses = args.lbfgs_losses if args.lbfgs_epochs > 0 else []
        for k, loss_name in enumerate(lbfgs_losses):
            print(f"\n--- Stage {j}: L-BFGS on {loss_name} ---")
            lbfgs_record, lbfgs_history = [], []
            start = time.time()
            params_lbfgs, lbfgs_loss = mlg.trainer.train_stage_lbfgs(
                model=model,
                dataset=fit_set,
                epochs=args.lbfgs_epochs,
                loss_metric=functools.partial(LOSSES[loss_name], amp=amp),
                params=params_adam,
                batch_size=args.lbfgs_batch_size,
                patience=args.lbfgs_patience,
                min_delta=stop_delta(loss_name, args.lbfgs_min_delta),
                loss_record=lbfgs_record,
                seed=stage_seed,
                verbose=True,
                history=lbfgs_history,
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
                                   'loss_record': lbfgs_record,
                                   'eval': evaluate(metrics_lbfgs, sets, f'stage {j}, L-BFGS {loss_name}')})
            if k == 0:
                chain_metrics, chain_file = metrics_lbfgs, file_name
            write_summary()

        # The loss curves of the stage: Adam by epoch, and each L-BFGS run by iteration.
        curves = {'adam_' + args.adam_loss: np.asarray(stage['adam']['loss_record'])}
        for run in stage['lbfgs']:
            curves['lbfgs_' + run['loss']] = np.asarray(run['loss_record'])
        np.savez(os.path.join(args.save_dir, f'stage{j}_curves.npz'), amp=amp, **curves)

        # The chain continues with the first L-BFGS result.
        for name, ds in sets.items():
            ds['cymetric'] = chain_metrics[name]
        print_dtypes(f"fit set with the stored metric of stages 0 to {j}", fit_set)
        stage['dtypes_stored_metric'] = {name: str(ds['cymetric'].dtype) for name, ds in sets.items()}
        stage['eval'] = stage['lbfgs'][0]['eval'] if stage['lbfgs'] else stage['adam']['eval']
        stage['chain_params'] = chain_file
        with open(os.path.join(args.save_dir, f'stage{j}.json'), 'w') as fp:
            json.dump({'stage': j, 'layers': layers, 'amp': amp, 'network': args.network,
                       'siren_float32': args.siren_float32, 'chain_params': chain_file}, fp, indent=1)
        write_summary()

    print(f"\nRun finished in {time.time() - run_start:.1f}s. Summary: {summary_path}")


if __name__ == '__main__':
    main()
