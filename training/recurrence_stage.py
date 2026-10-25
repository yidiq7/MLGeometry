"""
Fits a stage of the units of a learned recurrence on stored datasets, after the stages of an earlier run.

The stage is UnitRecurrence (training/models.py) on the features of degree 0 of
recurrence_features: z_a z_b_bar times the base weight exp(-Phi_prev / k0) of the frozen
potential (--features base), or times 1 / |z|^2 (--features fs). calibrate_recurrence draws
random matrices A and B with the seed --draw_seed and orthonormalizes the units on the fit
points. Then solve_stage_units solves the coefficients c exactly. With --lm_steps,
train_stage_varpro then fits A and B by Levenberg-Marquardt (LM), and computes the
orthonormalization and c again after each accepted step. The validation set selects only
the parameters of LM; the hold-out set gives the final numbers.

Example (from the root folder of the repository):
    PYTHONPATH=src python training/recurrence_stage.py --fit data/fit.npz --validation data/val.npz \
        --holdout data/holdout.npz --load_dir runs/stage1 --save_dir runs/rec128 --width 128
"""

import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import unfreeze

import MLGeometry as mlg
from models import (UnitRecurrence, recurrence_features, calibrate_recurrence, recurrence_gram_error,
                    reduced_potential_mean, zerolayer)
from spectral_stages import (load_dataset, print_dtypes, dtypes_of, evaluate, n_params_of, build_model,
                             make_phi_prev, load_pickle, save_pickle)


# --- Argument Parsing ---
def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Fit a stage of the units of a learned recurrence.")
    parser.add_argument('--fit', type=str, required=True, help='The .npz file of the fit set.')
    parser.add_argument('--validation', type=str, required=True, help='The .npz file of the validation set.')
    parser.add_argument('--holdout', type=str, required=True, help='The .npz file of the hold-out set.')
    parser.add_argument('--load_dir', type=str, required=True,
                        help='The save_dir of an earlier run of spectral_stages.py with the frozen stages.')
    parser.add_argument('--save_dir', type=str, required=True, help='Folder for the parameters and summary.json.')
    parser.add_argument('--width', type=int, required=True, help='Number of products of each level.')
    parser.add_argument('--degree', type=int, default=16, help='Number of levels, the top degree L.')
    parser.add_argument('--features', type=str, default='base', choices=['base', 'fs'],
                        help='Weight of the features: the base weight of the frozen stages, or 1 / |z|^2.')
    parser.add_argument('--seed', type=int, default=42, help='The new stage j uses the stage seed seed + j - 1.')
    parser.add_argument('--draw_seed', type=int, default=None,
                        help='Seed of the random matrices A and B. Default: the stage seed.')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Smallest new part of a product, relative to its norm, that enters as a unit.')
    parser.add_argument('--solves', type=int, default=3, help='Number of exact solves of c.')
    parser.add_argument('--lm_steps', type=int, default=0, help='Maximum number of LM steps on A and B. 0 skips LM.')
    parser.add_argument('--lm_patience', type=int, default=None,
                        help='Stop LM after this number of steps with no relative decrease of --lm_min_delta '
                             'in the validation sigma.')
    parser.add_argument('--lm_min_delta', type=float, default=0.001)
    parser.add_argument('--lm_damping', type=float, default=1e-3, help='Initial damping of LM.')
    parser.add_argument('--varpro_solves', type=int, default=2, help='Solves of c after each accepted LM step.')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size of the metric, of the sums of the solves and of LM.')
    parser.add_argument('--calib_points', type=int, default=10000,
                        help='Number of fit points for the shift of the base weight.')
    return parser.parse_args()


# --- Evaluation ---
def stage_metric(model, params, dataset: dict, batch_size: int) -> jnp.ndarray:
    """The stored metric of the earlier stages plus the metric of the model, in batches."""
    step = jax.jit(lambda p, b: mlg.loss.compute_cy_metric(model, p, b))
    keys = ('points', 'restriction')
    parts = [step(params, {k: dataset[k][i:i + batch_size] for k in keys})
             for i in range(0, dataset['points'].shape[0], batch_size)]
    return dataset['cymetric'] + jnp.concatenate(parts, axis=0)


def check_structure(model, variables: dict, d: int):
    """Raises an error if the variables do not have the structure and the shapes of model.init."""
    dummy = jnp.ones((1, d), dtype=mlg.config.complex_dtype)
    expected = unfreeze(jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), dummy)))
    same = (jax.tree_util.tree_structure(expected) == jax.tree_util.tree_structure(variables)
            and all(e.shape == v.shape for e, v in zip(jax.tree_util.tree_leaves(expected),
                                                       jax.tree_util.tree_leaves(variables))))
    if not same:
        raise ValueError("The calibrated variables do not match the parameters of UnitRecurrence.")


# --- Main ---
def main():
    """Fits the stage and writes the parameters and summary.json."""
    args = parse_args()
    run_start = time.time()
    mlg.set_precision(64)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"JAX {jax.__version__}, devices: {jax.devices()}, default float dtype: {jnp.ones(1).dtype}")

    sets, data_info = {}, {}
    for name, path in [('fit', args.fit), ('validation', args.validation), ('holdout', args.holdout)]:
        sets[name], data_info[name] = load_dataset(path)
    fit_set, val_set, holdout_set = sets['fit'], sets['validation'], sets['holdout']
    print(', '.join(f"{name}: {info['n_points']} points (seed {info['seed']})" for name, info in data_info.items()))
    d = int(fit_set['points'].shape[-1])

    summary = {'args': vars(args), 'jax': jax.__version__, 'devices': [str(dev) for dev in jax.devices()],
               'data': data_info, 'stages': []}
    summary_path = os.path.join(args.save_dir, 'summary.json')

    def write_summary():
        summary['wall_time_s'] = time.time() - run_start
        with open(summary_path, 'w') as fp:
            json.dump(summary, fp, indent=1)

    # --- Stage 0 and the frozen stages of load_dir ---
    with open(os.path.join(args.load_dir, 'stage1.json')) as fp:
        base = json.load(fp).get('base', 'fs')
    fs_model = zerolayer(n_units=[])
    fs_params = mlg.trainer.init_params(fs_model, (d,))
    for name, ds in sets.items():
        metric = mlg.loss.compute_cy_metric_batched(fs_model, fs_params, ds, 10000)
        ds['cymetric'] = metric if base == 'fs' else jnp.zeros_like(metric)
    frozen, k0, j = [], 1, 1
    while os.path.exists(os.path.join(args.load_dir, f'stage{j}.json')):
        with open(os.path.join(args.load_dir, f'stage{j}.json')) as fp:
            meta = json.load(fp)
        k0 = meta.get('k0', 1)
        phi_prev = make_phi_prev(base, frozen) if meta.get('weight') == 'base' else None
        model = build_model(meta, d, phi_prev)
        params = load_pickle(os.path.join(args.load_dir, meta['chain_params']))
        metrics = {name: ds['cymetric'] + mlg.loss.compute_cy_metric_batched(model, params, ds, 10000)
                   for name, ds in sets.items()}
        for name, ds in sets.items():
            ds['cymetric'] = metrics[name]
        frozen.append((model, params))
        summary['stages'].append({'stage': j, 'loaded_from': os.path.join(args.load_dir, meta['chain_params']),
                                  'network': meta['network'], 'k0': k0, 'n_params': n_params_of(params),
                                  'eval': evaluate(metrics, sets, f'stage {j}, loaded')})
        j += 1
    print_dtypes("fit set with the stored metric of the frozen stages", fit_set)
    summary['dtypes'] = {name: dtypes_of(ds) for name, ds in sets.items()}
    write_summary()

    # --- The new stage ---
    stage_seed = args.seed + j - 1
    draw_seed = stage_seed if args.draw_seed is None else args.draw_seed
    phi_prev = make_phi_prev(base, frozen) if args.features == 'base' else None
    stride = max(1, fit_set['points'].shape[0] // args.calib_points)
    phi_shift = (reduced_potential_mean(phi_prev, k0, fit_set['points'][::stride][:args.calib_points])
                 if phi_prev is not None else 0.0)
    feature_fn = jax.jit(lambda z: recurrence_features(z, phi_prev, k0, phi_shift))
    x_fit = jnp.concatenate([feature_fn(fit_set['points'][i:i + 10000])
                             for i in range(0, fit_set['points'].shape[0], 10000)], axis=0)
    weights_fit = fit_set['mass'] / jnp.sum(fit_set['mass'])
    print(f"\n--- Stage {j}: recurrence, features {args.features}, width {args.width}, degree {args.degree}, "
          f"k0 {k0}, shift {phi_shift:.6f}, draw seed {draw_seed} ---")

    start = time.time()
    cal = calibrate_recurrence(x_fit, weights_fit, args.width, args.degree, key=jax.random.PRNGKey(draw_seed),
                               tol=args.tol)
    time_cal = time.time() - start
    model = UnitRecurrence(widths=cal['widths'], n_features=d * d, phi_prev=phi_prev, k0=k0, phi_shift=phi_shift)
    variables = cal['variables']
    check_structure(model, variables, d)
    n_units = int(sum(cal['widths']))
    n_hidden = int(sum(variables['params'][k].size for k in variables['params'] if k != 'c'))
    levels = [{'level': l + 1, 'units': len(parts), 'median': float(np.median(parts)), 'min': float(np.min(parts)),
               'max': float(np.max(parts))} for l, parts in enumerate(cal['new_parts'])]
    print(f"Calibration in {time_cal:.1f}s: units of the levels {cal['widths']}, {n_units} units")
    for entry in levels:
        print(f"  level {entry['level']}: {entry['units']} units, new part median {entry['median']:.3e}, "
              f"min {entry['min']:.3e}, max {entry['max']:.3e}")
    start = time.time()
    orthogonality = {name: recurrence_gram_error(model, variables, sets[name]['points'], sets[name]['mass'],
                                                 batch_size=5000) for name in ('fit', 'holdout')}
    print(f"Orthogonality error max abs(U^T W U - I): fit {orthogonality['fit']:.3e}, "
          f"hold-out {orthogonality['holdout']:.3e} ({time.time() - start:.1f}s)")
    stage = {'stage': j, 'network': 'recurrence', 'features': args.features, 'width': args.width,
             'degree': args.degree, 'widths': cal['widths'], 'n_units': n_units, 'n_hidden': n_hidden,
             'k0': k0, 'phi_shift': phi_shift, 'seed': stage_seed, 'draw_seed': draw_seed, 'tol': args.tol,
             'n_points_fit': data_info['fit']['n_points'],
             'calibration': {'time_s': time_cal, 'levels': levels,
                             'new_parts': [p.tolist() for p in cal['new_parts']]},
             'orthogonality': orthogonality}
    summary['stages'].append(stage)
    write_summary()

    # Exact solves of c.
    print(f"\n--- Stage {j}: {args.solves} exact solves of c ---")
    solve_record = []
    start = time.time()
    variables, _ = mlg.trainer.solve_stage_units(
        model=model, dataset=fit_set, params=variables, solves=args.solves, batch_size=args.batch_size,
        val_dataset=val_set, val_batch_size=args.batch_size, step_record=solve_record, verbose=True)
    time_solve = time.time() - start
    save_pickle(variables, os.path.join(args.save_dir, f'stage{j}_solve.pkl'))
    start = time.time()
    metrics = {name: stage_metric(model, variables, ds, args.batch_size) for name, ds in sets.items()}
    stage['solve'] = {'solves': args.solves, 'time_s': time_solve, 'record': solve_record,
                      'eval': evaluate(metrics, sets, f'stage {j}, exact solve'), 'time_eval_s': time.time() - start}
    stage['n_params'] = n_units
    stage['eval'] = stage['solve']['eval']
    write_summary()

    # LM on A and B, with the orthonormalization and c again after each accepted step.
    if args.lm_steps > 0:
        print(f"\n--- Stage {j}: LM on A and B with variable projection ---")

        def recalibrate(p):
            a_list = [p['params'][f'A_{l}'] for l in range(len(cal['widths']))]
            b_list = [p['params'][f'B_{l}'] for l in range(len(cal['widths']))]
            new = calibrate_recurrence(x_fit, weights_fit, args.width, len(cal['widths']), a_list=a_list,
                                       b_list=b_list)
            return {**p, 'orth': new['variables']['orth']}

        lm_record = []
        start = time.time()
        variables, _ = mlg.trainer.train_stage_varpro(
            model=model, dataset=fit_set, epochs=args.lm_steps, params=variables, recalibrate=recalibrate,
            solves=args.varpro_solves, batch_size=args.batch_size, damping=args.lm_damping,
            patience=args.lm_patience, min_delta=args.lm_min_delta, val_dataset=val_set,
            val_batch_size=args.batch_size, step_record=lm_record, verbose=True)
        time_lm = time.time() - start
        save_pickle(variables, os.path.join(args.save_dir, f'stage{j}_lm.pkl'))
        metrics = {name: stage_metric(model, variables, ds, args.batch_size) for name, ds in sets.items()}
        stage['lm'] = {'steps': len(lm_record), 'time_s': time_lm, 'record': lm_record,
                       'eval': evaluate(metrics, sets, f'stage {j}, LM'),
                       'orthogonality': {name: recurrence_gram_error(model, variables, sets[name]['points'],
                                                                     sets[name]['mass'], batch_size=5000)
                                         for name in ('fit', 'holdout')}}
        stage['n_params'] = n_units + n_hidden
        stage['eval'] = stage['lm']['eval']
        write_summary()

    print(f"\nRun finished in {time.time() - run_start:.1f}s. Summary: {summary_path}")
    write_summary()


if __name__ == '__main__':
    main()
