"""
Fits potentials on the full bases of sections of low degree, with no symmetry.

--task sections
    Evaluates all monomials of each degree k of --degrees at the fit points, divided by
    |z|^k, and orthonormalizes them in the mass-weighted inner product of the fit points,
    <a, b> = sum_i w_i conj(a_i) b_i with w_i = mass_i / sum(mass). A pivoted QR
    factorization of the rows sqrt(w_i) m(z_i) gives the basis: the monomials with a pivot
    above --rank_tol times the first pivot enter. A Cholesky factor of the Gram matrix of the
    result gives a second pass. The file --sections holds the exponents and the linear map T_k
    of each degree, so sigma_k(z) = m_k(z) T_k / |z|^k at all points.
--task hn
    Fits the HN potential (k0 / k) log(sigma_k^H H sigma_k) + k0 log|z|^2 of the degree k =
    --degree by Levenberg-Marquardt (LM) on the MSPE of the fit set, with the validation stop
    of train_stage_lm. H starts at I, or with --base_dir at the lift of that HN base: the
    weighted least-squares fit of sigma_k^H H sigma_k to p_0^(k / k_b), relative to it, with
    p_0 the density of the base of degree k_b divided by its mean. The folder gets the
    stage1.json of spectral_stages.py, so that a stage 2 of that driver can use the potential
    as its base.
--task correction
    Solves a correction psi = sum_L w_L sigma_L^H C_L sigma_L on the HN base of --base_dir
    exactly with solve_stage_units, with w_L = p_0^(-L / k_b) (--weight base) or 1 (--weight
    fs). The solves repeat until the MSPE of the fit set changes by not more than --solve_rtol
    of its value, at most --max_solves times. For a single level, --ranks truncates C_L to the
    eigenvalues of the largest modulus and gives sigma with no new fit.

After each fit, the eigenvalues of J^T J at the result give cond(J), with and without the
scale of the columns. Eigenvalues below --null_tol times the largest one count as null
directions: the scale of H in HN, and the constant function in the FS weight.

Example (from the root folder of the repository):
    PYTHONPATH=src python training/full_basis.py --task sections --fit data/fit.npz \
        --validation data/val.npz --holdout data/holdout.npz --sections runs/full/sections.npz \
        --save_dir runs/full/sections
"""

import argparse
import json
import math
import os
import time

import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp

import MLGeometry as mlg
from models import (HNPotential, HermitianCorrection, monomial_exponents, monomials, load_sections,
                    hermitian_units, hermitian_coefficients, hermitian_matrix)
from spectral_stages import load_dataset, print_dtypes, dtypes_of, evaluate, n_params_of, save_pickle, load_pickle


# --- Argument Parsing ---
def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Fit potentials on the full bases of sections of low degree.")
    parser.add_argument('--task', type=str, required=True, choices=['sections', 'hn', 'correction'])
    parser.add_argument('--fit', type=str, required=True, help='The .npz file of the fit set.')
    parser.add_argument('--validation', type=str, required=True, help='The .npz file of the validation set.')
    parser.add_argument('--holdout', type=str, required=True, help='The .npz file of the hold-out set.')
    parser.add_argument('--sections', type=str, required=True,
                        help='The .npz file of the sections: --task sections writes it, the other tasks read it.')
    parser.add_argument('--save_dir', type=str, required=True, help='Folder for the results and summary.json.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Keep every stride-th point of each set. Use it only for tests.')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size of the metric, of the sums of J^T J and of the solves.')
    parser.add_argument('--null_tol', type=float, default=1e-13,
                        help='Eigenvalues of J^T J below this times the largest one are null directions.')
    parser.add_argument('--skip_cond', action='store_true', help='Do not compute cond(J) after the fit.')

    # Sections
    parser.add_argument('--degrees', type=int, nargs='+', default=[3, 4, 5], help='Degrees of the sections.')
    parser.add_argument('--rank_tol', type=float, default=1e-10,
                        help='Smallest pivot of the QR factorization, relative to the first, that enters.')

    # HN
    parser.add_argument('--degree', type=int, default=3, help='Degree k of the sections of the HN potential.')
    parser.add_argument('--k0', type=int, default=3, help='Class k0 H of the HN potential.')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Folder of an HN fit: the start of --task hn (the lift), or the base of --task correction.')
    parser.add_argument('--lm_steps', type=int, default=200, help='Maximum number of LM steps.')
    parser.add_argument('--lm_patience', type=int, default=10,
                        help='Stop LM after this number of steps with no relative decrease of --lm_min_delta '
                             'in the validation sigma.')
    parser.add_argument('--lm_min_delta', type=float, default=1e-4)
    parser.add_argument('--lm_damping', type=float, default=1e-3, help='Initial damping of LM.')

    # Correction
    parser.add_argument('--levels', type=int, nargs='+', default=[5], help='Degrees L of the levels of the correction.')
    parser.add_argument('--weight', type=str, default='base', choices=['base', 'fs'],
                        help='Weight of each level: p_0^(-L / k_b) of the base, or 1.')
    parser.add_argument('--max_solves', type=int, default=10, help='Maximum number of exact solves.')
    parser.add_argument('--solve_rtol', type=float, default=1e-3,
                        help='Stop the solves when the MSPE of the fit set changes by not more than this relative amount.')
    parser.add_argument('--ranks', type=int, nargs='*', default=None,
                        help='Ranks of the truncated C_L of a single level, evaluated with no new fit.')
    return parser.parse_args()


# --- Data ---
def load_sets(args) -> tuple:
    """The fit, validation and hold-out sets, with every stride-th point."""
    sets, info = {}, {}
    for name, path in [('fit', args.fit), ('validation', args.validation), ('holdout', args.holdout)]:
        dataset, data_info = load_dataset(path)
        if args.stride > 1:
            dataset = {k: v[::args.stride] for k, v in dataset.items()}
            data_info['stride'] = args.stride
            data_info['n_points'] = int(dataset['points'].shape[0])
        sets[name], info[name] = dataset, data_info
    return sets, info


def mass_weights(dataset: dict) -> jnp.ndarray:
    """The weights w_i = mass_i / sum(mass)."""
    return dataset['mass'] / jnp.sum(dataset['mass'])


def pad_rows(x: jnp.ndarray, batch_size: int, value=None) -> jnp.ndarray:
    """Pads x to a multiple of batch_size with copies of its last row, or with `value`."""
    pad_len = (-x.shape[0]) % batch_size
    if pad_len == 0:
        return x
    filler = jnp.repeat(x[-1:], pad_len, axis=0) if value is None else jnp.full((pad_len,) + x.shape[1:], value, x.dtype)
    return jnp.concatenate([x, filler], axis=0)


def weighted_sum(fn, points: jnp.ndarray, weights: jnp.ndarray, batch_size: int):
    """sum_i weights_i fn(z_i), with fn from points (B, d) to an array (B, ...), in batches."""
    step = jax.jit(lambda z, w: jnp.tensordot(w, fn(z), axes=1))
    z_pad, w_pad = pad_rows(points, batch_size), pad_rows(weights, batch_size, 0.0)
    total = 0.0
    for i in range(0, z_pad.shape[0], batch_size):
        total = total + step(z_pad[i:i + batch_size], w_pad[i:i + batch_size])
    return total


# --- Evaluation ---
def make_metric_fn(model, batch_size: int):
    """The function (params, dataset) -> the stored metric plus the metric of the model, in batches."""
    step = jax.jit(lambda p, z, r: mlg.loss.compute_cy_metric(model, p, {'points': z, 'restriction': r}))

    def metric(params, dataset):
        z, r = dataset['points'], dataset['restriction']
        parts = [step(params, z[i:i + batch_size], r[i:i + batch_size]) for i in range(0, z.shape[0], batch_size)]
        return dataset['cymetric'] + jnp.concatenate(parts, axis=0)
    return metric


def evaluate_sets(metric_fn, params, sets: dict, label: str) -> dict:
    """metric_stats of the full metric on each set, as in spectral_stages.evaluate."""
    return evaluate({name: metric_fn(params, ds) for name, ds in sets.items()}, sets, label)


def condition_numbers(model, dataset: dict, params, batch_size: int, null_tol: float) -> dict:
    """cond(J) at params from the eigenvalues of J^T J, without and with the scale of the columns.

    The eigenvalues below null_tol times the largest one are null directions; cond is the square
    root of the largest over the smallest of the other eigenvalues.
    """
    start = time.time()
    jtj, _, _ = mlg.trainer.stage_normal_equations(model, dataset, params, batch_size)
    diag = jnp.diag(jtj)
    scale = 1.0 / jnp.sqrt(jnp.where(diag > 0, diag, 1.0))
    result = {'n_zero_diag': int(jnp.sum(diag <= 0))}
    for name in ('unscaled', 'scaled'):
        matrix = jtj if name == 'unscaled' else jtj * scale[:, None] * scale[None, :]
        ev = np.asarray(jnp.linalg.eigvalsh(matrix))
        top = ev[-1]
        null = ev <= null_tol * top
        result[name] = {'cond': float(np.sqrt(top / ev[~null][0])), 'n_null': int(np.sum(null)),
                        'smallest_over_largest': [float(x) for x in ev[:6] / top]}
        del matrix
    result['time_s'] = time.time() - start
    print(f"cond(J): {result['unscaled']['cond']:.3e} ({result['unscaled']['n_null']} null), "
          f"with scaled columns {result['scaled']['cond']:.3e} ({result['scaled']['n_null']} null), "
          f"smallest scaled eigenvalues over the largest {result['scaled']['smallest_over_largest'][:3]} "
          f"({result['time_s']:.1f}s)")
    return result


# --- HN potentials ---
def load_hn(base_dir: str) -> tuple:
    """The HN model, its parameters and its stage1.json of a --task hn folder."""
    with open(os.path.join(base_dir, 'stage1.json')) as fp:
        meta = json.load(fp)
    sections, n_sections = load_sections(meta['sections'], meta['degree'])
    model = HNPotential(sections=sections, n_sections=n_sections, degree=meta['degree'], k0=meta['k0'])
    params = load_pickle(os.path.join(base_dir, meta['chain_params']))
    return model, params, meta


def normalized_density(model, params, fit: dict, batch_size: int) -> tuple:
    """The density p_0 of an HN model divided by its weighted mean on the fit set, and that mean."""
    def density(z):
        return model.apply(params, z, method='density')
    mean = float(weighted_sum(density, fit['points'], mass_weights(fit), batch_size))

    def scaled(z):
        return density(z) / mean
    return scaled, mean


def lift_coefficients(sections, target, fit: dict, batch_size: int) -> tuple:
    """The weighted least-squares fit of hermitian_units(sigma) @ h to target(z), relative to target(z).

    It minimizes sum_i w_i (u_i @ h / t_i - 1)^2 with the normal equations, solved by an
    eigendecomposition.

    Returns:
        (h, a dict with cond of the rows, the relative rms and max of the residual, and the time).
    """
    start = time.time()
    weights = mass_weights(fit)

    def rows(z):
        return hermitian_units(sections(z)) / target(z)[:, None]

    @jax.jit
    def sums(z, w):
        u = rows(z)
        return u.T @ (w[:, None] * u), w @ u

    z_pad, w_pad = pad_rows(fit['points'], batch_size), pad_rows(weights, batch_size, 0.0)
    gram, rhs = 0.0, 0.0
    for i in range(0, z_pad.shape[0], batch_size):
        g, b = sums(z_pad[i:i + batch_size], w_pad[i:i + batch_size])
        gram, rhs = gram + g, rhs + b
    evals, evecs = jnp.linalg.eigh(gram)
    keep = evals > 1e-15 * evals[-1]
    h = evecs @ jnp.where(keep, (evecs.T @ rhs) / jnp.where(keep, evals, 1.0), 0.0)

    @jax.jit
    def residual(z, w):
        r = rows(z) @ h - 1.0
        return jnp.sum(w * r**2), jnp.max(jnp.where(w > 0, jnp.abs(r), 0.0))

    mse, max_r = 0.0, 0.0
    for i in range(0, z_pad.shape[0], batch_size):
        s, m = residual(z_pad[i:i + batch_size], w_pad[i:i + batch_size])
        mse, max_r = mse + float(s), max(max_r, float(m))
    info = {'cond_rows': float(jnp.sqrt(evals[-1] / evals[0])) if float(evals[0]) > 0 else float('inf'),
            'dropped': int(jnp.sum(~keep)), 'rms_rel_residual': math.sqrt(mse), 'max_rel_residual': max_r,
            'time_s': time.time() - start}
    print(f"Lift: cond of the rows {info['cond_rows']:.3e}, relative residual rms {info['rms_rel_residual']:.3e}, "
          f"max {info['max_rel_residual']:.3e} ({info['time_s']:.1f}s)")
    return h, info


# --- Tasks ---
def task_sections(args, sets: dict, summary: dict, write_summary):
    """Orthonormal sections of each degree from a pivoted QR factorization on the fit points."""
    fit, holdout = sets['fit'], sets['holdout']
    d = int(fit['points'].shape[-1])
    weights = mass_weights(fit)
    weights_h = mass_weights(holdout)
    stored, summary['degrees'] = {}, []
    for k in args.degrees:
        start = time.time()
        exponents = monomial_exponents(k, d)

        def normalized(z, exponents=exponents, k=k):
            norm_sq = jnp.sum(jnp.real(z * jnp.conj(z)), axis=-1, keepdims=True)
            return monomials(z, exponents) / norm_sq**(0.5 * k)
        step = jax.jit(normalized)
        m = jnp.concatenate([step(fit['points'][i:i + 50000]) for i in range(0, fit['points'].shape[0], 50000)])
        a = np.asarray(jnp.sqrt(weights)[:, None] * m)
        r, piv = scipy.linalg.qr(a, mode='r', pivoting=True)
        del a
        n_mon = len(exponents)
        r = r[:n_mon, :n_mon]
        rel = np.abs(np.diag(r)) / np.abs(r[0, 0])
        rank = int(np.sum(rel > args.rank_tol))
        transform = np.zeros((n_mon, rank), dtype=np.complex128)
        transform[piv[:rank]] = scipy.linalg.solve_triangular(r[:rank, :rank], np.eye(rank, dtype=r.dtype))
        time_qr = time.time() - start

        # The second pass: sigma L^-H is orthonormal for the Gram matrix L L^H of sigma.
        def gram_error(mvals, w, t):
            sigma = mvals @ jnp.asarray(t)
            gram = (jnp.conj(sigma).T * w[None, :]) @ sigma
            return gram, float(jnp.max(jnp.abs(gram - jnp.eye(gram.shape[0]))))
        gram, err_first = gram_error(m, weights, transform)
        chol = np.linalg.cholesky(np.asarray(gram))
        transform = transform @ np.linalg.inv(chol).conj().T
        _, err_fit = gram_error(m, weights, transform)
        m_h = step(holdout['points'])
        _, err_holdout = gram_error(m_h, weights_h, transform)
        del m, m_h

        h0 = math.comb(k + d - 1, d - 1) - math.comb(k - 1, d - 1)
        entry = {'degree': k, 'n_monomials': n_mon, 'h0': h0, 'rank': rank,
                 'smallest_kept_pivot': float(rel[rank - 1]),
                 'largest_dropped_pivot': float(rel[rank]) if rank < n_mon else None,
                 'dropped_monomials': [exponents[j].tolist() for j in piv[rank:]],
                 'orthonormality_error_first_pass': err_first, 'orthonormality_error_fit': err_fit,
                 'orthonormality_error_holdout': err_holdout, 'time_qr_s': time_qr,
                 'time_s': time.time() - start, 'relative_pivots': rel.tolist()}
        summary['degrees'].append(entry)
        print(f"Degree {k}: {n_mon} monomials, h0 = {h0}, numerical rank {rank}; smallest kept pivot "
              f"{entry['smallest_kept_pivot']:.3e}, largest dropped pivot {entry['largest_dropped_pivot']}, "
              f"dropped monomials {entry['dropped_monomials']}; max abs(G - I) {err_first:.2e} after the QR, "
              f"{err_fit:.2e} after the second pass, {err_holdout:.2e} on the hold-out ({entry['time_s']:.1f}s)")
        stored.update({f'exponents_{k}': exponents, f'transform_{k}': transform, f'pivots_{k}': piv,
                       f'relative_pivots_{k}': rel})
        write_summary()
    out_dir = os.path.dirname(args.sections)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.sections, **stored)
    print(f"Saved the sections to {args.sections}")


def task_hn(args, sets: dict, summary: dict, write_summary):
    """The HN potential of --degree by LM, from H = I or from the lift of the base of --base_dir."""
    d = int(sets['fit']['points'].shape[-1])
    for ds in sets.values():
        n_man = ds['restriction'].shape[1]
        ds['cymetric'] = jnp.zeros((ds['points'].shape[0], n_man, n_man), dtype=mlg.config.complex_dtype)
    sections, n_sections = load_sections(args.sections, args.degree)
    model = HNPotential(sections=sections, n_sections=n_sections, degree=args.degree, k0=args.k0)
    params = mlg.trainer.init_params(model, (d,))
    metric_fn = make_metric_fn(model, args.batch_size)
    stage = {'task': 'hn', 'degree': args.degree, 'k0': args.k0, 'n_sections': n_sections,
             'n_params': n_params_of(params), 'n_points_fit': int(sets['fit']['points'].shape[0])}
    summary['stage'] = stage
    print(f"\n--- HN at degree {args.degree} (class {args.k0} H): {n_sections} sections, "
          f"{stage['n_params']} real parameters ---")

    if args.base_dir is None:
        # With H = I, the density is the sum of abs(sigma_a)^2.
        z = sets['fit']['points'][:1000]
        dev = float(jnp.max(jnp.abs(model.apply(params, z, method='density')
                                    - jnp.sum(jnp.abs(sections(z))**2, axis=-1))))
        stage['start'] = {'from': 'identity', 'check_density_identity': dev}
        print(f"Start H = I: max abs(density - sum abs(sigma)^2) = {dev:.3e}")
    else:
        base_model, base_params, base_meta = load_hn(args.base_dir)
        density, mean = normalized_density(base_model, base_params, sets['fit'], args.batch_size)
        exponent = args.degree / base_meta['degree']
        h, lift = lift_coefficients(sections, lambda z: density(z)**exponent, sets['fit'], args.batch_size)
        params = {'params': {'h': h}}
        stage['start'] = {'from': 'lift', 'base_dir': args.base_dir, 'base_density_mean': mean,
                          'exponent': exponent, 'lift': lift}
    stage['start']['eval'] = evaluate_sets(metric_fn, params, sets, 'start')
    save_pickle(jax.device_get(params), os.path.join(args.save_dir, 'stage1_start.pkl'))
    write_summary()

    lm_record = []
    start = time.time()
    params, best_val = mlg.trainer.train_stage_lm(
        model=model, dataset=sets['fit'], epochs=args.lm_steps, params=params, batch_size=args.batch_size,
        damping=args.lm_damping, patience=args.lm_patience, min_delta=args.lm_min_delta,
        val_dataset=sets['validation'], val_batch_size=args.batch_size, step_record=lm_record, verbose=True)
    stage['lm'] = {'steps': len(lm_record), 'time_s': time.time() - start, 'best_val_sigma': best_val,
                   'record': lm_record}
    stage['eval'] = evaluate_sets(metric_fn, params, sets, f'HN {args.degree}, LM')
    save_pickle(jax.device_get(params), os.path.join(args.save_dir, 'stage1_lm.pkl'))
    write_summary()
    if not args.skip_cond:
        stage['cond'] = condition_numbers(model, sets['fit'], params, args.batch_size, args.null_tol)
        write_summary()

    meta = {'stage': 1, 'network': 'hn', 'base': 'none', 'degree': args.degree, 'k0': args.k0,
            'sections': args.sections, 'n_sections': n_sections, 'amp': 1.0, 'chain_params': 'stage1_lm.pkl'}
    with open(os.path.join(args.save_dir, 'stage1.json'), 'w') as fp:
        json.dump(meta, fp, indent=1)


def task_correction(args, sets: dict, summary: dict, write_summary):
    """The exact solve of a correction of the levels --levels on the HN base of --base_dir."""
    d = int(sets['fit']['points'].shape[-1])
    base_model, base_params, base_meta = load_hn(args.base_dir)
    base_metric = make_metric_fn(base_model, args.batch_size)
    for ds in sets.values():
        n_man = ds['restriction'].shape[1]
        ds['cymetric'] = jnp.zeros((ds['points'].shape[0], n_man, n_man), dtype=mlg.config.complex_dtype)
        ds['cymetric'] = base_metric(base_params, ds)
    print_dtypes("fit set with the stored metric of the base", sets['fit'])
    summary['dtypes'] = {name: dtypes_of(ds) for name, ds in sets.items()}
    stage = {'task': 'correction', 'levels': args.levels, 'weight': args.weight, 'base_dir': args.base_dir,
             'k0': base_meta['degree'], 'n_points_fit': int(sets['fit']['points'].shape[0])}
    summary['stage'] = stage
    stage['base_eval'] = evaluate(
        {name: ds['cymetric'] for name, ds in sets.items()}, sets, f"base HN {base_meta['degree']}")

    density = None
    if args.weight == 'base':
        density, mean = normalized_density(base_model, base_params, sets['fit'], args.batch_size)
        stage['base_density_mean'] = mean
    level_sections = [load_sections(args.sections, level) for level in args.levels]
    model = HermitianCorrection(sections=[s for s, _ in level_sections], n_sections=[n for _, n in level_sections],
                                degrees=args.levels, density=density, k0=base_meta['degree'])
    params = mlg.trainer.init_params(model, (d,))
    stage['n_sections'] = [n for _, n in level_sections]
    stage['n_params'] = n_params_of(params)
    print(f"\n--- Correction: levels {args.levels}, weight {args.weight}, sections {stage['n_sections']}, "
          f"{stage['n_params']} real coefficients ---")

    # The units of the model against the forms w_L sigma^H C_L sigma for random coefficients.
    c = jax.random.normal(jax.random.PRNGKey(0), params['params']['c'].shape, mlg.config.real_dtype)
    z = sets['fit']['points'][:256]
    psi = model.apply({'params': {'c': c}}, z)[:, 0]
    reference, offset = 0.0, 0
    for (sections, n), level in zip(level_sections, args.levels):
        matrix = jnp.asarray(hermitian_matrix(np.asarray(c[offset:offset + n * n]), n))
        sigma = sections(z)
        form = jnp.real(jnp.einsum('pa,ab,pb->p', jnp.conj(sigma), matrix, sigma))
        weight = 1.0 if density is None else density(z)**(-level / base_meta['degree'])
        reference, offset = reference + weight * form, offset + n * n
    stage['check_units'] = float(jnp.max(jnp.abs(psi - reference)) / jnp.max(jnp.abs(reference)))
    print(f"Check of the units: max abs(psi - forms) over max abs(forms) = {stage['check_units']:.3e}")
    write_summary()

    solve_record = []
    start = time.time()
    params, _ = mlg.trainer.solve_stage_units(
        model=model, dataset=sets['fit'], params=params, solves=args.max_solves, batch_size=args.batch_size,
        val_dataset=sets['validation'], val_batch_size=args.batch_size, step_record=solve_record,
        verbose=True, rtol=args.solve_rtol)
    stage['solve'] = {'solves': len(solve_record) - 1, 'time_s': time.time() - start, 'record': solve_record}
    metric_fn = make_metric_fn(model, args.batch_size)
    stage['eval'] = evaluate_sets(metric_fn, params, sets, f"levels {args.levels}, {args.weight} weight")
    save_pickle(jax.device_get(params), os.path.join(args.save_dir, 'correction.pkl'))
    write_summary()
    if not args.skip_cond:
        stage['cond'] = condition_numbers(model, sets['fit'], params, args.batch_size, args.null_tol)
        write_summary()

    if args.ranks and len(args.levels) == 1:
        n = stage['n_sections'][0]
        matrix = hermitian_matrix(np.asarray(params['params']['c']), n)
        lam, vec = np.linalg.eigh(matrix)
        order = np.argsort(-np.abs(lam))
        eval_sets = {name: sets[name] for name in ('validation', 'holdout')}
        rows = []
        for rank in args.ranks:
            keep = order[:rank]
            truncated = (vec[:, keep] * lam[keep]) @ vec[:, keep].conj().T
            p_rank = {'params': {'c': jnp.asarray(hermitian_coefficients(truncated), mlg.config.real_dtype)}}
            rows.append({'rank': rank, 'n_real_params': rank * (2 * n - rank),
                         'eval': evaluate_sets(metric_fn, p_rank, eval_sets, f'rank {rank}')})
        stage['ranks'] = {'abs_eigenvalues': [float(x) for x in np.abs(lam[order])],
                          'signs': [int(x) for x in np.sign(lam[order])], 'rows': rows}
        write_summary()


# --- Main ---
def main():
    """Runs the task and writes summary.json."""
    args = parse_args()
    run_start = time.time()
    mlg.set_precision(64)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"JAX {jax.__version__}, devices: {jax.devices()}, default float dtype: {jnp.ones(1).dtype}")

    sets, data_info = load_sets(args)
    print(', '.join(f"{name}: {info['n_points']} points (seed {info['seed']})" for name, info in data_info.items()))
    summary = {'args': vars(args), 'jax': jax.__version__, 'devices': [str(dev) for dev in jax.devices()],
               'data': data_info}
    summary_path = os.path.join(args.save_dir, 'summary.json')

    def write_summary():
        summary['wall_time_s'] = time.time() - run_start
        with open(summary_path, 'w') as fp:
            json.dump(summary, fp, indent=1)

    tasks = {'sections': task_sections, 'hn': task_hn, 'correction': task_correction}
    tasks[args.task](args, sets, summary, write_summary)
    write_summary()
    print(f"\nRun finished in {time.time() - run_start:.1f}s. Summary: {summary_path}")


if __name__ == '__main__':
    main()
