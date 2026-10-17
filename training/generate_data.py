"""
Generates a dataset of points on the Fermat quintic and stores it in a .npz file.

The points come from random lines in CP^4. Each line meets the quintic in 5 points,
so the dataset holds 5 * n_pairs points. The same n_pairs and seed give the same
points, so each run can use the same fit set and hold-out set.

Example (from the root folder of the repository):
    PYTHONPATH=src python training/generate_data.py --n_pairs 10000 --seed 1 --out data/fit.npz
"""

import argparse
import os
import time

import numpy as np
import sympy as sp

import MLGeometry as mlg


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a dataset on the Fermat quintic.")
    parser.add_argument('--n_pairs', type=int, required=True,
                        help='Number of random lines. The dataset holds 5 * n_pairs points.')
    parser.add_argument('--seed', type=int, required=True,
                        help='Seed of the numpy generator that gives the random lines.')
    parser.add_argument('--out', type=str, required=True, help='Path of the .npz file.')
    return parser.parse_args()


def main():
    """Generates the points and the dataset, checks them, and stores them."""
    args = parse_args()

    # The dtypes of the points and of the dataset come from this call, so it is first.
    mlg.set_precision(64)
    np.random.seed(args.seed)

    z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
    Z = [z0, z1, z2, z3, z4]
    f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5

    start_time = time.time()
    HS = mlg.hypersurface.Hypersurface(Z, f, args.n_pairs)
    dataset = mlg.dataset.generate_dataset(HS, jax=False)
    print(f"Generated {HS.n_points} points and the dataset in {time.time() - start_time:.1f}s.")

    # The residual of f relative to the size of its terms shows how close the points are to X.
    points = dataset['points']
    f_num = sp.lambdify(Z, f, 'numpy')
    scale = np.sum(np.abs(points)**5, axis=1)
    residual = np.abs(f_num(*points.T)) / scale
    print(f"max |f(z)| / sum |z_i|^5 = {np.max(residual):.3e}, "
          f"mean = {np.mean(residual):.3e}")

    for key, val in dataset.items():
        print(f"  {key}: shape {val.shape}, dtype {val.dtype}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, n_pairs=args.n_pairs, seed=args.seed, **dataset)
    print(f"Saved the dataset to {args.out}")


if __name__ == '__main__':
    main()
