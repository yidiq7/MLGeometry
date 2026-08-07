# Changelog

## [Unreleased]

## [2.2.0] - 2026-08-07

### Added

- A GitHub Actions workflow that uploads the package to PyPI when a release is published

### Changed

- `requires-python` is now `>=3.12`, to match JAX
- Renamed `num_kahler_metric_jax` to `num_FS_metric_jax`, since Fubini-Study is the only
  metric it can now compute

### Removed

- The unreachable general-`k` code paths, which have raised `AttributeError` since `set_k`
  was dropped in the switch to JAX. Only the Fubini-Study case remains. Computing the
  Donaldson ansatz at arbitrary `k` is left to a future efficient JAX implementation
- The `k` and `h_matrix` arguments, each of which had a single valid value. `num_s_J_jax`,
  `num_FS_metric_jax` and `num_FS_volume_form_jax` now take no arguments

## [2.1.0] - 2026-08-06

### Added

- Support for FP64. Call `mlg.set_precision(64)` before any other operation, or pass
  `--precision 64` to the training script, to run the whole package in double precision
- The loss functions `weighted_RMSE` and `max_abs_error`
- `Kahler_potential`, a model in the training script that takes the hidden layer sizes as
  a list and therefore supports an arbitrary number of layers
- `compute_cy_metric_batched`, which computes the metric in batches so that a large
  dataset does not run out of memory
- A `tolerence` argument for `train_lbfgs`, which defaults to 1e-8 instead of the
  previously hard-coded 1e-5

### Changed

- Renamed `weighted_MSE` to `weighted_MSPE`, since it computes the mean squared
  percentage error rather than the mean squared error
- Renamed the argument `max_iter` of `train_lbfgs` to `epochs`, to be consistent with the
  other training functions
- Fixed a bug that returns NaN whenever the number of points is not divisible by the batch
  size in the accumulated gradient mode
- Fixed a bug that stops the L-BFGS training at a tolerance of 1e-5 in the verbose mode,
  regardless of the tolerance requested
- Removed the small number added to the denominator of the mass formula, which biases the
  Monte Carlo weights
- The normalization factor of the volume form is now held constant during the
  backpropagation
- The multiprocessing pool now uses the 'spawn' context, which avoids a deadlock with JAX
  on Linux. A script that generates points now has to be guarded by
  `if __name__ == '__main__':`
- The losses are now printed in the scientific notation

### Removed

- The models `onelayer` to `fivelayers` in the training script, which are superseded by
  `Kahler_potential`

## [2.0.0] - 2025-12-31

### Changed

We have switched the backend from `Tensorflow` to `JAX` due to flexibility and popularity.
The API bas also been simplified accordingly. 
Please check the usage of the updated package in the latest version of `Guide.ipynb` and training script.

## [1.2.2] - 2025-06-26

### Changed

- Fixed the version info in pyproject.toml

## [1.2.1] - 2025-06-25

### Changed

- Fixed a bug that crashes the code when the solver can't find a solution to the polynomial

## [1.2.0] - 2025-03-07

### Changed

- Updated the package to be compatible with the lastest version of Tensorflow (2.18) and Keras 3
- The package can now be installed by pip 
- Moved the U1-invariant neural network from LOGML24 to the branch 'U1'

## [1.1.0] - 2023-11-20

### Added

- A new section to print out the metrics explicitly in Guide.ipynb
- Support for Calabi-Yau manifolds as the complete intersection of two hypersurfaces 
- Support for generating the real locus of a hypersurface with class RealHypersurface

### Changed

- Changed the default initialization of the SquareDense layer to be all-positive with an extra 
  abs function, which could help the training in certain cases
- Changed several functions in the hypersurface class from being private to public

### Removed

- An incorrect documentation for the complex hessian function
- The function to do numerical integration over the manifold and several related deprecated functions

## [1.0.2] - 2022-03-18

### Added

- A new argument d in the bihomogeneous layer for different dimensions
- Save and load models in the guide
- A tutorial for environment setup

### Removed

- The n_patches attribute in the Hypersurface class since it fails on subpatches
 
## [1.0.1] - 2020-12-20

### Added

- Multi-batch support for L-BFGS

[Unreleased]: https://github.com/yidiq7/MLGeometry/compare/v2.1.0...HEAD
[1.0.1]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.0.1
[1.0.2]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.0.2
[1.1.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.1.0
[1.2.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.2.0
[1.2.1]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.2.1
[1.2.2]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.2.2
[2.0.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v2.0.0
[2.1.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v2.1.0
