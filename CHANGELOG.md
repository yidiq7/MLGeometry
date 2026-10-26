# Changelog

## [Unreleased]

### Added

- The layers `Spectral`, `Dense` and `SirenDense` in `bihomoNN`. `Spectral` gives the
  bihomogeneous monomials divided by the Fubini-Study weight |z|^2. `Dense` is a dense
  layer with Xavier normal initialization. `SirenDense` is a dense layer with the sine
  activation and the SIREN initialization
- The loss functions `weighted_MAPE_amp_scaled` and `weighted_MSPE_amp_scaled`. They divide
  the MAPE by an amplitude and the MSPE by its square
- `compute_stage_loss`, `make_stage_loss_fn` and `evaluate_stage_dataset` in `loss`. They
  add the stored metric of the earlier stages, `dataset['cymetric']`, to the metric of the
  model of one stage
- `train_stage_optax` and `train_stage_lbfgs` in `trainer`, the forms of `train_optax` and
  `train_lbfgs` for one stage on top of the stored metric. They can stop when the loss
  does not decrease any more (`patience`, `min_delta`), and they can record the loss of
  each epoch or iteration (`loss_record`). `train_stage_lbfgs` returns the parameters with
  the lowest loss, and each parameter keeps its dtype: float32 parameters stay on the
  float32 grid. With a validation set (`val_dataset`), both functions return the
  parameters with the lowest sigma on it, and they can also stop when this sigma does not
  decrease any more (`val_patience`, `val_min_delta`)
- The models `SpectralNetwork` and `SpectralSirenNetwork` in `training/models.py`, the
  driver `training/spectral_stages.py` that fits the stages one after the other, and
  `training/generate_data.py` that stores a dataset of the Fermat quintic
- The polynomial correction in `training/models.py`: `PolynomialBranch`, a homogeneous
  polynomial of degree 2^d in its input from x^2 layers with no biases and a linear output,
  and `PolynomialCorrection`, a sum of such branches on z_i z_j_bar / |z|^2 with the
  Fubini-Study weight or the base weight exp(-(L/k0) (Phi_prev - k0 log|z|^2)) of the frozen
  earlier stages. `calibrate_polynomial` gives the fixed layer scales, and
  `reduced_potential_mean` gives the constant shift of the base weight. `SpectralNetwork`
  takes `out_std`, a small standard deviation of the initial output weights
- The options of `training/spectral_stages.py` for the comparison of the networks:
  `--network poly` and `--network bihom`, `--branches`, `--weight`, `--base none`,
  `--out_std`, `--validation`, `--val_patience` and `--lr_search`
- The model `SpectralWangLaiNetwork` in `training/models.py`, after Wang and Lai
  (arXiv:2307.08934). Its first hidden layer is `ScaledSineDense`, sin(kappa x W + b) with a
  fixed scale kappa on the kernel only. The tanh layers follow, and the output is linear
  with no bias. The driver option `--network wanglai` with `--kappa`, and the loop
  `training/wanglai_stage2.sh` of the control at stage 2
- `solve_stage_output` and `train_stage_lm` in `trainer`. `solve_stage_output` solves the
  linear output layer of a stage exactly with the hidden layers fixed, by QR solves of the
  linearized residual. `train_stage_lm` runs Levenberg-Marquardt on all parameters of a
  stage, full batch on MSPE: it sums J^T J and J^T r over batches, solves the damped normal
  equations with a Cholesky factorization, and adapts the damping by the gain ratio. In
  both, J includes the change of the mean of the volume form
- The model `SpectralLinearNetwork` in `training/models.py`, the tanh network with the
  linear output amp * (h w) with no sinh and no bias, and the driver options
  `--network tanh_linear`, `--init_params`, `--output_solves`, `--lm_steps`,
  `--lm_patience`, `--lm_min_delta`, `--lm_damping` and `--lm_batch_size`
- `solve_stage_units` and `train_stage_varpro` in `trainer`. `solve_stage_units` solves the
  coefficients c of a potential psi = units(z) @ c exactly. The rows of J come from second
  derivatives of the units along three directions at each point, so no Hessian of all units
  is stored. `train_stage_varpro` runs Levenberg-Marquardt on the other parameters of such a
  potential, and computes the fixed maps and c again after each accepted step (variable
  projection)
- The units of a learned recurrence in `training/models.py`: `recurrence_features` (the
  d^2 real features z_a z_b_bar of degree 0 with the base weight or the Fubini-Study weight),
  `UnitRecurrence` (products t = (x A^T) * (u B^T) of the last units, orthonormalized against
  all earlier units with fixed maps, and psi = sum c u), `calibrate_recurrence` (the maps on
  the fit points) and `recurrence_gram_error`. The driver `training/recurrence_stage.py` and
  its loop `training/recurrence_stage.sh`
- `stage_normal_equations` in `trainer`. It gives J^T J and J^T r of the MSPE of a stage for
  all parameters, with the rows of `train_stage_lm`, so that the eigenvalues give cond(J)
- The option `rtol` of `solve_stage_units`. The solves stop after the first one that changes
  the MSPE by not more than rtol times its value
- The full bases of sections in `training/models.py`: `monomial_exponents` and `monomials` (all
  monomials of one degree), `make_sections` and `load_sections` (sections m(z) T / |z|^k with a
  fixed map T), `hermitian_units`, `hermitian_coefficients`, `hermitian_matrix` and
  `identity_coefficients` (the real basis Re and Im of conj(sigma_a) sigma_b of the Hermitian
  forms), `HNPotential` (the HN potential (k0 / k) log(sigma^H H sigma) + k0 log|z|^2) and
  `HermitianCorrection` (psi = sum_L w_L sigma_L^H C_L sigma_L with the base weight or the
  Fubini-Study weight, linear in its coefficients). The driver `training/full_basis.py` (the
  orthonormal sections by a pivoted QR factorization, HN by LM, the exact solve of a correction
  and its truncation to a rank) and its loop `training/full_basis.sh`.
  `training/spectral_stages.py` can load an HN potential as the base of a stage

## [2.2.2] - 2026-08-10

### Changed

- The point generator was improved: the point solver is now much faster and more accurate
- Fixed a bug introduced in the translation to JAX in 2.0.0, which makes
  `CICYHypersurface` return no points at all
- Fixed an older bug that makes `CICYHypersurface` use one intersection of each plane
  rather than all of them, which makes its `mass` weights wrong

## [2.2.1] - 2026-08-07

### Removed

- Unexpected tracked files and old comments

## [2.2.0] - 2026-08-07

### Added

- A GitHub Actions workflow that uploads the package to PyPI when a release is published
- The arguments `memory_size` and `max_linesearch_steps` for `train_lbfgs`, which expose
  the two L-BFGS settings worth tuning per problem

### Changed

- `requires-python` is now `>=3.12`, to match JAX
- `train_lbfgs` now uses the L-BFGS in `optax` rather than the one in `jaxopt`, which is no
  longer maintained
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

[Unreleased]: https://github.com/yidiq7/MLGeometry/compare/v2.2.2...HEAD
[1.0.1]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.0.1
[1.0.2]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.0.2
[1.1.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.1.0
[1.2.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.2.0
[1.2.1]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.2.1
[1.2.2]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.2.2
[2.0.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v2.0.0
[2.1.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v2.1.0
[2.2.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v2.2.0
[2.2.1]: https://github.com/yidiq7/MLGeometry/releases/tag/v2.2.1
[2.2.2]: https://github.com/yidiq7/MLGeometry/releases/tag/v2.2.2
