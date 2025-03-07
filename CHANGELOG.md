# Changelog

## [Unreleased]

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

[Unreleased]: https://github.com/yidiq7/MLGeometry/compare/v1.2.0...HEAD
[1.0.1]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.0.1
[1.0.2]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.0.2
[1.1.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.1.0
[1.2.0]: https://github.com/yidiq7/MLGeometry/releases/tag/v1.2.0
