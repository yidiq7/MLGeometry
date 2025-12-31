# MLGeometry

Higher dimensional computational geometry using machine learning software 

- Kahler geometry and Kahler-Einstein metrics

More to come.

## Recent Changes

The backend of MLGeometry has been switched to 'JAX' from 'Tensorflow' due to the flexibility `JAX` provides and the current trend in the ML community. 
If you prefer the older version, please check the 'Using and Older Version' section below.

## Installation

*Note: This installs the CPU version of JAX by default. If you wish to use a GPU, it is recommended to install the appropriate version of JAX for your hardware before installing this package. Please refer to the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for instructions.*

You can install MLGeometry using one of the following methods:

### Via PyPI

    pip install MLGeometry-JAX

### Directly from Github

    pip install git+https://github.com/yidiq7/MLGeometry.git

### Using an Older Version

If you prefer to use an older version of MLGeometry based on Tensorflow 2.16+ and Keras 3, you can check out the previous release (v1.2.1) here: [Version 1.2.1 Release](https://github.com/yidiq7/MLGeometry/releases/tag/v1.2.1).

For an older version based on Tensorflow 2.12 and Keras 2, check [Version 1.1.0 Release](https://github.com/yidiq7/MLGeometry/releases/tag/v1.1.0).

Follow the installation instructions provided in that release's documentation. The compatible versions of Python and CUDA can be found [here](https://www.tensorflow.org/install/source#gpu).

## [Sample jupyter notebook](https://github.com/yidiq7/MLGeometry/blob/main/Guide.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yidiq7/MLGeometry/blob/main/Guide.ipynb)

## Citation

You can find our paper on [arxiv](https://arxiv.org/abs/2012.04797) or [PMLR](https://proceedings.mlr.press/v145/douglas22a.html). 
If you find our paper or package useful in your research or project, please cite it as follows:

```
@InProceedings{pmlr-v145-douglas22a,
  title =    {Numerical Calabi-Yau metrics from holomorphic networks},
  author =       {Douglas, Michael and Lakshminarasimhan, Subramanian and Qi, Yidi},
  booktitle =    {Proceedings of the 2nd Mathematical and Scientific Machine Learning Conference},
  pages =    {223--252},
  year =     {2022},
  editor =   {Bruna, Joan and Hesthaven, Jan and Zdeborova, Lenka},
  volume =   {145},
  series =   {Proceedings of Machine Learning Research},
  month =    {16--19 Aug},
  publisher =    {PMLR},
  pdf =      {https://proceedings.mlr.press/v145/douglas22a/douglas22a.pdf},
  url =      {https://proceedings.mlr.press/v145/douglas22a.html},
}
```
