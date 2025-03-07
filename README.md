# MLGeometry

Higher dimensional computational geometry using machine learning software 

- Kahler geometry and Kahler-Einstein metrics

More to come.

## Recent Changes

MLGeometry has been updated to be compatible with the lastest version of TensorFlow and Keras 3, and it can now be installed directly from PyPI. If you prefer the older version, please check the 'Using and Older Version' section below.

## Installation

### Prerequisites

MLGeometry requires Python 3.11 and TensorFlow (>=2.16).

Install TensorFlow by following the official installation guide: [TensorFlow Installation](https://www.tensorflow.org/install). 

On Linux with GPU, TensorFlow can be installed by

    pip install 'tensorflow[and-cuda]'

### Installing MLGeometry

You can install MLGeometry using one of the following methods:

#### Via PyPI

    pip install ML-Geometry

*Note: Use "ML-Geometry" with a hyphen when installing via pip, not "MLGeometry".*

#### Directly from Github

    pip install git+https://github.com/yidiq7/MLGeometry.git

#### Using an Older Version

If you prefer to use an older version of MLGeometry based on Tensorflow 2.12 and Keras 2, you can check out the previous release (v1.1.0) here: [Version 1.1.0 Release](https://github.com/yidiq7/MLGeometry/releases/tag/v1.1.0). Follow the installation instructions provided in that release's documentation. The compatible versions of Python and CUDA can be found [here](https://www.tensorflow.org/install/source#gpu).


## [Sample jupyter notebook](https://github.com/yidiq7/MLGeometry/blob/main/Guide.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yidiq7/blob/main/Guide.ipynb)

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
