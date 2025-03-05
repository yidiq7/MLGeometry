# MLGeometry

Higher dimensional computational geometry using machine learning software 

- Kahler geometry and Kahler-Einstein metrics

More to come.
## Set up the environment (Temporary Fix):
       conda create -n MLGeometry python=3.11 pip sympy matplotlib
       conda activate MLGeometry
       pip install tensorflow[and-cuda] tensorflow-probability
## Set up the environment

1. Install `conda` via [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Install the jupyter notebook in the base environment:

       conda install -c conda-forge notebook
       conda install -c conda-forge nb_conda_kernels
       conda install -c conda-forge cudatoolkit=11.8.0


4. Create the environment with necessary packages:

       conda create -n MLGeometry pip tensorflow-probability==0.14.0 sympy matplotlib ipykernel

5. Activate the environment and install Tensorflow:

       conda activate MLGeometry
       python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
       mkdir -p $CONDA_PREFIX/etc/conda/activate.d
       echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
       echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
       source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

6. Verify install:
   
       python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

7. Open Jupyter with `jupyter-notebook` in the command line, and change the kernel in Kernel -> Change kernel -> Python [conda env:MLGeometry]

8. Clone the repository

       git clone https://github.com/yidiq7/MLGeometry/

   Or download the released version [here](https://github.com/yidiq7/MLGeometry/releases) 

## [Sample jupyter notebook](https://github.com/yidiq7/MLGeometry/blob/master/Guide.ipynb)

<!--## [Leaderboard](https://github.com/yidiq7/MLGeometry/blob/master/Leaderboard.md)-->

## Citation

You can find our paper on [arxiv](https://arxiv.org/abs/2012.04797) or [PMLR](https://proceedings.mlr.press/v145/douglas22a.html). 
If you find our paper or package useful in your research or project, please cite it as follows:

```
@InProceedings{pmlr-v145-douglas22a,
  title = 	 {Numerical Calabi-Yau metrics from holomorphic networks},
  author =       {Douglas, Michael and Lakshminarasimhan, Subramanian and Qi, Yidi},
  booktitle = 	 {Proceedings of the 2nd Mathematical and Scientific Machine Learning Conference},
  pages = 	 {223--252},
  year = 	 {2022},
  editor = 	 {Bruna, Joan and Hesthaven, Jan and Zdeborova, Lenka},
  volume = 	 {145},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {16--19 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v145/douglas22a/douglas22a.pdf},
  url = 	 {https://proceedings.mlr.press/v145/douglas22a.html},
}
```
