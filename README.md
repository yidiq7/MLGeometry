# MLGeometry

Higher dimensional computational geometry using machine learning software 

- Kahler geometry and Kahler-Einstein metrics

More to come.

## Set up the environment

1. Install `conda` via [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Install the jupyter notebook in the base environment:

       conda install -c conda-forge notebook
       conda install -c conda-forge nb_conda_kernels
       conda install -c conda-forge cudatoolkit=11.8.0


4. Create the environment with necessary packages:

       conda create -n MLGeometry pip tensorflow-probability sympy matplotlib ipykernel

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


## [Leaderboard](https://github.com/yidiq7/MLGeometry/blob/master/Leaderboard.md)
