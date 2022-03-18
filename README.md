# MLGeometry

Higher dimensional computational geometry using machine learning software 

- Kahler geometry and Kahler-Einstein metrics

More to come.

## Set up the environment

1. Install `conda` via [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Install the jupyter notebook in the base environment:

       conda install -c conda-forge notebook
       conda install -c conda-forge nb_conda_kernels

3. Create the environment with necessary packages:

       conda create -n MLGeometry pip tensorflow-probability sympy matplotlib ipykernel

4. Activate the environment and install Tensorflow:

       conda activate MLGeometry
       pip install tensorflow-gpu

   Or if you do not have a GPU:

       conda activate MLGeometry
       pip install tensorflow

5. Open Jupyter with `jupyter-notebook` in the command line, and change the kernel in Kernel -> Change kernel -> Python [conda env:MLGeometry]

6. Clone the repository

       git clone https://github.com/yidiq7/MLGeometry/

   Or download the released version [here](https://github.com/yidiq7/MLGeometry/releases) 

## [Sample jupyter notebook](https://github.com/yidiq7/MLGeometry/blob/master/Guide.ipynb)


## [Leaderboard](https://github.com/yidiq7/MLGeometry/blob/master/Leaderboard.md)
