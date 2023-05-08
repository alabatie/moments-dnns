# Moments-DNNs

Code for the paper “Characterizing Well-behaved vs. Pathological Deep Neural Networks” published in 36th International Conference on Machine Learning (ICML 2019): "https://arxiv.org/abs/1811.03087".

## Setup

To install the package, first install poetry following: "https://python-poetry.org/docs/".

Then, simply run:

```sh
poetry install 
```

## Description

The package is built on top of TensorFlow Keras. At the core of the package, four types of custom layers perform the simultaneous propagation of signal and noise:

* Convolutional layers
* Batch Norm layers
* Activation layers
* Addition layers to merge residual and skip-connection branches for resnets

Custom layers are also introduced for the computation of the moments of signal and noise. Performing these computations inside the model rather than outside is much more effective both in terms of speed and memory usage.

The entry-point of the package is `run_experiments.py`. This file contains the function `run_experiment()` which runs an experiment with fixed parameters for a given number of simulations. The results of the experiment are saved as npz arrays in the folder `npz` with the parameter `name_experiment` set at the invocation of `run_experiment()`.

For an experiment with 1,000 simulations, `.npz` files typically occupy a space of a few MB. This space can be optionally reduced by calling the function `prune_experiment()` in the file `manage_experiments.py`. This function enables to only retain the moments relevant for a specific type of plot.

The file `plots.py` provides function to plot the results of the experiments in situations equivalent to Fig. 2, 3, 4, 5 of the paper.

## Notebooks

Notebooks provide a easy way of familiarizing with the package. To run these notebooks, simply navigate to [notebooks/](https://github.com/alabatie/moments-dnns/blob/master/notebooks/) and launch a kernel in the poetry's env (using `poetry run jupyter lab` or `poetry run jupyter notebook`).

The main notebook [Reproducing Fig. 2, 3, 4, 5.ipynb](https://github.com/alabatie/moments-dnns/blob/master/notebooks/Reproducing%20Fig.%202%2C%203%2C%204%2C%205.ipynb) shows the function calls to reproduce the results from Fig. 2, 3, 4, 5 from the paper.

There are two complementary notebooks:

* [Complements on width, boundary conditions, dataset, epsilon.ipynb](https://github.com/alabatie/moments-dnns/blob/master/notebooks/Complements%20on%20width%2C%20boundary%20conditions%2C%20dataset%2C%20epsilon.ipynb) discusses the effect of changing the width, boundary conditions of convolutional layers, input dataset and Batch Norm fuzz factor

* [Complements on fully-connected networks.ipynb](https://github.com/alabatie/moments-dnns/blob/master/notebooks/Complements%20on%20fully-connected%20networks.ipynb) discusses experiments equivalent to Fig. 2, 3, 4, 5 for fully-connected networks

These complementary notebooks confirm the results of the paper and provide additional insights and examples of usage of the function `run_experiment()`.
