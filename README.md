# Characterizing Well-behaved vs. Pathological Deep Neural Networks

Code for the paper “Characterizing Well-behaved vs. Pathological Deep Neural Networks”: https://arxiv.org/abs/1811.03087 

## Setup
This package has been tested with python 2.7 and 3.7.

First, you need to install all necessary python dependencies:
```sh
cd moments-dnns
sudo pip install -r requirements.txt   # optionally: sudo pip3 install -r requirements.txt
```
To run reasonably large experiments with convolutional networks, you will need to have tensorflow-gpu installed, e.g. with
```sh
sudo pip install tensorflow-gpu
```


## Description

The package is built on top of Keras. At the core of the package, four types of custom layers perform the simultaneous propagation of signal and noise:
* Convolutional layers
* Batch normalization layers
* Activation layers
* Addition layers to merge residual and skip-connection branches for resnets

There are also custom layers for the computation of the moments of signal and noise. Performing these computations inside the model rather than outside is much more effective both in terms of speed and memory usage.

The entry-point of the package is `run_experiment.py`. This file contains the function `run_experiment()` which runs an experiment with fixed parameters for a given number of realizations. The results of the experiment are saved as numpy arrays in the folder `npy/name_experiment/` with the parameter `name_experiment` set at the invocation of `run_experiment()`.

For an experiment with 1,000 realizations .npy files typically occupy a space of a few MB. This space can be optionally reduced by calling the function `prune_experiment()` in the file `manage_experiments.py`. This function enables to only retain the moments relevant for a specific type of plot.

The file `plots.py` provides function to plot the results of the experiments in situations equivalent to Fig. 2, 3, 4, 5 of the paper.


## Notebooks

A good way to familiarize with the package is to have a look at the notebooks. The *main notebook* `Reproducing Fig. 2, 3, 4, 5` shows the function calls as well as the results from Fig. 2, 3, 4, 5 of the paper.

There are two complementary notebooks 

- `Complements on boundary conditions, dataset, epsilon` discusses the influence of boundary conditions of convolutional layers, input dataset and batch normalization fuzz factor

- `Complements on fully-connected networks` discusses the results of experiments equivalent to Fig. 2, 3, 4, 5 for fully-connected networks

These complementary notebooks confirm the results of the paper and provide additional insights and examples of usage of the function `run_experiment()`.

