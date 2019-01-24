# Characterizing Well-behaved vs. Pathological Deep Neural Networks

Code for the paper “Characterizing Well-behaved vs. Pathological Deep Neural Networks“: https://arxiv.org/abs/1811.03087 

## Setup
This package has been tested with python 2.7 and 3.7.

First, you need to install all necessary python dependencies:
```sh
cd moments-dnns
sudo pip install -r requirements.txt
# optionally: sudo pip3 install -r requirements.txt
```

To run reasonably large experiments with convolutional networks, you will need however to have tensorflow-gpu installed, e.g. with
```sh
sudo pip install tensorflow-gpu
```


## Functioning of the package

The package is built on top of Keras. At the core of the package, there are custom layers which perform the simultaneous propagation of signal and noise. There are four layers of simultaneous propagation:
* convolution layers
* batch normalization
* activation layers
* addition layers (used to merge residual and skip-connection branches for resnets)

There are also custom layers performing the computation of moments of signal and noise. Performing the computation inside the models was much more effective both in terms of speed and memory usage.

The entry-point of the package is the file `main.py`. This file contains the function `run_experiment()` which runs an experiment with fixed parameters for a given number of realizations. The final results are saved in the folder `npy/name_experiment/` with the parameter `name_experiment` set at the invocation of `run_experiment()`.

For an experiment with 1,000 realizations .npy files typically occupy a space of a few MB. This space can be optionally reduced by calling the function `prune_experiment()` which only retains the moments relevant for a specified type of plot.

The file `plots.py` provides function to plot the results of the experiments in situations equivalent to the Fig. 2, 3, 4, 5 of the paper.


## How to familiarize with the package

The best way to familiarize with the package is to have a look at the notebooks. The **main notebook** `Reproducing Fig. 2, 3, 4, 5` provides the functional calls as well as the figures from the experiments in the paper.

The other notebook `Complements -- boundary conditions, dataset, epsilon` discusses the influence of boundary conditions of convolutions, input dataset and batch normalization $\epsilon$ fuzz factor. This complementary notebook confirms the results of the paper and may provide additional insights. In addition, it provides additional examples of usage of the function `run_experiment()`.
