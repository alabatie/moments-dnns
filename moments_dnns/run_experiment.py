"""Entrypoint to run experiments."""
import inspect
import logging

import fire
import numpy as np
from tqdm.auto import tqdm

from moments_dnns.main_utils import (
    check_args,
    get_name_moments,
    get_submodel_constants,
    load_dataset,
)
from moments_dnns.manage_experiments import save_experiment
from moments_dnns.models import (
    init_ff_model,
    init_orig_model,
    init_res_model,
    reset_model,
)


def run_experiment(
    architecture: str,
    total_depth: int,
    kernel_size: int,
    num_channels: int,
    batch_size: int,
    num_sims: int,
    name_experiment: str | None,
    boundary: str = "periodic",
    dataset: str = "cifar10",
    epsilon: float = 0.001,
    res_depth: int = 2,
    num_computations: int = 100,
    numpy_seed: int = 0,
    compute_reff_signal: bool = True,
    compute_reff_noise: bool = True,
):
    """Entry point of the repo to run experiments.

    # Steps
        - Check that experiment arguments are valid
        - Load data
        - Get name of moments to be computed
        - Initialize models
        - For each simulation, propagate noise and signal, and fetch moments
        - Save moments as .npz files

    # Usage
        - This function can be imported as a standard python function
        - Or executed directly as a script with fire, e.g.
            ```python run_experiment.py --architecture=bn_ff
              --total_depth=200 --kernel_size=3 --num_channels=512
              --boundary=periodic --dataset=cifar10 --batch_size=64
              --num_sims=1000  --name_experiment=bn_ff```

    # Args
        architecture: 'vanilla' or 'bn_ff' or 'bn_res'
        total_depth: total depth of the model
        kernel_size: spatial extent of convolutional kernels
        num_channels: number of channels
        batch_size: number of images considered for each simulation
            (i.e. 1 simulation = 1 batch)
        num_sims: number of simulations in the experiment
            (i.e. number of randomly initialized propagation of signal and noise)
        name_experiment: name of experiment and directory to save results
            (if directory already exists, it is deleted and created again)
        boundary: 'periodic' or 'symmetric' or 'zero_padding'
            (only relevant if kernel_size > 1)
        dataset: 'cifar10' or 'mnist'
        epsilon: fuzz factor of Batch Norm
            (only relevant if architecture = 'bn_ff' or 'bn_res')
        res_depth: feedforward depth of residual units
            (only relevant if architecture = 'bn_res')
        num_computations: total number of moment computations
            (computation occurs once every total depth // num_computations layers)
        numpy_seed:
            - seed to reproduce image selection
            - it does not lead to fully deterministic behaviour either,
                but this is not a problem since we are only concerned
                in expectations and 1-sigma intervals
        compute_reff_signal: whether reff is computed for the signal
        compute_reff_noise: whether reff is computed for the noise
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    logger = logging.getLogger(__name__)
    frame = inspect.currentframe()
    args, _, _, param_values = inspect.getargvalues(frame)
    logger.info("Running experiment with parameters:")
    for name_param in args:
        logger.info("%s = %s", name_param, param_values[name_param])

    # check that arguments are valid
    check_args(
        architecture=architecture,
        kernel_size=kernel_size,
        num_channels=num_channels,
        boundary=boundary,
        total_depth=total_depth,
        dataset=dataset,
        num_computations=num_computations,
        batch_size=batch_size,
    )

    # load data (all images are flattened if kernel_size = 1)
    signal_orig, orig_strides, orig_num, orig_size, orig_channels = load_dataset(
        dataset, kernel_size
    )

    # get name of moments to be computed
    name_moments, locs, num_moments, num_moments_loc = get_name_moments(
        architecture, compute_reff_signal, compute_reff_noise
    )

    # get submodel constants
    spatial_size, num_submodels, sub_depth, delta_moments = get_submodel_constants(
        orig_size, orig_strides, total_depth, num_computations
    )

    # initialize original model performing a single convolution
    original_model = init_orig_model(
        orig_size=orig_size,
        kernel_size=kernel_size,
        orig_channels=orig_channels,
        num_channels=num_channels,
        boundary=boundary,
        orig_strides=orig_strides,
    )

    match architecture:
        case "vanilla":
            submodel = init_ff_model(
                spatial_size=spatial_size,
                kernel_size=kernel_size,
                num_channels=num_channels,
                boundary=boundary,
                sub_depth=sub_depth,
                delta_moments=delta_moments,
                name_moments=name_moments,
                epsilon=epsilon,
                batch_norm=False,
            )
        case "bn_ff":
            submodel = init_ff_model(
                spatial_size=spatial_size,
                kernel_size=kernel_size,
                num_channels=num_channels,
                boundary=boundary,
                sub_depth=sub_depth,
                delta_moments=delta_moments,
                name_moments=name_moments,
                epsilon=epsilon,
                batch_norm=True,
            )
        case "bn_res":
            submodel = init_res_model(
                spatial_size=spatial_size,
                kernel_size=kernel_size,
                num_channels=num_channels,
                boundary=boundary,
                sub_depth=sub_depth,
                res_depth=res_depth,
                delta_moments=delta_moments,
                name_moments=name_moments,
                epsilon=epsilon,
            )

    # Fix numpy seed for image selection
    np.random.seed(numpy_seed)

    # this dict aggregates all moments from all simulations
    moments_all = {
        "depth": total_depth // num_computations * np.arange(1, num_computations + 1),
        "res_depth": res_depth,
    }

    for _ in tqdm(range(num_sims)):
        # randomly sample original signal and noise
        ind_sim = np.random.permutation(orig_num)[:batch_size]
        signal = signal_orig[ind_sim]

        # Start with unit variance noise.
        # This later avoids the additional normalization mu2(dx^0) in chi^l.
        # This works since all pathologies are invariant to original noise scaling
        # and we use the right equations of propagation (linear in the input noise).
        noise = np.random.normal(
            0, 1, (batch_size, orig_size, orig_size, orig_channels)
        )

        # Normalize with constant rescaling to have mu2(x^0) = 1.
        # This later avoids the additional normalization mu2(x^0) in chi^l.
        signal = (signal - signal.mean(axis=(0, 1, 2), keepdims=True)) / signal.std(
            axis=(0, 1, 2), keepdims=True
        )

        # pass original signal and noise through original model
        inputs = [signal, noise]
        reset_model(original_model)
        outputs = original_model.predict(inputs, batch_size=batch_size, verbose=False)

        # incorporate logarithm of mu2(dx^l)
        log_noise = np.zeros((batch_size, 1, 1, 1))  # start at zero log
        inputs = outputs + [log_noise]

        # pass through the same keras submodel, each time re-initialized
        moments = []
        for _ in range(num_submodels):  # total depth divided in submodels
            reset_model(submodel)  # re-initialize submodel
            outputs = submodel.predict(inputs)

            moments += outputs[3:]  # fetch signal, noise, log_noise
            inputs = outputs[:3]  # fetch moments

        # add locs to moments
        moments_sim = {}
        for iloc, loc in enumerate(locs):
            for iraw, name_moment in enumerate(name_moments):
                imoment_loc = iloc * num_moments + iraw
                moment = moments[imoment_loc::num_moments_loc]

                # convert to float128 to deal with large values
                moment = np.array(moment, dtype=np.longdouble)

                # Average over fake batch dimension
                # (this is just a dummy dimension added by keras).
                # Outputs are already constants with respect to this dim.
                moment = moment.mean(1)
                if "mu2_noise" in name_moment:
                    # take exp for mu_2_noise, since it comes in log scale
                    # to avoid overflow inside model
                    moment = np.exp(moment)

                # add loc
                moments_sim[name_moment + "_" + loc] = moment

            # compute normalized sensitivity
            moments_sim["chi_" + loc] = np.sqrt(
                moments_sim["mu2_noise_" + loc] / moments_sim["mu2_signal_" + loc]
            )

        # add to aggregation
        for name_moment, moment in moments_sim.items():
            if name_moment not in moments_all:
                moments_all[name_moment] = []  # initialize array
            moments_all[name_moment].append(moment)

    for name_moment in moments_all:
        if isinstance(moments_all[name_moment], list):
            moments_all[name_moment] = np.stack(moments_all[name_moment])

    # save experiment
    if name_experiment is not None:
        save_experiment(moments_all, name_experiment)


if __name__ == "__main__":
    # fire enables to run this function directly in CLI
    fire.Fire(run_experiment)
