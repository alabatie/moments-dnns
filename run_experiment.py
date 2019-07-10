import numpy as np
from tqdm.auto import tqdm
import fire
import inspect

from manage_experiments import save_experiment
from moments_dnns.main_utils import get_name_moments, get_submodel_constants
from moments_dnns.main_utils import load_dataset, make_asserts

from tensorflow.compat.v1.logging import set_verbosity, ERROR
set_verbosity(ERROR)  # remove tensorflow deprecated warnings

from moments_dnns.models import init_original_model, reset_model
from moments_dnns.models import init_ff_model, init_res_model


def run_experiment(architecture, total_depth, kernel_size, num_channels,
                   batch_size, num_realizations, name_experiment,
                   boundary='periodic', dataset='cifar10',
                   epsilon=0.001, res_depth=2,
                   num_computations=100,
                   numpy_seed=0, verbose=True,
                   compute_reff_signal=True, compute_reff_noise=True):
    """ run_experiment
    Entry point of the code to run experiments
    Steps:
        - Assert that experiment constants are valid
        - Load data
        - Get name of moments to be computed
        - Initialize keras models
        - For each realization, propagate noise and signal and fetch moments
        - Save moments in results/name_experiment/ as .npy files
    This function can be:
        - either imported as a standard python function
        - or executed directly as a script, thanks to the fire interface, e.g.
            'python run_experiment.py --architecture=bn_ff
              --total_depth=200 --kernel_size=3 --num_channels=512
              --boundary=periodic --dataset=cifar10 --batch_size=64
              --num_realizations=1000  --name_experiment=bn_ff'

    Inputs:
        architecture: 'vanilla' or 'bn_ff' or 'bn_res'
        total_depth: total depth of the experiment
        kernel_size: spatial extent of convolutional kernel
        num_channels: number of channels
        batch_size: number of images considered for each realization
            (in other words, 1 realization = 1 batch)
        num_realizations: number of realizations in the experiment,
            i.e. number of randomly initialized simultaneous propagation of
            signal on noise with computation of moments
        name_experiment: name of the experiment = name of directory
            to save results (if a directory already exists, it will be
            deleted and created again)
        boundary: boundary condition among 'periodic' or 'symmetric'
            or 'zero_padding' (only relevant if kernel_size > 1)
        dataset: 'cifar10' or 'mnist'
        epsilon: batch normalization fuzz factor
            (only relevant if architecture = 'bn_ff' or 'bn_res')
        res_depth: feedforward depth of residual units
            (only relevant if architecture='bn_res')
        num_computations: total number of moment computations
            (there will be a moment computation every
            total depth // num_computations layers)
        numpy_seed:
            - seed to reproduce image selection
            - note that it does not lead to fully deterministic behaviour
                either, but this is not a problem since we are only concerned
                in expectations and 1-sigma intervals
        verbose: whether parameter values are printed
        compute_reff_signal: whether reff is computed for the signal
        compute_reff_noise: whether reff is computed for the noise
    """
    if verbose:
        # print parameter names and values
        frame = inspect.currentframe()
        args, _, _, param_values = inspect.getargvalues(frame)
        print('Running experiment with parameters:')
        for name_param in args:
            print('    {} = {}'.format(name_param, param_values[name_param]))

    # assertions
    make_asserts(architecture=architecture, kernel_size=kernel_size,
                 num_channels=num_channels, boundary=boundary,
                 total_depth=total_depth, dataset=dataset,
                 num_computations=num_computations, batch_size=batch_size)

    # load data (all images are flattened if kernel_size = 1)
    signal_original, (original_strides,
                      original_num,
                      original_size,
                      original_channels) = load_dataset(dataset, kernel_size)

    # get name of moments to be computed
    name_moments_raw, locs, (num_moments_raw, num_moments) \
        = get_name_moments(architecture,
                           compute_reff_signal,
                           compute_reff_noise)

    # get submodel constants
    spatial_size, num_submodels, sub_depth, delta_moments = \
        get_submodel_constants(original_size, original_strides, total_depth,
                               num_computations)

    # initialize original model
    original_model = init_original_model(original_size=original_size,
                                         kernel_size=kernel_size,
                                         original_channels=original_channels,
                                         num_channels=num_channels,
                                         boundary=boundary,
                                         original_strides=original_strides)

    if architecture == 'vanilla':
        # vanilla net
        submodel = init_ff_model(spatial_size=spatial_size,
                                 kernel_size=kernel_size,
                                 num_channels=num_channels,
                                 boundary=boundary,
                                 sub_depth=sub_depth,
                                 delta_moments=delta_moments,
                                 name_moments_raw=name_moments_raw,
                                 batch_normalization=False)
    elif architecture == 'bn_ff':
        # batch normalized feedforward net
        submodel = init_ff_model(spatial_size=spatial_size,
                                 kernel_size=kernel_size,
                                 num_channels=num_channels,
                                 boundary=boundary,
                                 sub_depth=sub_depth,
                                 delta_moments=delta_moments,
                                 name_moments_raw=name_moments_raw,
                                 batch_normalization=True)
    elif architecture == 'bn_res':
        # batch normalized resnet
        submodel = init_res_model(spatial_size=spatial_size,
                                  kernel_size=kernel_size,
                                  num_channels=num_channels,
                                  boundary=boundary,
                                  sub_depth=sub_depth,
                                  res_depth=res_depth,
                                  delta_moments=delta_moments,
                                  name_moments_raw=name_moments_raw)

    # Fix numpy seed for image selection
    np.random.seed(numpy_seed)

    # this dict will aggregate all moments from all realizations
    moments = {}

    # save depth associated with each computation of moments
    moments['depth'] = total_depth // num_computations \
        * np.arange(1, num_computations + 1)

    # save res_depth (only relevant for resnets in the power law fit for plots)
    moments['res_depth'] = res_depth

    for ireal in tqdm(range(num_realizations)):
        # randomly sample original signal and noise
        ind_real = np.random.permutation(original_num)[:batch_size]
        signal = signal_original[ind_real, ]

        # Start with unit variance noise
        # since all pathologies are invariant to original noise scaling and
        # since we use the right equations of propagation  - linear in
        # the input noise - this works, and later avoids the normalization
        # mu2(dx^0) in chi^l
        noise = np.random.normal(0, 1, (batch_size,
                                        original_size,
                                        original_size,
                                        original_channels))

        # normalize with constant rescaling to have mu2_signal = 1
        # this later avoids the additional normalization mu2(x^0) in chi^l
        mean_signal = signal.mean(axis=(0, 1, 2), keepdims=True)
        std_signal = signal.std(axis=(0, 1, 2), keepdims=True)
        signal = (signal - mean_signal) / std_signal

        # pass original signal and noise through original model
        inputs = [signal, noise]
        reset_model(original_model)
        outputs = original_model.predict(inputs, batch_size=batch_size)

        # incorporate logarithm of mu2(dx^l)
        log_noise = np.zeros((batch_size, 1, 1, 1))  # start at zero log
        inputs = outputs + [log_noise]

        # pass through the same keras submodel, each time reinitialized
        moments_raw = []
        for imodel in range(num_submodels):  # total depth divided in submodels
            reset_model(submodel)  # reinitialize submodel
            outputs = submodel.predict(inputs, batch_size=batch_size)

            moments_raw += outputs[3:]  # fetch signal, noise, log_noise
            inputs = outputs[:3]  # fetch moments

        # add locs to moments
        moments_real = {}
        for iloc, loc in enumerate(locs):
            for iraw, name_moment_raw in enumerate(name_moments_raw):
                imoment = iloc * num_moments_raw + iraw
                moment = moments_raw[imoment::num_moments]

                # convert to float128 to deal with large values
                moment = np.array(moment, dtype=np.float128)

                # average over fake batch dimension
                #  - this is just a dummy dimension added by keras,
                #     which necessarily returns an array (batch_size,)
                #  - outputs are already constants with respect to this dim
                moment = moment.mean(1)
                if 'mu2_noise' in name_moment_raw:
                    # take exp for mu_2_noise, since it comes in log scale
                    # to avoid overflow inside model
                    moment = np.exp(moment)

                # add loc
                name_moment = name_moment_raw + '_' + loc
                moments_real[name_moment] = moment

            # compute normalized sensitivity
            chi_square = \
                moments_real['mu2_noise_' + loc] \
                / moments_real['mu2_signal_' + loc]
            moments_real['chi_' + loc] = np.sqrt(chi_square)

        # add to aggregation
        for name_moment, moment in moments_real.items():
            if (name_moment not in moments):  # initialize array
                moments[name_moment] = np.empty((0, num_computations))
            moments[name_moment] = np.vstack((moments[name_moment], moment))

    # save experiment
    save_experiment(moments, name_experiment)


if __name__ == '__main__':
    # fire enables to run this function directly in bash
    fire.Fire(run_experiment)
