import numpy as np
from tqdm.auto import tqdm
import fire
import os
import shutil
import inspect

from moments_dnns.main_utils import get_name_moments, get_submodel_constants
from moments_dnns.main_utils import load_dataset, make_asserts
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
        - Save moments as .npy files in results/name_experiment/
    This function can be:
        - either imported as a standard python function
        - or executed directly as a script, thanks to the fire interface, e.g.
            'python3 main.py run_experiment --architecture=BN_FF \
              --total_depth=200 --kernel_size=3 --num_channels=512 \
              --boundary=periodic --dataset=cifar10 --batch_size=64 \
              --num_realizations=1000  --name_experiment=BN_FF'

    Inputs:
        architecture: 'vanilla' or 'BN_FF' or 'BN_Res'
        total_depth: total depth of the experiment
        kernel_size: spatial extent of convolution kernel
        num_channels: number of channels
        batch_size: number of images considered for each realization
            (in other words, 1 realization = 1 batch)
        num_realizations: number of realizations in the experiment,
            i.e. number of randomly initialized simultaneous propagation of
            signal on noise up to the total depth
        name_experiment: name of directory to save results
            (if a directory already exists, it will be deleted
            and created again)
        boundary: boundary condition among 'periodic' or 'symmetric'
            or 'zero_padding' (only relevant if kernel_size > 1)
        dataset: 'cifar10' or 'mnist'
        epsilon: fuzz factor of batch normalization
            (only relevant if architecture='BN_FF' or architecture='BN_Res')
        res_depth: feedforward depth of residual units
            (only relevant if architecture='BN_Res')
        num_computations: total number of moment computations
            (thus there will be a moment computation every
            total depth // num_computations layers)
        numpy_seed:
            -> seed to reproduce image selection
            -> note that it does not lead to fully deterministic behaviour
                either
            -> since we are only concerned in expectations and 1-sigma
                intervals, this is not a problem
        verbose: whether parameter values are printed
        compute_reff_signal: whether reff is computed for the signal
            -> computation bottleneck
        compute_reff_noise: whether reff is computed for the noise
            -> computation bottleneck

    Outputs:
        moments: dict of moments computed in all realizations of the experiment
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

    # initialize original
    original_model = init_original_model(original_size=original_size,
                                         kernel_size=kernel_size,
                                         original_channels=original_channels,
                                         num_channels=num_channels,
                                         boundary=boundary,
                                         original_strides=original_strides)

    if architecture == 'vanilla':
        submodel = init_ff_model(spatial_size=spatial_size,
                                 kernel_size=kernel_size,
                                 num_channels=num_channels,
                                 boundary=boundary,
                                 sub_depth=sub_depth,
                                 delta_moments=delta_moments,
                                 name_moments_raw=name_moments_raw,
                                 batch_normalization=False)
    elif architecture == 'BN_FF':
        submodel = init_ff_model(spatial_size=spatial_size,
                                 kernel_size=kernel_size,
                                 num_channels=num_channels,
                                 boundary=boundary,
                                 sub_depth=sub_depth,
                                 delta_moments=delta_moments,
                                 name_moments_raw=name_moments_raw,
                                 batch_normalization=True)
    elif architecture == 'BN_Res':
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
        #  -> all pathologies are invariant to original noise scaling and
        #  -> we use the right equations of propagation, linear in input noise
        #  -> so it works
        # This later simplifies the computation of chi^l
        # since it avoids the term mu2(dx^0)
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

        # pass through the same keras model, each time reinitialized
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
                #  -> this is just a dummy dimension added by keras,
                #     which necessarily returns an array (batch_size,)
                #  -> outputs are already constants in this direction
                moment = moment.mean(1)
                if 'mu2_noise' in name_moment_raw:
                    # take exp for mu_2_noise,
                    # since it comes in log scale
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

        break

    # save experiment
    save_experiment(moments, name_experiment)


def merge_experiments(name_experiments, name_merged):
    """ merge_experiments
    Merge the results of different experiments
    Assert that:
        - All moment names coincide
        - Depth values coincide

    Inputs:
        name_experiments: list of names of experiments to merge
        name_merged: name of the merged experiments

    Outputs:
        moments: moments of the merged experiments
    """
    moments = {}
    for iexperiment, name_experiment in enumerate(name_experiments):
        moments_experiment = load_experiment(name_experiment)

        if iexperiment == 0:
            moments = moments_experiment
        else:
            assert np.allclose(moments['depth'], moments_experiment['depth'])
            del moments_experiment['depth']
            for name_moment, moment_experiment in moments_experiment.items():
                moments[name_moment] = np.vstack((moments[name_moment],
                                                  moment_experiment))

    # save merged experiment
    save_experiment(moments, name_merged)


def prune_experiment(type_plot, name_experiment):
    """ prune_experiment
    Only keep moments relevant for the plots
    This function is used to limit hard disk use of saved results

    Inputs:
        type_plot: type of plot corresponding to the pruning
            ('vanilla_histo' or 'vanilla' or 'BN_FF' or 'BN_Res')
        name_experiment: name of the experiment
    """
    assert type_plot in ['vanilla_histo', 'vanilla', 'BN_FF', 'BN_Res']

    pruned_list = ['depth']
    if type_plot == 'vanilla_histo':
        pruned_list += ['nu2_signal_loc3', 'mu2_noise_loc3']
    elif type_plot == 'vanilla':
        pruned_list += ['chi_loc3', 'chi_loc1', 'reff_signal_loc3']
    elif type_plot == 'BN_FF':
        pruned_list += ['chi_loc1', 'chi_loc3', 'chi_loc4',
                        'reff_noise_loc4', 'reff_signal_loc4',
                        'mu4_signal_loc3', 'nu1_abs_signal_loc3']
    else:
        pruned_list += ['chi_loc4', 'chi_loc2', 'chi_loc1', 'chi_loc5',
                        'reff_noise_loc3', 'reff_signal_loc3',
                        'mu4_signal_loc2', 'nu1_abs_signal_loc2']
        pruned_list += ['res_depth']

    moments = load_experiment(name_experiment)
    moments = {name_moment: moment for name_moment, moment in moments.items()
               if name_moment in pruned_list}

    # save pruned experiment
    save_experiment(moments, name_experiment)


def save_experiment(moments, name_experiment):
    """ save_experiment
    Save moments in directory results/name_experiment/
    If directory already exists, it is deleted and created again

    Inputs:
        moments: moments of the experiment
        name_experiment: name of the experiment
    """
    name_dir = os.path.join('npy', name_experiment)
    if os.path.isdir(name_dir):
        shutil.rmtree(name_dir)
    os.makedirs(name_dir)  # create a new dir

    # save different moments as different numpy files
    for name_moment, moment in moments.items():
        path_file = os.path.join(name_dir, name_moment)
        np.save(path_file, moment)


def load_experiment(name_experiment):
    """ load_experiment
    Load moments from directory: results/name_experiment/

    Inputs:
        name_experiment: name of the experiment

    Outputs:
        moments: moments of the experiment
    """
    name_dir = os.path.join('npy', name_experiment)
    assert os.path.isdir(name_dir)

    moments = {}
    for name_file in os.listdir(name_dir):
        name_moment = name_file.split('.')[0]
        path_file = os.path.join(name_dir, name_file)
        moments[name_moment] = np.load(path_file)
    return moments


if __name__ == '__main__':
    # fire enables to run these functions directly in bash
    fire.Fire({'run_experiment': run_experiment,
               'merge_experiment': load_experiment})
