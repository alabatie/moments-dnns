import numpy as np
import os
import shutil


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
    This function is used to limit disk space taken by .npy results

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
    Save moments in directory npy/name_experiment/
    If directory already exists, it is deleted and created again

    Inputs:
        moments: moments of the experiment
        name_experiment: name of the experiment
    """
    name_dir = os.path.join('npy', name_experiment)
    if os.path.isdir(name_dir):
        shutil.rmtree(name_dir)
    os.makedirs(name_dir)  # create a new dir

    # save different moments as different .npy files
    for name_moment, moment in moments.items():
        path_file = os.path.join(name_dir, name_moment)
        np.save(path_file, moment)


def load_experiment(name_experiment):
    """ load_experiment
    Load moments from directory: npy/name_experiment/

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
