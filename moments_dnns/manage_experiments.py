import shutil
from pathlib import Path

import numpy as np


def merge_experiments(name_experiments: list[str], name_merged: str):
    """Merge the results of different experiments.

    Assert that:
        - All moment names match
        - Depth values match

    # Args
        name_experiments: names of experiments to merge
        name_merged: name of the merged experiments

    # Returns
        moments: dictionary of moments of merged experiments
    """
    moments = {}
    for iexperiment, name_experiment in enumerate(name_experiments):
        moments_experiment = load_experiment(name_experiment)

        if iexperiment == 0:
            moments = moments_experiment
        else:
            assert np.allclose(moments["depth"], moments_experiment["depth"])
            del moments_experiment["depth"]
            for name_moment, moment_experiment in moments_experiment.items():
                moments[name_moment] = np.vstack(
                    (moments[name_moment], moment_experiment)
                )

    # save merged experiment
    save_experiment(moments, name_merged)


def prune_experiment(type_plot: str, name_experiment: str):
    """Only keep moments relevant for a given plot/

    This function is used to limit disk space taken by .npy results.

    # Args
        type_plot: type of plot corresponding to the pruning
            ('vanilla_histo' or 'vanilla' or 'bn_ff' or 'bn_res')
        name_experiment: name of the experiment
    """
    assert type_plot in ["vanilla_histo", "vanilla", "bn_ff", "bn_res"]

    pruned_list = ["depth"]
    if type_plot == "vanilla_histo":
        pruned_list += ["nu2_signal_loc3", "mu2_noise_loc3"]
    elif type_plot == "vanilla":
        pruned_list += ["chi_loc3", "chi_loc1", "reff_signal_loc3"]
    elif type_plot == "bn_ff":
        pruned_list += [
            "chi_loc1",
            "chi_loc3",
            "chi_loc4",
            "reff_noise_loc4",
            "reff_signal_loc4",
            "mu4_signal_loc3",
            "nu1_abs_signal_loc3",
        ]
    elif type_plot == "bn_res":
        pruned_list += [
            "chi_loc4",
            "chi_loc2",
            "chi_loc1",
            "chi_loc5",
            "reff_noise_loc3",
            "reff_signal_loc3",
            "mu4_signal_loc2",
            "nu1_abs_signal_loc2",
        ]
        pruned_list += ["res_depth"]

    moments = load_experiment(name_experiment)
    moments = {
        name_moment: moment
        for name_moment, moment in moments.items()
        if name_moment in pruned_list
    }

    # save pruned experiment
    save_experiment(moments, name_experiment)


def save_experiment(moments: dict[str, np.ndarray], name_experiment: str):
    """Save moments in npy/name_experiment/.

    If directory already exists, it is deleted and re-created.

    # Arguments
        moments: moments of the experiment
        name_experiment: name of the experiment
    """
    npy_dir = Path(__file__).parent.parent / "npy"
    exp_dir = npy_dir / name_experiment
    if exp_dir.is_dir():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir()  # create a new dir

    # save different moments as different .npy files
    for name_moment, moment in moments.items():
        path_file = exp_dir / name_moment
        np.save(path_file, moment)


def load_experiment(name_experiment: str) -> dict[str, np.ndarray]:
    """Load moments from npy/name_experiment/.

    # Args
        name_experiment: name of the experiment
    """
    npy_dir = Path(__file__).parent.parent / "npy"
    exp_dir = npy_dir / name_experiment
    if not exp_dir.is_dir():
        raise ValueError("Experiment folder does not exist")

    moments = {}
    for path_file in exp_dir.glob("*"):
        name_moment = path_file.stem
        moments[name_moment] = np.load(path_file)
    return moments
