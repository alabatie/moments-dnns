"""Utils to manage experiments."""
import numpy as np

from moments_dnns import ROOT_DIR


def merge_experiments(name_experiments: list[str], name_merged: str):
    """Merge the results of different experiments.

    # Args
        name_experiments: names of experiments to merge
        name_merged: name of the merged experiments

    # Returns
        moments: dictionary of moments of merged experiments
    """
    moments = {}
    for iexperiment, name_experiment in enumerate(name_experiments):
        moments_experiment = load_experiment(name_experiment)
        for name_moment, moments in moments_experiment.items():
            if name_moment not in moments:
                moments[name_moment] = []
            moments[name_moment].append(moments)

    if any(res_depth != moments["res_depth"][0] for res_depth in moments["res_depth"]):
        raise ValueError("Residual depths do not match.")
    if any(len(depth) != len(moments["depth"][0]) for depth in moments["depth"]):
        raise ValueError("Depth arrays do not match.")

    for name_moment in moments:
        if name_moment in ("depth", "res_depth"):
            moments[name_moment] = moments[name_moment][0]
        else:
            moments[name_moment] = np.concatenate(moments[name_moment], axis=0)
    save_experiment(moments, name_merged)


def prune_experiment(type_plot: str, name_experiment: str):
    """Only keep relevant moments for a given plot.

    This enables to limit disk space taken by .npz results.

    # Args
        type_plot: type of plot associated with the pruning
            ('vanilla_histo' or 'vanilla' or 'bn_ff' or 'bn_res')
        name_experiment: name of the experiment
    """
    if type_plot not in {"vanilla_histo", "vanilla", "bn_ff", "bn_res"}:
        raise ValueError(f"Unknown type of plot: {type_plot}")

    pruned_list = ["depth"]
    match type_plot:
        case "vanilla_histo":
            pruned_list += ["nu2_signal_loc3", "mu2_noise_loc3"]
        case "vanilla":
            pruned_list += ["chi_loc3", "chi_loc1", "reff_signal_loc3"]
        case "bn_ff":
            pruned_list += [
                "chi_loc1",
                "chi_loc3",
                "chi_loc4",
                "reff_noise_loc4",
                "reff_signal_loc4",
                "mu4_signal_loc3",
                "nu1_abs_signal_loc3",
            ]
        case "bn_res":
            pruned_list += [
                "chi_loc4",
                "chi_loc2",
                "chi_loc1",
                "chi_loc5",
                "reff_noise_loc3",
                "reff_signal_loc3",
                "mu4_signal_loc2",
                "nu1_abs_signal_loc2",
                "res_depth",
            ]

    moments = load_experiment(name_experiment)
    moments = {
        name_moment: moment
        for name_moment, moment in moments.items()
        if name_moment in pruned_list
    }
    save_experiment(moments, name_experiment)


def save_experiment(moments: dict[str, np.ndarray], name_experiment: str):
    """Save moments in npz folder.

    # Arguments
        moments: moments of the experiment
        name_experiment: name of the experiment
    """
    npz_dir = ROOT_DIR / "npz"
    path_experiment = npz_dir / f"{name_experiment}.npz"
    np.savez(path_experiment, **moments)


def load_experiment(name_experiment: str) -> dict[str, np.ndarray]:
    """Load moments from npz folder.

    # Args
        name_experiment: name of the experiment
    """
    npz_dir = ROOT_DIR / "npz"
    path_experiment = npz_dir / f"{name_experiment}.npz"
    moments = np.load(path_experiment)

    return dict(moments)


def delete_experiment(name_experiment: str):
    """Delete moments from npz folder.

    # Args
        name_experiment: name of the experiment
    """
    npz_dir = ROOT_DIR / "npz"
    path_experiment = npz_dir / f"{name_experiment}.npz"
    path_experiment.unlink()
