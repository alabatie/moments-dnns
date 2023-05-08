import numpy as np

from moments_dnns.manage_experiments import delete_experiment, load_experiment
from moments_dnns.run_experiment import run_experiment
from tests import TEST_BATCH_SIZE, TEST_NUM_CHANNELS


def test_run_experiment_vanilla():
    architecture = "vanilla"
    run_experiment(
        architecture=architecture,
        total_depth=10,
        kernel_size=1,
        num_channels=TEST_NUM_CHANNELS,
        batch_size=TEST_BATCH_SIZE,
        num_sims=1,
        name_experiment=f"test_{architecture}",
        dataset="mnist",  # MNIST is lighter to download
        num_computations=1,
    )
    moments = load_experiment(name_experiment=f"test_{architecture}")
    delete_experiment(name_experiment=f"test_{architecture}")

    assert isinstance(moments, dict)
    assert len(moments) == 26
    assert all(isinstance(moments[name_moment], np.ndarray) for name_moment in moments)


def test_run_experiment_bn_ff():
    architecture = "bn_ff"
    run_experiment(
        architecture=architecture,
        total_depth=10,
        kernel_size=1,
        num_channels=TEST_NUM_CHANNELS,
        batch_size=TEST_BATCH_SIZE,
        num_sims=1,
        name_experiment=f"test_{architecture}",
        dataset="mnist",  # MNIST is lighter to download
        num_computations=1,
    )
    moments = load_experiment(name_experiment=f"test_{architecture}")
    delete_experiment(name_experiment=f"test_{architecture}")

    assert isinstance(moments, dict)
    assert len(moments) == 34
    assert all(isinstance(moments[name_moment], np.ndarray) for name_moment in moments)


def test_run_experiment_bn_res():
    architecture = "bn_res"
    run_experiment(
        architecture=architecture,
        total_depth=10,
        kernel_size=1,
        num_channels=TEST_NUM_CHANNELS,
        batch_size=TEST_BATCH_SIZE,
        num_sims=1,
        name_experiment=f"test_{architecture}",
        dataset="mnist",  # MNIST is lighter to download
        num_computations=1,
    )
    moments = load_experiment(name_experiment=f"test_{architecture}")
    delete_experiment(name_experiment=f"test_{architecture}")

    assert isinstance(moments, dict)
    assert len(moments) == 42
    assert all(isinstance(moments[name_moment], np.ndarray) for name_moment in moments)
