import numpy as np
import pytest

from moments_dnns.manage_experiments import (
    delete_experiment,
    load_experiment,
    prune_experiment,
)
from tests import TEST_DATA_DIR


@pytest.fixture
def moments_vanilla_raw():
    moments = np.load(f"{TEST_DATA_DIR}/moments_vanilla_raw.npz")
    return dict(moments)


@pytest.fixture
def moments_bn_ff_raw():
    moments = np.load(f"{TEST_DATA_DIR}/moments_bn_ff_raw.npz")
    return dict(moments)


@pytest.fixture
def moments_bn_res_raw():
    moments = np.load(f"{TEST_DATA_DIR}/moments_bn_res_raw.npz")
    return dict(moments)


def test_prune_vanilla_experiment(moments_vanilla_raw, mocker):
    mocker.patch(
        "moments_dnns.manage_experiments.load_experiment",
        return_value=moments_vanilla_raw,
    )
    type_plot = "vanilla_histo"
    prune_experiment(type_plot=type_plot, name_experiment=f"test_{type_plot}")
    moments_vanilla_histo = load_experiment(name_experiment=f"test_{type_plot}")
    delete_experiment(name_experiment=f"test_{type_plot}")

    type_plot = "vanilla"
    prune_experiment(type_plot=type_plot, name_experiment=f"test_{type_plot}")
    moments_vanilla = load_experiment(name_experiment=f"test_{type_plot}")
    delete_experiment(name_experiment=f"test_{type_plot}")

    assert len(moments_vanilla_histo) == 3
    assert len(moments_vanilla) == 4


def test_prune_bn_ff_experiment(moments_bn_ff_raw, mocker):
    mocker.patch(
        "moments_dnns.manage_experiments.load_experiment",
        return_value=moments_bn_ff_raw,
    )
    type_plot = "bn_ff"
    prune_experiment(type_plot=type_plot, name_experiment=f"test_{type_plot}")
    moments_bn_ff = load_experiment(name_experiment=f"test_{type_plot}")
    delete_experiment(name_experiment=f"test_{type_plot}")

    assert len(moments_bn_ff) == 8


def test_prune_bn_res_experiment(moments_bn_res_raw, mocker):
    mocker.patch(
        "moments_dnns.manage_experiments.load_experiment",
        return_value=moments_bn_res_raw,
    )
    type_plot = "bn_res"
    prune_experiment(type_plot=type_plot, name_experiment=f"test_{type_plot}")
    moments_bn_res = load_experiment(name_experiment=f"test_{type_plot}")
    delete_experiment(name_experiment=f"test_{type_plot}")

    assert len(moments_bn_res) == 10
