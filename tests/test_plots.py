import pytest
import numpy as np

from moments_dnns import ROOT_DIR
from moments_dnns.plots import plot_vanilla, plot_bn_ff, plot_bn_res
from moments_dnns.plot_utils import delete_figure
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


def test_plot_vanilla(moments_vanilla_raw):
    name_fig = "test_vanilla_plot"
    plot_vanilla(moments_vanilla_raw, use_tex=False, name_fig=name_fig)
    is_pdf = (ROOT_DIR / "figures" / "pdf" / f"{name_fig}.pdf").is_file()
    is_png = (ROOT_DIR / "figures" / "png" / f"{name_fig}.png").is_file()
    delete_figure(name_fig=name_fig)

    assert is_pdf
    assert is_png


def test_plot_bn_ff(moments_bn_ff_raw):
    name_fig = "test_bn_ff_plot"
    plot_bn_ff(moments_bn_ff_raw, use_tex=False, name_fig=name_fig)
    is_pdf = (ROOT_DIR / "figures" / "pdf" / f"{name_fig}.pdf").is_file()
    is_png = (ROOT_DIR / "figures" / "png" / f"{name_fig}.png").is_file()
    delete_figure(name_fig=name_fig)

    assert is_pdf
    assert is_png


def test_plot_bn_res(moments_bn_res_raw):
    name_fig = "test_bn_res_plot"
    plot_bn_res(moments_bn_res_raw, use_tex=False, name_fig=name_fig)
    is_pdf = (ROOT_DIR / "figures" / "pdf" / f"{name_fig}.pdf").is_file()
    is_png = (ROOT_DIR / "figures" / "png" / f"{name_fig}.png").is_file()
    delete_figure(name_fig=name_fig)

    assert is_pdf
    assert is_png
