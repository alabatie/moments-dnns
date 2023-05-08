import numpy as np
import pytest

from moments_dnns.models import init_ff_model, init_res_model, reset_model
from tests import TEST_BATCH_SIZE, TEST_NUM_CHANNELS


@pytest.fixture
def orig_inputs():
    orig_signal = np.random.normal(size=(TEST_BATCH_SIZE, 1, 1, TEST_NUM_CHANNELS))
    orig_noise = np.random.normal(size=(TEST_BATCH_SIZE, 1, 1, TEST_NUM_CHANNELS))
    orig_log_noise = np.zeros((TEST_BATCH_SIZE, 1, 1, 1))
    return [orig_signal, orig_noise, orig_log_noise]


def test_vanilla(orig_inputs):
    model = init_ff_model(
        spatial_size=1,
        kernel_size=1,
        num_channels=TEST_NUM_CHANNELS,
        boundary="periodic",
        sub_depth=10,
        delta_moments=10,
        name_moments=["mu2_signal", "mu2_noise"],
        batch_norm=False,
    )
    reset_model(model)
    outputs = model.predict(orig_inputs, batch_size=TEST_BATCH_SIZE)
    signal, noise = outputs[:2]

    assert len(outputs) == 9
    assert signal.shape == (TEST_BATCH_SIZE, 1, 1, TEST_NUM_CHANNELS)
    assert noise.shape == (TEST_BATCH_SIZE, 1, 1, TEST_NUM_CHANNELS)
    assert (signal >= 0).all()


def test_bn_ff(orig_inputs):
    model = init_ff_model(
        spatial_size=1,
        kernel_size=1,
        num_channels=TEST_NUM_CHANNELS,
        boundary="periodic",
        sub_depth=10,
        delta_moments=10,
        name_moments=["nu2_signal", "mu2_signal"],
        batch_norm=True,
    )
    reset_model(model)
    outputs = model.predict(orig_inputs, batch_size=TEST_BATCH_SIZE)
    signal, noise, log_noise = outputs[:3]
    noise *= np.sqrt(np.exp(log_noise))

    assert len(outputs) == 11
    assert signal.shape == (TEST_BATCH_SIZE, 1, 1, TEST_NUM_CHANNELS)
    assert noise.shape == (TEST_BATCH_SIZE, 1, 1, TEST_NUM_CHANNELS)
    assert (noise**2).mean() > signal.var()


def test_bn_res(orig_inputs):
    model = init_res_model(
        spatial_size=1,
        kernel_size=1,
        num_channels=TEST_NUM_CHANNELS,
        boundary="periodic",
        sub_depth=10,
        res_depth=2,
        delta_moments=10,
        name_moments=["nu2_signal", "mu2_signal"],
    )
    reset_model(model)
    outputs = model.predict(orig_inputs, batch_size=TEST_BATCH_SIZE)
    signal, noise, log_noise = outputs[:3]
    noise *= np.sqrt(np.exp(log_noise))

    assert len(outputs) == 13
    assert signal.shape == (TEST_BATCH_SIZE, 1, 1, TEST_NUM_CHANNELS)
    assert noise.shape == (TEST_BATCH_SIZE, 1, 1, TEST_NUM_CHANNELS)
    assert (noise**2).mean() > signal.var()
