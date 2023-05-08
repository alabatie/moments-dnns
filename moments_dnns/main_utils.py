"""Utils to run experiments."""
import numpy as np
import tensorflow as tf


def check_args(
    architecture: str,
    kernel_size: int,
    total_depth: int,
    num_computations: int,
    num_channels: int,
    boundary: str,
    dataset: str,
    batch_size: int,
):
    """Check that experiment's arguments are valid.

    # Conditions
        - kernel_size, num_channels, total_depth, batch_size must be integers
        - architecture must be 'vanilla' or 'bn_ff' or 'bn_res'
        - dataset must be 'cifar10' or 'mnist'
        - boundary must be  'periodic' or 'symmetric' or 'zero_padding'
        - 'symmetric' boundary rae only compatible with odd kernel size
        - total depth must be a multiple of the number of moment computations
    """
    if not all(
        isinstance(arg, int)
        for arg in [kernel_size, num_channels, total_depth, batch_size]
    ):
        raise ValueError(
            "kernel_size, num_channels, total_depth, and batch_size must be integers"
        )

    if architecture not in {"vanilla", "bn_ff", "bn_res"}:
        raise ValueError("Architecture must be 'vanilla' or 'bn_ff' or 'bn_res'")

    if dataset not in {"cifar10", "mnist"}:
        raise ValueError("Dataset must be 'cifar10' or 'mnist'")

    if boundary not in {"periodic", "symmetric", "zero_padding"}:
        raise ValueError("Boundary must be 'periodic' or 'symmetric' or 'zero_padding'")

    if boundary == "symmetric" and kernel_size % 2 == 0:
        raise ValueError(
            "Symmetric boundary conditions are only compatible with odd kernel size"
        )

    if total_depth % num_computations != 0:
        raise ValueError(
            "Total depth must be a multiple of the number of moment computations"
        )


def get_submodel_constants(
    orig_size: int, orig_strides: int, total_depth: int, num_computations: int
) -> tuple[int, int, int, int]:
    """Compute constants for submodel.

    # Args
      orig_size: spatial extent of original images
      orig_strides strides of first downsampling conv layer
      total_depth: total depth of the experiment
      num_computations: total number of moment computations

    # Returns
      spatial_size: spatial size of images in submodel
      num_submodels: number of submodels subdividing the total depth
          - each time the same Keras model is reused as submodel
          - each time it is randomly re-initialized
          - this leads to exactly the same behaviour as a randomly
              initialized model of depth equal to total_depth
          - but it requires less memory
      sub_depth: submodel depth
      delta_moments: interval between computation of moments
    """
    # num_submodels = 10 if 10 divides both num_computations and total_depth,
    # otherwise num_submodels = num_computations
    num_submodels = (
        10
        if ((num_computations % 10 == 0) and (total_depth % 10 == 0))
        else num_computations
    )

    spatial_size = orig_size // orig_strides
    sub_depth = total_depth // num_submodels
    delta_moments = total_depth // num_computations

    return spatial_size, num_submodels, sub_depth, delta_moments


def get_name_moments(
    architecture: str, compute_reff_signal: bool, compute_reff_noise: bool
) -> tuple[list[str], list[str], int, int]:
    """Create list of moment names.

    # Creates
        list of raw moments to be computed.
        list of locs, depending on the architecture:
            - vanilla: ['loc1', 'loc2', 'loc3']
            - bn_ff:   ['loc1', 'loc2', 'loc3', 'loc4']
            - bn_res:  ['loc1', 'loc2', 'loc3', 'loc4', 'loc5']

    # Args
        architecture: 'vanilla' or 'bn_ff' or 'bn_res'
        compute_reff_signal: whether reff is computed for signal
        compute_reff_noise: whether reff is computed for noise

    # Returns
        name_moments: names of raw (i.e. without locs) moments
        locs: locs
        num_moments: number of raw moments
        num_moments_loc: total number of moments
            (equals number of raw moments * number of locs)
    """
    name_moments = [
        "nu1_abs_signal",
        "nu2_signal",
        "mu2_signal",
        "mu4_signal",
        "mu2_noise",
    ]
    if compute_reff_signal:
        name_moments += ["reff_signal"]
    if compute_reff_noise:
        name_moments += ["reff_noise"]
    num_moments = len(name_moments)

    match architecture:
        case "vanilla":
            num_locs = 3
        case "bn_ff":
            num_locs = 4
        case "bn_res":
            num_locs = 5
        case _:
            raise NotImplementedError
    locs = ["loc" + str(iloc) for iloc in range(1, num_locs + 1)]
    num_moments_loc = num_locs * num_moments

    return name_moments, locs, num_moments, num_moments_loc


def load_dataset(
    dataset: str, kernel_size: int
) -> tuple[np.ndarray, int, int, int, int]:
    """Load_dataset.

    Cifar images are 32 x 32 x 3.
    Mnist images are 28 x 28, and must be reshaped to 28 x 28 x 1.
    When kernel_size = 1, images are flattened to have spatial size n = 1
        (fully-connected case)

     # Args
        dataset: 'cifar1O' or 'mnist'
        kernel_size: used to deal with the fully-connected case

     # Returns
        signal_orig: suitably reshaped original images
        orig_strides: strides of first downsampling conv layer
            (= 2 except in the fully-connected case)
        orig_num: number of original images
        orig_spatial: spatial size of original images
        orig_channels: number of channels in original images
    """
    match dataset:
        case "cifar10":
            (signal_orig, _), (_, _) = tf.keras.datasets.cifar10.load_data()
        case "mnist":
            (signal_orig, _), (_, _) = tf.keras.datasets.mnist.load_data()
            signal_orig = np.expand_dims(signal_orig, -1)
        case _:
            raise NotImplementedError()

    # if kernel_size = 1 (fully-connected case), then flatten inputs
    if kernel_size == 1:
        signal_orig = signal_orig.reshape((signal_orig.shape[0], 1, 1, -1))

    orig_strides = 2 if (kernel_size > 1) else 1  # strides of first conv
    orig_num = signal_orig.shape[0]  # number of original images
    orig_spatial = signal_orig.shape[1]  # original spatial extent
    orig_channels = signal_orig.shape[-1]  # original num channels

    return (
        signal_orig,
        orig_strides,
        orig_num,
        orig_spatial,
        orig_channels,
    )
