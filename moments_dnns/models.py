"""Model initialization and management."""
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

from moments_dnns.computation_layers import MomentsLayer, RescaleLayer
from moments_dnns.propagation_layers import (
    ActivationLayer,
    AddLayer,
    BatchNormLayer,
    ConvLayer,
)


def init_orig_model(
    orig_size: int,
    kernel_size: int,
    orig_channels: int,
    num_channels: int,
    boundary: str,
    orig_strides: int,
) -> Model:
    """Initialize model performing an original convolution
        from (orig_size, orig_size, orig_channels)
        to (orig_size // orig_strides, orig_size // orig_strides, num_channels).

    The convolution is initialized with 'LeCun normal' since no ReLU follows.
    When kernel_size > 1, orig_strides = 2 to reduce spatial extent.
    When kernel_size = 1, orig_strides = 1 since images already have spatial size of 1.

    # Args
        orig_size: spatial extent of original images
        kernel_size: spatial extent of convolutional kernel
        orig_channels: number of channels in original images
        num_channels: number of channels in the propagated tensors
        boundary: boundary conditions
        orig_strides: strides of convolution

    # Returns
        [signal, noise]
    """
    orig_shape = (orig_size, orig_size, orig_channels)
    signal = Input(shape=orig_shape)
    noise = Input(shape=orig_shape)
    inputs = [signal, noise]

    # convolutional layer, initialized with 'LeCun normal'
    conv_layer = ConvLayer(
        input_size=orig_size,
        kernel_size=kernel_size,
        input_channels=orig_channels,
        output_channels=num_channels,
        boundary=boundary,
        strides=orig_strides,
        fac_weigths=1.0,
    )
    signal, noise = conv_layer([signal, noise])

    outputs = [signal, noise]
    return Model(inputs=inputs, outputs=outputs)


def init_ff_model(
    spatial_size: int,
    kernel_size: int,
    num_channels: int,
    boundary: str,
    sub_depth: int,
    delta_moments: int,
    name_moments: list[str],
    epsilon: float = 0.001,
    batch_norm: bool = False,
) -> Model:
    """Initialize feedforward model.

    # Computations
        - Every delta_moments layers
        - Locs vanilla: 'loc1' -> Conv -> 'loc2' -> Activation -> 'loc3'
        - Locs bn_ff: 'loc1' -> Conv -> 'loc2' -> BN -> 'loc3' -> Activation -> 'loc4'
        - Effective rank is only computed after activation, else it is set to -1
            since it is not needed for the plots

    # Args
        spatial_size: spatial extent of propagated tensors
        kernel_size: spatial extent of convolutional kernel
        num_channels: number of channels in the propagated tensors
        boundary: boundary conditions 'periodic' or 'symmetric'
            or 'zero_padding'
        sub_depth: number of layers inside submodel
        delta_moments: interval between computation of moments
        name_moments: names of raw moments to be computed
        epsilon: fuzz factor of Batch Norm
            (only relevant if batch_norm = True)
        batch_norm: True for 'bn_ff', False for 'vanilla'
    """
    input_shape = (spatial_size, spatial_size, num_channels)
    signal = Input(shape=input_shape)
    noise = Input(shape=input_shape)
    log_noise = Input(shape=(1,) * 3)
    inputs = [signal, noise, log_noise]

    moments = []  # list of output moments
    for ilayer in range(1, sub_depth + 1):
        # instantiate layers
        compute_moments = (ilayer % delta_moments) == 0
        moments_layer = MomentsLayer(name_moments, compute_moments, compute_reff=False)
        reff_moments_layer = MomentsLayer(
            name_moments, compute_moments, compute_reff=True
        )
        conv_layer = ConvLayer(
            input_size=spatial_size,
            input_channels=num_channels,
            output_channels=num_channels,
            kernel_size=kernel_size,
            boundary=boundary,
            strides=1,
        )

        # 'loc1' moments
        moments += moments_layer([signal, noise, log_noise])

        # convolution step
        signal, noise = conv_layer([signal, noise])

        # 'loc2' moments
        moments += moments_layer([signal, noise, log_noise])

        if batch_norm:
            # batch norm step
            signal, noise = BatchNormLayer(epsilon)([signal, noise])

            # 'loc3' moments if batch norm is used
            moments += moments_layer([signal, noise, log_noise])

        # activation step
        signal, noise = ActivationLayer()([signal, noise])

        # 'loc4' moments if batch norm is used, otherwise 'loc3' moments
        # only location where we really compute reff
        moments += reff_moments_layer([signal, noise, log_noise])

        # rescale to avoid overflow
        noise, log_noise = RescaleLayer()([noise, log_noise])

    outputs = [signal, noise, log_noise] + moments
    return Model(inputs=inputs, outputs=outputs)


def init_res_model(
    spatial_size: int,
    kernel_size: int,
    num_channels: int,
    boundary: str,
    sub_depth: int,
    res_depth: int,
    delta_moments: int,
    name_moments: list[str],
    epsilon: float = 0.001,
) -> Model:
    """Initialize ResNet model.

    For each residual unit, residual branch goes through res_depth feedforward layers.

    # Computations
        - every delta_moments residual units, in the first ff layer
            of the residual unit and finally at 'loc5'
        - locs: 'loc1' -> BN -> 'loc2' -> Activation -> 'loc3'
            -> Conv -> 'loc4' -> ... -> 'loc5' (just after the addition)
        - only compute reff after activation, else bypass and return -1
        - rescale noise when branches are merged

    # Args
        spatial_size: spatial extent of propagated tensors
        kernel_size: spatial extent of convolutional kernel
        num_channels: number of channels
        boundary: boundary conditions
        sub_depth: number of residual units in the submodel
        res_depth: total ff depth in each residual unit
        delta_moments: interval between computation of moments
        name_moments: names of raw moments to be computed
        epsilon: fuzz factor of Batch Norm
    """
    input_shape = (spatial_size, spatial_size, num_channels)
    signal = Input(shape=input_shape)
    noise = Input(shape=input_shape)
    log_noise = Input(shape=(1,) * 3)
    inputs = [signal, noise, log_noise]

    moments = []  # list of output moments
    for ilayer in range(1, sub_depth + 1):
        # skip-connection branch
        signal_skip, noise_skip = signal, noise

        # residual branch
        for ires in range(1, res_depth + 1):
            # instantiate layers
            compute_moments_unit = (ilayer % delta_moments) == 0
            compute_moments_res = compute_moments_unit and (ires == 1)
            moments_layer = MomentsLayer(
                name_moments, compute_moments_res, compute_reff=False
            )
            reff_moments_layer = MomentsLayer(
                name_moments, compute_moments_res, compute_reff=True
            )
            conv_layer = ConvLayer(
                input_size=spatial_size,
                input_channels=num_channels,
                output_channels=num_channels,
                kernel_size=kernel_size,
                boundary=boundary,
                strides=1,
            )

            # 'loc1' moments
            moments += moments_layer([signal, noise, log_noise])

            # batch norm step
            signal, noise = BatchNormLayer(epsilon)([signal, noise])

            # 'loc2' moments
            moments += moments_layer([signal, noise, log_noise])

            # activation step
            signal, noise = ActivationLayer()([signal, noise])

            # 'loc3' moments, only location where we really compute reff
            moments += reff_moments_layer([signal, noise, log_noise])

            # convolution step
            signal, noise = conv_layer([signal, noise])

            # loc4' moments
            moments += moments_layer([signal, noise, log_noise])

        # merge branches
        signal, noise = AddLayer()([signal, noise, signal_skip, noise_skip])

        # 'loc5' moments
        moments_layer = MomentsLayer(
            name_moments, compute_moments_unit, compute_reff=False
        )
        moments += moments_layer([signal, noise, log_noise])

        # rescale to avoid overflow (must happen when all branches are merged)
        noise, log_noise = RescaleLayer()([noise, log_noise])

    outputs = [signal, noise, log_noise] + moments
    return Model(inputs=inputs, outputs=outputs)


def reset_model(model: Model):
    """Reinitialize model parameters.

    Since only convolutional layers contain random parameters in our analysis:
        - Loop through all layers
        - Reinitialize 'kernel' attribute of each convolutional layer
    """
    for layer in model.layers:
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))
