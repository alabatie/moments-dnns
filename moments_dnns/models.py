from keras.models import Model
from keras.layers import Input
import keras.backend as K

from moments_dnns.propagation_layers import ConvLayer, BatchNormLayer
from moments_dnns.propagation_layers import ActivationLayer, AddLayer
from moments_dnns.computation_layers import MomentsLayer, RescaleLayer


def init_original_model(original_size, kernel_size,
                        original_channels, num_channels,
                        boundary, original_strides):
    """ init_original_model
    Construct the model performing the original convolution
        from (original_size, original_size, original_channels)
        to (original_size // original_strides,
            original_size // original_strides,
            num_channels)
    The convolution is initialized with 'LeCun normal' since no ReLU follows
    In the case kernel_size > 1, original_strides = 2 in order to reduce
        the spatial extent
    In the case kernel_size = 1, original_strides = 1 since images already
        have spatial size equal to 1

    Inputs:
        original_size: spatial extent of original images
        kernel_size: spatial extent of convolutional kernel
        original_channels: number of channels in original images
        num_channels: number of channels in the propagated tensors
        boundary: boundary conditions
        original_strides: strides of convolution

    Outputs:
        [signal, noise] fed later as inputs to the main submodels of the
            simultaneous propagation
    """
    original_shape = (original_size, original_size, original_channels)
    signal = Input(shape=original_shape)
    noise = Input(shape=original_shape)
    inputs = [signal, noise]

    # convolutional layer, initialized with 'LeCun normal'
    conv_layer = ConvLayer(input_size=original_size,
                           kernel_size=kernel_size,
                           input_channels=original_channels,
                           output_channels=num_channels,
                           boundary=boundary,
                           strides=original_strides,
                           fac_weigths=1.)
    signal, noise = conv_layer([signal, noise])

    outputs = [signal, noise]
    return Model(inputs=inputs, outputs=outputs)


def init_ff_model(spatial_size, kernel_size, num_channels, boundary,
                  sub_depth, delta_moments, name_moments_raw,
                  epsilon=0.001, batch_normalization=False):
    """ init_ff_model
    Construct feedforward model
    Moments are computed:
        - every delta_moments layers
        - locs vanilla: 'loc1' -> Conv -> 'loc2' -> Activation -> 'loc3'
        - locs BN_FF: 'loc1' -> Conv -> 'loc2' -> BN -> 'loc3'
            -> Activation -> 'loc4'
        - reff is only computed after activation, else it is set to -1
            since it is not needed for the plots

    Inputs:
        spatial_size: spatial extent of propagated tensors
        kernel_size: spatial extent of convolutional kernel
        num_channels: number of channels in the propagated tensors
        boundary: boundary conditions 'periodic' or 'symmetric'
            or 'zero_padding'
        sub_depth: number of layers inside submodel
        delta_moments: interval between computation of moments
        name_moments_raw: list of raw moments to be computed
        epsilon = batch normalization fuzz factor
            (only relevant if batch_normalization = True)
        batch_normalization: True for 'BN_FF', False for 'vanilla'

    Outputs:
        [signal, noise, log_noise]: serve as inputs to the next submodel
        moments_raw: all moments computed in this submodel
    """
    input_shape = (spatial_size, spatial_size, num_channels)
    signal = Input(shape=input_shape)
    noise = Input(shape=input_shape)
    log_noise = Input(shape=(1, ) * 3)
    inputs = [signal, noise, log_noise]

    moments_raw = []  # list of output moments
    for ilayer in range(1, sub_depth + 1):
        # instantiate layers
        moments_computation = (ilayer % delta_moments) == 0
        moments_layer = MomentsLayer(name_moments_raw,
                                     moments_computation,
                                     reff_computation=False)
        reff_moments_layer = MomentsLayer(name_moments_raw,
                                          moments_computation,
                                          reff_computation=True)
        conv_layer = ConvLayer(input_size=spatial_size,
                               input_channels=num_channels,
                               output_channels=num_channels,
                               kernel_size=kernel_size,
                               boundary=boundary,
                               strides=1)
        batch_norm_layer = BatchNormLayer(epsilon)
        activation_layer = ActivationLayer()
        rescale_layer = RescaleLayer()

        # 'loc1' moments
        moments_raw += moments_layer([signal, noise, log_noise])

        # convolution step
        signal, noise = conv_layer([signal, noise])

        # 'loc2' moments
        moments_raw += moments_layer([signal, noise, log_noise])

        if batch_normalization:
            # batch normalization step
            signal, noise = batch_norm_layer([signal, noise])

            # 'loc3' moments if batch norm is used
            moments_raw += moments_layer([signal, noise, log_noise])

        # activation step
        signal, noise = activation_layer([signal, noise])

        # 'loc4' moments if batch norm is used, otherwise 'loc3' moments
        # only location where we really compute reff
        moments_raw += reff_moments_layer([signal, noise, log_noise])

        # rescale to avoid overflow
        noise, log_noise = rescale_layer([noise, log_noise])

    outputs = [signal, noise, log_noise] + moments_raw
    return Model(inputs=inputs, outputs=outputs)


def init_res_model(spatial_size, kernel_size, num_channels, boundary,
                   sub_depth, res_depth, delta_moments, name_moments_raw,
                   epsilon=0.001):
    """ init_res_model
    Construct resnet model
    For each residual unit:
        - The residual branch goes through a number of feedforward layers
            equal to res_depth
        - The skip-connection branch is the identity
        - Both branches are merged by addition
    Moments:
        - computed every delta_moments residual units, in the first feedforward
            layer of the residual unit and finally at 'loc5' after the addition
        - locations: 'loc1' -> BN -> 'loc2' -> Activation -> 'loc3'
            -> Conv -> 'loc4' -> ... -> 'loc5' (just after the addition)
        - only compute reff after activation, else bypass and return -1
        - rescale noise only when branches are merged

    Inputs:
        spatial_size: spatial extent of propagated tensors
        kernel_size: spatial extent of convolutional kernel
        num_channels: number of channels
        boundary: boundary conditions
        sub_depth: number of residual units in the submodel
        res_depth: total feedforward depth in each residual unit
        delta_moments: interval between computation of moments
        name_moments_raw: list of raw moments to be computed
        epsilon: fuzz factor of batch normalization

    Outputs:
        [signal, noise, log_noise]: serve as inputs to next submodel
        moments_raw: all moments computed in this submodel
    """
    input_shape = (spatial_size, spatial_size, num_channels)
    signal = Input(shape=input_shape)
    noise = Input(shape=input_shape)
    log_noise = Input(shape=(1, ) * 3)
    inputs = [signal, noise, log_noise]

    moments_raw = []   # list of output moments
    for ilayer in range(1, sub_depth + 1):
        # skip-connection branch
        signal_skip, noise_skip = signal, noise

        # residual branch
        for ires in range(1, res_depth + 1):
            # instantiate layers
            moments_computation_unit = (ilayer % delta_moments) == 0
            moments_computation_res = moments_computation_unit and (ires == 1)
            moments_layer = MomentsLayer(name_moments_raw,
                                         moments_computation_res,
                                         reff_computation=False)
            reff_moments_layer = MomentsLayer(name_moments_raw,
                                              moments_computation_res,
                                              reff_computation=True)
            conv_layer = ConvLayer(input_size=spatial_size,
                                   input_channels=num_channels,
                                   output_channels=num_channels,
                                   kernel_size=kernel_size,
                                   boundary=boundary,
                                   strides=1)
            batch_norm_layer = BatchNormLayer(epsilon)
            activation_layer = ActivationLayer()

            # 'loc1' moments
            moments_raw += moments_layer([signal, noise, log_noise])

            # batch normalization step
            signal, noise = batch_norm_layer([signal, noise])

            # 'loc2' moments
            moments_raw += moments_layer([signal, noise, log_noise])

            # activation step
            signal, noise = activation_layer([signal, noise])

            # 'loc3' moments, only location where we really compute reff
            moments_raw += reff_moments_layer([signal, noise, log_noise])

            # convolution step
            signal, noise = conv_layer([signal, noise])

            # loc4' moments
            moments_raw += moments_layer([signal, noise, log_noise])

        # merge branches
        signal, noise = AddLayer()([signal, noise, signal_skip, noise_skip])

        # 'loc5' moments
        moments_layer = MomentsLayer(name_moments_raw,
                                     moments_computation_unit,
                                     reff_computation=False)
        moments_raw += moments_layer([signal, noise, log_noise])

        # rescale to avoid overflow (must happen when all branches are merged)
        noise, log_noise = RescaleLayer()([noise, log_noise])

    outputs = [signal, noise, log_noise] + moments_raw
    return Model(inputs=inputs, outputs=outputs)


def reset_model(model):
    """ reset_model
    Reinitialize model
    Since only convolutional layers contain random parameters in our analysis:
        -> Loop through all layers
        -> reinitialize “kernel“ attribute of each convolutional layer
    """
    session = K.get_session()
    for layer in model.layers:
        if isinstance(layer, ConvLayer):
            # reinitialize kernel attribute
            layer.kernel.initializer.run(session=session)
