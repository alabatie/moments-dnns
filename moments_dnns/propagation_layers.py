import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

from math import sqrt

# remove tf deprecated warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ConvLayer(Layer):
    """ ConvLayer
    Convolution step in the simultaneous propagation of signal and noise
    Biases are taken to be zeros as in He initialization
    When boundary cond. are 'zero_padding', use conv2d with padding = 'same'
    When boundary cond. are 'periodic' or 'symmetric':
        -> first pad signal and noise
        -> then use conv2d with padding = 'valid'
    Kernel is stored as attribute 'kernel' and reinitialized for every submodel

    # Initialization
        input_size (int): spatial extent of input
        kernel_size (int): spatial extent of convolutional kernel
        input_channels (int): number of inputs channels
        output_channels (int): number of output channels
        boundary (str): boundary conditions
        strides (int): strides of convolution
        fac_weigths (float): variance of weights equal to fac_weights / fan_in
            (default value of fac_weights is 2. as in He initialization)

    # Arguments
        [signal, noise]

    # Returns
        [signal, noise]
    """
    def __init__(self, input_size, kernel_size, input_channels,
                 output_channels, boundary, strides, fac_weigths=2.):
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.boundary = boundary
        self.padding = 'valid' \
            if (self.boundary in ['periodic', 'symmetric']) else 'same'
        self.strides = strides
        self.kernel_shape = (self.kernel_size,
                             self.kernel_size,
                             self.input_channels,
                             self.output_channels)

        fan_in = self.input_channels * self.kernel_size**2
        std_weights = sqrt(fac_weigths / float(fan_in))
        self.kernel_initializer = \
            tf.compat.v1.keras.initializers.RandomNormal(stddev=std_weights)
        super(ConvLayer, self).__init__()

    def build(self, input_shape):
        # create kernel
        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      dtype=K.floatx())
        super(ConvLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return [(None,
                 self.input_size // self.strides,
                 self.input_size // self.strides,
                 self.output_channels)] * 2

    def pad_periodic(self, x):
        # pad with periodic boundary conditions
        pad_size = self.kernel_size - 1
        x = K.concatenate([x[:, -pad_size:, :, :], x], axis=1)
        x = K.concatenate([x[:, :, -pad_size:, :], x], axis=2)
        return x

    def pad_symmetric(self, x):
        # pad with symmetric boundary conditions (kernel size must be odd)
        pad_size = (self.kernel_size - 1) // 2
        x = K.concatenate([x[:, :pad_size, :, :][:, ::-1, :, :], x,
                           x[:, -pad_size:, :, :][:, ::-1, :, :]], axis=1)
        x = K.concatenate([x[:, :, :pad_size, :][:, :, ::-1, :], x,
                           x[:, :, -pad_size:, :][:, :, ::-1, :]], axis=2)
        return x

    def call(self, inputs):
        signal, noise = inputs
        if (self.boundary == 'periodic') and (self.kernel_size > 1):
            signal = self.pad_periodic(signal)
            noise = self.pad_periodic(noise)
        elif (self.boundary == 'symmetric') and (self.kernel_size > 1):
            signal = self.pad_symmetric(signal)
            noise = self.pad_symmetric(noise)

        # convolve signal and noise with the same kernel
        signal = K.conv2d(signal,
                          self.kernel,
                          strides=(self.strides, ) * 2,
                          padding=self.padding,
                          data_format='channels_last')
        noise = K.conv2d(noise,
                         self.kernel,
                         strides=(self.strides, ) * 2,
                         padding=self.padding,
                         data_format='channels_last')
        return [signal, noise]


class BatchNormLayer(Layer):
    """ BatchNormLayer
    Batch norm step in the simultaneous propagation of signal and noise
        -> signal is centered and normalized
        -> noise is normalized
        -> each channel is normalized by sqrt(var_signal + epsilon)

    # Initialization
        epsilon (float): batch norm fuzz factor

    # Arguments
        [signal, noise]

    # Returns
        [signal, noise]
    """
    def __init__(self, epsilon):
        self.K_epsilon = K.constant(epsilon)
        super(BatchNormLayer, self).__init__()

    def call(self, inputs):
        signal, noise = inputs
        mean_signal = K.mean(signal, axis=(0, 1, 2), keepdims=True)
        centered_signal = signal - mean_signal
        var_signal = K.mean(K.pow(centered_signal, 2),
                            axis=(0, 1, 2), keepdims=True)

        # signal is centered and normalized,
        # noise is only normalized
        signal = centered_signal / K.sqrt(var_signal + self.K_epsilon)
        noise = noise / K.sqrt(var_signal + self.K_epsilon)
        return [signal, noise]


class ActivationLayer(Layer):
    """ ActivationLayer
    Activation step in the simultaneous propagation of signal and noise
        - signal goes through relu
        - noise goes through element-wise multiplication by relu derivative

    # Arguments
        [signal, noise]

    # Returns
        [signal, noise]
    """
    def call(self, inputs):
        signal, noise = inputs
        signal_diff = K.cast(K.greater(signal, K.constant(0.0)), K.floatx())

        signal = K.relu(signal)
        noise = noise * signal_diff
        return [signal, noise]


class AddLayer(Layer):
    """ AddLayer
    Addition step in the simultaneous propagation of signal and noise
        (only used in resnets)

    # Arguments
        [signal, noise, signal_skip, noise_skip]

    # Returns
        [signal, noise]
    """
    def compute_output_shape(self, input_shape):
        return input_shape[:2]

    def call(self, inputs):
        signal, noise, signal_skip, noise_skip = inputs
        signal = signal + signal_skip
        noise = noise + noise_skip
        return [signal, noise]
