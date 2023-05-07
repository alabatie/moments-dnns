import tensorflow as tf
from tensorflow.python.keras.layers import Layer

from math import sqrt

# remove tf deprecated warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ConvLayer(Layer):
    """Convolution step in the simultaneous propagation of signal and noise.

    Biases are taken to be zeros as in He initialization.
    When boundary cond. are 'zero_padding', use conv2d with padding = 'same'.
    When boundary cond. are 'periodic' or 'symmetric':
        -> First pad signal and noise
        -> Then use conv2d with padding = 'valid'
    Kernel is stored as attribute 'kernel' and reinitialized for every submodel.
    """

    def __init__(
        self,
        input_size: int,
        kernel_size: int,
        input_channels: int,
        output_channels: int,
        boundary: str,
        strides: int,
        fac_weigths: float = 2.0,
    ):
        """Initialize layer.

        # Args
            input_size: spatial extent of input
            kernel_size: spatial extent of convolutional kernel
            input_channels: number of inputs channels
            output_channels: number of output channels
            boundary: boundary conditions
            strides: strides of convolution
            fac_weigths: variance of weights equal to fac_weights / fan_in
                (default value of fac_weights is 2. as in He initialization)
        """
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.boundary = boundary
        self.padding = (
            "VALID" if (self.boundary in ["periodic", "symmetric"]) else "SAME"
        )
        self.strides = strides
        self.kernel_shape = (
            self.kernel_size,
            self.kernel_size,
            self.input_channels,
            self.output_channels,
        )

        fan_in = self.input_channels * self.kernel_size**2
        std_weights = sqrt(fac_weigths / float(fan_in))
        self.kernel_initializer = tf.compat.v1.keras.initializers.RandomNormal(
            stddev=std_weights
        )
        super(ConvLayer, self).__init__()

    def build(self, input_shape: tuple[int, int, int, int]):
        # create kernel
        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            name="kernel",
            initializer=self.kernel_initializer,
            dtype=tf.float32,
        )
        super(ConvLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape) -> list[tuple]:
        return [
            (
                None,
                self.input_size // self.strides,
                self.input_size // self.strides,
                self.output_channels,
            )
        ] * 2

    def pad_periodic(self, x: tf.Tensor) -> tf.Tensor:
        # pad with periodic boundary conditions
        pad_size = self.kernel_size - 1
        x = tf.concat([x[:, -pad_size:, :, :], x], axis=1)
        x = tf.concat([x[:, :, -pad_size:, :], x], axis=2)
        return x

    def pad_symmetric(self, x: tf.Tensor) -> tf.Tensor:
        # pad with symmetric boundary conditions (kernel size must be odd)
        pad_size = (self.kernel_size - 1) // 2
        x = tf.concat(
            [
                x[:, :pad_size, :, :][:, ::-1, :, :],
                x,
                x[:, -pad_size:, :, :][:, ::-1, :, :],
            ],
            axis=1,
        )
        x = tf.concat(
            [
                x[:, :, :pad_size, :][:, :, ::-1, :],
                x,
                x[:, :, -pad_size:, :][:, :, ::-1, :],
            ],
            axis=2,
        )
        return x

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        signal, noise = inputs
        if (self.boundary == "periodic") and (self.kernel_size > 1):
            signal = self.pad_periodic(signal)
            noise = self.pad_periodic(noise)
        elif (self.boundary == "symmetric") and (self.kernel_size > 1):
            signal = self.pad_symmetric(signal)
            noise = self.pad_symmetric(noise)

        # convolve signal and noise with the same kernel
        signal = tf.nn.conv2d(
            signal,
            self.kernel,
            strides=(self.strides,) * 2,
            padding=self.padding,
            data_format="NHWC",
        )
        noise = tf.nn.conv2d(
            noise,
            self.kernel,
            strides=(self.strides,) * 2,
            padding=self.padding,
            data_format="NHWC",
        )
        return signal, noise


class BatchNormLayer(Layer):
    """Batch Norm step in the simultaneous propagation of signal and noise."""

    def __init__(self, epsilon: float):
        """Initialize layer.

        # Args
            epsilon: fuzz factor of Batch Norm
        """
        self.epsilon = epsilon
        super(BatchNormLayer, self).__init__()

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        signal, noise = inputs
        mean_signal = tf.reduce_mean(signal, axis=(0, 1, 2), keepdims=True)
        centered_signal = signal - mean_signal
        var_signal = tf.reduce_mean(
            tf.pow(centered_signal, 2), axis=(0, 1, 2), keepdims=True
        )

        # signal is centered and normalized,
        # noise is only normalized
        signal = centered_signal / tf.sqrt(var_signal + self.epsilon)
        noise = noise / tf.sqrt(var_signal + self.epsilon)
        return signal, noise


class ActivationLayer(Layer):
    """Activation step in the simultaneous propagation of signal and noise."""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        signal, noise = inputs
        signal_diff = tf.cast(tf.math.greater(signal, 0), tf.float32)

        signal = tf.nn.relu(signal)
        noise = noise * signal_diff
        return signal, noise


class AddLayer(Layer):
    """Addition step in the simultaneous propagation of signal and noise."""

    def compute_output_shape(self, input_shape: list[tuple]) -> list[tuple]:
        return input_shape[:2]

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> tuple[tf.Tensor, tf.Tensor]:
        signal, noise, signal_skip, noise_skip = inputs
        signal = signal + signal_skip
        noise = noise + noise_skip
        return signal, noise
