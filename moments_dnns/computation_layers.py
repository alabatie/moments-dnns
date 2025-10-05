"""Layers of moment computation."""
import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class MomentsLayer(Layer):
    """Compute raw moments at given layer and given 'loc'.

    # Computes
        'nu1_abs_signal': 1st-order non-central moment of abs value of signal
        'nu2_signal': 2nd-order non-central moment of signal
        'nu4_signal': 4th-order non-central moment of signal
        'mu2_signal': 2nd-order central moment of signal
        'mu4_signal': 4th-order central moment of signal
        'mu2_noise': 2nd-order central moment of noise
            -> also equal 2nd-order non-central moment since noise is centered
            -> to avoid overflow inside model, it is computed in log space
        'reff_signal': effective rank of signal
        'reff_noise': effective rank of noise
    """

    # pylint: disable=abstract-method

    def __init__(
        self, name_moments: list[str], compute_moments: bool, compute_reff: bool
    ):
        """Init layer.

        # Args
            name_moments: names of raw moments to be computed
            compute_moments: if True compute moments, else return []
            compute_reff: if False, bypass reff computation by returning -1
                (computational bottleneck)
        """
        super().__init__()
        self.name_moments = name_moments
        self.compute_moments = compute_moments
        self.compute_reff = compute_reff
        self.num_moments = len(self.name_moments)

    def compute_output_shape(self, input_shape: list[tuple]) -> list[tuple]:
        """Return output shapes."""
        return (
            [(None,) for _ in range(self.num_moments)] if self.compute_moments else []
        )

    def compute_effective_rank(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Compute effective rank."""
        if self.compute_reff:
            # fetch feature vectors from every input x, dx and spatial position
            feat_vecs = tf.reshape(input_tensor, (-1, tf.shape(input_tensor)[-1]))
            centered_feat_vecs = feat_vecs - tf.reduce_mean(
                feat_vecs, axis=0, keepdims=True
            )

            # singular value decomposition
            sing_vals = tf.linalg.svd(centered_feat_vecs, compute_uv=False)

            # eig. values of covariance matrix are sing. values squared
            eig_vals = tf.pow(sing_vals, 2)
            reff = tf.reduce_sum(eig_vals) / tf.reduce_max(eig_vals)
        else:
            reff = tf.constant([-1], dtype=tf.float32)

        return reff

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], *args, **kwargs
    ) -> tf.Tensor:
        """Call layer."""
        signal, noise, log_noise = inputs
        centered_signal = signal - tf.reduce_mean(signal, axis=[0, 1, 2], keepdims=True)
        log_noise = tf.reduce_mean(log_noise)  # squeeze dimensions

        moments = []  # stores all moments computed
        name_moments = self.name_moments if self.compute_moments else []
        for name_moment in name_moments:
            match name_moment:
                case "nu1_abs_signal":
                    moment = tf.reduce_mean(tf.abs(signal))
                case "nu2_signal":
                    moment = tf.reduce_mean(tf.pow(signal, 2))
                case "nu4_signal":
                    moment = tf.reduce_mean(tf.pow(signal, 4))
                case "mu2_signal":
                    moment = tf.reduce_mean(tf.pow(centered_signal, 2))
                case "mu4_signal":
                    moment = tf.reduce_mean(tf.pow(centered_signal, 4))
                case "mu2_noise":
                    # computation in log scale here to avoid overflow
                    # noise is always centered -> mu2_noise = nu2_noise
                    moment = tf.math.log(tf.reduce_mean(tf.pow(noise, 2))) + log_noise
                case "reff_signal":
                    moment = self.compute_effective_rank(signal)
                case "reff_noise":
                    moment = self.compute_effective_rank(noise)
                case _:
                    raise NotImplementedError()
            moments.append(tf.reshape(moment, (1,)))

        return moments


class RescaleLayer(Layer):
    """Rescale noise to avoid overflow inside model.

    Log of mu2_noise is then stored in log_noise
        (this value is reused afterwards in the computation of moments)
    """

    # pylint: disable=abstract-method

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], *args, **kwargs):
        """Call layer."""
        noise, log_noise = inputs
        mu2_noise = tf.reduce_mean(tf.pow(noise, 2), keepdims=True)

        # rescale noise to avoid overflow
        log_noise += tf.math.log(mu2_noise)
        noise = noise / tf.sqrt(mu2_noise)
        return noise, log_noise
