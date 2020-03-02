import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class MomentsLayer(Layer):
    """ MomentsLayer
    Compute raw moments at given layer and given 'loc'

    # Computations
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

    # Initialization
        name_moment_raw (list): names of raw moments to be computed
        compute_moments (bool): if True compute raw moments, else return []
        bypass_reff (bool): if True bypass reff computation by returning -1
            (computational bottleneck)

    # Arguments
        [signal, noise, log_noise]

    # Returns
        moments_raw
    """
    def __init__(self, name_moments_raw, moments_computation,
                 reff_computation):
        self.name_moments_raw = name_moments_raw
        self.num_moments_raw = len(self.name_moments_raw)

        # - if compute_moments = False, no computation will take place
        # - if compute_moments = True and compute_reff = False,
        #     moments will be computed but effective ranks won't be computed
        # - if compute_moments = True and compute_reff = True,
        #     both moments and effective ranks will be computed
        self.moments_computation = moments_computation
        self.reff_computation = reff_computation
        super(MomentsLayer, self).__init__()

    def compute_output_shape(self, input_shape):
        return [(None, ) for _ in range(self.num_moments_raw)] \
            if self.moments_computation else []

    def compute_reff(self, x):
        if self.reff_computation:
            # fetch feature vectors from every input x, dx and spatial position
            feat_vecs = K.reshape(x, (-1, K.shape(x)[-1]))
            mean_feat_vecs = K.mean(feat_vecs, axis=0, keepdims=True)
            centered_feat_vecs = feat_vecs - mean_feat_vecs

            # singular value decomposition
            sing_vals = tf.linalg.svd(centered_feat_vecs, compute_uv=False)

            # eig. values of covariance matrix are sing. values squared
            eig_vals = K.pow(sing_vals, 2)
            reff = K.sum(eig_vals) / K.max(eig_vals)
        else:
            reff = K.constant(-1)

        return reff

    def call(self, inputs):
        signal, noise, log_noise = inputs
        moments_raw = []  # stores all moments computed
        if self.moments_computation:
            mean_signal = K.mean(signal, axis=[0, 1, 2], keepdims=True)
            centered_signal = signal - mean_signal
            log_noise = K.mean(log_noise)  # squeeze dimensions

            for name_moment_raw in self.name_moments_raw:
                if name_moment_raw == 'nu1_abs_signal':
                    moment_raw = K.mean(K.abs(signal))
                elif name_moment_raw == 'nu2_signal':
                    moment_raw = K.mean(K.pow(signal, 2))
                elif name_moment_raw == 'nu4_signal':
                    moment_raw = K.mean(K.pow(signal, 4))
                elif name_moment_raw == 'mu2_signal':
                    moment_raw = K.mean(K.pow(centered_signal, 2))
                elif name_moment_raw == 'mu4_signal':
                    moment_raw = K.mean(K.pow(centered_signal, 4))
                elif name_moment_raw == 'mu2_noise':
                    # computation in log scale here to avoid overflow
                    # noise is always centered -> mu2_noise = nu2_noise
                    moment_raw = K.log(K.mean(K.pow(noise, 2))) + log_noise
                elif name_moment_raw == 'reff_signal':
                    moment_raw = self.compute_reff(signal)
                elif name_moment_raw == 'reff_noise':
                    moment_raw = self.compute_reff(noise)
                else:
                    raise NotImplementedError()
                moments_raw.append(K.reshape(moment_raw, (1, )))

        return moments_raw


class RescaleLayer(Layer):
    """ RescaleLayer
    Rescale noise to avoid overflow inside model
    Log of mu2_noise stored in log_noise
        (this value is reused afterwards in the computation of moments)

    # Arguments
        [noise, log_noise]

    # Returns
        [noise, log_noise]
    """
    def call(self, inputs):
        noise, log_noise = inputs
        mu2_noise = K.mean(K.pow(noise, 2), keepdims=True)

        # rescale noise to avoid overflow, store current value in log_noise
        log_noise += K.log(mu2_noise)
        noise = noise / K.sqrt(mu2_noise)
        return [noise, log_noise]
