"""
APIs for continuous spaces.
"""

import math

import numpy as np
import tensorflow as tf

from .base import Distribution

class BoxGaussian(Distribution):
    """
    A probability distribution over continuous variables,
    parameterized as a diagonal gaussian.
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high

    @property
    def out_shape(self):
        return self.low.shape

    def to_vecs(self, space_elements):
        return np.array(space_elements)

    @property
    def param_shape(self):
        return self.low.shape + (2,)

    def sample(self, param_batch):
        params = np.array(param_batch)
        means, log_stddevs = self._mean_and_log_stddevs(params)
        stddevs = np.exp(log_stddevs)
        return np.random.normal(loc=means, scale=stddevs)

    def log_prob(self, param_batch, sample_vecs):
        means, log_stddevs = self._mean_and_log_stddevs(param_batch)
        constant_factor = 0.5 * math.log(2 * math.pi)
        diff = 0.5 * tf.square((means - sample_vecs) / tf.exp(log_stddevs))
        neg_log_probs = constant_factor + log_stddevs + diff
        return _reduce_sums(tf.negative(neg_log_probs))

    def entropy(self, param_batch):
        _, log_stddevs = self._mean_and_log_stddevs(param_batch)
        constant_factor = 0.5 * (math.log(2 * math.pi) + 1)
        return _reduce_sums(constant_factor + log_stddevs)

    def kl_divergence(self, param_batch_1, param_batch_2):
        means_1, log_stddevs_1 = self._mean_and_log_stddevs(param_batch_1)
        means_2, log_stddevs_2 = self._mean_and_log_stddevs(param_batch_2)
        # log(s2/s1) + (s1^2 + (u1 - u2)^2)/(2*s2^2) - 0.5
        term_1 = log_stddevs_2 - log_stddevs_1
        term_2_num = tf.exp(2 * log_stddevs_1) + tf.square(means_1 - means_2)
        term_2_denom = 2 * tf.exp(2 * log_stddevs_2)
        return _reduce_sums(term_1 + term_2_num/term_2_denom - 0.5)

    def _mean_and_log_stddevs(self, param_batch):
        """
        Compute the means and variances for a batch of
        parameters.
        """
        means = param_batch[..., 0]
        log_stddevs = param_batch[..., 1]
        bias = (self.high + self.low) / 2
        scale = (self.high - self.low) / 2
        return means + bias, log_stddevs + np.log(scale)

def _reduce_sums(batch):
    """
    Reduce a batch of shape [batch x out_shape] to a
    batch of scalars.
    """
    dims = list(range(1, len(batch.shape)))
    return tf.reduce_sum(batch, axis=dims)
