"""
APIs for continuous spaces.
"""

import math

import numpy as np
import tensorflow as tf

from .base import Distribution, Vectorizer


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

    def mode(self, param_batch):
        params = np.array(param_batch)
        return self._mean_and_log_stddevs(params)[0]

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
        return means * scale + bias, log_stddevs + np.log(scale)


class BoxBeta(Distribution):
    """
    A probability distribution over continuous variables,
    parameterized as a component-wise scaled beta
    distribution.

    By default, inputs to the distribution are fed through
    `1 + softplus` to ensure that they are valid.
    """

    def __init__(self, low, high, softplus=True):
        self.low = low
        self.high = high
        self.softplus = softplus

    @property
    def out_shape(self):
        return self.low.shape

    def to_vecs(self, space_elements):
        return np.array(space_elements)

    @property
    def param_shape(self):
        return self.low.shape + (2,)

    def sample(self, param_batch):
        params = self._squash_inputs(np.array(param_batch))
        raw = np.random.beta(params[..., 0], params[..., 1])
        return raw * (self.high - self.low) + self.low

    def mode(self, param_batch):
        params = self._squash_inputs(np.array(param_batch))
        alpha, beta = params[..., 0], params[..., 1]
        raw = (alpha - 1) / (alpha + beta - 2)
        return raw * (self.high - self.low) + self.low

    def log_prob(self, param_batch, sample_vecs):
        scaled_samples = (sample_vecs - self.low) / (self.high - self.low)
        epsilon = 1e-20
        scaled_samples = tf.clip_by_value(scaled_samples, 0+epsilon, 1-epsilon)
        raw_probs = self._create_dist(param_batch).log_prob(scaled_samples)
        return _reduce_sums(raw_probs - np.log(self.high - self.low))

    def entropy(self, param_batch):
        raw_ents = self._create_dist(param_batch).entropy()
        return _reduce_sums(raw_ents + np.log(self.high - self.low))

    def kl_divergence(self, param_batch_1, param_batch_2):
        # KL is scale-invariant.
        return _reduce_sums(tf.contrib.distributions.kl_divergence(
            self._create_dist(param_batch_1),
            self._create_dist(param_batch_2)))

    def _create_dist(self, param_batch):
        params = self._squash_inputs(param_batch)
        return tf.contrib.distributions.Beta(params[..., 0], params[..., 1])

    def _squash_inputs(self, inputs):
        if not self.softplus:
            return inputs
        if isinstance(inputs, np.ndarray):
            softplus = np.log(1 + np.exp(inputs))
            non_linear = (inputs < 30)
            return 1 + np.where(non_linear, softplus, inputs)
        return 1 + tf.nn.softplus(inputs)


class BoxStacker(Vectorizer):
    """
    An observation vectorizer that concatenates lists of
    numpy arrays along the inner-most direction.
    """

    def __init__(self, box_shape, num_dimensions):
        self._out_shape = tuple(box_shape[:-1]) + (box_shape[-1] * num_dimensions,)

    @property
    def out_shape(self):
        return self._out_shape

    def to_vecs(self, space_elements):
        return [np.concatenate(x, axis=-1) for x in space_elements]


def _reduce_sums(batch):
    """
    Reduce a batch of shape [batch x out_shape] to a
    batch of scalars.
    """
    dims = list(range(1, len(batch.shape)))
    return tf.reduce_sum(batch, axis=dims)
