"""
APIs for categorical (discrete) spaces.
"""

import numpy as np
import tensorflow as tf

from .base import Distribution


class CategoricalSoftmax(Distribution):
    """
    A probability distribution that uses softmax to decide
    between a discrete number of options.
    """

    def __init__(self, num_options, low=0):
        self.num_options = num_options
        self.low = low

    @property
    def out_shape(self):
        return (self.num_options,)

    def to_vecs(self, space_elements):
        res = np.zeros((len(space_elements), self.num_options))
        res[np.arange(len(space_elements)), np.array(space_elements)-self.low] = 1
        return res

    @property
    def param_shape(self):
        return (self.num_options,)

    def sample(self, param_batch):
        dist = softmax(np.array(param_batch))
        cumulative_dist = np.cumsum(dist, axis=-1)
        sampled = np.random.rand(len(param_batch), 1)
        large_enoughs = cumulative_dist > sampled
        return self.low + np.argmax(large_enoughs, axis=-1)

    def mode(self, param_batch):
        return self.low + np.argmax(param_batch, axis=-1)

    def log_prob(self, param_batch, sample_vecs):
        if hasattr(tf.nn, 'softmax_cross_entropy_with_logits_v2'):
            loss_func = tf.nn.softmax_cross_entropy_with_logits_v2
        else:
            loss_func = tf.nn.softmax_cross_entropy_with_logits
        return tf.negative(loss_func(labels=sample_vecs, logits=param_batch))

    def entropy(self, param_batch):
        log_probs = tf.nn.log_softmax(param_batch)
        probs = tf.exp(log_probs)
        return tf.negative(tf.reduce_sum(log_probs * probs, axis=-1))

    def kl_divergence(self, param_batch_1, param_batch_2):
        log_probs_1 = tf.nn.log_softmax(param_batch_1)
        log_probs_2 = tf.nn.log_softmax(param_batch_2)
        probs = tf.exp(log_probs_1)
        return tf.reduce_sum(probs * (log_probs_1 - log_probs_2), axis=-1)


class NaturalSoftmax(CategoricalSoftmax):
    """
    A softmax distribution with natural gradients through
    log_prob into the parameters.

    The forward outputs are like CategoricalSoftmax.
    However, the gradient through log_prob is artificially
    filled in as the natural gradient.
    """

    def __init__(self, num_options, low=0, epsilon=1e-4):
        super(NaturalSoftmax, self).__init__(num_options, low=low)
        self.epsilon = epsilon

    def log_prob(self, param_batch, sample_vecs):
        log_probs = super(NaturalSoftmax, self).log_prob(param_batch, sample_vecs)
        probs = tf.exp(log_probs) + self.epsilon
        neg_grads = -1 / (self.num_options * probs)
        natural_grads = tf.tile(tf.expand_dims(neg_grads, axis=1), (1, self.num_options))
        natural_grads -= self.num_options * natural_grads * sample_vecs
        dots = tf.reduce_sum(param_batch*tf.stop_gradient(natural_grads), axis=-1)
        return tf.stop_gradient(log_probs) + dots - tf.stop_gradient(dots)


def softmax(param_batch):
    """
    Compute a batched softmax on the minor dimension.
    """
    col_shape = (len(param_batch), 1)
    max_vals = np.reshape(param_batch.max(axis=-1), col_shape)
    unnorm = np.exp(param_batch - max_vals)
    return unnorm / np.reshape(np.sum(unnorm, axis=-1), col_shape)
