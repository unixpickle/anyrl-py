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

    def log_prob(self, param_batch, sample_vecs):
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

def softmax(param_batch):
    """
    Compute a batched softmax on the minor dimension.
    """
    col_shape = (len(param_batch), 1)
    max_vals = np.reshape(param_batch.max(axis=-1), col_shape)
    unnorm = np.exp(param_batch - max_vals)
    return unnorm / np.reshape(np.sum(unnorm, axis=-1), col_shape)
