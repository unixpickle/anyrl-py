"""
APIs for binary and multi-binary spaces.
"""

import numpy as np
import tensorflow as tf

from .base import Distribution


class MultiBernoulli(Distribution):
    """
    A multi-dimensional binary distribution parameterized
    as Bernoulli distributions with probabilities from
    sigmoids.
    """

    def __init__(self, num_bits):
        self.num_bits = num_bits

    @property
    def out_shape(self):
        return (self.num_bits,)

    def to_vecs(self, space_elements):
        return np.array(space_elements)

    @property
    def param_shape(self):
        return (self.num_bits,)

    def sample(self, param_batch):
        probs = 1 / (1 + np.exp(np.negative(np.array(param_batch))))
        rand = np.random.uniform(size=probs.shape)
        return (probs > rand).astype('int')

    def mode(self, param_batch):
        probs = 1 / (1 + np.exp(np.negative(np.array(param_batch))))
        return (probs > 0.5).astype('int')

    def log_prob(self, param_batch, sample_vecs):
        sample_vecs = tf.cast(sample_vecs, param_batch.dtype)
        log_probs_on = tf.log_sigmoid(param_batch) * sample_vecs
        log_probs_off = tf.log_sigmoid(-param_batch) * (1-sample_vecs)
        return tf.reduce_sum(log_probs_on + log_probs_off, axis=-1)

    def entropy(self, param_batch):
        ent_on = tf.log_sigmoid(param_batch) * tf.sigmoid(param_batch)
        ent_off = tf.log_sigmoid(-param_batch) * tf.sigmoid(-param_batch)
        return tf.negative(tf.reduce_sum(ent_on + ent_off, axis=-1))

    def kl_divergence(self, param_batch_1, param_batch_2):
        probs_on = tf.sigmoid(param_batch_1)
        probs_off = tf.sigmoid(-param_batch_1)
        log_diff_on = tf.log_sigmoid(param_batch_1) - tf.log_sigmoid(param_batch_2)
        log_diff_off = tf.log_sigmoid(-param_batch_1) - tf.log_sigmoid(-param_batch_2)
        kls = probs_on*log_diff_on + probs_off*log_diff_off
        return tf.reduce_sum(kls, axis=-1)
