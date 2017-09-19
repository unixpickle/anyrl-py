"""
Parameteric categorical distributions.
"""

import numpy as np
import tensorflow as tf

from .interfaces import Distribution

class CategoricalSoftmax(Distribution):
    """
    A probability distribution that uses softmax to decide
    between a discrete number of options.
    """
    def __init__(self, num_options):
        self.num_options = num_options

    def param_size(self):
        return self.num_options

    def sample(self, param_batch):
        probs = np.exp(np.array(param_batch))
        return [np.random.choice(len(p), p=p) for p in probs]

    def log_probs(self, param_batch, samples):
        loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits
        return tf.negative(loss_func(labels=samples, logits=param_batch))

    def entropy(self, param_batch):
        log_probs = tf.nn.log_softmax(param_batch)
        probs = tf.exp(log_probs)
        tf.negative(tf.reduce_sum(log_probs * probs, axis=-1))

    def kl_divergence(self, param_batch_1, param_batch_2):
        log_probs_1 = tf.nn.log_softmax(param_batch_1)
        log_probs_2 = tf.nn.log_softmax(param_batch_2)
        probs = tf.exp(log_probs_1)
        return tf.reduce_sum(probs * (log_probs_1 - log_probs_2))
