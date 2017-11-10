"""
APIs for categorical (discrete) spaces.
"""

from functools import partial

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

class NaturalSoftmax(CategoricalSoftmax):
    """
    A softmax distribution with natural gradients through
    log_prob into the parameters.

    The forward outputs are like CategoricalSoftmax.
    However, the gradient through log_prob is artificially
    filled in as the natural gradient.
    """
    def log_prob(self, param_batch, sample_vecs):
        probs = super(NaturalSoftmax, self).log_prob(param_batch, sample_vecs)

        grads = tf.gradients(probs, param_batch)[0]
        fishers = self._fisher_matrices(param_batch)
        natural_grads = tf.matmul(tf.reshape(grads, (-1, 1, self.num_options)),
                                  _pseudo_inverses(fishers))
        natural_grads = tf.reshape(natural_grads, (-1, self.num_options))
        dots = tf.reduce_sum(param_batch*tf.stop_gradient(natural_grads), axis=-1)
        return tf.stop_gradient(probs) + dots - tf.stop_gradient(dots)

    def _fisher_matrices(self, param_batch):
        """
        Compute a Fisher matrix for each softmax.
        """
        num_params = tf.shape(param_batch)[0]
        arr = tf.TensorArray(param_batch.dtype, size=num_params)
        idx = tf.constant(0, dtype=tf.int32)
        def loop_body(obj, idx, arr): # pylint: disable=C0111
            row = param_batch[idx]
            kl_div = obj.kl_divergence(tf.stop_gradient(row), row)
            return idx+1, arr.write(idx, tf.hessians(kl_div, row)[0])
        _, arr = tf.while_loop(cond=lambda idx, _: idx < num_params,
                               body=partial(loop_body, self),
                               loop_vars=(idx, arr))
        return arr.stack()

def softmax(param_batch):
    """
    Compute a batched softmax on the minor dimension.
    """
    col_shape = (len(param_batch), 1)
    max_vals = np.reshape(param_batch.max(axis=-1), col_shape)
    unnorm = np.exp(param_batch - max_vals)
    return unnorm / np.reshape(np.sum(unnorm, axis=-1), col_shape)

def _pseudo_inverses(matrices, epsilon=1e-8):
    """
    Compute the pseudo-inverses for a batch of matrices.
    """
    singulars, left, right = tf.svd(matrices)
    zero = tf.less_equal(singulars, epsilon)
    inv_singulars = tf.matrix_diag(tf.where(zero, singulars, 1 / singulars))
    return tf.matmul(tf.matmul(left, inv_singulars), tf.transpose(right, perm=(0, 2, 1)))
