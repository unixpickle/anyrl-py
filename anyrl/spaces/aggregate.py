"""
APIs for aggregate spaces.
"""

import numpy as np
import tensorflow as tf

from .base import Distribution


class TupleDistribution(Distribution):
    """
    A distribution that consists of an ordered list of
    sub-distributions.
    """

    def __init__(self, sub_dists, to_sample=lambda x: x):
        self.tuple = tuple(sub_dists)
        self.to_sample = to_sample

    @property
    def out_shape(self):
        return (int(sum([np.prod(d.out_shape) for d in self.tuple])),)

    def to_vecs(self, space_elements):
        per_dist = zip(*space_elements)
        sub_vecs = [d.to_vecs(list(l)) for d, l in zip(self.tuple, per_dist)]
        flat_shape = (len(space_elements), -1)
        flat_batches = [np.reshape(np.array(v), flat_shape) for v in sub_vecs]
        return np.concatenate(flat_batches, axis=-1)

    @property
    def param_shape(self):
        return (int(sum([np.prod(d.param_shape) for d in self.tuple])),)

    def sample(self, param_batch):
        per_dist = self.unpack_params(np.array(param_batch))
        samples = [d.sample(p) for d, p in zip(self.tuple, per_dist)]
        return self.to_sample(list(zip(*samples)))

    def mode(self, param_batch):
        per_dist = self.unpack_params(np.array(param_batch))
        modes = [d.mode(p) for d, p in zip(self.tuple, per_dist)]
        return self.to_sample(list(zip(*modes)))

    def log_prob(self, param_batch, sample_vecs):
        per_dist = self.unpack_params(param_batch)
        samples = self.unpack_outs(sample_vecs)
        sub_probs = [d.log_prob(p, s) for d, p, s
                     in zip(self.tuple, per_dist, samples)]
        return tf.add_n(sub_probs)

    def entropy(self, param_batch):
        per_dist = self.unpack_params(param_batch)
        sub_ents = [d.entropy(p) for d, p in zip(self.tuple, per_dist)]
        return tf.add_n(sub_ents)

    def kl_divergence(self, param_batch_1, param_batch_2):
        per_dist_1 = self.unpack_params(param_batch_1)
        per_dist_2 = self.unpack_params(param_batch_2)
        sub_kls = [d.kl_divergence(p1, p2) for d, p1, p2
                   in zip(self.tuple, per_dist_1, per_dist_2)]
        return tf.add_n(sub_kls)

    def unpack_outs(self, sample_vecs):
        """
        Unpack tensors from to_vecs into a tuple of
        properly-shaped tensors for each sub-space.

        Takes either a numpy array or a TF tensor.
        """
        return _unpack(sample_vecs, [d.out_shape for d in self.tuple])

    def unpack_params(self, param_batch):
        """
        Unpack parameter tensors into a tuple of
        properly-shaped tensors for each sub-space.

        Takes either a numpy array or a TF tensor.
        """
        return _unpack(param_batch, [d.param_shape for d in self.tuple])


def _unpack(vecs, shapes):
    """
    Unpack the flattened and packed vectors into a tuple.
    """
    offset = 0
    res = ()
    for shape in shapes:
        size = int(np.prod(shape))
        sub_slice = vecs[:, offset:offset+size]
        offset += size
        if isinstance(vecs, np.ndarray):
            sub_shape = (len(vecs),) + shape
            res += (np.reshape(sub_slice, sub_shape),)
        else:
            sub_shape = (tf.shape(vecs)[0],) + shape
            res += (tf.reshape(sub_slice, sub_shape),)
    return res
