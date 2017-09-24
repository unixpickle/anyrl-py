"""
Helpers for implementing models.
"""

import numpy as np
import tensorflow as tf

def mini_batches(size_per_index, batch_size=None):
    """
    Generate mini-batches of size batch_size.

    The size_per_index list is the size of each batch
    element.
    Batches are generated such that the sum of the sizes
    of the batch elements is at least batch_size.
    """
    if batch_size is None or sum(size_per_index) <= batch_size:
        while True:
            yield list(range(len(size_per_index)))
    cur_indices = []
    cur_size = 0
    for idx in _infinite_random_shuffle(len(size_per_index)):
        cur_indices.append(idx)
        cur_size += size_per_index[idx]
        if cur_size >= batch_size:
            yield cur_indices
            cur_indices = []
            cur_size = 0

def _infinite_random_shuffle(num_elements):
    """
    Continually permute the elements and yield all of the
    permuted indices.
    """
    while True:
        for elem in np.random.permutation(num_elements):
            yield elem

def mix_init_states(is_init, init_states, start_states):
    """
    Mix initial variables with start state placeholders.
    """
    if isinstance(init_states, tuple):
        assert isinstance(start_states, tuple)
        res = []
        for sub_init, sub_start in zip(init_states, start_states):
            res.append(mix_init_states(is_init, sub_init, sub_start))
        return tuple(res)
    batch_size = tf.shape(start_states)[0]
    return tf.where(is_init, _batchify(batch_size, init_states), start_states)

def _batchify(batch_size, tensor):
    """
    Repeat a tensor the given number of times in the outer
    dimension.
    """
    batchable = tf.reshape(tensor, tf.concat([[1], tf.shape(tensor)], axis=0))
    ones = tf.ones(tensor.shape.ndims, dtype=tf.int32)
    repeat_count = tf.concat([[batch_size], ones], axis=0)
    return tf.tile(batchable, repeat_count)

def product(vals):
    """
    Compute the product of values in a list-like object.
    """
    prod = 1
    for val in vals:
        prod *= val
    return prod
