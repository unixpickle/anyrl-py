"""
Helpers for implementing models.
"""

import math

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


def nature_cnn(obs_batch, dense=tf.layers.dense):
    """
    Apply the CNN architecture from the Nature DQN paper.

    The result is a batch of feature vectors.
    """
    conv_kwargs = {
        'activation': tf.nn.relu,
        'kernel_initializer': tf.orthogonal_initializer(gain=math.sqrt(2))
    }
    with tf.variable_scope('layer_1'):
        cnn_1 = tf.layers.conv2d(obs_batch, 32, 8, 4, **conv_kwargs)
    with tf.variable_scope('layer_2'):
        cnn_2 = tf.layers.conv2d(cnn_1, 64, 4, 2, **conv_kwargs)
    with tf.variable_scope('layer_3'):
        cnn_3 = tf.layers.conv2d(cnn_2, 64, 3, 1, **conv_kwargs)
    flat_size = product([x.value for x in cnn_3.get_shape()[1:]])
    flat_in = tf.reshape(cnn_3, (tf.shape(cnn_3)[0], int(flat_size)))

    # The orthogonal initializer appears to be unstable
    # for large matrices. With ortho init, I see huge
    # max outputs for some environments.
    del conv_kwargs['kernel_initializer']

    return dense(flat_in, 512, **conv_kwargs)


def impala_cnn(images, depths=(16, 32, 32)):
    """
    Apply the CNN architecture from the IMPALA paper.

    The result is a batch of feature vectors.
    """
    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value
        out = tf.nn.relu(inputs)
        out = tf.layers.conv2d(out, depth, 3, padding='same', activation=tf.nn.relu)
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        return out + inputs

    def conv_sequence(inputs, depth):
        out = tf.layers.conv2d(inputs, depth, 3, padding='same')
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = images
    for depth in depths:
        out = conv_sequence(out, depth)
    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu)

    return out


def nature_huber_loss(residuals):
    """
    Compute the Huber loss as used for Nature DQN.

    Args:
      residuals: a batch of TD errors.

    Returns:
      A batch of loss values.
    """
    absolute = tf.abs(residuals)
    return tf.where(absolute < 1, 0.5 * tf.square(residuals), absolute - 0.5)


def product(vals):
    """
    Compute the product of values in a list-like object.
    """
    prod = 1
    for val in vals:
        prod *= val
    return prod


def simple_mlp(inputs, layer_sizes, activation, dense=tf.layers.dense):
    """
    Apply a simple multi-layer perceptron model to the
    batch of inputs.

    Args:
      inputs: the batch of inputs. This may have any shape
        with at least two dimensions, provided all the
        sizes are known ahead of time besides the batch
        size.
      layer_sizes: a sequence of hidden layer sizes.
      activation: the activation function.
      dense: the dense layer implementation.
    """
    layer_in_size = product([x.value for x in inputs.get_shape()[1:]])
    layer_in = tf.reshape(inputs, (tf.shape(inputs)[0], layer_in_size))
    for layer_idx, out_size in enumerate(layer_sizes):
        with tf.variable_scope(None, default_name='layer_' + str(layer_idx)):
            layer_in = dense(layer_in, out_size, activation=activation)
    return layer_in


def take_vector_elems(vectors, indices):
    """
    For a batch of vectors, take a single vector component
    out of each vector.

    Args:
      vectors: a [batch x dims] Tensor.
      indices: an int32 Tensor with `batch` entries.

    Returns:
      A Tensor with `batch` entries, one for each vector.
    """
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))
