"""
Tests for MPI-related routines.
"""

import numpy as np
import pytest
import tensorflow as tf

from anyrl.algos.mpi import MPIOptimizer


@pytest.mark.parametrize('loss_fn', [lambda x: tf.reduce_mean(tf.square(x)),
                                     # Specifically test for sparse gradients:
                                     lambda x: tf.reduce_mean(tf.gather(x, [0, 1]))])
def test_mpi_optimizer(loss_fn):
    """
    Test that the MPIOptimizer is equivalent to its
    encapsulated optimizer.
    """
    with tf.Graph().as_default():
        x = tf.get_variable('x', shape=[10, 15], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        loss = loss_fn(x)
        optim = tf.train.AdamOptimizer(learning_rate=0.1)
        mpi_optim = MPIOptimizer(optim, loss)
        minimize_op = optim.minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            old_vars = sess.run(tf.global_variables())
            expected_loss = sess.run(loss)

            sess.run(minimize_op)
            expected_x = sess.run(x)
            sess.run([tf.assign(x, old_x) for x, old_x in zip(tf.global_variables(), old_vars)])

            actual_loss = mpi_optim.minimize(sess, terms=[loss])[0]
            actual_x = sess.run(x)

            np.testing.assert_allclose(expected_loss, actual_loss)
            np.testing.assert_allclose(expected_x, actual_x)
