"""
Utilities for distributed training with MPI.
"""

from mpi4py import MPI
import numpy as np
import tensorflow as tf

# pylint: disable=E1101

# pylint: disable=R0903
class MPIOptimizer:
    """
    Wraps a TensorFlow optimizer to use MPI allreduce.
    """
    def __init__(self, optimizer, loss, var_list=None):
        self.grads = optimizer.compute_gradients(loss, var_list=var_list)

        # TODO: make sure gradients will be ordered
        # deterministically.

        self.placeholders = []
        apply_in = []
        for grad, var in self.grads:
            placeholder = tf.placeholder(dtype=grad.dtype, shape=grad.shape)
            self.placeholders.append(placeholder)
            apply_in.append((placeholder, var))
        self.apply = optimizer.apply_gradients(apply_in)

    def minimize(self, sess, feed_dict=None):
        """
        Compute the gradients, aggregate them, and apply
        them using the wrapped optimizer.
        """
        if not feed_dict:
            feed_dict = {}
        grad_outs = sess.run([x[0] for x in self.grads], feed_dict=feed_dict)
        extra_feed = feed_dict.copy()
        for grad_out, placeholder in zip(grad_outs, self.placeholders):
            mean_grad = np.zeros(grad_out.shape, dtype='float32')
            send_grad = np.array(grad_out, dtype='float32')
            MPI.COMM_WORLD.Allreduce(send_grad, mean_grad, op=MPI.SUM)
            mean_grad /= MPI.COMM_WORLD.Get_size()
            extra_feed[placeholder] = mean_grad
        sess.run(self.apply, feed_dict=extra_feed)

def mpi_ppo(ppo, optimizer, rollouts, batch_size=None, num_iter=12):
    """
    Run the PPO inner loop with an MPI optimizer.
    """
    batch_idx = 0
    batches = ppo.model.batches(rollouts, batch_size=batch_size)
    for batch in batches:
        feed_dict = ppo.feed_dict(rollouts, batch)
        optimizer.minimize(ppo.model.session, feed_dict=feed_dict)
        batch_idx += 1
        if batch_idx == num_iter:
            break
