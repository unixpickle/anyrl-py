"""
Utilities for distributed training with MPI.
"""

from mpi4py import MPI
import numpy as np
import tensorflow as tf

# pylint: disable=E1101


class MPIOptimizer:
    """
    Wraps a TensorFlow optimizer to use MPI allreduce.
    """

    def __init__(self, optimizer, loss, var_list=None):
        old_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.grads = [pair for pair in optimizer.compute_gradients(loss, var_list=var_list)
                      if pair[0] is not None]

        # TODO: make sure gradients will be ordered
        # deterministically.

        self.placeholders = []
        apply_in = []
        for grad, var in self.grads:
            placeholder = tf.placeholder(dtype=grad.dtype, shape=grad.shape)
            self.placeholders.append(placeholder)
            apply_in.append((placeholder, var))
        self.apply = optimizer.apply_gradients(apply_in)
        optimizer_vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                          if v not in old_variables]
        self._var_sync = VarSync([v for _, v in self.grads] + optimizer_vars)

    def minimize(self, sess, feed_dict=None, terms=None):
        """
        Compute the gradients, aggregate them, and apply
        them using the wrapped optimizer.

        Arguments:
          sess: the TensorFlow session.
          feed_dict: the TensorFlow feed_dict for the
            objective.
          terms: a list of scalar Tensors to run at the
             same time as the gradient computation.

        Returns:
          A tuple containing the mean values for each
          entry in terms.
        """
        if not feed_dict:
            feed_dict = {}
        if not terms:
            terms = []
        outs = sess.run(terms + [x[0] for x in self.grads],
                        feed_dict=feed_dict)
        grad_outs = outs[len(terms):]
        term_outs = outs[:len(terms)]

        extra_feed = feed_dict.copy()
        for grad_out, placeholder in zip(grad_outs, self.placeholders):
            mean_grad = np.zeros(grad_out.shape, dtype='float32')
            send_grad = np.array(grad_out, dtype='float32')
            MPI.COMM_WORLD.Allreduce(send_grad, mean_grad, op=MPI.SUM)
            mean_grad /= MPI.COMM_WORLD.Get_size()
            extra_feed[placeholder] = mean_grad
        sess.run(self.apply, feed_dict=extra_feed)

        result = []
        for term in term_outs:
            total = MPI.COMM_WORLD.allreduce(term, op=MPI.SUM)
            result.append(total / MPI.COMM_WORLD.Get_size())
        return tuple(result)

    def sync_from_root(self, sess):
        """
        Send the root node's parameters to every worker.

        This synchronizes both the model parameters and
        any extra variables created by the optimizer.

        Arguments:
          sess: the TensorFlow session.
        """
        self._var_sync.sync(sess)


class VarSync:
    """
    An object that can synchronize TensorFlow variables
    between MPI nodes.
    """

    def __init__(self, variables):
        self._variables = variables
        self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                              for v in variables]
        self._assigns = [tf.assign(v, ph) for v, ph in zip(variables, self._placeholders)]

    def sync(self, sess):
        """
        Synchronize all the variables.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            for val in sess.run(self._variables):
                MPI.COMM_WORLD.Bcast(val)
        else:
            feed = {}
            for ph in self._placeholders:
                buf = np.zeros([x.value for x in ph.get_shape()], dtype=ph.dtype.as_numpy_dtype)
                MPI.COMM_WORLD.Bcast(buf)
                feed[ph] = buf
            sess.run(self._assigns, feed_dict=feed)


def mpi_ppo(ppo, optimizer, rollouts, batch_size=None, num_iter=12, log_fn=None,
            extra_feed_dict=None):
    """
    Run the PPO inner loop with an MPI optimizer.

    If log_fn is set, logging is done on rank 0.

    This is exactly like to PPO.run_optimize(), except
    that it uses an MPIOptimizer. Since this is not a
    method on the PPO class, the first argument is a PPO
    instance.
    """
    def _optimize_fn(batch_idx, batch, feed_dict):
        terms = optimizer.minimize(ppo.model.session,
                                   feed_dict=feed_dict,
                                   terms=[ppo.actor_loss, ppo.explained_var, ppo.entropy,
                                          ppo.clipped_frac, ppo.value_clipped_frac])
        if log_fn and MPI.COMM_WORLD.Get_rank() == 0:
            log_fn('batch %d: actor=%f explained=%f entropy=%f clipped=%f value_clipped=%f' %
                   (batch_idx, -terms[0], terms[1], terms[2], terms[3], terms[4]))
        batch_size = len(batch['timestep_idxs'])
        batch_size = MPI.COMM_WORLD.allreduce(batch_size, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        return terms + (batch_size,)
    return ppo._training_loop(optimize_fn=_optimize_fn,
                              rollouts=rollouts,
                              batch_size=batch_size,
                              num_iter=num_iter,
                              extra_feed_dict=extra_feed_dict)
