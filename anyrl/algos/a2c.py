"""
A synchronous version of advantage actor-critic.
"""

import tensorflow as tf

from .advantages import GAE
from .util import select_from_batch, select_acts_from_batch

class A2C:
    """
    Train TensorFlow actor-critic models using A2C.

    This works with any model that implements
    anyrl.models.TFActorCritic.

    Thanks to:
    https://github.com/openai/baselines/blob/699919f1cf2527b184f4445a3758a773f333a1ba/baselines/a2c/a2c.py
    """
    # pylint: disable=R0913
    def __init__(self,
                 model,
                 vf_coeff=0.5,
                 entropy_reg=0.01,
                 adv_est=GAE(lam=0.95, discount=0.99),
                 variables=None):
        self.model = model
        self._adv_est = adv_est

        self.variables = variables
        if variables is None:
            self.variables = tf.trainable_variables()

        self._advs = tf.placeholder(tf.float32, (None,))
        self._target_vals = tf.placeholder(tf.float32, (None,))
        self._actions = tf.placeholder(tf.float32, (None,))

        actor, critic = model.batch_outputs()

        actor_loss = self._advs * model.action_dist.log_probs(actor, self._actions)
        critic_loss = tf.square(critic - self._target_vals)
        regularization = entropy_reg * model.action_dist.entropy(actor)
        self.objective = tf.reduce_mean(actor_loss - vf_coeff*critic_loss +
                                        regularization)

    def feed_dict(self, rollouts):
        """
        Generate a TensorFlow feed_dict that feeds the
        rollouts into the objective.
        """
        batch = next(self.model.batches(rollouts))
        advs = self._adv_est.advantages(rollouts)
        targets = self._adv_est.targets(rollouts)
        actions = select_acts_from_batch(rollouts, batch)
        feed_dict = batch['feed_dict']
        feed_dict[self._advs] = select_from_batch(advs, batch)
        feed_dict[self._target_vals] = select_from_batch(targets, batch)
        feed_dict[self._actions] = self.model.action_dist.to_vecs(actions)
        return feed_dict

    def maximize(self,
                 max_grad_norm=0.5,
                 learning_rate=7e-4,
                 rms_decay=0.99,
                 rms_epsilon=1e-5):
        """
        Create an operation that trains the model based on
        values given by feed_dict.
        """
        grads = tf.gradients(tf.negative(self.objective), self.variables)
        if max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                            decay=rms_decay,
                                            epsilon=rms_epsilon)
        return trainer.apply_gradients(list(zip(grads, self.variables)))

    # TODO: API that supports schedules and runs the
    # entire training loop for us.
    # TODO: instance attributes for different parts of
    # objective, for logging.
