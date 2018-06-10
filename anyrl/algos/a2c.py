"""
A synchronous version of advantage actor-critic.
"""

import tensorflow as tf

from .advantages import GAE
from . import util


class A2C:
    """
    Train TensorFlow actor-critic models using A2C.

    This works with any model that implements
    anyrl.models.TFActorCritic.

    Thanks to:
    https://github.com/openai/baselines/blob/699919f1cf2527b184f4445a3758a773f333a1ba/baselines/a2c/a2c.py
    """

    def __init__(self,
                 model,
                 vf_coeff=0.5,
                 entropy_reg=0.01,
                 adv_est=GAE(lam=0.95, discount=0.99),
                 variables=None):
        self.model = model
        self.adv_est = adv_est

        self.variables = variables
        if variables is None:
            self.variables = tf.trainable_variables()

        self._advs = tf.placeholder(tf.float32, (None,))
        self._target_vals = tf.placeholder(tf.float32, (None,))
        self._actions = tf.placeholder(tf.float32,
                                       (None,) + model.action_dist.out_shape)

        self._create_objective(vf_coeff, entropy_reg)

    def feed_dict(self, rollouts, batch=None, advantages=None, targets=None):
        """
        Generate a TensorFlow feed_dict that feeds the
        rollouts into the objective.

        If no batch is specified, all rollouts are used.

        If advantages or targets are specified, then they
        are used instead of using the advantage estimator.
        """
        if batch is None:
            batch = next(self.model.batches(rollouts))
        advantages = advantages or self.adv_est.advantages(rollouts)
        targets = targets or self.adv_est.targets(rollouts)
        actions = util.select_model_out_from_batch('actions', rollouts, batch)
        feed_dict = batch['feed_dict']
        feed_dict[self._advs] = util.select_from_batch(advantages, batch)
        feed_dict[self._target_vals] = util.select_from_batch(targets, batch)
        feed_dict[self._actions] = self.model.action_dist.to_vecs(actions)
        return feed_dict

    def optimize(self,
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

    def _create_objective(self, vf_coeff, entropy_reg):
        """
        Build up the objective graph.
        """
        actor, critic = self.model.batch_outputs()
        dist = self.model.action_dist
        log_probs = dist.log_prob(actor, self._actions)
        entropies = dist.entropy(actor)
        critic_error = self._target_vals - critic
        self.actor_loss = -tf.reduce_mean(log_probs * self._advs)
        self.critic_loss = tf.reduce_mean(tf.square(critic_error))
        self.entropy = tf.reduce_mean(entropies)
        self.objective = (entropy_reg * self.entropy - self.actor_loss -
                          vf_coeff * self.critic_loss)
        self.explained_var = self._compute_explained_var()

    def _compute_explained_var(self):
        variance = (tf.reduce_mean(tf.square(self._target_vals)) -
                    tf.square(tf.reduce_mean(self._target_vals)))
        return 1 - (self.critic_loss / variance)

    # TODO: API that supports schedules and runs the
    # entire training loop for us.
