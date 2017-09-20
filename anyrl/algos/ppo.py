"""
Proximal policy optimization.
"""

import tensorflow as tf

from .advantages import GAE
from . import util

# pylint: disable=R0902
class PPO:
    """
    Train TensorFlow actor-critic models using PPO.

    This works with any model that implements
    anyrl.models.TFActorCritic.
    """
    # pylint: disable=R0913
    def __init__(self,
                 model,
                 vf_coeff=0.5,
                 entropy_reg=0.01,
                 adv_est=GAE(lam=0.95, discount=0.99),
                 epsilon=0.2,
                 variables=None):
        self.model = model
        self._adv_est = adv_est

        self.variables = variables
        if variables is None:
            self.variables = tf.trainable_variables()

        self._advs = tf.placeholder(tf.float32, (None,))
        self._target_vals = tf.placeholder(tf.float32, (None,))
        self._actions = tf.placeholder(tf.float32, (None,))

        param_size = model.action_dist.param_size
        self._orig_action_params = tf.placeholder(tf.float32, (None, param_size))

        actor, critic = model.batch_outputs()

        new_log_probs = model.action_dist.log_probs(actor, self._actions)
        old_log_probs = model.action_dist.log_probs(self._orig_action_params,
                                                    self._actions)

        self.actor_loss = _clipped_objective(new_log_probs, old_log_probs,
                                             self._advs, epsilon)
        self.critic_loss = tf.reduce_mean(tf.square(critic - self._target_vals))
        self.regularization = (tf.reduce_mean(model.action_dist.entropy(actor)) *
                               entropy_reg)
        self.objective = (self.actor_loss + self.regularization -
                          vf_coeff*self.critic_loss)

    def feed_dict(self, rollouts, batch):
        """
        Generate a TensorFlow feed_dict that feeds the
        mini-batch into the objective.
        """
        advs = self._adv_est.advantages(rollouts)
        targets = self._adv_est.targets(rollouts)
        actions = util.select_model_out_from_batch('actions', rollouts, batch)
        orig_outs = util.select_model_out_from_batch('action_params', rollouts, batch)
        feed_dict = batch['feed_dict']
        feed_dict[self._advs] = util.select_from_batch(advs, batch)
        feed_dict[self._target_vals] = util.select_from_batch(targets, batch)
        feed_dict[self._actions] = self.model.action_dist.to_vecs(actions)
        feed_dict[self._orig_action_params] = orig_outs
        return feed_dict

    def optimize(self, learning_rate=1e-3):
        """
        Create an operation that trains the model based on
        values given by feed_dict.
        """
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return trainer.minimize(-self.objective)

    def run_optimize(self, optimize_op, rollouts, batch_size=None, num_iter=12):
        """
        Run several steps of training with mini-batches.
        """
        remaining_iter = num_iter
        batches = self.model.batches(rollouts, batch_size=batch_size)
        for batch in batches:
            self.model.session.run(optimize_op, self.feed_dict(rollouts, batch))
            remaining_iter -= 1
            if remaining_iter == 0:
                break

    # TODO: API that supports schedules and runs the
    # entire training loop for us.

def _clipped_objective(new_log_probs, old_log_probs, advs, epsilon):
    """
    Compute the mean clipped PPO objective.
    """
    prob_ratio = tf.exp(new_log_probs - old_log_probs)
    clipped_ratio = tf.clip_by_value(prob_ratio, 1-epsilon, 1+epsilon)
    mins = tf.minimum(advs*clipped_ratio, advs*prob_ratio)
    return tf.reduce_mean(mins)
