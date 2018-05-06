"""
Proximal policy optimization.
"""

import tensorflow as tf

from .a2c import A2C
from . import util


class PPO(A2C):
    """
    Train TensorFlow actor-critic models using PPO.

    This works with any model that implements
    anyrl.models.TFActorCritic.

    For more on PPO, see:
    https://arxiv.org/abs/1707.06347
    """

    def __init__(self,
                 model,
                 epsilon=0.2,
                 **a2c_kwargs):
        self._epsilon = epsilon
        param_shape = (None,) + model.action_dist.param_shape
        self._orig_action_params = tf.placeholder(tf.float32, param_shape)
        super(PPO, self).__init__(model, **a2c_kwargs)

    def _create_objective(self, vf_coeff, entropy_reg):
        actor, critic, mask = self.model.batch_outputs()

        dist = self.model.action_dist
        new_log_probs = dist.log_prob(actor, self._actions)
        old_log_probs = dist.log_prob(self._orig_action_params, self._actions)
        clipped_obj = clipped_objective(new_log_probs, old_log_probs,
                                        self._advs, self._epsilon)
        clipped_samples = _clipped_samples(new_log_probs, old_log_probs,
                                           self._advs, self._epsilon)
        critic_error = self._target_vals - critic
        self.actor_loss = -util.masked_mean(mask, clipped_obj)
        self.critic_loss = util.masked_mean(mask, tf.square(critic_error))
        self.entropy = util.masked_mean(mask, dist.entropy(actor))
        self.num_clipped = tf.cast(util.masked_sum(mask, clipped_samples), tf.int32)
        self.objective = (entropy_reg * self.entropy - self.actor_loss -
                          vf_coeff * self.critic_loss)
        self.explained_var = self._compute_explained_var(mask)

    def feed_dict(self, rollouts, batch=None, advantages=None, targets=None):
        if batch is None:
            batch = next(self.model.batches(rollouts))
        feed_dict = super(PPO, self).feed_dict(rollouts,
                                               batch=batch,
                                               advantages=advantages,
                                               targets=targets)
        orig_outs = util.select_model_out_from_batch('action_params', rollouts, batch)
        feed_dict[self._orig_action_params] = orig_outs
        return feed_dict

    # pylint: disable=W0221
    def optimize(self, learning_rate=1e-3):
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return trainer.minimize(-self.objective, var_list=self.variables)

    def run_optimize(self, optimize_op, rollouts, batch_size=None, num_iter=12,
                     log_fn=None, extra_feed_dict=None):
        """
        Run several steps of training with mini-batches.

        If log_fn is set, intermediate progress is logged
        by calling log_fn with string log messages.

        Returns:
          A list of tuples, where each tuple is:
            (actor_loss, explained, entropy, clipped).

          The returned list of tuples correspond to each
            iteration of training.
        """
        batch_idx = 0
        batches = self.model.batches(rollouts, batch_size=batch_size)
        advantages = self.adv_est.advantages(rollouts)
        targets = self.adv_est.targets(rollouts)
        result = []
        for batch in batches:
            terms = (self.actor_loss, self.explained_var, self.entropy,
                     self.num_clipped, optimize_op)
            feed_dict = self.feed_dict(rollouts, batch,
                                       advantages=advantages,
                                       targets=targets)
            if extra_feed_dict:
                feed_dict.update(extra_feed_dict)
            terms = self.model.session.run(terms, feed_dict)
            if log_fn is not None:
                log_fn('batch %d: actor=%f explained=%f entropy=%f clipped=%d' %
                       (batch_idx, -terms[0], terms[1], terms[2], terms[3]))
            result.append(terms[:4])
            batch_idx += 1
            if batch_idx == num_iter:
                break
        return result

    # TODO: API that supports schedules and runs the
    # entire training loop for us.


def clipped_objective(new_log_probs, old_log_probs, advs, epsilon):
    """
    Compute the component-wise clipped PPO objective.
    """
    prob_ratio = tf.exp(new_log_probs - old_log_probs)
    clipped_ratio = tf.clip_by_value(prob_ratio, 1-epsilon, 1+epsilon)
    return tf.minimum(advs*clipped_ratio, advs*prob_ratio)


def _clipped_samples(new_log_probs, old_log_probs, advs, epsilon):
    """
    Count the number of samples that are clipped by the
    clipped PPO objective.
    """
    prob_ratio = tf.exp(new_log_probs - old_log_probs)
    clipped_ratio = tf.clip_by_value(prob_ratio, 1-epsilon, 1+epsilon)
    return tf.cast(advs*clipped_ratio < advs*prob_ratio, tf.float32)
