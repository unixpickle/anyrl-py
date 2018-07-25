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
                 value_epsilon=None,
                 value_clip_fn=util.symmetric_clipped_value_loss,
                 **a2c_kwargs):
        self._epsilon = epsilon
        self._value_epsilon = value_epsilon
        self._value_clip_fn = value_clip_fn
        param_shape = (None,) + model.action_dist.param_shape
        self._orig_action_params = tf.placeholder(tf.float32, param_shape)
        self._orig_values = tf.placeholder(tf.float32, (None,))
        super(PPO, self).__init__(model, **a2c_kwargs)

    def _create_objective(self, vf_coeff, entropy_reg):
        actor, critic = self.model.batch_outputs()

        dist = self.model.action_dist
        new_log_probs = dist.log_prob(actor, self._actions)
        old_log_probs = dist.log_prob(self._orig_action_params, self._actions)
        clipped_obj = clipped_objective(new_log_probs, old_log_probs,
                                        self._advs, self._epsilon)
        self.critic_losses, self.value_clipped_frac = self._value_clip_fn(critic,
                                                                          self._orig_values,
                                                                          self._target_vals,
                                                                          self._value_epsilon)
        self.actor_loss = -tf.reduce_mean(clipped_obj)
        self.critic_loss = tf.reduce_mean(self.critic_losses)
        self.entropy = tf.reduce_mean(dist.entropy(actor))
        self.clipped_frac = _clipped_frac(new_log_probs, old_log_probs, self._advs, self._epsilon)
        self.objective = (entropy_reg * self.entropy - self.actor_loss -
                          vf_coeff * self.critic_loss)
        self.explained_var = self._compute_explained_var()

    def feed_dict(self, rollouts, batch=None, advantages=None, targets=None):
        if batch is None:
            batch = next(self.model.batches(rollouts))
        feed_dict = super(PPO, self).feed_dict(rollouts,
                                               batch=batch,
                                               advantages=advantages,
                                               targets=targets)
        orig_outs = util.select_model_out_from_batch('action_params', rollouts, batch)
        orig_vals = util.select_model_out_from_batch('values', rollouts, batch)
        feed_dict[self._orig_action_params] = orig_outs
        feed_dict[self._orig_values] = orig_vals
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
          A dict of metadata from training, with keys:
            'actor_loss': a list containing the mean actor
              loss at each iteration.
            'explained_var': a list containing the
              explained variation at each iteration.
            'entropy': a list containing the mean entropy
              at each iteration.
            'clipped': a list containing the fraction of
              clipped samples at each iteration.
            'value_clipped': a list containing the
              fraction of clipped value function targets
              at each iterations.
            'batch_size': a list containing the number of
              timestep samples per batch.

          More keys may be added in the future.
        """
        def _optimize_fn(batch_idx, batch, feed_dict):
            terms = (self.actor_loss, self.explained_var, self.entropy,
                     self.clipped_frac, self.value_clipped_frac, optimize_op)
            terms = self.model.session.run(terms, feed_dict)
            if log_fn is not None:
                log_fn('batch %d: actor=%f explained=%f entropy=%f clipped=%f value_clipped=%f' %
                       (batch_idx, -terms[0], terms[1], terms[2], terms[3], terms[4]))
            return terms[:-1] + (len(batch['timestep_idxs']),)
        return self._training_loop(optimize_fn=_optimize_fn,
                                   rollouts=rollouts,
                                   batch_size=batch_size,
                                   num_iter=num_iter,
                                   extra_feed_dict=extra_feed_dict)

    def _training_loop(self, optimize_fn, rollouts, batch_size, num_iter, extra_feed_dict):
        """
        Run a generic PPO training loop.

        The optimize_fn takes (batch_idx, batch, feed_dict)
        and returns a dict of meta-data compatible with
        run_optimize().
        """
        assert num_iter > 0
        batch_idx = 0
        batches = self.model.batches(rollouts, batch_size=batch_size)
        advantages = self.adv_est.advantages(rollouts)
        targets = self.adv_est.targets(rollouts)
        term_names = ['actor_loss', 'explained_var', 'entropy', 'clipped', 'batch_size',
                      'value_clipped']
        result = {key: [] for key in term_names}
        for batch in batches:
            feed_dict = self.feed_dict(rollouts, batch,
                                       advantages=advantages,
                                       targets=targets)
            if extra_feed_dict:
                feed_dict.update(extra_feed_dict)
            terms = optimize_fn(batch_idx, batch, feed_dict)
            assert len(terms) == 6, 'missing or extraneous step results'
            for name, term in zip(term_names, terms):
                result[name].append(term)
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


def _clipped_frac(new_log_probs, old_log_probs, advs, epsilon):
    """
    Count the fraction of samples that are clipped by the
    clipped PPO objective.
    """
    prob_ratio = tf.exp(new_log_probs - old_log_probs)
    clipped_ratio = tf.clip_by_value(prob_ratio, 1-epsilon, 1+epsilon)
    num_samples = tf.cast(tf.shape(new_log_probs)[0], tf.float32)
    num_clipped = tf.reduce_sum(tf.cast(advs*clipped_ratio < advs*prob_ratio, tf.float32))
    return num_clipped / num_samples
