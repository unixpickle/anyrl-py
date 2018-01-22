"""
Distributional Q-learning models.
"""

import numpy as np
import tensorflow as tf

from .util import put_vector_elems

class DiscreteDist:
    """
    A discrete reward distribution.
    """
    def __init__(self, num_atoms, min_val, max_val):
        assert num_atoms >= 2
        assert max_val > min_val
        self.num_atoms = num_atoms
        self.min_val = min_val
        self.max_val = max_val
        self._delta = (self.max_val - self.min_val) / (self.num_atoms - 1)

    def atom_values(self):
        """Get the reward values for each atom."""
        return [self.min_val + i * self._delta for i in range(0, self.num_atoms)]

    def mean(self, log_probs):
        """Get the mean rewards for the distributions."""
        return tf.reduce_sum(tf.exp(log_probs) * tf.constant(self.atom_values), axis=-1)

    def add_rewards(self, log_probs, rewards, discounts):
        """
        Compute new distributions after adding rewards to
        old distributions.

        Args:
          log_probs: a batch of log probability vectors.
          rewards: a batch of rewards.
          discounts: the discount factors to apply to the
            distribution rewards.

        Returns:
          A new batch of log probability vectors.
        """
        minus_inf = tf.zeros_like(log_probs) - tf.constant(np.inf)
        new_probs = minus_inf
        for i, atom_rew in enumerate(self.atom_values()):
            old_probs = log_probs[:, i]
            # If the position is exactly 0, rounding up
            # and subtracting 1 would cause problems.
            new_idxs = ((rewards + discounts * atom_rew) - self.min_val) / self._delta
            new_idxs = tf.clip_by_value(new_idxs, 1e-5, float(self.num_atoms - 1))
            index1 = tf.cast(tf.ceil(new_idxs) - 1, tf.int32)
            frac1 = tf.abs(tf.ceil(new_idxs) - new_idxs)
            for indices, frac in [(index1, frac1), (index1 + 1, 1 - frac1)]:
                prob_offset = put_vector_elems(indices, old_probs - 1 + tf.log(frac),
                                               self.num_atoms)
                prob_offset = tf.where(tf.equal(prob_offset, 0), minus_inf, prob_offset + 1)
                new_probs = _add_log_probs(new_probs, prob_offset)
        return new_probs

def _add_log_probs(probs1, probs2):
    return tf.reduce_logsumexp(tf.stack([probs1, probs2]), axis=0)
