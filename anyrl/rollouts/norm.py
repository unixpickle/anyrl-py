"""
Reward normalization schemes.
"""

from math import sqrt

# pylint: disable=R0903
class RewardNormalizer:
    """
    Normalize rewards in rollouts with a gradually
    updating divisor.
    """
    def __init__(self, update_rate=0.05, epsilon=1e-5):
        """
        Create a reward normalizer.

        Arguments:
          update_rate: the speed at which the normalizing
            coefficient updates (0 through 1)
          epsilon: used to avoid dividing by 0
        """
        self._update_rate = update_rate
        self._epsilon = epsilon
        self._coeff = None

    def update(self, rollouts):
        """
        Update the statistics using the rollouts and
        return a normalized copy of the rollouts.
        """
        squares = [rew**2 for r in rollouts for rew in r.rewards]
        new_mean = sum(squares) / len(squares)
        if self._coeff is None:
            self._coeff = new_mean
        else:
            self._coeff += self._update_rate * (new_mean - self._coeff)
        return [self._normalized_rollout(r) for r in rollouts]

    def _normalized_rollout(self, rollout):
        """
        Normalize the rollout.
        """
        scale = 1 / (self._epsilon + sqrt(self._coeff))
        rollout = rollout.copy()
        rollout.rewards = [r*scale for r in rollout.rewards]
        return rollout
