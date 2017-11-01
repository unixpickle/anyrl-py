"""
Reward normalization schemes.
"""

from math import sqrt

import numpy as np

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
            coefficient updates (0 through 1). Set to None
            to use a running average over all rewards.
          epsilon: used to avoid dividing by 0
        """
        self._average = OnlineAverage(rate=update_rate)
        self._epsilon = epsilon
        self._coeff = None

    def update(self, rollouts):
        """
        Update the statistics using the rollouts and
        return a normalized copy of the rollouts.
        """
        squares = [rew**2 for r in rollouts for rew in r.rewards]
        self._average.update(squares)
        return [self._normalized_rollout(r) for r in rollouts]

    def _normalized_rollout(self, rollout):
        """
        Normalize the rollout.
        """
        scale = 1 / (self._epsilon + sqrt(self._average.value))
        rollout = rollout.copy()
        rollout.rewards = [r*scale for r in rollout.rewards]
        return rollout

class OnlineAverage:
    """
    A moving or running average.

    Running averages are unbiased and compute the mean of
    a list of values, even if those values come in in a
    stream.

    Moving averages are biased towards newer values,
    updating in a way that forgets the distant past.
    """
    def __init__(self, rate=None):
        """
        Create a new OnlineAverage.

        Args:
          rate: the moving average update rate. Used in
            update as `rate*(new-old)`, where new is the
            average of a new batch. If None, a dynamic
            update rate is chosen such that the online
            average is a running average over all the
            samples.
        """
        self.rate = rate
        self._current = 0
        self._num_samples = 0

    @property
    def value(self):
        """
        Get the current average value.
        """
        return self._current

    def update(self, values):
        """
        Update the moving average with the value batch.

        Args:
          values: a sequence of numerics.

        Returns:
          The new online average.
        """
        if self.rate is not None:
            rate = self.rate
            if self._num_samples == 0:
                rate = 1
        else:
            rate = len(values) / (len(values) + self._num_samples)
        self._current += rate * (np.mean(values) - self._current)
        self._num_samples += len(values)
        return self._current
