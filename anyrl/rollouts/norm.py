"""
Reward normalization schemes.
"""

from math import sqrt

import numpy as np


class RewardNormalizer:
    """
    Normalize rewards in rollouts with a gradually
    updating divisor.
    """

    def __init__(self, update_rate=0.05, discount=0.0, scale=1.0, epsilon=1e-5):
        """
        Create a reward normalizer.

        Arguments:
          update_rate: the speed at which the normalizing
            coefficient updates (0 through 1). Set to None
            to use a running average over all rewards.
          discount: the discount factor to use. Using 0
            means that rewards themselves are normalized.
            Using a larger value means that a geometric
            sum over rewards is normalized.
          scale: a scalar to multiply rewards by after
            normalizing.
          epsilon: used to avoid dividing by 0
        """
        self._average = OnlineAverage(rate=update_rate)
        self._discount = discount
        self._scale = scale
        self._epsilon = epsilon

    def update(self, rollouts):
        """
        Update the statistics using the rollouts and
        return a normalized copy of the rollouts.
        """
        squares = [x**2 for x in self._advantages(rollouts)]
        self._average.update(squares)
        return [self._normalized_rollout(r) for r in rollouts]

    def _normalized_rollout(self, rollout):
        """
        Normalize the rollout.
        """
        scale = self._scale / (self._epsilon + sqrt(self._average.value))
        rollout = rollout.copy()
        rollout.rewards = [r*scale for r in rollout.rewards]
        return rollout

    def _advantages(self, rollouts):
        if self._discount == 0.0:
            return [rew for r in rollouts for rew in r.rewards]
        result = []
        for rollout in rollouts:
            if rollout.trunc_end and 'values' in rollout.model_outs[-1]:
                accumulator = rollout.model_outs[-1]['values'][0]
            else:
                accumulator = 0
            for reward in rollout.rewards[::-1]:
                accumulator *= self._discount
                accumulator += reward
                result.append(accumulator)
        return result


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
