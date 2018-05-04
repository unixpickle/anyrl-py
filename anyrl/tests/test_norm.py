"""
Test reward normalizers.
"""

import numpy as np

from anyrl.algos import GAE
from anyrl.rollouts import Rollout
from anyrl.rollouts.norm import RewardNormalizer, OnlineAverage


def test_advantage_norm():
    """
    Test advantage normalization when discount != 0.
    """
    rollout = Rollout(observations=[None] * 7,
                      model_outs=[{'values': [0]}] * 6 + [{'values': [3.0]}],
                      rewards=[1, 2, 0, 2, 1, -3],
                      start_state=None)
    factor = np.sqrt(np.mean(np.square(GAE(lam=1, discount=0.7).advantages([rollout])[0])))
    normalizer = RewardNormalizer(update_rate=None, discount=0.7, epsilon=0)
    normed = normalizer.update([rollout])
    assert np.allclose(normed[0].rewards, np.array(rollout.rewards) / factor)


def test_running_average():
    """
    Test a running, unbiased average.
    """
    avg = OnlineAverage(rate=None)
    values = np.random.normal(size=15)
    for subset in [values[:3], values[3:9], values[9:14], values[14:]]:
        avg.update(subset)
    assert np.allclose(avg.value, np.mean(values))


def test_moving_average():
    """
    Test an exponential moving average.
    """
    avg = OnlineAverage(rate=0.05)
    values = np.random.normal(size=15)
    subsets = [values[:3], values[3:9], values[9:14], values[14:]]
    for subset in subsets:
        avg.update(subset)
    expected = np.mean(subsets[0])
    for subset in subsets[1:]:
        expected += 0.05 * (np.mean(subset) - expected)
    assert np.allclose(avg.value, expected)
