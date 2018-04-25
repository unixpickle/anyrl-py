"""
Test reward normalizers.
"""

import numpy as np

from anyrl.rollouts.norm import OnlineAverage

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
