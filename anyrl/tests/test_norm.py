"""
Test reward normalizers.
"""

import unittest

import numpy as np

from anyrl.rollouts.norm import OnlineAverage

class TestOnlineAverage(unittest.TestCase):
    """
    Test online averages.
    """
    def test_running_average(self):
        """
        Test a running, unbiased average.
        """
        avg = OnlineAverage(rate=None)
        values = np.random.normal(size=15)
        for subset in [values[:3], values[3:9], values[9:14], values[14:]]:
            avg.update(subset)
        self.assertTrue(np.allclose(avg.value, np.mean(values)))

    def test_moving_average(self):
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
        self.assertTrue(np.allclose(avg.value, expected))

if __name__ == '__main__':
    unittest.main()
