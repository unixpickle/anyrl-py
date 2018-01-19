"""
Tests for experience replay buffers.
"""

from collections import Counter
import unittest

import numpy as np

from anyrl.rollouts import PrioritizedReplayBuffer

class PrioritizedReplayTest(unittest.TestCase):
    """
    Tests for PrioritizedReplayBuffer.
    """
    def test_uniform_sampling(self):
        """
        Test the buffer when it's configured to sample
        uniformly.
        """
        buf = PrioritizedReplayBuffer(capacity=10, alpha=0, beta=1)
        for i in range(10):
            sample = {'obs': 0, 'action': 0, 'reward': 0, 'new_obs': 0, 'steps': 1, 'idx': i}
            buf.add_sample(sample)
        sampled_idxs = []
        for _ in range(10000):
            samples = buf.sample(3)
            sampled_idxs.extend([s['idx'] for s in samples])
            buf.update_weights(samples, [s['idx'] for s in samples])
        counts = Counter(sampled_idxs)
        for i in range(10):
            frac = counts[i] / len(sampled_idxs)
            self.assertGreater(frac, 0.09)
            self.assertLess(frac, 0.11)

    def test_prioritized_sampling(self):
        """
        Test the buffer in a simple prioritized setting.
        """
        buf = PrioritizedReplayBuffer(capacity=10, alpha=1.5, beta=1, epsilon=0.5)
        for i in range(10):
            sample = {'obs': 0, 'action': 0, 'reward': 0, 'new_obs': 0, 'steps': 1, 'idx': i}
            buf.add_sample(sample, init_weight=i)
        sampled_idxs = []
        for _ in range(50000):
            samples = buf.sample(3)
            sampled_idxs.append(samples[0]['idx'])
        counts = Counter(sampled_idxs)
        probs = np.power(np.arange(10).astype('float64') + 0.5, 1.5)
        probs /= np.sum(probs)
        for i, prob in enumerate(probs):
            frac = counts[i] / len(sampled_idxs)
            self.assertGreater(frac, prob - 0.01)
            self.assertLess(frac, prob + 0.01)

    def test_simple_importance_sampling(self):
        """
        Test importance sampling with beta=1.
        """
        buf = PrioritizedReplayBuffer(capacity=10, alpha=1.5, beta=1, epsilon=0.5)
        for i in range(10):
            sample = {'obs': 0, 'action': 0, 'reward': 0, 'new_obs': 0, 'steps': 1, 'idx': i}
            buf.add_sample(sample, init_weight=i)
        sampled_weights = [0.0] * 10
        for _ in range(50000):
            samples = buf.sample(10)
            for sample in samples:
                sampled_weights[sample['idx']] += sample['weight']
        sampled_weights = np.array(sampled_weights) / max(sampled_weights)
        weights = 1 / np.power(np.arange(10).astype('float64') + 0.5, 1.5)
        weights /= np.max(weights)
        self.assertTrue(np.allclose(sampled_weights, weights, atol=1e-2, rtol=1e-2))

if __name__ == '__main__':
    unittest.main()
