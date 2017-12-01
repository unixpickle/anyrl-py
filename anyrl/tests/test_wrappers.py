"""
Tests for environment wrappers.
"""

import unittest

import gym
import numpy as np

from anyrl.tests import SimpleEnv
from anyrl.wrappers import RL2Env, DownsampleEnv, GrayscaleEnv, FrameStackEnv

class RL2EnvTest(unittest.TestCase):
    """
    Tests for RL2Env.
    """
    def test_num_eps(self):
        """
        Test that the meta-episode contains the right
        number of sub-episodes.
        """
        real_env = SimpleEnv(1, (3, 4), 'uint8')
        env = RL2Env(real_env, real_env.action_space.sample(),
                     num_eps=3, warmup_eps=1)
        done_eps = 0
        env.reset()
        while done_eps < 3:
            obs, _, done, _ = env.step(env.action_space.sample())
            if obs[3]:
                done_eps += 1
            if done_eps == 3:
                self.assertTrue(done)
                break
            else:
                self.assertFalse(done)

    def test_rewards(self):
        """
        Test that rewards are masked properly.
        """
        real_env = SimpleEnv(1, (3, 4), 'uint8')
        env = RL2Env(real_env, real_env.action_space.sample(), num_eps=5, warmup_eps=-2)
        env.reset()
        done_eps = 0
        while done_eps < 3:
            obs, rew, _, _ = env.step(env.action_space.sample())
            self.assertEqual(rew, 0)
            if obs[3]:
                done_eps += 1
        while True:
            _, rew, done, _ = env.step(env.action_space.sample())
            self.assertNotEqual(rew, 0)
            if done:
                break

class DownsampleEnvTest(unittest.TestCase):
    """
    Tests for DownsampleEnv.
    """
    def test_rate_1(self):
        """
        Test with rate=1.
        """
        low = np.array([[1, 2], [3, 4]])
        high = np.array([[3, 4], [5, 6]])
        env = DownsampleEnv(ShapeEnv(low, high), 1)
        self.assertTrue((env.observation_space.low == low).all())
        self.assertTrue((env.observation_space.high == high).all())

    def test_rate_2(self):
        """
        Test with rate=2.
        """
        low = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        high = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])
        env = DownsampleEnv(ShapeEnv(low, high), 2)
        self.assertEqual(env.observation_space.shape, (1, 2))
        self.assertTrue((env.observation_space.low == np.array([[1, 3]])).all())
        self.assertTrue((env.observation_space.high == np.array([[9, 11]])).all())

        low = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 6, 7, 8]])
        high = np.array([[9, 10, 11, 12], [13, 14, 15, 16], [16, 14, 15, 16]])
        env = DownsampleEnv(ShapeEnv(low, high), 2)
        self.assertEqual(env.observation_space.shape, (2, 2))
        self.assertTrue((env.observation_space.low == np.array([[1, 3], [9, 7]])).all())
        self.assertTrue((env.observation_space.high == np.array([[9, 11], [16, 15]])).all())

class GrayscaleEnvTest(unittest.TestCase):
    """
    Tests for GrayscaleEnv.
    """
    def test_integers(self):
        """
        Tests for integer observations.
        """
        low = np.array([[[0, 0, 0], [16, 16, 50]]])
        high = np.array([[[19, 255, 13], [255, 255, 255]]])
        env = GrayscaleEnv(ShapeEnv(low, high))
        self.assertEqual(env.observation_space.shape, (1, 2, 1))
        self.assertTrue((env.observation_space.low == np.array([[[0], [26]]])).all())
        self.assertTrue((env.observation_space.high == np.array([[[95], [255]]])).all())

class FrameStackEnvTest(unittest.TestCase):
    """
    Tests for FrameStackEnv.
    """
    def test_stack_3(self):
        """
        Test stacking 3 frames.
        """
        low = np.zeros((4, 5, 2))
        high = np.zeros((4, 5, 2)) + 0xff
        env = FrameStackEnv(ShapeEnv(low, high), 3)
        self.assertEqual(env.observation_space.shape, (4, 5, 6))
        obs1 = env.reset()
        obs2, _, _, _ = env.step(0)
        obs3, _, _, _ = env.step(0)
        self.assertFalse((obs1 == obs2).all())
        self.assertFalse((obs1 == obs3).all())
        self.assertFalse((obs2 == obs3).all())
        self.assertTrue((obs1[:, :, 2:] == obs2[:, :, :4]).all())
        self.assertTrue((obs2[:, :, 2:] == obs3[:, :, :4]).all())

class ShapeEnv(gym.Env):
    """
    An environment with a pre-defined observation shape.
    """
    def __init__(self, low, high):
        super(ShapeEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low, high)

    def _reset(self):
        return self.observation_space.sample()

    def _step(self, action):
        return self.observation_space.sample(), 0, False, {}

if __name__ == '__main__':
    unittest.main()
