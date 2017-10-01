"""
Tests for environment wrappers.
"""

import unittest

from anyrl.tests import SimpleEnv
from anyrl.wrappers import RL2Env

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

if __name__ == '__main__':
    unittest.main()
