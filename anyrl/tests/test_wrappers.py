"""
Tests for environment wrappers.
"""

import os
import shutil
import tempfile
import unittest

import gym
import numpy as np
import pandas
import tensorflow as tf

from anyrl.envs import batched_gym_env
from anyrl.envs.wrappers import (RL2Env, DownsampleEnv, GrayscaleEnv, FrameStackEnv,
                                 MaxEnv, ResizeImageEnv, BatchedFrameStack, ObservationPadEnv,
                                 LoggedEnv)
from anyrl.tests import SimpleEnv

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

    def test_stack_3_no_concat(self):
        """
        Test stacking 3 frames with no concatenation.
        """
        low = np.zeros((4, 5, 2))
        high = np.zeros((4, 5, 2)) + 0xff
        env = FrameStackEnv(ShapeEnv(low, high), 3, concat=False)
        self.assertEqual(env.observation_space.box.shape, (4, 5, 2))
        self.assertEqual(env.observation_space.count, 3)
        obs1 = env.reset()
        obs2, _, _, _, = env.step(0)
        self.assertTrue(np.allclose(obs2[0], obs1[0]))
        self.assertFalse(np.allclose(obs2[-1], obs1[0]))

class MaxEnvTest(unittest.TestCase):
    """
    Tests for MaxEnv.
    """
    def test_max_2(self):
        """
        Test maxing 2 frames.
        """
        env = SimpleEnv(5, (3, 2, 5), 'float32')
        actions = [env.action_space.sample() for _ in range(4)]
        frame1 = env.reset()
        frame2 = env.step(actions[0])[0]
        frame3 = env.step(actions[1])[0]
        frame4 = env.step(actions[2])[0]
        frame5 = env.step(actions[3])[0]
        wrapped = MaxEnv(env, num_images=2)
        max1 = wrapped.reset()
        max2 = wrapped.step(actions[0])[0]
        max3 = wrapped.step(actions[1])[0]
        max4 = wrapped.step(actions[2])[0]
        max5 = wrapped.step(actions[3])[0]
        self.assertTrue((max1 == frame1).all())
        self.assertTrue((max2 == np.max([frame1, frame2], axis=0)).all())
        self.assertTrue((max3 == np.max([frame2, frame3], axis=0)).all())
        self.assertTrue((max4 == np.max([frame3, frame4], axis=0)).all())
        self.assertTrue((max5 == np.max([frame4, frame5], axis=0)).all())

class ResizeImageEnvTest(unittest.TestCase):
    """
    Tests for ResizeImageEnv.
    """
    def test_resize_even(self):
        """
        Test resizing for an even number of pixels.
        """
        env = SimpleEnv(5, (13, 5, 3), 'float32')
        frame = env.reset()
        actual = ResizeImageEnv(env, size=(5, 4)).reset()
        expected = tf.Session().run(
            tf.image.resize_images(frame, [5, 4], method=tf.image.ResizeMethod.AREA))
        self.assertEqual(actual.shape, (5, 4, 3))
        self.assertTrue(np.allclose(actual, expected))

class BatchedFrameStackTest(unittest.TestCase):
    """
    Tests for BatchedFrameStack.
    """
    def test_equivalence(self):
        """
        Test that BatchedFrameStack is equivalent to a
        regular batched FrameStackEnv.
        """
        self._test_equivalence(True)
        self._test_equivalence(False)

    def _test_equivalence(self, concat):
        envs = [lambda idx=i: SimpleEnv(idx+2, (3, 2, 5), 'float32') for i in range(6)]
        env1 = BatchedFrameStack(batched_gym_env(envs, num_sub_batches=3, sync=True),
                                 concat=concat)
        env2 = batched_gym_env([lambda env=e: FrameStackEnv(env(), concat=concat) for e in envs],
                               num_sub_batches=3, sync=True)
        for j in range(50):
            for i in range(3):
                if j == 0 or (j + i) % 17 == 0:
                    env1.reset_start(sub_batch=i)
                    env2.reset_start(sub_batch=i)
                    obs1 = env1.reset_wait(sub_batch=i)
                    obs2 = env2.reset_wait(sub_batch=i)
                    self.assertTrue(np.allclose(obs1, obs2))
                actions = [env1.action_space.sample() for _ in range(2)]
                env1.step_start(actions, sub_batch=i)
                env2.step_start(actions, sub_batch=i)
                obs1, rews1, dones1, _ = env1.step_wait(sub_batch=i)
                obs2, rews2, dones2, _ = env2.step_wait(sub_batch=i)
                self.assertTrue(np.allclose(obs1, obs2))
                self.assertTrue(np.array(rews1 == rews2).all())
                self.assertTrue(np.array(dones1 == dones2).all())

class ObservationPadEnvTest(unittest.TestCase):
    """
    Tests for ObservationPadEnv.
    """
    def test_centered(self):
        """
        Test centered padding.
        """
        env = ShapeEnv(np.array([[1, 2, 5], [3, 4, 0]]),
                       np.array([[12, 21, 15], [31, 14, 10]]))
        padded = ObservationPadEnv(env, (3, 5))
        self.assertEqual(padded.observation_space.shape, (3, 5))
        self.assertTrue(np.allclose(padded.observation_space.low,
                                    np.array([[0, 1, 2, 5, 0], [0, 3, 4, 0, 0], [0]*5])))
        self.assertTrue(np.allclose(padded.observation_space.high,
                                    np.array([[0, 12, 21, 15, 0], [0, 31, 14, 10, 0], [0]*5])))

    def test_uncentered(self):
        """
        Test uncentered padding.
        """
        env = ShapeEnv(np.array([[1, 2, 5], [3, 4, 0]]),
                       np.array([[12, 21, 15], [31, 14, 10]]))
        padded = ObservationPadEnv(env, (3, 5), center=False)
        self.assertEqual(padded.observation_space.shape, (3, 5))
        self.assertTrue(np.allclose(padded.observation_space.low,
                                    np.array([[1, 2, 5, 0, 0], [3, 4, 0, 0, 0], [0]*5])))
        self.assertTrue(np.allclose(padded.observation_space.high,
                                    np.array([[12, 21, 15, 0, 0], [31, 14, 10, 0, 0], [0]*5])))

class LoggedEnvTest(unittest.TestCase):
    """
    Tests for LoggedEnv.
    """
    def test_single_env(self):
        """
        Test monitoring for a single environment.
        """
        dirpath = tempfile.mkdtemp()
        try:
            log_file = os.path.join(dirpath, 'monitor.csv')
            env = LoggedEnv(SimpleEnv(2, (3,), 'float32'), log_file)
            for _ in range(4):
                env.reset()
                while not env.step(env.action_space.sample())[2]:
                    pass
            env.close()
            with open(log_file, 'rt'):
                log_contents = pandas.read_csv(log_file)
                self.assertEqual(list(log_contents['r']), [2] * 4)
                self.assertEqual(list(log_contents['l']), [3] * 4)
        finally:
            shutil.rmtree(dirpath)

    def test_multi_env(self):
        """
        Test monitoring for concurrent environments.
        """
        dirpath = tempfile.mkdtemp()
        try:
            log_file = os.path.join(dirpath, 'monitor.csv')
            env1 = LoggedEnv(SimpleEnv(2, (3,), 'float32'), log_file, use_locking=True)
            env2 = LoggedEnv(SimpleEnv(3, (3,), 'float32'), log_file, use_locking=True)

            env1.reset()
            env2.reset()
            for _ in range(13):
                for env in [env1, env2]:
                    if env.step(env.action_space.sample())[2]:
                        env.reset()
            env1.close()
            env2.close()

            with open(log_file, 'rt'):
                log_contents = pandas.read_csv(log_file)
                self.assertEqual(list(log_contents['r']), [2, 2.5, 2, 2.5, 2, 2, 2.5])
                self.assertEqual(list(log_contents['l']), [3, 4, 3, 4, 3, 3, 4])
        finally:
            shutil.rmtree(dirpath)

class ShapeEnv(gym.Env):
    """
    An environment with a pre-defined observation shape.
    """
    def __init__(self, low, high):
        super(ShapeEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low, high, dtype=low.dtype)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0, False, {}

    def render(self, mode='human'):
        pass

if __name__ == '__main__':
    unittest.main()
