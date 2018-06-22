"""
Tests for environment wrappers.
"""

import os
import tempfile

import gym
import numpy as np
import pandas
import pytest
import tensorflow as tf

from anyrl.envs import batched_gym_env
from anyrl.envs.wrappers import (RL2Env, DownsampleEnv, GrayscaleEnv, FrameSkipEnv, FrameStackEnv,
                                 MaxEnv, ResizeImageEnv, BatchedFrameStack, ObservationPadEnv,
                                 LoggedEnv)
from anyrl.tests import SimpleEnv


def test_rl2_num_eps():
    """
    Test that RL^2 meta-episodes contain the right number
    of sub-episodes.
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
            assert done
            break
        else:
            assert not done


def test_rl2_rewards():
    """
    Test that rewards in RL^2 are masked properly.
    """
    real_env = SimpleEnv(1, (3, 4), 'uint8')
    env = RL2Env(real_env, real_env.action_space.sample(), num_eps=5, warmup_eps=-2)
    env.reset()
    done_eps = 0
    while done_eps < 3:
        obs, rew, _, _ = env.step(env.action_space.sample())
        assert rew == 0
        if obs[3]:
            done_eps += 1
    while True:
        _, rew, done, _ = env.step(env.action_space.sample())
        assert rew != 0
        if done:
            break


def test_downsample_rate_1():
    """
    Test DownsampleEnv with rate=1.
    """
    low = np.array([[1, 2], [3, 4]])
    high = np.array([[3, 4], [5, 6]])
    env = DownsampleEnv(ShapeEnv(low, high), 1)
    assert (env.observation_space.low == low).all()
    assert (env.observation_space.high == high).all()


def test_downsample_rate_2():
    """
    Test DownsampleEnv with rate=2.
    """
    low = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    high = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])
    env = DownsampleEnv(ShapeEnv(low, high), 2)
    assert env.observation_space.shape == (1, 2)
    assert (env.observation_space.low == np.array([[1, 3]])).all()
    assert (env.observation_space.high == np.array([[9, 11]])).all()

    low = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 6, 7, 8]])
    high = np.array([[9, 10, 11, 12], [13, 14, 15, 16], [16, 14, 15, 16]])
    env = DownsampleEnv(ShapeEnv(low, high), 2)
    assert env.observation_space.shape == (2, 2)
    assert (env.observation_space.low == np.array([[1, 3], [9, 7]])).all()
    assert (env.observation_space.high == np.array([[9, 11], [16, 15]])).all()


def test_grayscale_integers():
    """
    Tests GrayscaleEnv for integer observations.
    """
    low = np.array([[[0, 0, 0], [16, 16, 50]]])
    high = np.array([[[19, 255, 13], [255, 255, 255]]])
    env = GrayscaleEnv(ShapeEnv(low, high))
    assert env.observation_space.shape == (1, 2, 1)
    assert (env.observation_space.low == np.array([[[0], [26]]])).all()
    assert (env.observation_space.high == np.array([[[95], [255]]])).all()


def test_stack_3():
    """
    Test FrameStackEnv for 3 frames.
    """
    low = np.zeros((4, 5, 2))
    high = np.zeros((4, 5, 2)) + 0xff
    env = FrameStackEnv(ShapeEnv(low, high), 3)
    assert env.observation_space.shape == (4, 5, 6)
    obs1 = env.reset()
    obs2, _, _, _ = env.step(0)
    obs3, _, _, _ = env.step(0)
    assert not (obs1 == obs2).all()
    assert not (obs1 == obs3).all()
    assert not (obs2 == obs3).all()
    assert (obs1[:, :, 2:] == obs2[:, :, :4]).all()
    assert (obs2[:, :, 2:] == obs3[:, :, :4]).all()


def test_stack_3_strided():
    """
    Test FrameStackEnv for 3 frames with a stride of 2.
    """
    low = np.zeros((4, 5, 2))
    high = np.zeros((4, 5, 2)) + 0xff
    env = FrameStackEnv(ShapeEnv(low, high), 3, stride=2)
    assert env.observation_space.shape == (4, 5, 6)
    obses = [env.reset()]
    for _ in range(5):
        new_obs, _, _, _ = env.step(0)
        for obs in obses:
            assert not (obs == new_obs).all()
        obses.append(new_obs)
    assert (obses[-1][:, :, :-2] == obses[-3][:, :, 2:]).all()
    assert (obses[-2][:, :, :-2] == obses[-4][:, :, 2:]).all()
    assert (obses[-3][:, :, :-2] == obses[-5][:, :, 2:]).all()


def test_stack_3_no_concat():
    """
    Test FrameStackEnv for 3 frames with no concatenation.
    """
    low = np.zeros((4, 5, 2))
    high = np.zeros((4, 5, 2)) + 0xff
    env = FrameStackEnv(ShapeEnv(low, high), 3, concat=False)
    assert env.observation_space.box.shape == (4, 5, 2)
    assert env.observation_space.count == 3
    obs1 = env.reset()
    obs2, _, _, _, = env.step(0)
    assert np.allclose(obs2[0], obs1[0])
    assert not np.allclose(obs2[-1], obs1[0])


def test_stack_3_no_concat_strided():
    """
    Test FrameStackEnv for 3 frames with no concatenation
    and a stride of 2.
    """
    low = np.zeros((4, 5, 2))
    high = np.zeros((4, 5, 2)) + 0xff
    env = FrameStackEnv(ShapeEnv(low, high), 3, concat=False, stride=2)
    assert env.observation_space.box.shape == (4, 5, 2)
    assert env.observation_space.count == 3
    obses = [env.reset()]
    for _ in range(5):
        new_obs, _, _, _ = env.step(0)
        new_obs = np.array(new_obs)
        for obs in obses:
            assert not (obs == new_obs).all()
        obses.append(new_obs)
    assert (obses[-1][:-1] == obses[-3][1:]).all()
    assert (obses[-2][:-1] == obses[-4][1:]).all()
    assert (obses[-3][:-1] == obses[-5][1:]).all()


def test_skip():
    """
    Test a FrameSkipEnv wrapper.
    """
    # Timestep limit is 5.
    env = SimpleEnv(4, (3, 2, 5), 'float32')
    act1 = np.random.uniform(high=255.0, size=(3, 2, 5))
    act2 = np.random.uniform(high=255.0, size=(3, 2, 5))
    obs1 = env.reset()
    rew1 = 0.0
    rew2 = 0.0
    for _ in range(3):
        obs2, rew, _, _ = env.step(act1)
        rew1 += rew
    for _ in range(2):
        obs3, rew, done, _ = env.step(act2)
        rew2 += rew
    assert done

    env = FrameSkipEnv(env, num_frames=3)
    actual_obs1 = env.reset()
    assert np.allclose(actual_obs1, obs1)
    actual_obs2, actual_rew1, done, _ = env.step(act1)
    assert not done
    assert actual_rew1 == rew1
    assert np.allclose(actual_obs2, obs2)
    actual_obs3, actual_rew2, done, _ = env.step(act2)
    assert done
    assert actual_rew2 == rew2
    assert np.allclose(actual_obs3, obs3)


def test_max_2():
    """
    Test MaxEnv for 2 frames.
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
    assert (max1 == frame1).all()
    assert (max2 == np.max([frame1, frame2], axis=0)).all()
    assert (max3 == np.max([frame2, frame3], axis=0)).all()
    assert (max4 == np.max([frame3, frame4], axis=0)).all()
    assert (max5 == np.max([frame4, frame5], axis=0)).all()


def test_resize_even():
    """
    Test ResizeImageEnv for an even number of pixels.
    """
    env = SimpleEnv(5, (13, 5, 3), 'float32')
    frame = env.reset()
    actual = ResizeImageEnv(env, size=(5, 4)).reset()
    expected = tf.Session().run(
        tf.image.resize_images(frame, [5, 4], method=tf.image.ResizeMethod.AREA))
    assert actual.shape == (5, 4, 3)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize('concat,stride', [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_batched_stack(concat, stride):
    """
    Test that BatchedFrameStack is equivalent to a regular
    batched FrameStackEnv.
    """
    envs = [lambda idx=i: SimpleEnv(idx+2, (3, 2, 5), 'float32') for i in range(6)]
    env1 = BatchedFrameStack(batched_gym_env(envs, num_sub_batches=3, sync=True),
                             concat=concat, stride=stride)
    env2 = batched_gym_env([lambda env=e: FrameStackEnv(env(), concat=concat, stride=stride)
                            for e in envs],
                           num_sub_batches=3, sync=True)
    for j in range(50):
        for i in range(3):
            if j == 0 or (j + i) % 17 == 0:
                env1.reset_start(sub_batch=i)
                env2.reset_start(sub_batch=i)
                obs1 = env1.reset_wait(sub_batch=i)
                obs2 = env2.reset_wait(sub_batch=i)
                assert np.allclose(obs1, obs2)
            actions = [env1.action_space.sample() for _ in range(2)]
            env1.step_start(actions, sub_batch=i)
            env2.step_start(actions, sub_batch=i)
            obs1, rews1, dones1, _ = env1.step_wait(sub_batch=i)
            obs2, rews2, dones2, _ = env2.step_wait(sub_batch=i)
            assert np.allclose(obs1, obs2)
            assert np.array(rews1 == rews2).all()
            assert np.array(dones1 == dones2).all()


def test_obs_pad_centered():
    """
    Test ObservationPadEnv with centered padding.
    """
    env = ShapeEnv(np.array([[1, 2, 5], [3, 4, 0]]),
                   np.array([[12, 21, 15], [31, 14, 10]]))
    padded = ObservationPadEnv(env, (3, 5))
    assert padded.observation_space.shape == (3, 5)
    assert np.allclose(padded.observation_space.low,
                       np.array([[0, 1, 2, 5, 0], [0, 3, 4, 0, 0], [0]*5]))
    assert np.allclose(padded.observation_space.high,
                       np.array([[0, 12, 21, 15, 0], [0, 31, 14, 10, 0], [0]*5]))


def test_obs_pad_uncentered():
    """
    Test ObservationPadEnv with uncentered padding.
    """
    env = ShapeEnv(np.array([[1, 2, 5], [3, 4, 0]]),
                   np.array([[12, 21, 15], [31, 14, 10]]))
    padded = ObservationPadEnv(env, (3, 5), center=False)
    assert padded.observation_space.shape == (3, 5)
    assert np.allclose(padded.observation_space.low,
                       np.array([[1, 2, 5, 0, 0], [3, 4, 0, 0, 0], [0]*5]))
    assert np.allclose(padded.observation_space.high,
                       np.array([[12, 21, 15, 0, 0], [31, 14, 10, 0, 0], [0]*5]))


def test_logged_single_env():
    """
    Test LoggedEnv for a single environment.
    """
    with tempfile.TemporaryDirectory() as dirpath:
        log_file = os.path.join(dirpath, 'monitor.csv')
        env = LoggedEnv(SimpleEnv(2, (3,), 'float32'), log_file)
        for _ in range(4):
            env.reset()
            while not env.step(env.action_space.sample())[2]:
                pass
        env.close()
        with open(log_file, 'rt'):
            log_contents = pandas.read_csv(log_file)
            assert list(log_contents['r']) == [2] * 4
            assert list(log_contents['l']) == [3] * 4


def test_multi_env():
    """
    Test monitoring for concurrent environments.
    """
    with tempfile.TemporaryDirectory() as dirpath:
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
            assert list(log_contents['r']) == [2, 2.5, 2, 2.5, 2, 2, 2.5]
            assert list(log_contents['l']) == [3, 4, 3, 4, 3, 3, 4]


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
