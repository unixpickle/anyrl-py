"""
Wrappers for meta-learning.
"""

import random

import gym
import gym.spaces as spaces
import numpy as np

# pylint: disable=E0202,W0221


class RL2Env(gym.Wrapper):
    """
    A wrapper which augments the observation space to
    include the action, reward, and done value.

    Creates a tuple observation space:
      (observation, action, reward, done).

    See: https://arxiv.org/abs/1611.02779.
    """

    def __init__(self, env, first_action, num_eps=1, warmup_eps=0):
        """
        Parameters:
          env: the environment to wrap.
          first_action: the action to include in the first
            observation.
          num_eps: episodes per meta-episode.
          warmup_eps: the number of episodes at the start
            of a meta-episode for which rewards are 0.
            Negative values are added to num_eps.
        """
        if warmup_eps < 0:
            warmup_eps += num_eps
        super(RL2Env, self).__init__(env)
        self.first_action = first_action
        self.observation_space = spaces.Tuple([
            env.observation_space,
            env.action_space,
            spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype='float'),
            spaces.MultiBinary(1)
        ])
        self.num_eps = num_eps
        self.warmup_eps = warmup_eps
        self._done_eps = 0

    def reset(self, **kwargs):
        self._done_eps = 0
        obs = self.env.reset(**kwargs)
        return (obs, self.first_action, np.array([0.0]), np.array([0]))

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        aug_obs = (obs, action, np.array([rew]), np.array([int(done)]))
        if self._done_eps < self.warmup_eps:
            rew = 0.0
        if done:
            self._done_eps += 1
            if self._done_eps < self.num_eps:
                done = False
                aug_obs = (self.env.reset(),) + aug_obs[1:]
        return aug_obs, rew, done, info


class SwitchableEnv(gym.Env):
    """
    An environment that proxies calls to another
    environment that can be swapped out at any time.

    This is useful in conjunction with RL2Env to swap out
    the environment after every meta-episode.
    """

    def __init__(self, first_env):
        self.env = first_env
        self.action_space = first_env.action_space
        self.observation_space = first_env.observation_space

    def switch_env(self, new_env):
        """
        Switch to a new environment.

        The new environment must have the same spaces as
        the old one.
        """
        self.env = new_env

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def seed(self, seed=None):
        return self.env.seed(seed=seed)


class JointEnv(gym.Env):
    """
    An environment that samples a new sub-environment at
    every episode boundary.

    This can be used for joint-training.
    """

    def __init__(self, env_fns, env_names=None):
        """
        Create a joint environment.

        Args:
          env_fns: a sequence of callables that construct
            new environments. All environments must have
            the same spaces.
          env_names: if specified, a sequence of names for
            the env_fns. The name is included in the info
            dict under 'env_name'.
        """
        self.env_fns = env_fns
        self.env_names = env_names
        self.env = None
        self.env_idx = 0
        env = env_fns[0]()
        try:
            self.action_space = env.action_space
            self.observation_space = env.observation_space
        finally:
            env.close()

    def reset(self, **kwargs):
        if self.env is not None:
            self.env.close()
        self.env_idx = random.randrange(len(self.env_fns))
        self.env = self.env_fns[self.env_idx]()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info = info.copy()
        info['env_idx'] = self.env_idx
        if self.env_names is not None:
            info['env_name'] = self.env_names[self.env_idx]
        return obs, rew, done, info

    def render(self, mode='human'):
        if self.env is None:
            return
        return self.env.render(mode=mode)

    def seed(self, seed=None):
        if self.env is None:
            return
        return self.env.seed(seed=seed)
