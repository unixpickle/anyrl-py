"""
Wrappers for RL^2.

See: https://arxiv.org/abs/1611.02779.
"""

import gym
import gym.spaces as spaces
import numpy as np

class RL2Env(gym.Wrapper):
    """
    A wrapper which augments the observation space to
    include the action, reward, and done value.

    Creates a tuple observation space:
      (observation, action, reward, done).
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
            spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            spaces.MultiBinary(1)
        ])
        self.num_eps = num_eps
        self.warmup_eps = warmup_eps
        self._done_eps = 0

    def _reset(self):
        self._done_eps = 0
        obs = self.env.reset()
        return (obs, self.first_action, np.array([0.0]), np.array([0]))

    def _step(self, action):
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
