"""
Utilities for RL-related tests.
"""

import gym
import gym.spaces as spaces
import numpy as np

from anyrl.models import Model


class SimpleEnv(gym.Env):
    """
    An environment with a pre-determined observation space
    and RNG seed.
    """

    def __init__(self, seed, shape, dtype):
        np.random.seed(seed)
        self._dtype = dtype
        self._start_obs = np.array(np.random.randint(0, 0x100, size=shape),
                                   dtype=dtype)
        self._max_steps = seed + 1
        self._cur_obs = None
        self._cur_step = 0
        self.action_space = gym.spaces.Box(0, 0xff, shape=shape, dtype=dtype)
        self.observation_space = self.action_space

    def step(self, action):
        flat_act = np.array(action, dtype=self._dtype)
        self._cur_obs = self._cur_obs + np.roll(flat_act, 1)
        self._cur_step += 1
        done = self._cur_step >= self._max_steps
        reward = self._cur_step / self._max_steps
        return self._cur_obs, reward, done, {'foo': 'bar' + str(reward)}

    def reset(self):
        self._cur_obs = self._start_obs
        self._cur_step = 0
        return self._cur_obs

    def render(self, mode='human'):
        pass


class SimpleModel(Model):
    """
    A stateful, deterministic model which is compatible
    with a SimpleEnv.
    """

    def __init__(self, shape, stateful=False, state_tuple=True):
        self.shape = shape
        self._stateful = stateful
        self.state_tuple = state_tuple

    @property
    def stateful(self):
        return self._stateful

    def start_state(self, batch_size):
        if not self.stateful:
            return None
        if self.state_tuple:
            return (np.zeros((batch_size,) + self.shape),
                    np.zeros((batch_size, 1)))
        return np.zeros((batch_size,) + self.shape)

    def step(self, observations, states):
        actions = []
        values = []
        if not self.stateful:
            new_states = None
        elif not self.state_tuple:
            new_states = []
        else:
            new_states = ([], [])
        for i, obs in enumerate(observations):
            if not self.stateful:
                actions.append(obs*3)
            elif self.state_tuple:
                actions.append((states[0][i] + obs) * states[1][i])
                new_states[0].append(states[0][i] + 2*obs)
                new_states[1].append(3 - states[1][i])
            else:
                actions.append(states[i] + obs)
                new_states.append(states[i] + 2*obs)
            values.append(np.sum(obs))
        if new_states:
            if self.state_tuple:
                new_states = tuple(np.array(x) for x in new_states)
            else:
                new_states = np.array(new_states)
        return {
            'actions': actions,
            'values': values,
            'states': new_states
        }


class TupleCartPole(gym.Env):
    """
    A version of Gym's CartPole-v0 with tuple spaces.

    Intended to stress test models against weird kinds of
    spaces.
    """

    def __init__(self):
        self._inner_env = gym.make('CartPole-v0')
        self.action_space = spaces.Tuple([spaces.Discrete(2), spaces.Discrete(3)])
        obs_space = self._inner_env.observation_space
        self.observation_space = spaces.Tuple([
            spaces.Box(obs_space.low[:3], obs_space.high[:3], dtype=obs_space.dtype),
            spaces.Box(obs_space.low[3:], obs_space.high[3:], dtype=obs_space.dtype)
        ])

    def reset(self):
        return self._split_obs(self._inner_env.reset())

    def step(self, action):
        obs, rew, done, info = self._inner_env.step(action[0])
        return self._split_obs(obs), rew, done, info

    def render(self, mode='human'):
        pass

    def _split_obs(self, obs):
        obs = np.array(obs)
        return (obs[:3], obs[3:])
