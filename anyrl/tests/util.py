"""
Utilities for RL-related tests.
"""

import gym
import numpy as np

from anyrl.rollouts import BatchedEnv
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
        self.action_space = gym.spaces.Box(0, 0xff, shape=shape)
        self.observation_space = self.action_space

    def _step(self, action):
        flat_act = np.array(action, dtype=self._dtype)
        self._cur_obs = self._cur_obs + np.roll(flat_act, 1)
        self._cur_step += 1
        done = self._cur_step >= self._max_steps
        reward = self._cur_step / self._max_steps
        return self._cur_obs, reward, done, {'foo': 'bar' + str(reward)}

    def _reset(self):
        self._cur_obs = self._start_obs
        self._cur_step = 0
        return self._cur_obs

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
        return {
            'actions': actions,
            'values': values,
            'states': new_states
        }

class DummyBatchedEnv(BatchedEnv):
    """
    The simplest possible batched environment.
    """
    def __init__(self, env_fns, num_sub_batches):
        env_fn_batches = []
        batch = len(env_fns) // num_sub_batches
        for i in range(num_sub_batches):
            env_fn_batches.append(env_fns[i*batch : (i+1)*batch])
        self._envs = [[f() for f in fs] for fs in env_fn_batches]
        self._step_actions = [None] * len(self._envs)

    @property
    def num_sub_batches(self):
        return len(self._envs)

    @property
    def num_envs_per_sub_batch(self):
        return len(self._envs[0])

    def reset_start(self, sub_batch=0):
        pass

    def reset_wait(self, sub_batch=0):
        return [env.reset() for env in self._envs[sub_batch]]

    def step_start(self, actions, sub_batch=0):
        self._step_actions[sub_batch] = actions

    def step_wait(self, sub_batch=0):
        obses, rews, dones, infos = ([], [], [], [])
        for env, action in zip(self._envs[sub_batch], self._step_actions[sub_batch]):
            obs, rew, done, info = env.step(action)
            if done:
                obs = env.reset()
            obses.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        return obses, rews, dones, infos

    def close(self):
        for batch in self._envs:
            for env in batch:
                env.close()
