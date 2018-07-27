"""
Ways of running gym environments.
"""

from multiprocessing import Array, Pipe, Process

import cloudpickle
import gym.spaces
import numpy as np

from .base import AsyncEnv, BatchedEnv, BatchedAsyncEnv


def batched_gym_env(env_fns, observation_space=None, num_sub_batches=1, sync=False):
    """
    Create a BatchedEnv that controls a set of Gym
    environments.

    Environments are passed via env_fns, which is a list
    of functions that create environments.
    These functions must be picklable with cloudpickle.

    If the observation space is not specified, the first
    function in env_fns will be called an extra time to
    figure out the observation space.

    On top of the normal BatchedEnv interface, the result
    has action_space and observation_space attributes.
    """
    assert len(env_fns) % num_sub_batches == 0
    if not sync and observation_space is None:
        observation_space = _fetch_obs_space(env_fns[0])
    sub_batches = []
    batch_size = len(env_fns) // num_sub_batches
    for i in range(num_sub_batches):
        batch_fns = env_fns[i * batch_size:(i + 1) * batch_size]
        if sync:
            envs = [fn() for fn in batch_fns]
        else:
            envs = [AsyncGymEnv(fn, observation_space) for fn in batch_fns]
        sub_batches.append(envs)
    if sync:
        return BatchedGymEnv(sub_batches)
    return BatchedAsyncEnv(sub_batches)


def _fetch_obs_space(env_fn):
    def _worker_fn(pipe, fn_data):
        try:
            env = cloudpickle.loads(fn_data)()
        except BaseException as exc:
            pipe.send((None, exc))
            return
        try:
            space = env.observation_space
            pipe.send((space, None))
        finally:
            env.close()
    parent_pipe, child_pipe = Pipe()
    proc = Process(target=_worker_fn, args=(child_pipe, cloudpickle.dumps(env_fn)))
    proc.start()
    child_pipe.close()
    try:
        space, exc = parent_pipe.recv()
    except EOFError:
        raise RuntimeError('worker has died')
    finally:
        proc.join()
    if exc is not None:
        raise RuntimeError('exception from environment') from exc
    return space


class AsyncGymEnv(AsyncEnv):
    """
    An AsyncEnv that controls a Gym environment in a
    subprocess.

    On top of the AsyncEnv interface, the resulting object
    has observation_space and action_space attributes.
    """

    def __init__(self, make_env, observation_space):
        self.observation_space = observation_space
        if isinstance(observation_space, gym.spaces.Box):
            num_elems = len(np.array(observation_space.low).flatten())
            zeros = [0] * num_elems
            self._obs_buf = Array('b', zeros)
        else:
            self._obs_buf = None
        self._pipe, other_end = Pipe()
        self._proc = Process(target=self._worker,
                             args=(other_end, self._obs_buf),
                             daemon=True)
        self._proc.start()
        self._running_cmd = None
        other_end.close()
        self._pipe.send(cloudpickle.dumps(make_env))
        self.action_space = self._get_response()

    def reset_start(self):
        assert self._running_cmd is None
        self._running_cmd = 'reset'
        self._pipe.send(('reset', None))

    def reset_wait(self):
        assert self._running_cmd == 'reset'
        res = self._decode_observation(self._get_response())
        self._running_cmd = None
        return res

    def step_start(self, action):
        assert self._running_cmd is None
        self._running_cmd = 'step'
        self._pipe.send(('step', action))

    def step_wait(self):
        assert self._running_cmd == 'step'
        obs, rew, done, info = self._get_response()
        obs = self._decode_observation(obs)
        self._running_cmd = None
        return obs, rew, done, info

    def close(self):
        if self._running_cmd == 'reset':
            self.reset_wait()
        elif self._running_cmd == 'step':
            self.step_wait()
        assert self._running_cmd is None
        self._running_cmd = 'close'
        self._pipe.send(('close', None))
        self._proc.join()
        self._pipe.close()

    def _get_response(self):
        """
        Read a value from the response queue and make sure
        it is not an exception.
        """
        try:
            resp_obj = self._pipe.recv()
        except EOFError:
            raise RuntimeError('worker has died')
        if isinstance(resp_obj, BaseException):
            raise RuntimeError('exception on worker') from resp_obj
        return resp_obj

    def _decode_observation(self, obs):
        """
        Get an observation out of an object that the
        worker sent.
        """
        if obs is not None:
            return obs
        obs = np.frombuffer(self._obs_buf.get_obj(), dtype='uint8')
        shape = self.observation_space.low.shape
        return obs.reshape(shape).copy()

    @classmethod
    def _worker(cls, pipe, obs_buf):
        """
        Entry-point for the sub-process.
        """
        make_env = pipe.recv()
        try:
            env = cloudpickle.loads(make_env)()
        except BaseException as exc:
            pipe.send(exc)
            return
        pipe.send(env.action_space)
        try:
            while True:
                cmd, action = pipe.recv()
                if cmd == 'reset':
                    pipe.send(cls._sendable_observation(env.reset(), obs_buf))
                elif cmd == 'step':
                    obs, rew, done, info = env.step(action)
                    if done:
                        obs = env.reset()
                    obs = cls._sendable_observation(obs, obs_buf)
                    pipe.send((obs, rew, done, info))
                elif cmd == 'close':
                    return
                else:
                    raise ValueError('unknown command: ' + cmd)
        except BaseException as exc:
            pipe.send(exc)
        finally:
            env.close()

    @staticmethod
    def _sendable_observation(obs, obs_buf):
        """
        Efficiently communicate the observation to the parent
        process, either via shared memory or by returning a
        picklable object.
        """
        if obs_buf is None:
            return obs
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if obs.dtype != 'uint8':
            return obs
        dst = obs_buf.get_obj()
        dst_np = np.frombuffer(dst, dtype=obs.dtype).reshape(obs.shape)
        np.copyto(dst_np, obs)
        return None


class BatchedGymEnv(BatchedEnv):
    """
    A BatchedEnv that wraps a bunch of existing Gym
    environments and runs them synchronously.
    """

    def __init__(self, envs):
        self.action_space = envs[0][0].action_space
        self.observation_space = envs[0][0].observation_space
        self.envs = envs
        self._step_actions = [None] * len(self.envs)
        self._states = [None] * len(self.envs)

    @property
    def num_sub_batches(self):
        return len(self.envs)

    @property
    def num_envs_per_sub_batch(self):
        return len(self.envs[0])

    def reset_start(self, sub_batch=0):
        assert self._states[sub_batch] is None
        self._states[sub_batch] = 'reset'

    def reset_wait(self, sub_batch=0):
        assert self._states[sub_batch] == 'reset'
        self._states[sub_batch] = None
        return [env.reset() for env in self.envs[sub_batch]]

    def step_start(self, actions, sub_batch=0):
        assert len(actions) == self.num_envs_per_sub_batch
        assert self._states[sub_batch] is None
        self._states[sub_batch] = 'step'
        self._step_actions[sub_batch] = actions

    def step_wait(self, sub_batch=0):
        assert self._step_actions[sub_batch] is not None
        assert self._states[sub_batch] == 'step'
        self._states[sub_batch] = None
        obses, rews, dones, infos = ([], [], [], [])
        for env, action in zip(self.envs[sub_batch], self._step_actions[sub_batch]):
            obs, rew, done, info = env.step(action)
            if done:
                obs = env.reset()
            obses.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        self._step_actions[sub_batch] = None
        return obses, rews, dones, infos

    def close(self):
        for batch in self.envs:
            for env in batch:
                env.close()
