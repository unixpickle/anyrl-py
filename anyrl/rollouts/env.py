"""
Ways of running gym environments.
"""

from abc import ABC, abstractmethod, abstractproperty
from multiprocessing import Queue, Array, Process

import cloudpickle
import gym.spaces
import numpy as np

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
        env = env_fns[0]()
        try:
            observation_space = env.observation_space
        finally:
            env.close()
    sub_batches = []
    batch_size = len(env_fns) // num_sub_batches
    for i in range(num_sub_batches):
        batch_fns = env_fns[i*batch_size : (i+1)*batch_size]
        if sync:
            envs = [fn() for fn in batch_fns]
        else:
            envs = [AsyncGymEnv(fn, observation_space) for fn in batch_fns]
        sub_batches.append(envs)
    if sync:
        return BatchedGymEnv(sub_batches)
    return BatchedAsyncEnv(sub_batches)

class AsyncEnv(ABC):
    """
    An asynchronous environment.
    """
    @abstractmethod
    def reset_start(self):
        """
        Start resetting the environment.

        This should not be called while any other
        asynchronous operations are taking place.
        """
        pass

    @abstractmethod
    def reset_wait(self):
        """
        Wait for a reset_start() to finish.

        Returns an observation.

        The resulting array belongs to the caller.
        It should not be modified after-the-fact by the
        environment.
        """
        pass

    @abstractmethod
    def step_start(self, action):
        """
        Start taking a step in the environment.

        This should not be called while any other
        asynchronous operations are taking place.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for a step_start() to finish.

        Returns (observation, reward, done, info).

        If done is true, then the environment was reset
        and the new observation was taken.

        The resulting arrays belong to the caller.
        They should not be modified after-the-fact by the
        environment.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environment's resources.

        Waits for any pending operations to complete.
        """
        pass

class BatchedEnv(ABC):
    """
    A set of environments running in batches.

    The batch of environments is divided up into
    equally sized sub-batches.
    Each sub-batch is a set of environments that run in
    lockstep.

    Different BatchedEnvs may schedule jobs in different
    ways, but generally it will be FIFO order.
    Thus, it's best to wait for jobs in the same order
    that you started them.
    """
    @property
    def num_envs(self):
        """
        The total number of environments.
        """
        return self.num_sub_batches * self.num_envs_per_sub_batch

    @abstractproperty
    def num_sub_batches(self):
        """
        The number of sub-batches.
        """
        pass

    @abstractproperty
    def num_envs_per_sub_batch(self):
        """
        The number of environments per sub-batch.
        """
        pass

    @abstractmethod
    def reset_start(self, sub_batch=0):
        """
        Start resetting the sub-batch.

        This should not be called while any other
        operations are taking place for the sub-batch.
        """
        pass

    @abstractmethod
    def reset_wait(self, sub_batch=0):
        """
        Wait for a reset_start() to finish.

        Returns a list-like object of observations.

        The resulting array belongs to the caller.
        It should not be modified after-the-fact by the
        environment.
        """
        pass

    @abstractmethod
    def step_start(self, actions, sub_batch=0):
        """
        Start taking a step in the batch of environments.
        Takes a list-like object of actions.

        This should not be called while any other
        asynchronous operations are taking place.
        """
        pass

    @abstractmethod
    def step_wait(self, sub_batch=0):
        """
        Wait for a step_start() to finish.

        Returns (observations, rewards, dones, infos),
        where all those are list-like objects.

        If a done value is true, then the environment was
        automatically reset and the new observation was
        returned.

        The resulting arrays belong to the caller.
        They should not be modified after-the-fact by the
        environment.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.

        Waits for any pending operations to complete.
        """
        pass

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
        self._req_queue = Queue()
        self._resp_queue = Queue()
        self._proc = Process(target=self._worker,
                             args=(self._req_queue,
                                   self._resp_queue,
                                   self._obs_buf,
                                   _CloudpickleFunc(make_env)))
        self._proc.start()
        self._running_cmd = None
        self._req_queue.put(('action_space', None))
        self.action_space = self._get_response()

    def reset_start(self):
        assert self._running_cmd is None
        self._running_cmd = 'reset'
        self._req_queue.put(('reset', None))

    def reset_wait(self):
        assert self._running_cmd == 'reset'
        res = self._decode_observation(self._get_response())
        self._running_cmd = None
        return res

    def step_start(self, action):
        assert self._running_cmd is None
        self._running_cmd = 'step'
        self._req_queue.put(('step', action))

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
        self._req_queue.put(('close', None))
        self._proc.join()

    def _get_response(self):
        """
        Read a value from the response queue and make sure
        it is not an exception.
        """
        resp_obj = self._resp_queue.get()
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
    def _worker(cls, req_queue, resp_queue, obs_buf, make_env):
        """
        Entry-point for the sub-process.
        """
        env = make_env()
        try:
            while True:
                cmd, action = req_queue.get()
                if cmd == 'action_space':
                    resp_queue.put(env.action_space)
                elif cmd == 'reset':
                    resp_queue.put(cls._sendable_observation(env.reset(), obs_buf))
                elif cmd == 'step':
                    obs, rew, done, info = env.step(action)
                    if done:
                        obs = env.reset()
                    obs = cls._sendable_observation(obs, obs_buf)
                    resp_queue.put((obs, rew, done, info))
                elif cmd == 'close':
                    return
                else:
                    raise ValueError('unknown command: ' + cmd)
        except BaseException as exc:
            resp_queue.put(exc)
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

class BatchedAsyncEnv(BatchedEnv):
    """
    A BatchedEnv that controls AsyncEnvs.

    If the first AsyncEnv has an action_space and/or
    observation_space attribute, those attributes are
    copied.
    """
    def __init__(self, sub_batches):
        # pylint: disable=C1801
        assert len(sub_batches) > 0
        first_len = len(sub_batches[0])
        assert all([len(x) == first_len for x in sub_batches])
        self._sub_batches = sub_batches

        self.action_space = None
        self.observation_space = None

        first_env = sub_batches[0][0]
        if hasattr(first_env, 'action_space'):
            self.action_space = sub_batches[0][0].action_space
        if hasattr(first_env, 'observation_space'):
            self.observation_space = sub_batches[0][0].observation_space

    @property
    def num_sub_batches(self):
        return len(self._sub_batches)

    @property
    def num_envs_per_sub_batch(self):
        return len(self._sub_batches[0])

    def reset_start(self, sub_batch=0):
        for env in self._sub_batches[sub_batch]:
            env.reset_start()

    def reset_wait(self, sub_batch=0):
        obses = []
        for env in self._sub_batches[sub_batch]:
            obses.append(env.reset_wait())
        return obses

    def step_start(self, actions, sub_batch=0):
        assert len(actions) == self.num_envs_per_sub_batch
        for env, action in zip(self._sub_batches[sub_batch], actions):
            env.step_start(action)

    def step_wait(self, sub_batch=0):
        obses, rews, dones, infos = ([], [], [], [])
        for env in self._sub_batches[sub_batch]:
            obs, rew, done, info = env.step_wait()
            obses.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        return obses, rews, dones, infos

    def close(self):
        for batch in self._sub_batches:
            for env in batch:
                env.close()

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

    @property
    def num_sub_batches(self):
        return len(self.envs)

    @property
    def num_envs_per_sub_batch(self):
        return len(self.envs[0])

    def reset_start(self, sub_batch=0):
        pass

    def reset_wait(self, sub_batch=0):
        return [env.reset() for env in self.envs[sub_batch]]

    def step_start(self, actions, sub_batch=0):
        assert len(actions) == self.num_envs_per_sub_batch
        self._step_actions[sub_batch] = actions

    def step_wait(self, sub_batch=0):
        assert self._step_actions[sub_batch] is not None
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

# pylint: disable=R0903
class _CloudpickleFunc:
    """
    A function that cloudpickle will serialize.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getstate__(self):
        return cloudpickle.dumps(self.func)

    def __setstate__(self, val):
        self.func = cloudpickle.loads(val)
