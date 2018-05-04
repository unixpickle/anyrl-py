"""
Ways of running gym environments.
"""

from abc import ABC, abstractmethod, abstractproperty


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


class BatchedAsyncEnv(BatchedEnv):
    """
    A BatchedEnv that controls AsyncEnvs.

    If the first AsyncEnv has an action_space and/or
    observation_space attribute, those attributes are
    copied.
    """

    def __init__(self, sub_batches):
        assert len(sub_batches) > 0
        first_len = len(sub_batches[0])
        assert all([len(x) == first_len for x in sub_batches])
        self._sub_batches = sub_batches

        self.action_space = None
        self.observation_space = None

        first_env = sub_batches[0][0]
        if hasattr(first_env, 'action_space'):
            self.action_space = first_env.action_space
        if hasattr(first_env, 'observation_space'):
            self.observation_space = first_env.observation_space

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
