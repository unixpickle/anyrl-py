"""
Wrappers that act on batched environments.

These can be useful in cases where computations are more
efficient in batches in the parent process, rather than
individually in each environment sub-process.

For example, suppose you want to feed screen observations
through a pre-trained CNN before passing them to your RL
model. If you don't want to fine-tune this CNN, then it is
most efficient to make the CNN part of the environment.
It makes sense to do this CNN as a batched wrapper, and it
may even be desirable to use a batched framestack wrapper
on top of the CNN wrapper.
"""

from abc import ABCMeta, abstractmethod
import gym
import numpy as np

from anyrl.spaces import StackedBoxSpace
from ..base import BatchedEnv


class BatchedWrapper(BatchedEnv):
    """
    A BatchedEnv that, by default, forwards all calls to a
    wrapped BatchedEnv.
    """

    def __init__(self, env):
        self.env = env
        if hasattr(env, 'observation_space'):
            self.observation_space = env.observation_space
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space

    @property
    def num_sub_batches(self):
        return self.env.num_sub_batches

    @property
    def num_envs_per_sub_batch(self):
        return self.env.num_envs_per_sub_batch

    def reset_start(self, sub_batch=0):
        self.env.reset_start(sub_batch=sub_batch)

    def reset_wait(self, sub_batch=0):
        return self.env.reset_wait(sub_batch=sub_batch)

    def step_start(self, actions, sub_batch=0):
        self.env.step_start(actions, sub_batch=sub_batch)

    def step_wait(self, sub_batch=0):
        return self.env.step_wait(sub_batch=sub_batch)

    def close(self):
        self.env.close()


class BatchedFrameStack(BatchedWrapper):
    """
    The batched analog of FrameStackEnv.
    """

    def __init__(self, env, num_images=2, concat=True, stride=1):
        super(BatchedFrameStack, self).__init__(env)
        self.concat = concat
        if hasattr(self, 'observation_space'):
            old = self.observation_space
            if concat:
                self.observation_space = gym.spaces.Box(np.repeat(old.low, num_images, axis=-1),
                                                        np.repeat(old.high, num_images, axis=-1),
                                                        dtype=old.dtype)
            else:
                self.observation_space = StackedBoxSpace(old, num_images)
        self._num_images = num_images
        self._stride = stride
        self._history_size = 1 + (num_images - 1) * stride
        self._history = [None] * env.num_sub_batches

    def reset_wait(self, sub_batch=0):
        obses = super(BatchedFrameStack, self).reset_wait(sub_batch=sub_batch)
        self._history[sub_batch] = [[o]*self._history_size for o in obses]
        return self._packed_obs(sub_batch)

    def step_wait(self, sub_batch=0):
        obses, rews, dones, infos = super(BatchedFrameStack, self).step_wait(sub_batch=sub_batch)
        for i, (obs, done) in enumerate(zip(obses, dones)):
            if done:
                self._history[sub_batch][i] = [obs] * self._history_size
            else:
                self._history[sub_batch][i].append(obs)
                self._history[sub_batch][i] = self._history[sub_batch][i][1:]
        return self._packed_obs(sub_batch), rews, dones, infos

    def _packed_obs(self, sub_batch):
        """
        Pack the sub-batch's observation along the
        inner dimension.
        """
        if self.concat:
            return [np.concatenate(o[::self._stride], axis=-1) for o in self._history[sub_batch]]
        return [o[::self._stride].copy() for o in self._history[sub_batch]]


class BatchedObservationWrapper(BatchedWrapper):
    """
    The batched analog of ObservationWrapper.

    Different from `ObsWrapperBatcher` as this calls
    observation() once with all of the observations from the
    batched environments, allowing you to perform optimized
    observation updates on all observations at once.

    Be sure to modify self.observation_space in your
    __init__ function if necessary
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def observation(self, obses):
        """
        Modifies given batched observations into output
        batched observations

        Args:
          obses: a list or array of observations.

        Returns:
          A list or array of modified observations.
        """
        pass

    def reset_wait(self, **args):
        obses = super().reset_wait(**args)
        obses = self.observation(obses)
        return obses

    def step_wait(self, **args):
        obses, rews, dones, infos = super().step_wait(**args)
        obses = self.observation(obses)
        return obses, rews, dones, infos


class ObsWrapperBatcher(BatchedObservationWrapper):
    """
    Converts a gym.ObservationWrapper into a
    BatchedObservationWrapper

    This assumes that the ObservationWrapper is stateless.
    """

    def __init__(self, env, wrap_fn, *args, **kwargs):
        """
        Create a new instance.

        Args:
          env: the BatchedEnv to wrap.
          wrap_fn: the constructor for the wrapper.
          args: arguments to pass to the wrapper, after
            the initial env argument.
          kwargs: extra arguments to pass to the wrapper.
        """
        super(ObsWrapperBatcher, self).__init__(env)
        self.wrapper = wrap_fn(_AttrEnv(env), *args, **kwargs)
        self.observation_space = self.wrapper.observation_space

    def observation(self, obses):
        return map(self.wrapper.observation, obses)


class ActWrapperBatcher(BatchedWrapper):
    """
    A batched version of any gym.ActionWrapper.

    This assumes that the ActionWrapper is stateless.
    """

    def __init__(self, env, wrap_fn, *args, **kwargs):
        """
        Create a new instance.

        Args:
          env: the BatchedEnv to wrap.
          wrap_fn: the constructor for the wrapper.
          args: arguments to pass to the wrapper, after
            the initial env argument.
          kwargs: extra arguments to pass to the wrapper.
        """
        super(ActWrapperBatcher, self).__init__(env)
        self.wrapper = wrap_fn(_AttrEnv(env), *args, **kwargs)
        self.action_space = self.wrapper.action_space

    def step_start(self, actions, sub_batch=0):
        super(ActWrapperBatcher, self).step_start(map(self.wrapper.action, actions),
                                                  sub_batch=sub_batch)


class _AttrEnv(gym.Env):
    """
    An environment that inherits various attributes from a
    batched environment.
    """

    def __init__(self, env):
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    # pylint: disable=W0221

    def reset(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        pass
