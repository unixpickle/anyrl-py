"""
Wrappers for image preprocessing.
"""

import gym
import numpy as np

class DownsampleEnv(gym.ObservationWrapper):
    """
    An environment that downsamples its image inputs.
    """
    def __init__(self, env, rate):
        """
        Create a downsampling wrapper.

        Args:
          env: the environment to wrap.
          rate: (int) the downsample rate.
            New sizes are floor(old_size/rate).
        """
        super(DownsampleEnv, self).__init__(env)
        self._rate = rate
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(self._observation(old_space.low),
                                                self._observation(old_space.high))

    def _observation(self, observation):
        observation = observation[:1 + ((observation.shape[0] - 1) // self._rate) * self._rate,
                                  :1 + ((observation.shape[1] - 1) // self._rate) * self._rate]
        return observation[::self._rate, ::self._rate]

class GrayscaleEnv(gym.ObservationWrapper):
    """
    An environment that turns RGB images into grayscale.
    """
    def __init__(self, env, keep_depth=True, integers=True):
        """
        Create a grayscaling wrapper.

        Args:
          env: the environment to wrap.
          keep_depth: if True, a depth dimension is kept.
            Otherwise, the output is 2-D.
          integers: if True, the pixels are in [0, 255].
            Otherwise, they are in [0.0, 1.0].
        """
        super(GrayscaleEnv, self).__init__(env)
        old_space = env.observation_space
        self._integers = integers
        self._keep_depth = keep_depth
        self.observation_space = gym.spaces.Box(self._observation(old_space.low),
                                                self._observation(old_space.high))

    def _observation(self, observation):
        if self._integers:
            observation = observation // 3
        else:
            observation = observation / 3
        return np.sum(observation, axis=-1, keepdims=self._keep_depth)

class FrameStackEnv(gym.Wrapper):
    """
    An environment that stacks images.

    Input images should be 3-D, and are stacked along the
    inner-most dimension.

    The stacking is ordered from oldest to newest.

    At the beginning of an episode, the first observation
    is repeated in order to complete the stack.
    """
    def __init__(self, env, num_images=2):
        """
        Create a frame stacking environment.

        Args:
          env: the environment to wrap.
          num_images: the number of images to stack.
            This includes the current observation.
        """
        super(FrameStackEnv, self).__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(np.repeat(old_space.low, num_images, axis=-1),
                                                np.repeat(old_space.high, num_images, axis=-1))
        self._num_images = num_images
        self._history = []

    def _reset(self, **kwargs):
        obs = super(FrameStackEnv, self)._reset(**kwargs)
        self._history = [obs] * self._num_images
        return np.concatenate(self._history, axis=-1)

    def _step(self, action):
        obs, rew, done, info = super(FrameStackEnv, self)._step(action)
        self._history.append(obs)
        self._history = self._history[1:]
        return np.concatenate(self._history, axis=-1), rew, done, info
