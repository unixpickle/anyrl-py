"""
Wrappers for padding spaces.
"""

import gym
import numpy as np


class ObservationPadEnv(gym.ObservationWrapper):
    """
    An environment that zero-pads the observation.

    Supports any Box observation space.
    """

    def __init__(self, env, padded_shape, center=True):
        """
        Create a padded environment.

        Args:
          env: the environment to wrap.
          padded_shape: the shape after padding.
          center: if True, attempt to center the original
            observation in the padded one. Otherwise, put
            the original image at the beginning of the
            padded image (e.g. the top-left corner).
        """
        super(ObservationPadEnv, self).__init__(env)
        self._padded_shape = padded_shape
        self._center = center
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(low=self.observation(old_space.low),
                                                high=self.observation(old_space.high),
                                                dtype=old_space.dtype)

    def observation(self, observation):
        """
        Pad the observation.
        """
        total_pads = tuple(target - cur for target, cur
                           in zip(self._padded_shape, observation.shape))
        assert all(x >= 0 for x in total_pads)
        if self._center:
            pad_width = []
            for pad in total_pads:
                if pad % 2 == 0:
                    pad_width.append((pad//2, pad//2))
                else:
                    pad_width.append((pad//2, pad//2+1))
        else:
            pad_width = [(0, pad) for pad in total_pads]
        return np.pad(observation, pad_width, 'constant')


class MultiBinaryPadEnv(gym.ActionWrapper):
    """
    An environment that adds no-op actions to a
    multi-binary action space.

    This is useful for unifying several different
    environments with different controls.
    """

    def __init__(self, env, num_actions):
        """
        Create a padded environment.

        Args:
          env: the environment to wrap.
          num_actions: the number of actions in the padded
            action space.
        """
        super(MultiBinaryPadEnv, self).__init__(env)
        assert num_actions >= env.action_space.n
        self._num_actions = num_actions
        self.action_space = gym.spaces.MultiBinary(num_actions)

    def action(self, action):
        return action[:self.env.action_space.n]

    def reverse_action(self, action):
        res = np.array([self.action_space.n], dtype='bool')
        res[:self.env.action_space.n] = action
        return res
