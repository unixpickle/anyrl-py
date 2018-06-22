"""
Frame-skipping wrappers.
"""

import gym

# pylint: disable=E0202


class FrameSkipEnv(gym.Wrapper):
    """
    A wrapper that replays each action for a fixed number
    of timesteps and yields the final observation.
    """

    def __init__(self, env, num_frames=4):
        super(FrameSkipEnv, self).__init__(env)
        self.num_frames = num_frames

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        total_rew = 0.0
        for _ in range(self.num_frames):
            obs, rew, done, info = self.env.step(action)
            total_rew += rew
            if done:
                break
        return obs, total_rew, done, info
