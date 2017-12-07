"""
A set of APIs for using and manipulating RL environments.
"""

from .base import AsyncEnv, BatchedEnv, BatchedAsyncEnv
from .gym import batched_gym_env, AsyncGymEnv, BatchedGymEnv

from .batched_wrappers import BatchedWrapper, BatchedFrameStack
from .image_wrappers import DownsampleEnv, FrameStackEnv, GrayscaleEnv
from .meta_wrappers import RL2Env, SwitchableEnv
