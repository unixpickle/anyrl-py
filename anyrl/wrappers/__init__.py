"""
Helpful wrappers for Reinforcement Learning.
"""

from .batched import BatchedWrapper, BatchedFrameStack
from .image import DownsampleEnv, FrameStackEnv, GrayscaleEnv
from .meta import RL2Env, SwitchableEnv
