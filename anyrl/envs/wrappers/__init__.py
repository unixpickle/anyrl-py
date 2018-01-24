"""
Environments that wrap and modify other environments.
"""

from .batched import BatchedWrapper, BatchedFrameStack
from .image import DownsampleEnv, FrameStackEnv, GrayscaleEnv, MaxEnv, ResizeImageEnv
from .logs import LoggedEnv
from .meta import RL2Env, SwitchableEnv, JointEnv
from .padding import ObservationPadEnv, MultiBinaryPadEnv
