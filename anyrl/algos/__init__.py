"""
Learning algorithms for RL.
"""

from .a2c import A2C
from .advantages import AdvantageEstimator, GAE
from .dqn import DQN
from .ppo import PPO
from .schedules import TFSchedule, LinearTFSchedule, TFScheduleValue

# Don't import mpi by default since that breaks when
# you don't run your program with `mpirun`.

__all__ = dir()
