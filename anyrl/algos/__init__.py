"""
Learning algorithms for RL.
"""

from .advantages import AdvantageEstimator, GAE
from .a2c import A2C
from .ppo import PPO
from .schedules import TFSchedule, LinearTFSchedule

# Don't import mpi by default since that breaks when
# you don't run your program with `mpirun`.

__all__ = dir()
