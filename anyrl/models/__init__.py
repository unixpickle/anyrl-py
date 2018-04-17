"""
Various models for Reinforcement Learning agents.
"""

from .base import Model, TFActorCritic
from .feedforward_ac import FeedforwardAC, MLP, CNN
from .misc import RandomAgent
from .dqn_dist import DistQNetwork, MLPDistQNetwork, NatureDistQNetwork, rainbow_models
from .dqn_scalar import (ScalarQNetwork, MLPQNetwork, NatureQNetwork, EpsGreedyQNetwork,
                         noisy_net_dense)
from .recurrent_ac import RecurrentAC, RNNCellAC, CNNRNNCellAC

__all__ = dir()
