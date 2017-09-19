"""
A set of APIs for parameterizing probability distributions
in Reinforcement Learning.
"""

from .interfaces import Distribution
from .categorical import CategoricalSoftmax
from .spaces import gym_space_distribution
