"""
A set of APIs for parameterizing probability distributions
in Reinforcement Learning.
"""

from .base import Vectorizer, Distribution
from .categorical import CategoricalSoftmax, NaturalSoftmax
from .continuous import BoxGaussian
from .binary import MultiBernoulli
from .aggregate import TupleDistribution
from .gym import gym_space_distribution, gym_space_vectorizer

__all__ = dir()
