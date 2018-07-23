"""
A set of APIs for parameterizing probability distributions
in Reinforcement Learning.
"""

from .base import Vectorizer, Distribution
from .categorical import CategoricalSoftmax, NaturalSoftmax
from .continuous import BoxGaussian, BoxBeta, BoxStacker
from .binary import MultiBernoulli
from .aggregate import TupleDistribution
from .gym import StackedBoxSpace, gym_space_distribution, gym_space_vectorizer, gym_spaces

__all__ = dir()
