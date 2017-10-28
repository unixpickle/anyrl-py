"""
Conversions to and from Gym spaces.
"""

import gym.spaces as spaces

from .aggregate import TupleDistribution
from .binary import MultiBernoulli
from .categorical import CategoricalSoftmax
from .continuous import BoxGaussian

class UnsupportedGymSpace(Exception):
    """
    Thrown when a Gym space cannot be used as an action
    space.
    """
    def __init__(self, space):
        msg = 'unsupported Gym space: ' + str(space)
        super(UnsupportedGymSpace, self).__init__(msg)
        self.space = space

def gym_space_distribution(space):
    """
    Create a Distribution from a gym.Space.

    If the space is not supported, throws an
    UnsupportedActionSpace exception.
    """
    if isinstance(space, spaces.Discrete):
        return CategoricalSoftmax(space.n)
    elif isinstance(space, spaces.Box):
        return BoxGaussian(space.low, space.high)
    elif isinstance(space, spaces.MultiBinary):
        return MultiBernoulli(space.n)
    elif isinstance(space, spaces.Tuple):
        sub_dists = tuple(gym_space_distribution(s) for s in space.spaces)
        return TupleDistribution(sub_dists)
    elif isinstance(space, spaces.MultiDiscrete):
        discretes = tuple(CategoricalSoftmax(high-low+1, low=low)
                          for low, high in zip(space.low, space.high))
        return TupleDistribution(discretes)
    raise UnsupportedGymSpace(space)

def gym_space_vectorizer(space):
    """
    Create a Vectorizer from a gym.Space.

    If the space is not supported, throws an
    UnsupportedActionSpace exception.
    """
    return gym_space_distribution(space)
