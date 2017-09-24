"""
Conversions to and from Gym spaces.
"""

import gym.spaces as spaces

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
    raise UnsupportedGymSpace(space)

def gym_space_vectorizer(space):
    """
    Create a Vectorizer from a gym.Space.

    If the space is not supported, throws an
    UnsupportedActionSpace exception.
    """
    return gym_space_distribution(space)
