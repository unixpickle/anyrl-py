"""
Conversions to and from Gym spaces.
"""

import gym.spaces as spaces

from .categorical import CategoricalSoftmax

class UnsupportedSpace(Exception):
    """
    Thrown when a Gym space cannot be used somewhere.
    """
    def __init__(self, space):
        super(UnsupportedSpace, self).__init__('unsupported space: ' + str(space))
        self.space = space

def gym_space_distribution(space):
    """
    Create a Distribution from a gym.Space.

    If the space is not supported, throws an
    UnsupportedSpace exception.
    """
    if isinstance(space, spaces.Discrete):
        return CategoricalSoftmax(space.n)
    raise UnsupportedSpace(space)
