"""
Conversions to and from Gym spaces.
"""

import gym.spaces as spaces

from .categorical import CategoricalSoftmax

class UnsupportedActionSpace(Exception):
    """
    Thrown when a Gym space cannot be used as an action
    space.
    """
    def __init__(self, space):
        msg = 'unsupported action space: ' + str(space)
        super(UnsupportedActionSpace, self).__init__(msg)
        self.space = space

def gym_space_distribution(space):
    """
    Create a Distribution from a gym.Space.

    If the space is not supported, throws an
    UnsupportedActionSpace exception.
    """
    if isinstance(space, spaces.Discrete):
        return CategoricalSoftmax(space.n)
    raise UnsupportedActionSpace(space)
