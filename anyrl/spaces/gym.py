"""
Conversions to and from Gym spaces.
"""

import gym
import gym.spaces as spaces
import numpy as np

from .aggregate import TupleDistribution
from .binary import MultiBernoulli
from .categorical import CategoricalSoftmax
from .continuous import BoxGaussian, BoxStacker


class UnsupportedGymSpace(Exception):
    """
    Thrown when a Gym space cannot be used as an action
    space.
    """

    def __init__(self, space):
        msg = 'unsupported Gym space: ' + str(space)
        super(UnsupportedGymSpace, self).__init__(msg)
        self.space = space


class StackedBoxSpace(gym.Space):
    """
    A gym.Space representing a list of gym.Box elements.

    This space is used for frame stacking when no
    concatenation is used. This way, the observation
    vectorizer can concatenate the frames right when
    turning them into vectors for a model.
    """

    def __init__(self, box, count):
        super(StackedBoxSpace, self).__init__(shape=None, dtype=box.dtype)
        self.box = box
        self.count = count

    def sample(self):
        return [self.box.sample() for _ in range(self.count)]

    def contains(self, x):
        if not isinstance(x, list) or len(x) != self.count:
            return False
        return all(self.box.contains(e) for e in x)

    def to_jsonable(self, sample_n):
        return [self.box.to_jsonable(sample) for sample in zip(*sample_n)]

    def from_jsonable(self, sample_n):
        return list(list(l) for l in zip(*[self.box.from_jsonable(sample) for sample in sample_n]))

    def __repr__(self):
        return "StackedBox" + str(self.box.shape)


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
        discretes = tuple(CategoricalSoftmax(n) for n in space.nvec)
        return TupleDistribution(discretes, to_sample=lambda x: np.array(x, dtype=space.dtype))
    raise UnsupportedGymSpace(space)


def gym_space_vectorizer(space):
    """
    Create a Vectorizer from a gym.Space.

    If the space is not supported, throws an
    UnsupportedActionSpace exception.
    """
    if isinstance(space, StackedBoxSpace):
        return BoxStacker(space.box.shape, space.count)
    return gym_space_distribution(space)


def gym_spaces(env):
    """
    Get an action distribution and an observation
    vectorizer for a gym environment.

    Args:
      env: any object with an observation_space and
        action_space attribute.

    Returns:
      A tuple (action_dist, obs_vectorizer):
        action_dist: a Distribution for actions.
        obs_vectorizer: a Vectorizer for observations.
    """
    return gym_space_distribution(env.action_space), gym_space_vectorizer(env.observation_space)
