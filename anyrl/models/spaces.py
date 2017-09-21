"""
Conversions from observation spaces to vectors.
"""

from abc import ABC, abstractmethod, abstractproperty

import gym.spaces as spaces
import numpy as np

class UnsupportedObsSpace(Exception):
    """
    Thrown when a Gym space cannot be used for
    observations.
    """
    def __init__(self, space):
        msg = 'unsupported observaotion space: ' + str(space)
        super(UnsupportedObsSpace, self).__init__(msg)
        self.space = space

def gym_space_vectorizer(space):
    """
    Analyze the gym.Space to create an appropriate
    SpaceVectorizer for it.
    """
    if isinstance(space, spaces.Box):
        return BoxVectorizer(space)
    raise UnsupportedObsSpace(space)

class SpaceVectorizer(ABC):
    """
    Convert gym.Space elements to tensors.
    """
    @abstractproperty
    def space(self):
        """
        Get the space for which this SpaceVectorizer is
        intended to be used.
        """
        pass

    @abstractproperty
    def shape(self):
        """
        Get the shape of resulting tensors.
        """
        pass

    @abstractmethod
    def vectorize(self, elements):
        """
        Convert a batch of space elements to tensors.
        Returns a numpy array.
        """
        pass

class BoxVectorizer(SpaceVectorizer):
    """
    Convert gym.Box space elements to tensors.
    """
    def __init__(self, space):
        self._space = space

    @property
    def space(self):
        return self._space

    @property
    def shape(self):
        return self.space.low.shape

    def vectorize(self, elements):
        return np.array(elements)
