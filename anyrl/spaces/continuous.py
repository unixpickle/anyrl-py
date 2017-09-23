"""
APIs for continuous spaces.
"""

import numpy as np

from .base import Vectorizer

class BoxVectorizer(Vectorizer):
    """
    Convert gym.Box space elements to tensors.
    """
    def __init__(self, shape):
        self._shape = shape

    @property
    def out_shape(self):
        return self._shape

    def to_vecs(self, space_elements):
        return np.array(space_elements)
