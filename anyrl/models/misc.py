"""
Miscellaneous models.
"""

from .base import Model


class RandomAgent(Model):
    """
    A Model that takes random actions using a sampling
    function.
    """

    def __init__(self, sample_fn):
        self.sample_fn = sample_fn

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        return {
            'actions': [self.sample_fn() for _ in observations],
            'states': None
        }
