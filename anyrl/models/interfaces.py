"""
Abstractions for RL policies and value functions.
"""

from abc import ABC, abstractmethod, abstractproperty

import tensorflow as tf

class Model(ABC):
    """
    An abstract RL policy and (optional) value function.
    """
    @abstractproperty
    def stateful(self):
        """
        Return whether or not the model has states.
        """
        pass

    @abstractmethod
    def start_state(self, batch_size):
        """
        Return a batch of start states.

        State batches are represented as an array of shape
        [batch_size x D], or as a tuple of i elements of
        shapes [batch_size x Di].

        If the Model is not stateful, return None.
        """
        pass

    @abstractmethod
    def step(self, observations, states):
        """
        Apply the model for a single timestep in a batch
        of environments.

        Returns a dict with an 'action' key which maps to
        a batch of actions (one per environment).

        If the model implements a value function, the 'value'
        'value' key should be set to the predicted value.

        The returned dict may include an 'action_params'
        key as well, indicating the parameters that were
        fed into an action distribution.
        """
        pass

class TFActorCritic(Model):
    """
    An actor-critic model which is differentiable via
    TensorFlow.
    """
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default(): # pylint: disable=E1129
            self._build()

    @abstractmethod
    def actor_log_probs(self, rollouts):
        """
        Compute the action log probabilities for a list of
        rollouts.

        The result is a Tensor of log probabilities in the
        flattened ordering defined for rollouts.
        """
        pass

    @abstractmethod
    def critic_predictions(self, rollouts):
        """
        Compute the predicted value function for a list of
        rollouts.

        The result is a Tensor of value predictions in the
        flattened ordering defined for rollouts.
        """
        pass

    @abstractmethod
    def _build(self, **kwargs):
        """
        Build up a graph for the model.

        Called from within a model-specific graph.
        """
        pass
