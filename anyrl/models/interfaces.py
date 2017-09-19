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
    def batches(self, rollouts, batch_size=None):
        """
        Create an iterator of mini-batches for training
        the actor and the critic.

        Each mini-batch is a dict with these keys:
         - 'rollout_idxs': rollout index for each sample
         - 'timestep_idxs': timestep index for each asmple
         - 'log_probs': log-probability for each action
                        (as a 1-D Tensor)
         - 'pred_vals': predicted value for each sample

        The mini-batch may have another key 'action_params'
        which is the raw Tensor that is fed into an action
        distribution.
        This can be used to compute entropy, KL, etc.

        Arguments:
        rollouts -- a list of (partial) rollouts
        batch_size -- the approximate mini-batch size
        """
        pass

    @abstractmethod
    def _build(self, **kwargs):
        """
        Build up a graph for the model.

        Called from within a model-specific graph.
        """
        pass
