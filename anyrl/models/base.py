"""
Abstractions for RL policies and value functions.
"""

from abc import ABC, abstractmethod, abstractproperty

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

        Returns a dict with the following keys:
          'actions': batch of actions, one per env
          'states': new states after the step
          'values': (optional) predicted value function
          'action_params': (optional) parameters that were
            fed into the action distribution.
        """
        pass

class TFActorCritic(Model):
    """
    An actor-critic model which is differentiable via
    TensorFlow.

    Every TFActorCritic has an action distribution and
    observation vectorizer, which can be accessed via
    model.action_dist and model.obs_vectorizer.
    """
    def __init__(self, session, action_dist, obs_vectorizer):
        self.session = session
        self.action_dist = action_dist
        self.obs_vectorizer = obs_vectorizer

    @abstractmethod
    def batch_outputs(self):
        """
        Return three TF tensors: actor_outs, critic_outs,
        mask.

        The mask is a Tensor of 0's and 1's, where 1
        indicates that the sample is valid.

        Both mask and critic_outs should be 1-D.
        The actor_outs shape depends on the shape of
        action distribution parameters.

        These tensors are used in conjunction with the
        feed_dict returned by batches().
        """
        pass

    @abstractmethod
    def batches(self, rollouts, batch_size=None):
        """
        Create an iterator of mini-batches for training
        the actor and the critic.

        Each mini-batch is a dict with these keys:
          'rollout_idxs': rollout index for each sample
          'timestep_idxs': timestep index for each sample
          'feed_dict': inputs that the graph depends on

        There is a one-to-one correspondence between
        samples in the batch and values in the Tensors
        produced by batch_outputs.
        Masked samples should have 0's in rollout_idxs and
        timestep_idxs.

        Args:
          rollouts: a list of (partial) rollouts
          batch_size: the approximate mini-batch size
        """
        pass
