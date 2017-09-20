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
        - 'actions': batch of actions, one per env
        - 'states': new states after the step
        - 'values': (optional) predicted value function

        The returned dict may include an 'action_params'
        key as well, indicating the parameters that were
        fed into an action distribution.
        """
        pass

class TFActorCritic(Model):
    """
    An actor-critic model which is differentiable via
    TensorFlow.

    Every TFActorCritic has an action distribution.
    This can be accessed via model.action_dist.
    """
    def __init__(self, session, action_dist):
        self.session = session
        self.action_dist = action_dist

    @abstractmethod
    def batches(self, rollouts, batch_size=None):
        """
        Create an iterator of mini-batches for training
        the actor and the critic.

        Each mini-batch is a dict with these keys:
         - 'rollout_idxs': rollout index for each sample
         - 'timestep_idxs': timestep index for each sample
         - 'critic_outs': predicted value for each sample
         - 'actor_outs': action parameters for each sample
         - 'feed_dict': inputs that the graph depends on

        Arguments:
        rollouts -- a list of (partial) rollouts
        batch_size -- the approximate mini-batch size
        """
        pass
