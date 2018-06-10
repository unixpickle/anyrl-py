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
          'action_values': (optional) value predictions
            for each possible action.
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
        Return two TF tensors: actor_outs, critic_outs.

        The critic_outs Tensor should be 1-D.
        The actor_outs shape depends on the shape of
        action distribution parameters.

        These tensors are used in conjunction with the
        feed_dict returned by batches().

        This method may be called multiple times.
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

        Args:
          rollouts: a list of (partial) rollouts
          batch_size: the approximate mini-batch size
        """
        pass


class TFQNetwork(Model):
    """
    A Q-network model which is differentiable via
    TensorFlow.

    Attributes:
      session: the TF session for the model.
      num_actions: the size of the discrete action space.
      obs_vectorizer: used to convert observations into
        Tensors.
      name: the variable scope name for the network.
      variables: the trainable variables of the network.

    When a Q-network is instantiated, a graph is created
    that can be used by the step() method. This involves
    creating a set of variables and placeholders in the
    graph.

    After construction, other Q-network methods like
    transition_loss() reuse the variables that were made
    at construction time.
    """

    def __init__(self, session, num_actions, obs_vectorizer, name):
        """
        Construct a Q-network.

        Args:
          session: the TF session used by step().
          num_actions: the number of possible actions.
          obs_vectorizer: a vectorizer for the observation
            space.
          name: the scope name for the model. This should
            be different for the target and online models.
        """
        self.session = session
        self.num_actions = num_actions
        self.obs_vectorizer = obs_vectorizer
        self.name = name
        self.variables = []

    @abstractmethod
    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts):
        """
        Create a loss term for the Bellman update.

        This should be called on the "online network",
        since the variables of `self` are trained.

        Args:
          target_net: the companion TFQNetwork from which
            the target Q values should be computed.
          obses: the Tensor of starting observations.
          actions: the 1-D int32 Tensor of actions.
          rews: the 1-D Tensor of rewards.
          new_obses: the Tensor of final observations.
            For terminal transitions, the observation may
            be anything, e.g. a bunch of 0's.
          terminals: a 1-D boolean Tensor indicating which
            transitions are terminal.
          discounts: the 1-D Tensor of discount factors.
            For n-step Q-learning, this contains the true
            discount factor raised to the n-th power.

        Returns:
          A 1-D Tensor containing a loss value for each
            transition in the batch of transitions.
        """
        pass

    @abstractproperty
    def input_dtype(self):
        """
        Get the TF dtype to use for observation vectors.

        The returned dtype should be used for the Tensors
        that are passed into transition_loss().
        """
        pass
