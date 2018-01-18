"""
Ways of gathering state transitions to store in a replay
buffer.
"""

from abc import ABC, abstractmethod

# pylint: disable=R0903
class Player(ABC):
    """
    An object that runs a model on an environment and
    produces transitions that can be stored in a replay
    buffer.

    A Player serves a similar role in DQN as a Roller plays in
    policy-gradient algorithms.
    """
    @abstractmethod
    def play(self):
        """
        Gather a sequence of transition dicts.

        Each transition dict should have these keys:
          'obs': the starting observation.
          'action': the chosen action.
          'reward': the reward after taking the action.
          'new_obs': the new observation, or None if the
            'reward' field encompasses the rest of the
            episode's reward.
          'discount': the discount factor bridging rewards
            from the start and end observations.
            For n-step Q-learning, this is `gamma^n`.
          'start_state': the model's state when it chose
            to take the given action. This may be None.
            This is represented as a batch-size 1 value.
            For example, it might be [1 x n].
          'episode_id': a comparable value that identifies
            the particular episode for this timestep.
          'episode_step': the timestep of the initial
            observation, starting from 0.
          'end_time': the UNIX timestamp when the step was
            finished being taken.
          'is_last': a boolean indicating if this is the
            last transition for the episode.
          'total_rew': the total reward for the episode up
            to and including this transition.
            If 'new_obs' is None, then this is the total
            reward for the episode.
        """
        pass
