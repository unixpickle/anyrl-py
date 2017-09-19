"""
Types and functions for manipulating (partial) rollouts in
RL environments.
"""

def empty_rollout(start_state, trunc_start=False):
    """
    Create a Rollout with no timesteps.
    """
    return Rollout(observations=[], model_outs=[], rewards=[],
                   start_state=start_state, trunc_start=trunc_start)

class Rollout:
    """
    A sequence of observations, actions, and rewards that
    were recorded by running an agent in an environment.

    Rollouts may represent total episodes, but they may
    also represent a sliced version of the episode.
    The trunc_start and trunc_end fields indicate whether
    the start and end states were not the beginning or end
    of the episode.

    If a rollout is truncated at the end, it will have an
    extra observation and model_outs value.
    """
    # pylint: disable=R0913
    def __init__(self, observations, model_outs, rewards, start_state,
                 trunc_start=False, trunc_end=False):
        self.observations = observations
        self.model_outs = model_outs
        self.rewards = rewards
        self.start_state = start_state
        self.trunc_start = trunc_start
        self.trunc_end = trunc_end

    @property
    def num_steps(self):
        """
        Get the total number of timesteps (not including
        the extra observation for truncated episodes).
        """
        return len(self.rewards)

    @property
    def step_observations(self):
        """
        Returns a list of observations that does not
        include the trailing observation for truncated
        episodes.
        """
        return self.observations[:self.num_steps]

    @property
    def total_reward(self):
        """
        Get the total reward across all steps.
        """
        return sum(self.rewards)
