"""
Types and functions for manipulating (partial) rollouts in
RL environments.
"""


def empty_rollout(start_state, prev_steps=0, prev_reward=0):
    """
    Create a Rollout with no timesteps.
    """
    return Rollout(observations=[], model_outs=[], rewards=[],
                   start_state=start_state, prev_steps=prev_steps,
                   prev_reward=prev_reward)


class Rollout:
    """
    A sequence of observations, actions, and rewards that
    were recorded by running an agent in an environment.

    Model outputs are structured like the results of
    Model.step(), but they have a batch size of one.

    Rollouts may represent total episodes, but they may
    also represent a sliced version of the episode.
    The trunc_start and trunc_end properties indicate if
    the episode was truncated at the start or the end.
    When trunc_start is True, prev_reward and prev_steps
    indicate the amount of reward and number of steps in
    the episode before this Rollout.
    When trunc_end is true, there is an extra observation
    and model_outs entry.

    Rollouts may also have an end_time, which is a UNIX
    timestamp of when the rollout was done being made.
    """

    def __init__(self, observations, model_outs, rewards, start_state,
                 prev_steps=0, prev_reward=0, infos=None, end_time=0):
        assert len(observations) == len(model_outs)
        assert len(rewards) <= len(observations)
        assert len(observations) <= len(rewards)+1
        if infos is None:
            infos = [{}] * len(rewards)
        assert len(infos) == len(rewards)
        self.observations = observations
        self.model_outs = model_outs
        self.rewards = rewards
        self.start_state = start_state
        self.prev_steps = prev_steps
        self.prev_reward = prev_reward
        self.infos = infos
        self.end_time = end_time

    @property
    def trunc_end(self):
        """
        Get whether or not the episode completed in this
        Rollout.
        """
        return len(self.observations) > self.num_steps

    @property
    def trunc_start(self):
        """
        Get whether or not steps were taken in the episode
        before this Rollout.
        """
        return self.prev_steps > 0

    @property
    def num_steps(self):
        """
        Get the total number of timesteps (not including
        the extra observation or previous timesteps for
        truncated episodes).
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
    def step_model_outs(self):
        """
        Returns a list of model outputs that does not
        include the trailing output for truncated
        episodes.
        """
        return self.model_outs[:self.num_steps]

    @property
    def total_reward(self):
        """
        Get the total reward so far in the episode.
        This includes rewards from previous segments of
        this episode.
        """
        return sum(self.rewards) + self.prev_reward

    @property
    def total_steps(self):
        """
        Get the total number of completed timesteps in the
        episode.
        This includes timesteps from previous segments of
        this episode, but it does not include the final
        observation for truncated episodes.
        """
        return self.num_steps + self.prev_steps

    def predicted_value(self, timestep):
        """
        Get the predicted value at the given timestep.

        This only works if the model generated values.
        """
        return self.model_outs[timestep]['values'][0]

    def copy(self):
        """
        Create a shallow copy of the rollout.
        """
        return Rollout(self.observations,
                       self.model_outs,
                       self.rewards,
                       self.start_state,
                       prev_steps=self.prev_steps,
                       prev_reward=self.prev_reward,
                       infos=self.infos,
                       end_time=self.end_time)
