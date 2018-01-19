"""
Ways of gathering state transitions to store in a replay
buffer.
"""

from abc import ABC, abstractmethod
import time

# pylint: disable=R0902,R0903

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
            transition is terminal.
          'steps': the number of steps bridging the start
            and end observations. For n-step Q-learning,
            this is n.
          'info': the info dictionary from the step when
            the action was taken.
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
            last transition for the episode. This may not
            be True even if new_obs is None, since n-step
            Q-learning can result in multiple terminal
            transitions.
          'total_rew': the total reward for the episode up
            to and including this transition.
            If 'new_obs' is None, then this is the total
            reward for the episode.
        """
        pass

class BasicPlayer(Player):
    """
    A Player that uses a single Gym environment to gather
    sequential batches of transitions.
    """
    def __init__(self, env, model, batch_size=1):
        self.env = env
        self.model = model
        self.batch_size = batch_size
        self._needs_reset = True
        self._cur_state = None
        self._last_obs = None
        self._episode_id = 0
        self._episode_step = 0
        self._total_rew = 0.0

    def play(self):
        return [self._gather_transition() for _ in range(self.batch_size)]

    def _gather_transition(self):
        if self._needs_reset:
            self._needs_reset = False
            self._cur_state = self.model.start_state(1)
            self._last_obs = self.env.reset()
            self._episode_id += 1
            self._episode_step = 0
            self._total_rew = 0.0
        output = self.model.step([self._last_obs], self._cur_state)
        new_obs, rew, self._needs_reset, info = self.env.step(output['actions'][0])
        self._total_rew += rew
        res = {
            'obs': self._last_obs,
            'action': output['actions'][0],
            'reward': rew,
            'new_obs': (new_obs if not self._needs_reset else None),
            'steps': 1,
            'info': info,
            'start_state': self._cur_state,
            'episode_id': self._episode_id,
            'episode_step': self._episode_step,
            'end_time': time.time(),
            'is_last': self._needs_reset,
            'total_rew': self._total_rew
        }
        self._cur_state = output['states']
        self._last_obs = new_obs
        self._episode_step += 1
        return res
