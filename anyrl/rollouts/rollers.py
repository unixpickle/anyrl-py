"""
Ways of gathering rollouts.
"""

from abc import ABC, abstractmethod

from .rollout import empty_rollout

# pylint: disable=R0903
class Roller(ABC):
    """
    An object that gathers rollouts by running a model.
    """
    @abstractmethod
    def rollouts(self):
        """
        Return a list of Rollout objects.
        """
        pass

class BasicRoller(Roller):
    """
    Gathers episode rollouts from a Gym environment.
    """
    def __init__(self, env, model, min_episodes=1, min_steps=1):
        self.env = env
        self.model = model
        self.min_episodes = min_episodes
        self.min_steps = min_steps

    def rollouts(self):
        """
        Gather episodes until both self.min_episodes and
        self.min_steps are satisfied.
        """
        episodes = []
        num_steps = 0
        while num_steps < self.min_steps or len(episodes) < self.min_episodes:
            states = self.model.start_state(1)
            rollout = empty_rollout(states)
            obs = self.env.reset()
            while True:
                rollout.observations.append(obs)
                model_out = self.model.step([obs], states)
                rollout.model_outs.append(model_out)
                states = model_out['states']
                obs, rew, done, _ = self.env.step(model_out['actions'][0])
                rollout.rewards.append(rew)
                if done:
                    break
            num_steps += rollout.num_steps
            episodes.append(rollout)
        return episodes
