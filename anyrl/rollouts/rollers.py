"""
Ways of gathering rollouts.
"""

from abc import ABC, abstractmethod
import time

from .rollout import empty_rollout
from .util import reduce_states, inject_state, reduce_model_outs


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
                obs, rew, done, info = self.env.step(model_out['actions'][0])
                rollout.rewards.append(rew)
                rollout.infos.append(info)
                if done:
                    break
            num_steps += rollout.num_steps
            rollout.end_time = time.time()
            episodes.append(rollout)
        return episodes


class TruncatedRoller(Roller):
    """
    Gathers a fixed number of timesteps from each
    environment in a BatchedEnv.
    """

    def __init__(self, batched_env, model, num_timesteps, drop_states=False):
        """
        Create a new TruncatedRoller.

        Args:
          batched_env: a BatchedEnv to interact with.
          model: a Model to use for interaction.
          num_timesteps: the number of timesteps to run
            each sub-environment for.
          drop_states: if True, set model_outs['states']
            to None in the rollouts to save memory.
        """
        self.batched_env = batched_env
        self.model = model
        self.num_timesteps = num_timesteps
        self.drop_states = drop_states

        # These end up being batches of sub-batches.
        # Each sub-batch corresponds to a sub-batch of
        # environments.
        self._last_states = None
        self._last_obs = None
        self._prev_steps = None
        self._prev_reward = None

    def reset(self):
        """
        Reset the environments, model states, and partial
        trajectory information.

        This needn't be called on new TruncatedRollers.
        """
        inner_dim = self.batched_env.num_envs_per_sub_batch
        outer_dim = self.batched_env.num_sub_batches
        self._last_obs = []
        self._prev_steps = [[0]*inner_dim for _ in range(outer_dim)]
        self._prev_reward = [[0]*inner_dim for _ in range(outer_dim)]
        for i in range(outer_dim):
            self.batched_env.reset_start(sub_batch=i)
        for i in range(outer_dim):
            self._last_obs.append(self.batched_env.reset_wait(sub_batch=i))
        self._last_states = [self.model.start_state(inner_dim)
                             for _ in range(outer_dim)]

    def rollouts(self):
        """
        Gather (possibly truncated) rollouts.
        """
        if self._last_states is None:
            self.reset()
        completed_rollouts = []
        running_rollouts = self._starting_rollouts()
        for _ in range(self.num_timesteps):
            self._step(completed_rollouts, running_rollouts)
        self._step(completed_rollouts, running_rollouts, final_step=True)
        self._add_truncated(completed_rollouts, running_rollouts)
        return completed_rollouts

    def _starting_rollouts(self):
        """
        Create empty rollouts with the start states and
        initial observations.
        """
        rollouts = []
        for batch_idx, states in enumerate(self._last_states):
            rollout_batch = []
            for env_idx in range(self.batched_env.num_envs_per_sub_batch):
                sub_state = reduce_states(states, env_idx)
                prev_steps = self._prev_steps[batch_idx][env_idx]
                prev_reward = self._prev_reward[batch_idx][env_idx]
                rollout = empty_rollout(sub_state,
                                        prev_steps=prev_steps,
                                        prev_reward=prev_reward)
                rollout_batch.append(rollout)
            rollouts.append(rollout_batch)
        return rollouts

    def _step(self, completed, running, final_step=False):
        """
        Wait for the previous batched step to complete (or
        use self._last_obs) and start a new step.

        Updates the running rollouts to reflect new steps
        and episodes.

        Returns the newest batch of model outputs.
        """
        for batch_idx, obses in enumerate(self._last_obs):
            if obses is None:
                step_out = self.batched_env.step_wait(sub_batch=batch_idx)
                obses, rews, dones, infos = step_out
                for env_idx, (rew, done, info) in enumerate(zip(rews, dones, infos)):
                    running[batch_idx][env_idx].rewards.append(rew)
                    running[batch_idx][env_idx].infos.append(info)
                    if done:
                        self._complete_rollout(completed, running, batch_idx, env_idx)
                    else:
                        self._prev_steps[batch_idx][env_idx] += 1
                        self._prev_reward[batch_idx][env_idx] += rew

            states = self._last_states[batch_idx]
            model_outs = self.model.step(obses, states)
            for env_idx, (obs, rollout) in enumerate(zip(obses, running[batch_idx])):
                reduced_out = self._reduce_model_outs(model_outs, env_idx)
                rollout.observations.append(obs)
                rollout.model_outs.append(reduced_out)

            if final_step:
                self._last_obs[batch_idx] = obses
            else:
                self._last_states[batch_idx] = model_outs['states']
                self._last_obs[batch_idx] = None
                self.batched_env.step_start(model_outs['actions'], sub_batch=batch_idx)

    def _complete_rollout(self, completed, running, batch_idx, env_idx):
        """
        Finalize a rollout and start a new rollout.
        """
        running[batch_idx][env_idx].end_time = time.time()
        completed.append(running[batch_idx][env_idx])
        for prev in [self._prev_steps, self._prev_reward]:
            prev[batch_idx][env_idx] = 0
        start_state = self.model.start_state(1)
        inject_state(self._last_states[batch_idx], start_state, env_idx)

        rollout = empty_rollout(start_state)
        running[batch_idx][env_idx] = rollout

    def _add_truncated(self, completed, running):
        """
        Add partial but non-empty rollouts to completed.
        """
        for sub_running in running:
            for rollout in sub_running:
                if rollout.num_steps > 0:
                    rollout.end_time = time.time()
                    completed.append(rollout)

    def _reduce_model_outs(self, model_outs, env_idx):
        """
        Reduce the model_outs to be put into a Rollout.
        """
        res = reduce_model_outs(model_outs, env_idx)
        if self.drop_states:
            res['states'] = None
        return res


class EpisodeRoller(TruncatedRoller):
    """
    Gather rollouts from a BatchedEnv with step and
    episode quotas.

    An EpisodeRoller does not have any bias towards
    shorter episodes.
    As a result, it must gather at least as many episodes
    as there are environments in batched_env.
    """

    def __init__(self, batched_env, model, min_episodes=1, min_steps=1, drop_states=False):
        """
        Create a new EpisodeRoller.

        Args:
          batched_env: a BatchedEnv to interact with.
          model: a Model to use for interaction.
          min_episodes: the minimum number of episodes to
            rollout.
          min_steps: the minimum number of timesteps to
            rollout, across all environments.
          drop_states: if True, set model_outs['states']
            to None in the rollouts to save memory.
        """
        self.min_episodes = min_episodes
        self.min_steps = min_steps

        # A batch of booleans in the shape of the envs.
        # An environment gets masked out once we have met
        # all the criteria and aren't looking for another
        # episode.
        self._env_mask = None

        super(EpisodeRoller, self).__init__(batched_env, model, 0, drop_states=drop_states)

    def reset(self):
        super(EpisodeRoller, self).reset()
        inner_dim = self.batched_env.num_envs_per_sub_batch
        outer_dim = self.batched_env.num_sub_batches
        self._env_mask = [[True]*inner_dim for _ in range(outer_dim)]

    def rollouts(self):
        """
        Gather full-episode rollouts.
        """
        self.reset()
        completed_rollouts = []
        running_rollouts = self._starting_rollouts()
        while self._any_envs_running():
            self._step(completed_rollouts, running_rollouts)
        # Make sure we are ready for the next reset().
        for batch_idx in range(self.batched_env.num_sub_batches):
            self.batched_env.step_wait(sub_batch=batch_idx)
        return completed_rollouts

    def _complete_rollout(self, completed, running, batch_idx, env_idx):
        comp_dest = completed
        if not self._env_mask[batch_idx][env_idx]:
            comp_dest = []
        super(EpisodeRoller, self)._complete_rollout(comp_dest, running,
                                                     batch_idx, env_idx)
        if self._criteria_met(completed):
            self._env_mask[batch_idx][env_idx] = False

    def _criteria_met(self, completed):
        """
        Check if the stopping criteria are met.
        """
        total_steps = sum([r.num_steps for r in completed])
        total_eps = len(completed)
        return total_steps >= self.min_steps and total_eps >= self.min_episodes

    def _any_envs_running(self):
        """
        Check if any environment is not masked out.
        """
        return any([any(masks) for masks in self._env_mask])
