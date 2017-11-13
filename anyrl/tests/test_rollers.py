"""
Test various Roller implementations.
"""

import unittest

from anyrl.rollouts import (batched_gym_env, AsyncGymEnv, BasicRoller, TruncatedRoller,
                            EpisodeRoller, ThreadedRoller, Rollout)
from anyrl.tests import SimpleEnv, SimpleModel
import numpy as np

class TruncatedRollerTest(unittest.TestCase):
    """
    Tests for TruncatedRoller.
    """
    def test_basic_equivalence(self):
        """
        Test that TruncatedRoller is equivalent to a
        BasicRoller when used with a single environment.
        """
        self._test_basic_equivalence_case(False, False)
        self._test_basic_equivalence_case(True, False)
        self._test_basic_equivalence_case(True, True)

    def test_truncation(self):
        """
        Test that sequence truncation works correctly for
        a batch of one environment.
        """
        self._test_truncation_case(False, False)
        self._test_truncation_case(True, False)
        self._test_truncation_case(True, True)

    def test_batch_equivalence(self):
        """
        Test that doing things in batches is equivalent to
        doing things one at a time.
        """
        self._test_batch_equivalence_case(False, False)
        self._test_batch_equivalence_case(True, False)
        self._test_batch_equivalence_case(True, True)

    def _test_basic_equivalence_case(self, stateful, state_tuple):
        """
        Test BasicRoller equivalence for a specific set of
        model settings.
        """
        env_fn = lambda: SimpleEnv(3, (4, 5), 'uint8')
        env = env_fn()
        model = SimpleModel(env.action_space.low.shape,
                            stateful=stateful,
                            state_tuple=state_tuple)
        basic_roller = BasicRoller(env, model, min_episodes=5)
        expected = basic_roller.rollouts()
        total_timesteps = sum([x.num_steps for x in expected])

        batched_env = batched_gym_env([env_fn], sync=True)
        trunc_roller = TruncatedRoller(batched_env, model, total_timesteps)
        actual = trunc_roller.rollouts()
        _compare_rollout_batch(self, actual, expected)

    def _test_truncation_case(self, stateful, state_tuple):
        """
        Test rollout truncation and continuation for a
        specific set of model parameters.
        """
        env_fn = lambda: SimpleEnv(7, (5, 3), 'uint8')
        env = env_fn()
        model = SimpleModel(env.action_space.low.shape,
                            stateful=stateful,
                            state_tuple=state_tuple)
        basic_roller = BasicRoller(env, model, min_episodes=5)
        expected = basic_roller.rollouts()
        total_timesteps = sum([x.num_steps for x in expected])

        batched_env = batched_gym_env([env_fn], sync=True)
        trunc_roller = TruncatedRoller(batched_env, model, total_timesteps//2 + 1)
        actual1 = trunc_roller.rollouts()
        self.assertTrue(actual1[-1].trunc_end)
        actual2 = trunc_roller.rollouts()
        expected1, expected2 = _artificial_truncation(expected,
                                                      len(actual1) - 1,
                                                      actual1[-1].num_steps)
        self.assertEqual(len(actual2), len(expected2)+1)
        actual2 = actual2[:-1]
        _compare_rollout_batch(self, actual1, expected1)
        _compare_rollout_batch(self, actual2, expected2)

    def _test_batch_equivalence_case(self, stateful, state_tuple):
        """
        Test that doing things in batches is consistent,
        given the model parameters.
        """
        env_fns = [lambda seed=x: SimpleEnv(seed, (5, 3), 'uint8') for x in range(15)]
        model = SimpleModel((5, 3),
                            stateful=stateful,
                            state_tuple=state_tuple)

        unbatched_rollouts = []
        for env_fn in env_fns:
            batched_env = batched_gym_env([env_fn], sync=True)
            trunc_roller = TruncatedRoller(batched_env, model, 17)
            for _ in range(3):
                unbatched_rollouts.extend(trunc_roller.rollouts())

        batched_rollouts = []
        batched_env = batched_gym_env(env_fns, num_sub_batches=3, sync=True)
        trunc_roller = TruncatedRoller(batched_env, model, 17)
        for _ in range(3):
            batched_rollouts.extend(trunc_roller.rollouts())

        _compare_rollout_batch(self, unbatched_rollouts, batched_rollouts,
                               ordered=False)

class EpisodeRollerTest(unittest.TestCase):
    """
    Tests for EpisodeRoller.
    """
    def test_basic_equivalence(self):
        """
        Test that EpisodeRoller is equivalent to a
        BasicRoller when run on a single environment.
        """
        state_opts = [(False, False), (True, False), (True, True)]
        kwargs_opts = [{'min_episodes': 5}, {'min_steps': 7}]
        for state_opt in state_opts:
            for kwargs_opt in kwargs_opts:
                self._test_basic_equivalence_case(*state_opt, **kwargs_opt)

    def test_batch_equivalence(self):
        """
        Test that EpisodeRoller is equivalent to a
        BasicRoller when run on a batch of envs.
        """
        state_opts = [(False, False), (True, False), (True, True)]
        kwargs_opts = [{'min_episodes': 5}, {'min_steps': 7}]
        for state_opt in state_opts:
            for kwargs_opt in kwargs_opts:
                self._test_batch_equivalence_case(*state_opt, **kwargs_opt)

    def test_multiple_batches(self):
        """
        Make sure calling rollouts multiple times works.
        """
        env_fn = lambda: SimpleEnv(3, (4, 5), 'uint8')
        env = env_fn()
        try:
            model = SimpleModel(env.action_space.low.shape)
        finally:
            env.close()
        batched_env = batched_gym_env([env_fn], sync=True)
        try:
            ep_roller = EpisodeRoller(batched_env, model, min_episodes=5, min_steps=7)
            first = ep_roller.rollouts()
            for _ in range(3):
                _compare_rollout_batch(self, first, ep_roller.rollouts())
        finally:
            batched_env.close()

    def _test_basic_equivalence_case(self, stateful, state_tuple,
                                     **roller_kwargs):
        """
        Test BasicRoller equivalence for a single env in a
        specific case.
        """
        env_fn = lambda: SimpleEnv(3, (4, 5), 'uint8')
        env = env_fn()
        model = SimpleModel(env.action_space.low.shape,
                            stateful=stateful,
                            state_tuple=state_tuple)
        basic_roller = BasicRoller(env, model, **roller_kwargs)
        expected = basic_roller.rollouts()

        batched_env = batched_gym_env([env_fn], sync=True)
        ep_roller = EpisodeRoller(batched_env, model, **roller_kwargs)
        actual = ep_roller.rollouts()
        _compare_rollout_batch(self, actual, expected)

    def _test_batch_equivalence_case(self, stateful, state_tuple,
                                     **roller_kwargs):
        """
        Test BasicRoller equivalence when using a batch of
        environments.
        """
        env_fn = lambda: SimpleEnv(3, (4, 5), 'uint8')
        model = SimpleModel((4, 5),
                            stateful=stateful,
                            state_tuple=state_tuple)

        batched_env = batched_gym_env([env_fn]*21, num_sub_batches=7, sync=True)
        ep_roller = EpisodeRoller(batched_env, model, **roller_kwargs)
        actual = ep_roller.rollouts()

        total_steps = sum([r.num_steps for r in actual])
        self.assertTrue(len(actual) >= ep_roller.min_episodes)
        self.assertTrue(total_steps >= ep_roller.min_steps)

        if 'min_steps' not in roller_kwargs:
            num_eps = ep_roller.min_episodes + batched_env.num_envs - 1
            self.assertTrue(len(actual) == num_eps)

        basic_roller = BasicRoller(env_fn(), model, min_episodes=len(actual))
        expected = basic_roller.rollouts()

        _compare_rollout_batch(self, actual, expected)

class TestThreadedRoller(unittest.TestCase):
    """
    Tests for ThreadedRoller.
    """
    def test_truncated_equivalence(self):
        """
        Test that ThreadedRoller produces the same set of
        rollouts as TruncatedRoller.
        """
        env_fns = [lambda seed=x: SimpleEnv(seed, (5, 3), 'uint8') for x in range(15)]
        model = SimpleModel((5, 3), stateful=True, state_tuple=True)

        batched_env = batched_gym_env(env_fns, num_sub_batches=3, sync=True)
        trunc_roller = TruncatedRoller(batched_env, model, 17)

        obs_space = env_fns[0]().observation_space
        async_envs = [AsyncGymEnv(fn, obs_space) for fn in env_fns]
        thread_roller = ThreadedRoller(async_envs, model, 17)

        try:
            for _ in range(3):
                actual = thread_roller.rollouts()
                expected = trunc_roller.rollouts()
                _compare_rollout_batch(self, actual, expected, ordered=False)
        finally:
            for env in async_envs:
                env.close()
            thread_roller.close()

def _compare_rollout_batch(test, rs1, rs2, ordered=True):
    """
    Assert that batches of rollouts are the same.
    """
    test.assertEqual(len(rs1), len(rs2))
    if ordered:
        for rollout1, rollout2 in zip(rs1, rs2):
            test.assertEqual(_rollout_hash(rollout1), _rollout_hash(rollout2))
    else:
        hashes1 = [_rollout_hash(r) for r in rs1]
        hashes2 = [_rollout_hash(r) for r in rs2]
        for hash1 in hashes1:
            test.assertTrue(hash1 in hashes2)
            hashes2.remove(hash1)

def _rollout_hash(rollout):
    """
    Generate a string that uniquely identifies a rollout.
    """
    # Prevent ellipsis.
    np.set_printoptions(threshold=1e6)

    res = ''
    res += str(rollout.trunc_start)
    res += str(rollout.trunc_end)
    res += str(rollout.prev_steps)
    res += '|'
    res += str(rollout.prev_reward)
    res += str(rollout.observations)
    res += str(rollout.rewards)
    res += str(rollout.infos)
    for out in rollout.model_outs:
        res += 'output'
        for key in sorted(out.keys()):
            res += 'kv'
            res += key
            res += str(out[key])
    return res

def _artificial_truncation(rollouts, rollout_idx, timestep_idx):
    """
    Split up the rollouts into two batches by artificially
    truncating (and splitting) the given rollout before
    the given timestep.

    Returns (left_rollouts, right_rollouts)
    """
    to_split = rollouts[rollout_idx]
    left = Rollout(observations=to_split.observations[:timestep_idx+1],
                   model_outs=to_split.model_outs[:timestep_idx+1],
                   rewards=to_split.rewards[:timestep_idx],
                   start_state=to_split.start_state,
                   infos=to_split.infos[:timestep_idx])
    right = Rollout(observations=to_split.observations[timestep_idx:],
                    model_outs=to_split.model_outs[timestep_idx:],
                    rewards=to_split.rewards[timestep_idx:],
                    start_state=to_split.model_outs[timestep_idx-1]['states'],
                    prev_steps=timestep_idx,
                    prev_reward=left.total_reward,
                    infos=to_split.infos[timestep_idx:])
    return rollouts[:rollout_idx]+[left], [right]+rollouts[rollout_idx+1:]

if __name__ == '__main__':
    unittest.main()
