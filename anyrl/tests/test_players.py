"""
Test various Player implementations.
"""

import unittest

import numpy as np

from anyrl.envs import batched_gym_env
from anyrl.rollouts import BasicPlayer, NStepPlayer, BatchedPlayer
from anyrl.tests.util import SimpleEnv, SimpleModel

class NStepPlayerTest(unittest.TestCase):
    """
    Tests for NStepPlayer.
    """
    def test_one_step(self):
        """
        Test an NStepPlayer in the trivial, 1-step case.
        """
        make_env = lambda: SimpleEnv(15, (1, 2, 3), 'float32')
        make_agent = lambda: SimpleModel((1, 2, 3), stateful=True)
        make_basic = lambda: BasicPlayer(make_env(), make_agent(), batch_size=3)
        player1 = make_basic()
        player2 = NStepPlayer(make_basic(), 1)
        for _ in range(100):
            transes1 = player1.play()
            transes2 = player2.play()
            self.assertEqual(len(transes1), len(transes2))
            for trans1, trans2 in zip(transes1, transes2):
                self.assertTrue(_transitions_equal(trans1, trans2))

    def test_multi_step(self):
        """
        Test an NStepPlayer in the multi-step case.
        """
        make_env = lambda: SimpleEnv(9, (1, 2, 3), 'float32')
        make_agent = lambda: SimpleModel((1, 2, 3), stateful=True)
        make_basic = lambda: BasicPlayer(make_env(), make_agent(), batch_size=1)
        player1 = make_basic()
        player2 = NStepPlayer(make_basic(), 3)
        raw_trans = [t for _ in range(40) for t in player1.play()]
        nstep_trans = [t for _ in range(40) for t in player2.play()]
        for raw, multi in zip(raw_trans, nstep_trans):
            for key in ['episode_step', 'episode_id', 'is_last']:
                self.assertEqual(raw[key], multi[key])
            self.assertTrue(np.allclose(raw['model_outs']['actions'][0],
                                        multi['model_outs']['actions'][0]))
            self.assertTrue(np.allclose(raw['obs'], multi['obs']))
            self.assertEqual(raw['rewards'], multi['rewards'][:1])
            self.assertEqual(raw['total_reward'] + sum(multi['rewards'][1:]),
                             multi['total_reward'])
        for raw, multi in zip(raw_trans[3:], nstep_trans):
            if multi['new_obs'] is not None:
                self.assertTrue(np.allclose(multi['new_obs'], raw['obs']))
            else:
                self.assertNotEqual(multi['episode_id'], raw['episode_id'])

    def test_batch_invariance(self):
        """
        Test that the batch size of the underlying
        Player doesn't affect the NStepPlayer.
        """
        make_env = lambda: SimpleEnv(9, (1, 2, 3), 'float32')
        make_agent = lambda: SimpleModel((1, 2, 3), stateful=True)
        def _gather_transitions(batch_size):
            player = NStepPlayer(BasicPlayer(make_env(), make_agent(), batch_size=batch_size), 3)
            transitions = []
            while len(transitions) < 50:
                transitions.extend(player.play())
            # The NStepPlayer is not required to preserve
            # the order of transitions.
            return sorted(transitions, key=lambda t: (t['episode_id'], t['episode_step']))[:50]
        expected = _gather_transitions(1)
        for batch_size in range(2, 52):
            actual = _gather_transitions(batch_size)
            for trans1, trans2 in zip(expected, actual):
                self.assertTrue(_transitions_equal(trans1, trans2))

class BatchedPlayerTest(unittest.TestCase):
    """
    Tests for BatchedPlayer.
    """
    def test_single_batch(self):
        """
        Test when the batch size is 1.
        """
        make_env = lambda: SimpleEnv(9, (1, 2, 3), 'float32')
        make_agent = lambda: SimpleModel((1, 2, 3), stateful=True)
        basic_player = BasicPlayer(make_env(), make_agent(), 3)
        batched_player = BatchedPlayer(batched_gym_env([make_env]), make_agent(), 3)
        for _ in range(50):
            transes1 = basic_player.play()
            transes2 = batched_player.play()
            self.assertEqual(len(transes1), len(transes2))
            for trans1, trans2 in zip(transes1, transes2):
                self.assertTrue(_transitions_equal(trans1, trans2))

    def test_mixed_batch(self):
        """
        Test a batch with a bunch of different
        environments.
        """
        env_fns = [lambda s=seed: SimpleEnv(s, (1, 2, 3), 'float32')
                   for seed in [3, 3, 3, 3, 3, 3]] #[5, 8, 1, 9, 3, 2]]
        make_agent = lambda: SimpleModel((1, 2, 3), stateful=True)
        for num_sub in [1, 2, 3]:
            batched_player = BatchedPlayer(batched_gym_env(env_fns, num_sub_batches=num_sub),
                                           make_agent(), 3)
            expected_eps = []
            for player in [BasicPlayer(env_fn(), make_agent(), 3) for env_fn in env_fns]:
                transes = [t for _ in range(50) for t in player.play()]
                expected_eps.extend(_separate_episodes(transes))
            actual_transes = [t for _ in range(50) for t in batched_player.play()]
            actual_eps = _separate_episodes(actual_transes)
            self.assertEqual(len(expected_eps), len(actual_eps))
            for episode in expected_eps:
                found = False
                for i, actual in enumerate(actual_eps):
                    if _episodes_equivalent(episode, actual):
                        del actual_eps[i]
                        found = True
                        break
                self.assertTrue(found)

def _separate_episodes(transes):
    res = []
    for ep_id in set([t['episode_id'] for t in transes]):
        res.append([t for t in transes if t['episode_id'] == ep_id])
    return res

def _episodes_equivalent(transes1, transes2):
    if len(transes1) != len(transes2):
        return False
    for trans1, trans2 in zip(transes1, transes2):
        if not _transitions_equal(trans1, trans2, ignore_id=True):
            return False
    return True

def _transitions_equal(trans1, trans2, ignore_id=False):
    for key in ['episode_step', 'total_reward', 'is_last', 'rewards']:
        if trans1[key] != trans2[key] and (key != 'episode_id' or not ignore_id):
            return False
    if trans1['new_obs'] is None:
        if trans2['new_obs'] is not None:
            return False
    else:
        if not np.allclose(trans1['new_obs'], trans2['new_obs']):
            return False
    if (not np.allclose(trans1['model_outs']['actions'][0], trans2['model_outs']['actions'][0]) or
            not _states_equal(trans1['model_outs']['states'], trans2['model_outs']['states'])):
        return False
    if not np.allclose(trans1['obs'], trans2['obs']):
        return False
    return True

def _states_equal(states1, states2):
    if isinstance(states1, tuple):
        if not isinstance(states2, tuple):
            return False
        return all(np.allclose(x, y) for x, y in zip(states1, states2))
    else:
        return np.allclose(states1, states2)

if __name__ == '__main__':
    unittest.main()
