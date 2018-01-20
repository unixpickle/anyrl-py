"""
Test various Player implementations.
"""

import unittest

import numpy as np

from anyrl.rollouts import BasicPlayer, NStepPlayer
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
            for key in ['obs', 'action']:
                self.assertTrue(np.allclose(raw[key], multi[key]))
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

def _transitions_equal(trans1, trans2):
    for key in ['episode_step', 'episode_id', 'total_reward', 'is_last', 'rewards']:
        if trans1[key] != trans2[key]:
            return False
    for key in ['obs', 'action', 'new_obs']:
        if trans1[key] is None:
            if trans2[key] is not None:
                return False
        else:
            if not np.allclose(trans1[key], trans2[key]):
                return False
    return True

if __name__ == '__main__':
    unittest.main()
