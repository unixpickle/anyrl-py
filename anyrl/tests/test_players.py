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
                for key in ['episode_step', 'episode_id', 'total_reward', 'is_last', 'rewards']:
                    self.assertEqual(trans1[key], trans2[key])
                for key in ['obs', 'action', 'new_obs']:
                    if trans1[key] is None:
                        self.assertIs(trans2[key], None)
                    else:
                        self.assertTrue(np.allclose(trans1[key], trans2[key]))

    def test_multi_step(self):
        """
        Test an NStepPlayer in the multi-step case.
        """
        make_env = lambda: SimpleEnv(14, (1, 2, 3), 'float32')
        make_agent = lambda: SimpleModel((1, 2, 3), stateful=True)
        make_basic = lambda: BasicPlayer(make_env(), make_agent(), batch_size=2)
        player1 = make_basic()
        player2 = NStepPlayer(make_basic(), 3)
        raw_trans = [t for _ in range(20) for t in player1.play()]
        nstep_trans = [t for _ in range(20) for t in player2.play()]
        for trans1, trans2 in zip(raw_trans, nstep_trans):
            for key in ['episode_step', 'episode_id', 'is_last']:
                self.assertEqual(trans1[key], trans2[key])
            for key in ['obs', 'action']:
                self.assertTrue(np.allclose(trans1[key], trans2[key]))
            self.assertEqual(trans1['rewards'], trans2['rewards'][:1])
            self.assertEqual(trans1['total_reward'] + sum(trans2['rewards'][1:]),
                             trans2['total_reward'])

if __name__ == '__main__':
    unittest.main()
