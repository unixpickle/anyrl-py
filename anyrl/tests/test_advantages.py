"""
Tests for advantage routines.
"""

from anyrl.rollouts import Rollout
from anyrl.algos.advantages import GAE


def test_gae():
    """
    Test generalized advantage estimation.
    """
    rollouts = [
        _dummy_rollout([1, 0.5, 3], [0, -1.5, 2]),
        _dummy_rollout([3, 2, -7], [1, 9])
    ]
    judger = GAE(lam=0.7, discount=0.9)
    actual = judger.advantages(rollouts)
    expected = [[-0.5059, 0.07, -1], [0.241, 0.7]]
    for actual_seq, expected_seq in zip(actual, expected):
        assert len(actual_seq) == len(expected_seq)
        for act, exp in zip(actual_seq, expected_seq):
            assert abs(act-exp) < 1e-5


def _dummy_rollout(pred_vals, rewards):
    """
    Create a rollout with the predicted values and actual
    rewards.
    """
    model_outs = [{'values': [x]} for x in pred_vals]
    return Rollout(observations=[[1]]*len(pred_vals),
                   model_outs=model_outs,
                   rewards=rewards,
                   start_state=None)
