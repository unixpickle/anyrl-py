"""
Manipulate lists of rollouts.
"""


def mean_total_reward(rollouts):
    """
    Get the mean of the total rewards.
    """
    return sum([r.total_reward for r in rollouts]) / len(rollouts)


def mean_finished_reward(rollouts):
    """
    Get the mean of the total rewards in all completed
    episodes.
    """
    complete = [r for r in rollouts if not r.trunc_end]
    if not complete:
        return 0.0
    return sum([r.total_reward for r in complete]) / len(complete)
