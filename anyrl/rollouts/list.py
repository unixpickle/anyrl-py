"""
Manipulate lists of rollouts.
"""

def mean_total_reward(rollouts):
    """
    Get the mean of the total rewards.
    """
    return sum([r.total_reward for r in rollouts]) / len(rollouts)
