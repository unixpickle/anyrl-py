"""
Helper routines for training algorithms.
"""

def select_from_batch(advs, batch):
    """
    Take a rollout-shaped list of lists and select the
    indices from the mini-batch.
    """
    indices = zip(batch['rollout_idxs'], batch['timestep_idxs'])
    return [advs[x][y] for x, y in indices]

def select_acts_from_batch(rollouts, batch):
    """
    Select the actions corresponding to the indices from
    the mini-batch.
    """
    acts = [[m['action'] for m in r.model_outs] for r in rollouts]
    return select_from_batch(acts, batch)
