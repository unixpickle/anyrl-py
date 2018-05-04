"""
Helper routines for training algorithms.
"""

import tensorflow as tf


def select_from_batch(advs, batch):
    """
    Take a rollout-shaped list of lists and select the
    indices from the mini-batch.
    """
    indices = zip(batch['rollout_idxs'], batch['timestep_idxs'])
    return [advs[x][y] for x, y in indices]


def select_model_out_from_batch(key, rollouts, batch):
    """
    Select a model_outs key corresponding to the indices
    from the mini-batch.
    """
    vals = [[m[key][0] for m in r.model_outs] for r in rollouts]
    return select_from_batch(vals, batch)


def masked_mean(mask, vals):
    """
    Mask the values and compute the mean of the masked
    elements.
    """
    return masked_sum(mask, vals) / tf.reduce_sum(mask)


def masked_sum(mask, vals):
    """
    Mask the values and compute the sum of the masked
    elements.
    """
    masked = tf.where(tf.equal(mask, 0), tf.zeros_like(vals), vals)
    return tf.reduce_sum(masked)
