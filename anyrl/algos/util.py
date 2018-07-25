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


def symmetric_clipped_value_loss(new_values, old_values, targets, epsilon):
    """
    Compute the component-wise clipped value objective.

    The objective clips values that have changed by more
    than an epsilon in value.

    Args:
      new_values: the model's current value outputs.
      old_values: the original value outputs while
        gathering the rollouts.
      targets: the value targets.
      epsilon: the clipping epsilon.

    Returns:
      A tuple (losses, clip_frac).
    """
    if epsilon is None:
        return tf.square(new_values - targets), tf.constant(0.0, dtype=tf.float32)
    diffs = new_values - old_values
    clipped_diffs = tf.clip_by_value(diffs, -epsilon, epsilon)
    clipped_outputs = old_values + clipped_diffs
    clipped_flags = tf.not_equal(clipped_diffs, diffs)
    clip_frac = tf.reduce_mean(tf.cast(clipped_flags, tf.float32))
    return tf.square(clipped_outputs - targets), clip_frac
