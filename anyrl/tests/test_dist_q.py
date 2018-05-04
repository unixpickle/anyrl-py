"""
Tests for distributional Q-learning.
"""

import numpy as np
import tensorflow as tf

from anyrl.models.dqn_dist import ActionDist

# pylint: disable=E1129


def test_mean():
    """Test action mean calculations."""
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dist = ActionDist(7, -2.5, 3)
            log_probs = np.log(np.array([[[0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05],
                                          [0.1, 0.1, 0.3, 0.3, 0.05, 0.1, 0]],
                                         [[0, 0.2, 0.3, 0.1, 0.1, 0.05, 0.25],
                                          [0, 0.1, 0.3, 0.2, 0.05, 0.1, 0.2]]]))
            expected = np.sum(np.exp(log_probs) * np.linspace(-2.5, 3, num=7), axis=-1)
            actual = sess.run(dist.mean(log_probs))
            assert actual.shape == expected.shape
            assert np.allclose(actual, expected)


def test_add_rewards():
    """Test distribution shifting and projecting."""
    _test_add(ActionDist(4, -1, 2),
              [[0.2, 0.2, 0.35, 0.25], [0, 1, 0, 0]],
              [-1, 2],
              [1, 1],
              [[0.4, 0.35, 0.25, 0], [0, 0, 0, 1]])
    _test_add(ActionDist(4, -1, 2),
              [[0.2, 0.2, 0.35, 0.25], [0, 1, 0, 0]],
              [0.5, -0.75],
              [1, 1],
              [[0.1, 0.2, 0.275, 0.425], [0.75, 0.25, 0, 0]])
    _test_add(ActionDist(4, -1, 2),
              [[0.2, 0.2, 0.35, 0.25], [0, 0, 1, 0]],
              [1, 0.3],
              [0.5, 0.25],
              [[0, 0.1, 0.475, 0.425], [0, 0.45, 0.55, 0]])


def _test_add(dist, old_probs, rewards, discounts, new_probs):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            result = dist.add_rewards(tf.constant(old_probs, dtype=tf.float64),
                                      tf.constant(rewards, dtype=tf.float64),
                                      tf.constant(discounts, dtype=tf.float64))
            assert result.shape == np.array(new_probs).shape
            assert np.allclose(sess.run(result), new_probs)
            assert not (np.isnan(sess.run(result))).any()
