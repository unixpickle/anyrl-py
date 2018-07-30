"""
Test for the utilities module.
"""

import os
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

from anyrl.utils.tf_state import load_vars, save_vars


def test_save_restore():
    """
    Test TF state restoration.
    """
    with TemporaryDirectory() as out_dir:
        save_path = os.path.join(out_dir, 'save.pkl')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                v1 = tf.get_variable('test1', shape=(1, 2), dtype=tf.float64)
                v2 = tf.get_variable('test2', shape=(3, 4),
                                     initializer=tf.truncated_normal_initializer())
                sess.run(tf.global_variables_initializer())
                val1, val2 = sess.run((v1, v2))
                save_vars(sess, save_path)
        with tf.Graph().as_default():
            with tf.Session() as sess:
                v2 = tf.get_variable('test2', shape=(3, 4))
                v1 = tf.get_variable('test1', shape=(1, 2), dtype=tf.float64)
                sess.run(tf.global_variables_initializer())
                new_val1, new_val2 = sess.run((v1, v2))
                assert not np.allclose(val1, new_val1)
                assert not np.allclose(val2, new_val2)
                load_vars(sess, save_path, log_fn=None)
                res_val1, res_val2 = sess.run((v1, v2))
                assert np.allclose(val1, res_val1)
                assert np.allclose(val2, res_val2)
