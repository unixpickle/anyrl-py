"""
Saving/restoring TensorFlow state.
"""

import os
import pickle
import tensorflow as tf

from .atomic import atomic_pickle


def load_vars(sess, path, conditional=True, relaxed=False, log_fn=print, var_list=None):
    """
    Load TensorFlow variables from a file.

    Args:
      sess: the TF session for running assigns.
      path: the path to the input file.
      conditional: if set, then this checks if the file
        does not exist.
      relaxed: if True, ignore variables that are in the
        file but not in the graph.
      log_fn: the function to use for log messages.
      var_list: the variables to save. Defaults to all
        trainable variables.

    This is intended to be called once before training.
    It will create new nodes in the graph that cannot be
    easily reused.
    """
    var_list = var_list or tf.trainable_variables()
    log_fn = log_fn or (lambda x: None)
    if conditional:
        if not os.path.exists(path):
            log_fn('State path does not exist: ' + path)
            return
        log_fn('Loading variables from ' + path + ' ...')
    with open(path, 'rb') as in_file:
        obj = pickle.load(in_file)
        if isinstance(obj, list):
            obj = {v.name: val for v, val in zip(var_list, obj)}
    assigns = []
    feed = {}
    for name, value in obj.items():
        matches = [v for v in var_list if v.name == name]
        if not matches:
            if relaxed:
                log_fn('skipping missing variable: ' + name)
                continue
            raise RuntimeError('missing variable ' + name)
        var = matches[0]
        # Use placeholders so that we don't create
        # constants that are never used again.
        ph = tf.placeholder(dtype=var.dtype.base_dtype, shape=var.get_shape())
        assigns.append(tf.assign(var, ph))
        feed[ph] = value
    sess.run(assigns, feed_dict=feed)


def save_vars(sess, path, var_list=None):
    """
    Save the trainable variables to a file.

    Args:
      sess: the session to get values from.
      path: the path to the output file.
      var_list: the variables to save. Defaults to all
        trainable variables.
    """
    var_list = var_list or tf.trainable_variables()
    exported = {x.name: sess.run(x) for x in var_list}
    atomic_pickle(exported, path)
