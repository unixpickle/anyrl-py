"""
Popular RL models in the literature.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected # pylint: disable=E0611

from .interfaces import TFActorCritic

class FeedforwardAC(TFActorCritic):
    """
    A base class for any feed-forward actor-critic model.
    """
    def __init__(self, session, action_dist):
        """
        Construct a feed-forward model.

        Arguments:
        action_dist -- action probability distribution.
        base -- take an observation batch and produce a
                Tensor to be fed into actor and critic.
        actor -- take output of base and produce input for
                 action distribution.
        critic -- take output of base and produce value
                  prediction.
        """
        super(FeedforwardAC, self).__init__(action_dist)

        # Set these in your constructor.
        self._obs_placeholder = None
        self._actor_out = None
        self._critic_out = None

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        with self.graph.as_default(): # pylint: disable=E1129
            feed_dict = {self._obs_placeholder: observations}
            act, val = self.session.run((self._actor_out, self._critic_out), feed_dict)
            return {
                'actions': self.action_dist.sample(act),
                'states': None,
                'values': np.array(val)
            }

    def batches(self, rollouts, batch_size=None):
        # TODO: create mini-batches and apply the model.
        raise Exception('not yet implemented')

class MLP(FeedforwardAC):
    """
    A multi-layer perceptron actor-critic model.
    """
    def __init__(self, session, action_dist, obs_vectorizer, layer_sizes,
                 activation=tf.nn.relu):
        """
        Create an MLP model.

        Arguments:
        session -- TF session
        action_dist -- an action Distribution
        obs_vectorizer -- an observation SpaceVectorizer.
        layer_sizes -- list of hidden layer sizes.
        """
        super(MLP, self).__init__(session, action_dist)
        with self.graph.as_default(): # pylint: disable=E1129
            in_batch_shape = (None,) + obs_vectorizer.shape
            self._obs_placeholder = tf.placeholder(tf.float32, shape=in_batch_shape)

            # Iteratively generate hidden layers.
            layer_in_size = _product(obs_vectorizer.shape)
            layer_in = tf.reshape(self._obs_placeholder, (layer_in_size,))
            for out_size in layer_sizes:
                layer_in = fully_connected(layer_in, out_size, activation_fn=activation)
                layer_in_size = out_size

            self._actor_out = fully_connected(layer_in, action_dist.param_size,
                                              activation_fn=None,
                                              weights_initializer=tf.zeros_initializer())
            self._critic_out = fully_connected(layer_in, 1, activation_fn=None)

def _product(vals):
    prod = 1
    for val in vals:
        prod *= val
    return prod
