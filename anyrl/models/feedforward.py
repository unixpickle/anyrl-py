"""
Stateless neural network models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected # pylint: disable=E0611

from .base import TFActorCritic

# pylint: disable=E1129

class FeedforwardAC(TFActorCritic):
    """
    A base class for any feed-forward actor-critic model.
    """
    def __init__(self, session, action_dist):
        """
        Construct a feed-forward model.
        """
        super(FeedforwardAC, self).__init__(session, action_dist)

        # Set these in your constructor.
        self._obs_placeholder = None
        self._actor_out = None
        self._critic_out = None

    def scale_outputs(self, scale):
        """
        Scale the network outputs by the given amount.

        This may be called right after initializing the
        model to help deal with different reward scales.
        """
        self._critic_out *= scale
        self._actor_out *= scale

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        feed_dict = {self._obs_placeholder: observations}
        act, val = self.session.run((self._actor_out, self._critic_out), feed_dict)
        return {
            'action_params': act,
            'actions': self.action_dist.sample(act),
            'states': None,
            'values': np.array(val).flatten()
        }

    def batch_outputs(self):
        return self._actor_out, self._critic_out

    def batches(self, rollouts, batch_size=None):
        obses, rollout_idxs, timestep_idxs = _frames_from_rollouts(rollouts)
        while True:
            if batch_size is None or batch_size > len(obses):
                mini_indices = range(len(obses))
            else:
                mini_indices = np.random.choice(len(obses), size=batch_size,
                                                replace=False)
            obses = np.array(np.take(obses, mini_indices, axis=0))
            yield {
                'rollout_idxs': np.take(rollout_idxs, mini_indices),
                'timestep_idxs': np.take(timestep_idxs, mini_indices),
                'feed_dict': {self._obs_placeholder: obses}
            }

class MLP(FeedforwardAC):
    """
    A multi-layer perceptron actor-critic model.
    """
    # pylint: disable=R0913
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

        in_batch_shape = (None,) + obs_vectorizer.shape
        self._obs_placeholder = tf.placeholder(tf.float32, shape=in_batch_shape)

        # Iteratively generate hidden layers.
        layer_in_size = _product(obs_vectorizer.shape)
        vectorized_shape = (tf.shape(self._obs_placeholder)[0], layer_in_size)
        layer_in = tf.reshape(self._obs_placeholder, vectorized_shape)
        for layer_idx, out_size in enumerate(layer_sizes):
            with tf.variable_scope('layer_' + str(layer_idx)):
                layer_in = fully_connected(layer_in, out_size, activation_fn=activation)
            layer_in_size = out_size

        with tf.variable_scope('actor'):
            self._actor_out = fully_connected(layer_in, action_dist.param_size,
                                              activation_fn=None,
                                              weights_initializer=tf.zeros_initializer())

        with tf.variable_scope('critic'):
            self._critic_out = fully_connected(layer_in, 1, activation_fn=None)

def _product(vals):
    prod = 1
    for val in vals:
        prod *= val
    return prod

def _frames_from_rollouts(rollouts):
    """
    Flatten out the rollouts and produce a list of
    observations, rollout indices, and timestep indices.

    Does not include trailing observations for truncated
    rollouts.

    For example, [[obs1, obs2], [obs3, obs4, obs5]] would
    become ([obs1, obs2, ..., obs5], [0, 0, 1, 1, 1],
    [0, 1, 0, 1, 2])
    """
    all_obs = []
    rollout_indices = []
    timestep_indices = []
    for rollout_idx, rollout in enumerate(rollouts):
        for timestep_idx, obs in enumerate(rollout.step_observations):
            all_obs.append(obs)
            rollout_indices.append(rollout_idx)
            timestep_indices.append(timestep_idx)
    return all_obs, rollout_indices, timestep_indices
