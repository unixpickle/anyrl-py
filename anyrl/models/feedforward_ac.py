"""
Stateless neural network models.
"""

from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected  # pylint: disable=E0611

from .base import TFActorCritic
from .util import mini_batches, nature_cnn, product, simple_mlp

# pylint: disable=E1129


class FeedforwardAC(TFActorCritic):
    """
    A base class for any feed-forward actor-critic model.

    Subclasses should set several attributes on init:
      obs_ph: placeholder for observation batch
      actor_out: actor output batch
      critic_out: critic output batch. Should be of shape
        (None,).
    """
    # Set these in your constructor.
    obs_ph = None
    actor_out = None
    critic_out = None

    def scale_outputs(self, scale):
        """
        Scale the network outputs by the given amount.

        This may be called right after initializing the
        model to help deal with different reward scales.
        """
        self.critic_out *= scale
        self.actor_out *= scale

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        feed_dict = {
            self.obs_ph: self.obs_vectorizer.to_vecs(observations)
        }
        act, val = self.session.run((self.actor_out, self.critic_out), feed_dict)
        return {
            'action_params': act,
            'actions': self.action_dist.sample(act),
            'states': None,
            'values': val
        }

    def batch_outputs(self):
        return self.actor_out, self.critic_out

    def batches(self, rollouts, batch_size=None):
        obses, rollout_idxs, timestep_idxs = _frames_from_rollouts(rollouts)
        for mini_indices in mini_batches([1]*len(obses), batch_size):
            sub_obses = [obses[i] for i in mini_indices]
            yield {
                'rollout_idxs': np.take(rollout_idxs, mini_indices),
                'timestep_idxs': np.take(timestep_idxs, mini_indices),
                'feed_dict': {
                    self.obs_ph: self.obs_vectorizer.to_vecs(sub_obses)
                }
            }


class HeadFeedforwardAC(FeedforwardAC):
    """
    A base class for feedforward models that feed inputs
    into a base network which is then fed into actor and
    critic heads.
    """

    def __init__(self,
                 sess,
                 action_dist,
                 obs_vectorizer,
                 obs_batch,
                 actor_init,
                 critic_init):
        super(HeadFeedforwardAC, self).__init__(sess, action_dist, obs_vectorizer)
        with tf.variable_scope('base'):
            self.base_out = self.base(obs_batch)
        with tf.variable_scope('actor'):
            self.actor_out = self.actor(self.base_out, actor_init)
        with tf.variable_scope('critic'):
            self.critic_out = self.critic(self.base_out, critic_init)

    @abstractmethod
    def base(self, obs_batch):
        """
        Apply the shared part of the model.

        Returns:
          A batch that is fed into actor() and critic().
        """
        pass

    def actor(self, base, initializer):
        """
        Turn the output from base() into action params.
        """
        out_size = product(self.action_dist.param_shape)
        actor_out = fully_connected(base, out_size,
                                    activation_fn=None,
                                    weights_initializer=initializer)
        batch = tf.shape(actor_out)[0]
        return tf.reshape(actor_out, (batch,) + self.action_dist.param_shape)

    def critic(self, base, initializer):
        """
        Turn the output from base() into values.
        """
        critic_out = fully_connected(base, 1,
                                     activation_fn=None,
                                     weights_initializer=initializer)
        return tf.reshape(critic_out, (tf.shape(critic_out)[0],))


class MLP(HeadFeedforwardAC):
    """
    A multi-layer perceptron actor-critic model.
    """

    def __init__(self,
                 session,
                 action_dist,
                 obs_vectorizer,
                 layer_sizes,
                 activation=tf.nn.relu,
                 actor_init=tf.zeros_initializer(),
                 critic_init=tf.zeros_initializer()):
        """
        Create an MLP model.

        Args:
          session: TF session.
          action_dist: an action Distribution.
          obs_vectorizer: an observation SpaceVectorizer.
          layer_sizes: list of hidden layer sizes.
          activation: the activation function.
          actor_init: initializer for the actor head.
          critic_init: initializer for the critic head.
        """
        in_batch_shape = (None,) + obs_vectorizer.out_shape
        self.obs_ph = tf.placeholder(tf.float32, shape=in_batch_shape)
        self.layer_sizes = layer_sizes
        self.activation = activation
        super(MLP, self).__init__(session, action_dist, obs_vectorizer,
                                  self.obs_ph, actor_init, critic_init)

    def base(self, obs_batch):
        return simple_mlp(obs_batch, self.layer_sizes, self.activation)


class CNN(HeadFeedforwardAC):
    """
    A convolutional actor-critic model.

    Based on:
    https://github.com/openai/baselines/blob/699919f1cf2527b184f4445a3758a773f333a1ba/baselines/a2c/policies.py#L91
    """

    def __init__(self,
                 session,
                 action_dist,
                 obs_vectorizer,
                 input_scale=1/0xff,
                 input_dtype=tf.uint8,
                 actor_init=tf.zeros_initializer(),
                 critic_init=tf.zeros_initializer()):
        """
        Create a CNN model.

        Args:
          session: TF session.
          action_dist: an action Distribution.
          obs_vectorizer: an observation SpaceVectorizer.
          input_scale: factor to make inputs well-scaled.
          input_dtype: data-type for input placeholder.
          actor_init: initializer for the actor head.
          critic_init: initializer for the critic head.
        """
        in_batch_shape = (None,) + obs_vectorizer.out_shape
        self.obs_ph = tf.placeholder(input_dtype, shape=in_batch_shape)
        obs_batch = tf.cast(self.obs_ph, tf.float32) * input_scale
        super(CNN, self).__init__(session, action_dist, obs_vectorizer,
                                  obs_batch, actor_init, critic_init)

    def base(self, obs_batch):
        """
        Apply the shared part of the model.
        Return a batch that is fed into actor() and
        critic().
        """
        return nature_cnn(obs_batch)


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
