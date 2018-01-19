"""
Stateless Q-networks.
"""

from abc import abstractmethod
import random

import numpy as np
import tensorflow as tf

from .base import TFQNetwork
from .util import nature_cnn

# pylint: disable=R0913,R0903

class ScalarQNetwork(TFQNetwork):
    """
    An abstract Q-network that predicts action values as
    scalars (as opposed to distributions).

    Subclasses should override the base() and value_func()
    methods with specific neural network architectures.
    """
    def __init__(self, session, num_actions, obs_vectorizer, name):
        super(ScalarQNetwork, self).__init__(session, num_actions, obs_vectorizer, name)
        old_vars = tf.trainable_variables()
        with tf.variable_scope(name):
            self.step_obs_ph = tf.placeholder(self.input_dtype,
                                              shape=(None,) + obs_vectorizer.out_shape)
            self.step_base_out = self.base(self.step_obs_ph)
            self.step_values = self.value_func(self.step_base_out)
        self.variables = [v for v in tf.trainable_variables() if v not in old_vars]

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        feed = self.step_feed_dict(observations, states)
        values = self.session.run(self.step_values, feed_dict=feed)
        return {
            'actions': np.argmax(values, axis=1),
            'states': None,
            'action_values': values
        }

    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts):
        with tf.variable_scope(self.name, reuse=True):
            max_actions = tf.argmax(self.value_func(self.base(new_obses)), axis=1)
        with tf.variable_scope(target_net.name, reuse=True):
            target_preds = target_net.value_func(target_net.base(new_obses))
            target_preds = tf.where(terminals, tf.zeros_like(target_preds), target_preds)
        targets = rews + discounts * target_preds[tf.range(tf.shape(new_obses)[0]), max_actions]
        with tf.variable_scope(self.name, reuse=True):
            online_preds = self.value_func(self.base(new_obses))
            onlines = online_preds[tf.range(tf.shape(new_obses)[0]), max_actions]
            return tf.square(onlines - tf.stop_gradient(targets))

    @property
    def input_dtype(self):
        return tf.float32

    @abstractmethod
    def base(self, obs_batch):
        """
        Go from a Tensor of observations to a Tensor of
        feature vectors to feed into the final layer.

        Returns:
          A Tensor of shape [batch_size x num_features].
        """
        pass

    @abstractmethod
    def value_func(self, feature_batch):
        """
        Go from a Tensor of feature vectors to a Tensor of
        predicted action values.

        Returns:
          A Tensor of shape [batch_size x num_actions].
        """
        pass

    # pylint: disable=W0613
    def step_feed_dict(self, observations, states):
        """Produce a feed_dict for taking a step."""
        return {self.step_obs_ph: self.obs_vectorizer.to_vecs(observations)}

class NatureQNetwork(ScalarQNetwork):
    """
    A Q-network model based on the Nature DQN paper.
    """
    def __init__(self, session, num_actions, obs_vectorizer, name,
                 input_dtype=tf.uint8, input_scale=1/0xff):
        self._input_dtype = input_dtype
        self.input_scale = input_scale
        super(NatureQNetwork, self).__init__(self, session, num_actions, obs_vectorizer, name)

    @property
    def input_dtype(self):
        return self._input_dtype

    def base(self, obs_batch):
        obs_batch = tf.cast(obs_batch, tf.float32) * self.input_scale
        return nature_cnn(obs_batch)

    def value_func(self, feature_batch):
        return tf.layers.dense(feature_batch, self.num_actions)

class DuelingQNetwork:
    """
    A mixin that uses a dueling Q-network architecture.
    """
    num_actions = 0
    def value_func(self, feature_batch):
        """Dueling value function."""
        values = tf.layers.dense(feature_batch, 1)
        actions = tf.layers.dense(feature_batch, self.num_actions)
        actions -= tf.reduce_mean(actions, axis=1, keep_dims=True)
        return values + actions

class EpsGreedyQNetwork(TFQNetwork):
    """
    A wrapper around a Q-network that adds epsilon-greedy
    exploration to the actions.
    """
    def __init__(self, model, epsilon):
        super(EpsGreedyQNetwork, self).__init__(model.session, model.num_actions,
                                                model.obs_vectorizer, model.name)
        self.model = model
        self.epsilon = epsilon

    @property
    def stateful(self):
        return self.model.stateful

    def start_state(self, batch_size):
        return self.model.start_state(batch_size)

    def step(self, observations, states):
        result = self.model.step(observations, states)
        new_actions = []
        for action in result['actions']:
            if random.random() < self.epsilon:
                new_actions.append(random.randrange(self.num_actions))
            else:
                new_actions.append(action)
        result['actions'] = new_actions
        return result

    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts):
        return self.model.transition_loss(target_net, obses, actions, rews, new_obses, discounts)

    @property
    def input_dtype(self):
        return self.model.input_dtype
