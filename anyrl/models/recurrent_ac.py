"""
Stateful neural network models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import nest  # pylint: disable=E0611
from tensorflow.contrib.layers import fully_connected  # pylint: disable=E0611

from .base import TFActorCritic
from .util import mini_batches, product, mix_init_states, nature_cnn

# pylint: disable=E1129


class RecurrentAC(TFActorCritic):
    """
    A base class for any stateful actor-critic model.

    Sets the following special attributes:
      seq_lens_ph: placeholder of sequence lengths
      is_init_state_ph: placeholder of booleans indicating
        if sequences start with an initial state.
      mask_ph: placeholder for timestep mask. Is a 2-D
        boolean Tensor, [batch x timesteps].

    Subclasses should set several attributes on init:
      init_state_vars: variable or tuple of variables
        representing the start state vectors.
        This can be set automatically be calling
        create_state_fields().
      obs_ph: placeholder for input sequences
      first_state_ph: placeholder or tuple of placeholders
        for the states at the beginning of the sequences.
      actor_out: actor output sequences
      critic_out: critic output sequences. Should be of
        shape (None, None).
      states_out: resulting RNN states
    """

    def __init__(self, session, action_dist, obs_vectorizer):
        """
        Construct a recurrent model.
        """
        super(RecurrentAC, self).__init__(session, action_dist, obs_vectorizer)

        self.seq_lens_ph = tf.placeholder(tf.int32, shape=(None,))
        self.is_init_state_ph = tf.placeholder(tf.bool, shape=(None,))
        self.mask_ph = tf.placeholder(tf.bool, (None, None))

        # Set this to a variable or a tuple of variables
        # for your model's initial state.
        self.init_state_vars = None

        # Set this to a placeholder for a batch of
        # observation sequences.
        self.obs_ph = None

        # Set this to a placeholder or a tuple of
        # placeholders for the first state in each
        # sequence.
        #
        # If a value in _is_init_state is True, then the
        # corresponding entry here is ignored.
        self.first_state_ph = None

        # Set these to the model outputs.
        self.actor_out = None
        self.critic_out = None
        self.states_out = None

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
        return True

    def start_state(self, batch_size):
        if isinstance(self.init_state_vars, tuple):
            res = []
            # pylint: disable=E1133
            for var in self.init_state_vars:
                var_val = self.session.run(var)
                res.append(np.array([var_val] * batch_size))
            return tuple(res)
        var_val = self.session.run(self.init_state_vars)
        return np.array([var_val] * batch_size)

    def step(self, observations, states):
        vec_obs = self.obs_vectorizer.to_vecs(observations)
        feed_dict = {
            self.seq_lens_ph: [1] * len(observations),
            self.is_init_state_ph: [False] * len(observations),
            self.obs_ph: [[x] for x in vec_obs],
            self.mask_ph: [[1]] * len(observations)
        }

        if isinstance(self.first_state_ph, tuple):
            assert isinstance(states, tuple)
            for key, value in zip(self.first_state_ph, states):
                feed_dict[key] = value
        else:
            feed_dict[self.first_state_ph] = states

        acts, vals, states = self.session.run((self.actor_out,
                                               self.critic_out,
                                               self.states_out),
                                              feed_dict)
        action_params = [a[0] for a in acts]
        return {
            'action_params': action_params,
            'actions': self.action_dist.sample(action_params),
            'states': states,
            'values': np.array(vals).flatten()
        }

    def batch_outputs(self):
        seq_shape = tf.shape(self.actor_out)
        out_count = seq_shape[0] * seq_shape[1]
        actor_shape = (out_count,) + self.action_dist.param_shape
        critic_shape = (out_count,)
        masks = tf.reshape(self.mask_ph, critic_shape)
        return (tf.boolean_mask(tf.reshape(self.actor_out, actor_shape), masks),
                tf.boolean_mask(tf.reshape(self.critic_out, critic_shape), masks))

    def batches(self, rollouts, batch_size=None):
        sizes = [r.num_steps for r in rollouts]
        for rollout_indices in mini_batches(sizes, batch_size):
            batch = [rollouts[i] for i in rollout_indices]
            max_len = max([r.num_steps for r in batch])
            obs_seqs = []
            is_inits = []
            masks = []
            rollout_idxs = []
            timestep_idxs = []
            for rollout_idx, rollout in zip(rollout_indices, batch):
                obs_seq = list(self.obs_vectorizer.to_vecs(rollout.step_observations))
                empty_obs = np.zeros(np.array(obs_seq[0]).shape)
                obs_seqs.append(_pad(obs_seq, max_len, value=empty_obs))
                is_inits.append(not rollout.trunc_start)
                masks.append(_pad([True] * rollout.num_steps, max_len))
                rollout_idxs.extend([rollout_idx] * rollout.num_steps)
                timestep_idxs.extend(range(rollout.num_steps))
            feed_dict = {
                self.obs_ph: obs_seqs,
                self.is_init_state_ph: is_inits,
                self.mask_ph: masks,
                self.seq_lens_ph: [r.num_steps for r in batch]
            }
            self._add_first_states(feed_dict, batch)
            yield {
                'rollout_idxs': rollout_idxs,
                'timestep_idxs': timestep_idxs,
                'feed_dict': feed_dict
            }

    def create_state_fields(self, dtype, state_size):
        """
        Set self.first_state_ph and self.init_state_vars
        with the given TF datatype and the state size.

        The state size may be an integer, a TensorShape,
        or a tuple thereof.
        """
        if isinstance(state_size, tuple):
            self.first_state_ph = ()
            self.init_state_vars = ()
            for sub_shape in state_size:
                placeholder = tf.placeholder(dtype, _add_outer_none(sub_shape))
                variable = tf.Variable(tf.zeros(sub_shape))
                self.first_state_ph += (placeholder,)
                self.init_state_vars += (variable,)
        else:
            placeholder = tf.placeholder(dtype, _add_outer_none(state_size))
            variable = tf.Variable(tf.zeros(state_size))
            self.first_state_ph = placeholder
            self.init_state_vars = variable

    def _add_first_states(self, feed_dict, rollouts):
        """
        Add first state placeholders for the rollouts.
        """
        if isinstance(self.init_state_vars, tuple):
            for i, placeholder in enumerate(self.first_state_ph):
                first_states = [r.start_state[i][0] for r in rollouts]
                feed_dict[placeholder] = first_states
        else:
            first_states = [r.start_state[0] for r in rollouts]
            feed_dict[self.first_state_ph] = first_states


class RNNCellAC(RecurrentAC):
    """
    A recurrent actor-critic that uses a TensorFlow
    RNNCell.

    For RNNCells with nested tuple states, the tuple is
    flattened.
    """

    def __init__(self, session, action_dist, obs_vectorizer, make_cell, input_dtype=tf.float32):
        super(RNNCellAC, self).__init__(session, action_dist, obs_vectorizer)
        obs_seq_shape = (None, None) + obs_vectorizer.out_shape
        self.obs_ph = tf.placeholder(input_dtype, obs_seq_shape)

        with tf.variable_scope('cell_input'):
            cell_input = self.cell_input_sequences()

        with tf.variable_scope('cell'):
            cell = make_cell()
        with tf.variable_scope('states'):
            if isinstance(cell.state_size, tuple):
                self.create_state_fields(tf.float32, tuple(nest.flatten(cell.state_size)))
            else:
                self.create_state_fields(tf.float32, cell.state_size)
        init_state = mix_init_states(self.is_init_state_ph,
                                     self.init_state_vars,
                                     self.first_state_ph)
        with tf.variable_scope('base'):
            if isinstance(init_state, tuple):
                init_state = nest.pack_sequence_as(cell.state_size, init_state)
            self.base_out, states = tf.nn.dynamic_rnn(cell, cell_input,
                                                      sequence_length=self.seq_lens_ph,
                                                      initial_state=init_state)
        if isinstance(states, tuple):
            self.states_out = tuple(nest.flatten(states))
        else:
            self.states_out = states
        with tf.variable_scope('actor'):
            self.actor_out = self.actor(self.base_out)
        with tf.variable_scope('critic'):
            self.critic_out = self.critic(self.base_out)

    def cell_input_sequences(self):
        """
        Transform the observation sequence placeholder
        into a batch of sequences for the RNN cell.

        Default behavior is to flatten the observation
        sequence.
        """
        batch_size = tf.shape(self.obs_ph)[0]
        seq_len = tf.shape(self.obs_ph)[1]
        obs_size = product(self.obs_vectorizer.out_shape)
        flat_shape = (batch_size, seq_len, obs_size)
        return tf.reshape(self.obs_ph, flat_shape)

    def actor(self, cell_outputs):
        """
        Transform the cell outputs into action parameters.

        The default behavior is to use a zero
        fully-connected layer with no activation.
        """
        zeros = tf.zeros_initializer()
        out_size = product(self.action_dist.param_shape)
        actor_out_seq = fully_connected(cell_outputs, out_size,
                                        activation_fn=None,
                                        weights_initializer=zeros)
        seq_shape = tf.shape(actor_out_seq)
        new_shape = ((seq_shape[0], seq_shape[1]) +
                     self.action_dist.param_shape)
        return tf.reshape(actor_out_seq, new_shape)

    def critic(self, cell_outputs):
        """
        Transform the cell outputs into value predictions.

        The default behavior is to use a fully-connected
        layer with no activation.
        """
        zeros = tf.zeros_initializer()
        raw = fully_connected(cell_outputs, 1,
                              activation_fn=None,
                              weights_initializer=zeros)
        shape = tf.shape(raw)
        return tf.reshape(raw, (shape[0], shape[1]))


class CNNRNNCellAC(RNNCellAC):
    """
    An RNNCellAC that feeds inputs through a CNN.
    """

    def __init__(self, session, action_dist, obs_vectorizer, make_cell,
                 input_scale=1/0xff, input_dtype=tf.uint8, cnn_fn=nature_cnn):
        self.cnn_fn = cnn_fn
        self.input_scale = input_scale
        super(CNNRNNCellAC, self).__init__(session, action_dist, obs_vectorizer, make_cell,
                                           input_dtype=input_dtype)

    def cell_input_sequences(self):
        """
        Apply the CNN to get features for the RNN.
        """
        return self._cnn_output_seq(self.obs_ph)

    def _cnn_output_seq(self, obs_seq):
        batch_size = tf.shape(obs_seq)[0]
        seq_len = tf.shape(obs_seq)[1]
        obs_shape = [x.value for x in obs_seq.get_shape()[2:]]

        float_in = tf.cast(obs_seq, tf.float32) * self.input_scale
        flat_batch = tf.reshape(float_in, [batch_size * seq_len] + obs_shape)
        feature_batch = self.cnn_fn(flat_batch)

        seq_shape = (batch_size, seq_len, feature_batch.get_shape()[-1].value)
        return tf.reshape(feature_batch, seq_shape)


def _pad(unpadded, length, value=0):
    """
    Pad the list with the given value.
    """
    return unpadded + [value] * (length - len(unpadded))


def _add_outer_none(shape):
    """
    Add None as an outer dimension for the shape.
    """
    if isinstance(shape, tf.TensorShape):
        return [None] + shape.dims
    return [None, shape]
