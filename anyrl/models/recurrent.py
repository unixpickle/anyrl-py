"""
Stateful neural network models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected # pylint: disable=E0611

from .base import TFActorCritic
from .util import mini_batches, product

# pylint: disable=E1129

# pylint: disable=R0902
class RecurrentAC(TFActorCritic):
    """
    A base class for any stateful actor-critic model.
    """
    def __init__(self, session, action_dist, obs_vectorizer):
        """
        Construct a recurrent model.
        """
        super(RecurrentAC, self).__init__(session, action_dist, obs_vectorizer)

        self._seq_lens = tf.placeholder(tf.int32, shape=(None,))
        self._is_init_state = tf.placeholder(tf.bool, shape=(None,))

        # Set this to a variable or a tuple of variables
        # for your model's initial state.
        self._init_state_vars = None

        # Set this to a placeholder for a batch of
        # observation sequences.
        self._obs_seq_placeholder = None

        # Set this to a placeholder or a tuple of
        # placeholders for the first state in each
        # sequence.
        #
        # If a value in _is_init_state is True, then the
        # corresponding entry here is ignored.
        self._first_state_placeholders = None

        # Set this to a placeholder for a batch of mask
        # sequences to ignore timesteps in variable-length
        # sequences.
        self._mask_placeholder = None

        # Set these to the model outputs.
        self._actor_out_seq = None
        self._critic_out_seq = None
        self._states_out = None

    def scale_outputs(self, scale):
        """
        Scale the network outputs by the given amount.

        This may be called right after initializing the
        model to help deal with different reward scales.
        """
        self._critic_out_seq *= scale
        self._actor_out_seq *= scale

    @property
    def stateful(self):
        return True

    def start_state(self, batch_size):
        if isinstance(self._init_state_vars, tuple):
            res = []
            # pylint: disable=E1133
            for var in self._init_state_vars:
                var_val = self.session.run(var)
                res.append(np.array([var_val] * batch_size))
            return tuple(res)
        var_val = self.session.run(self._init_state_vars)
        return np.array([var_val] * batch_size)

    def step(self, observations, states):
        vec_obs = self.obs_vectorizer.to_vecs(observations)
        feed_dict = {
            self._seq_lens: [1] * len(observations),
            self._is_init_state: [False] * len(observations),
            self._obs_seq_placeholder: [[x] for x in vec_obs],
            self._mask_placeholder: [[[1]]] * len(observations)
        }

        if isinstance(self._first_state_placeholders, tuple):
            assert isinstance(states, tuple)
            for key, value in zip(self._first_state_placeholders, states):
                feed_dict[key] = value
        else:
            feed_dict[self._first_state_placeholders] = states

        acts, vals, states = self.session.run((self._actor_out_seq,
                                               self._critic_out_seq,
                                               self._states_out),
                                              feed_dict)
        return {
            'action_params': acts[0],
            'actions': self.action_dist.sample(acts[0]),
            'states': states,
            'values': np.array(vals[0]).flatten()
        }

    def batch_outputs(self):
        seq_shape = tf.shape(self._actor_out_seq)
        out_count = seq_shape[0] * seq_shape[1]
        actor_shape = tf.concat([[out_count], tf.shape(self._actor_out_seq)[2:]], axis=0)
        critic_shape = tf.concat([[out_count], tf.shape(self._critic_out_seq)[2:]], axis=0)
        return (tf.reshape(self._actor_out_seq, actor_shape),
                tf.reshape(self._critic_out_seq, critic_shape),
                tf.reshape(self._mask_placeholder, critic_shape))

    # pylint: disable=R0914
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
                obs_seq = rollout.step_observations
                empty_obs = np.zeros(np.array(obs_seq[0]).shape)
                obs_seqs.append(_pad(obs_seq, max_len, value=empty_obs))
                is_inits.append(not rollout.trunc_start)
                masks.append(_pad([[1]]*rollout.num_steps, max_len, value=[0]))
                rollout_idxs.extend(_pad([rollout_idx]*rollout.num_steps, max_len))
                timestep_idxs.extend(_pad(list(range(rollout.num_steps)), max_len))
            vec_obses = [self.obs_vectorizer.to_vecs(s) for s in obs_seqs]
            feed_dict = {
                self._obs_seq_placeholder: vec_obses,
                self._is_init_state: is_inits,
                self._mask_placeholder: masks,
                self._seq_lens: [r.num_steps for r in batch]
            }
            self._add_first_states(feed_dict, batch)
            yield {
                'rollout_idxs': rollout_idxs,
                'timestep_idxs': timestep_idxs,
                'feed_dict': feed_dict
            }

    def _add_first_states(self, feed_dict, rollouts):
        """
        Add first state placeholders for the rollouts.
        """
        if isinstance(self._init_state_vars, tuple):
            for i, placeholder in enumerate(self._first_state_placeholders):
                first_states = [r.start_state[i][0] for r in rollouts]
                feed_dict[placeholder] = first_states
        else:
            first_states = [r.start_state[0] for r in rollouts]
            feed_dict[self._first_state_placeholders] = first_states

    def _create_state_fields(self, dtype, state_size):
        """
        Set self._first_state_placeholders and
        self._init_state_vars.
        """
        if isinstance(state_size, tuple):
            self._first_state_placeholders = ()
            self._init_state_vars = ()
            for sub_shape in state_size:
                placeholder = tf.placeholder(dtype, _add_outer_none(sub_shape))
                variable = tf.Variable(tf.zeros(sub_shape))
                self._first_state_placeholders += (placeholder,)
                self._init_state_vars += (variable,)
        else:
            placeholder = tf.placeholder(dtype, _add_outer_none(state_size))
            variable = tf.Variable(tf.zeros(state_size))
            self._first_state_placeholders = placeholder
            self._init_state_vars = variable

class RNNCellAC(RecurrentAC):
    """
    A recurrent actor-critic that uses a TensorFlow
    RNNCell.
    """
    # pylint: disable=R0913
    # pylint: disable=R0914
    def __init__(self, session, action_dist, obs_vectorizer, make_cell,
                 make_state_tuple=lambda x: x):
        super(RNNCellAC, self).__init__(session, action_dist, obs_vectorizer)
        obs_seq_shape = (None, None) + obs_vectorizer.out_shape
        self._obs_seq_placeholder = tf.placeholder(tf.float32, obs_seq_shape)
        self._mask_placeholder = tf.placeholder(tf.float32, (None, None, 1))

        batch_size = tf.shape(self._obs_seq_placeholder)[0]
        seq_len = tf.shape(self._obs_seq_placeholder)[1]
        obs_size = product(obs_vectorizer.out_shape)
        flat_shape = (batch_size, seq_len, obs_size)
        flattened_seq = tf.reshape(self._obs_seq_placeholder, flat_shape)

        with tf.variable_scope('cell'):
            cell = make_cell()
        with tf.variable_scope('states'):
            self._create_state_fields(tf.float32, cell.state_size)
        init_state = _mix_init_states(self._is_init_state, self._init_state_vars,
                                      self._first_state_placeholders)
        with tf.variable_scope('base'):
            state_tuple = make_state_tuple(init_state)
            base, states = tf.nn.dynamic_rnn(cell, flattened_seq,
                                             sequence_length=self._seq_lens,
                                             initial_state=state_tuple)
        with tf.variable_scope('actor'):
            zeros = tf.zeros_initializer()
            out_size = product(action_dist.param_shape)
            actor_out_seq = fully_connected(base, out_size,
                                            activation_fn=None,
                                            weights_initializer=zeros)
            self._actor_out_seq = tf.reshape(actor_out_seq,
                                             ((batch_size, seq_len) +
                                              action_dist.param_shape))
        with tf.variable_scope('critic'):
            self._critic_out_seq = fully_connected(base, 1, activation_fn=None)
        self._states_out = states

def _pad(unpadded, length, value=0):
    """
    Pad the list with the given value.
    """
    return unpadded + [value] * (length - len(unpadded))

def _mix_init_states(is_init, init_states, start_states):
    """
    Mix initial variables with start state placeholders.
    """
    if isinstance(init_states, tuple):
        assert isinstance(start_states, tuple)
        res = []
        for sub_init, sub_start in zip(init_states, start_states):
            res.append(_mix_init_states(is_init, sub_init, sub_start))
        return tuple(res)
    batch_size = tf.shape(start_states)[0]
    return tf.where(is_init, _batchify(batch_size, init_states), start_states)

def _batchify(batch_size, tensor):
    """
    Repeat a tensor the given number of times in the outer
    dimension.
    """
    batchable = tf.reshape(tensor, tf.concat([[1], tf.shape(tensor)], axis=0))
    ones = tf.ones(tensor.shape.ndims, dtype=tf.int32)
    repeat_count = tf.concat([[batch_size], ones], axis=0)
    return tf.tile(batchable, repeat_count)

def _add_outer_none(shape):
    """
    Add None as an outer dimension for the shape.
    """
    if isinstance(shape, tf.TensorShape):
        return [None] + shape.dims
    return [None, shape]
