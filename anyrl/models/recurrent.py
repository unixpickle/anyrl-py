"""
Statefull neural network models.
"""

import numpy as np
import tensorflow as tf

from .base import TFActorCritic
from .util import mini_batches

# pylint: disable=R0902
class RecurrentAC(TFActorCritic):
    """
    A base class for any stateful actor-critic model.
    """
    def __init__(self, session, action_dist):
        """
        Construct a recurrent model.
        """
        super(RecurrentAC, self).__init__(session, action_dist)

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
        # If a value in _is_init_state_placeholder is 1,
        # the corresponding entry here is ignored.
        self._first_state_placeholders = None

        # Set this to a placeholder for a list of 0's and
        # 1's indicating whether a start state comes from
        # the initial state variable(s).
        self._is_init_state_placeholder = None

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
        feed_dict = {
            self._obs_seq_placeholder: observations,
            self._is_init_state_placeholder: [0] * len(observations),
            self._mask_placeholder: [[1] * len(observations)]
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
        actor_shape = tf.concat([out_count, tf.shape(self._actor_out_seq)[2:]], axis=0)
        critic_shape = tf.concat([out_count, tf.shape(self._critic_out_seq)[2:]], axis=0)
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
                empty_obs = np.array(np.array(obs_seq[0]).shape)
                obs_seqs.append(_pad(obs_seq, max_len, value=empty_obs))
                if rollout.trunc_start:
                    is_inits.append(1)
                else:
                    is_inits.append(0)
                masks.append(_pad([1]*rollout.num_steps, max_len))
                rollout_idxs.extend(_pad([rollout_idx]*rollout.num_steps, max_len))
                timestep_idxs.extend(_pad(list(range(rollout.num_steps)), max_len))
            feed_dict = {
                self._obs_seq_placeholder: obs_seqs,
                self._is_init_state_placeholder: is_inits,
                self._mask_placeholder: masks
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
                first_states = []
                for rollout in rollouts:
                    first_states.append(rollout.start_state[i])
                feed_dict[placeholder] = first_states
        else:
            first_states = []
            for rollout in rollouts:
                first_states.append(rollout.start_state)
            feed_dict[self._first_state_placeholders] = first_states

def _pad(unpadded, length, value=0):
    """
    Pad the list with the given value.
    """
    return unpadded + [value] * (length - len(unpadded))
