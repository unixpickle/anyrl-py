"""
Tests for TensorFlow actor-critic models.
"""

# pylint: disable=E0611
# pylint: disable=E1129

import unittest

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell

from anyrl.models import MLP, RNNCellAC
from anyrl.rollouts import BasicRoller
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer
from anyrl.tests import TupleCartPole

class ModelTester:
    """
    Test a TFActorCritic model.
    """
    def __init__(self, test_case, model_maker, add_noise=True):
        self.test_case = test_case
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            self.model = model_maker(self.session)
            self.session.run(tf.global_variables_initializer())
            if add_noise:
                for variable in tf.trainable_variables():
                    noise = 0.01 * tf.random_normal(tf.shape(variable))
                    self.session.run(tf.assign(variable, variable + noise))
            self.batch_outputs = self.model.batch_outputs()

    def test_all(self):
        """
        Run all tests.
        """
        with self.session:
            self.test_batches()
            self.test_batched_step()

    def test_batches(self):
        """
        Test that batches() gives consistent and correctly
        structured output.
        """
        with self.graph.as_default():
            for batch_size in [None, 10]:
                for trunc_start in [False, True]:
                    self._test_batches_consistency(batch_size, trunc_start)

    def test_batched_step(self):
        """
        Make sure that the outputs of batched steps are
        the correct shape.
        """
        with self.graph.as_default():
            env = TupleCartPole()
            try:
                obs_space = env.observation_space
            finally:
                env.close()
            batch_size = 7
            in_obses = [obs_space.sample() for _ in range(batch_size)]
            in_states = self.model.start_state(batch_size)
            outs = self.model.step(in_obses, in_states)
            self.test_case.assertEqual(len(outs['actions']), batch_size)
            self.test_case.assertEqual(_state_shape_uid(outs['states']),
                                       _state_shape_uid(in_states))
            if 'action_params' in outs:
                param_shape = self.model.action_dist.param_shape
                self.test_case.assertEqual(np.array(outs['action_params']).shape,
                                           (batch_size,)+param_shape)
            if 'values' in outs:
                self.test_case.assertEqual(np.array(outs['values']).shape,
                                           (batch_size,))

    def _test_batches_consistency(self, batch_size, trunc_start):
        """
        Make sure that batches() produces the same outputs
        that we got with step().
        """
        env = TupleCartPole()
        try:
            roller = BasicRoller(env, self.model, min_episodes=7)
            rollouts = roller.rollouts()
            if trunc_start:
                rollouts = self._truncate_first(rollouts)
            num_batches = 10
            for batch in self.model.batches(rollouts, batch_size=batch_size):
                num_batches -= 1
                if num_batches == 0:
                    break
                self._test_batch(rollouts, batch)
        finally:
            env.close()

    def _test_batch(self, rollouts, batch):
        """
        Test that the batch is consistent with the
        rollouts and with its own mask.
        """
        feed_dict = batch['feed_dict']
        actor_outs, critic_outs, mask = self.session.run(self.batch_outputs,
                                                         feed_dict)
        for i, (action, value) in enumerate(zip(actor_outs, critic_outs)):
            rollout_idx = batch['rollout_idxs'][i]
            timestep_idx = batch['timestep_idxs'][i]
            self.test_case.assertTrue(mask[i] == 0 or mask[i] == 1)
            if mask[i] == 0:
                self.test_case.assertEqual(rollout_idx, 0)
                self.test_case.assertEqual(timestep_idx, 0)
                continue
            self.test_case.assertEqual(mask[i], 1)
            model_out = rollouts[rollout_idx].model_outs[timestep_idx]
            step_action = np.array(model_out['action_params'][0])
            step_value = model_out['values'][0]
            self.test_case.assertEqual(step_action.shape, np.array(action).shape)
            self.test_case.assertEqual(step_action.shape,
                                       self.model.action_dist.param_shape)
            self.test_case.assertTrue(np.amax(step_action - np.array(action)) < 1e-4)
            self.test_case.assertTrue(abs(value - step_value) < 1e-4)

    def _truncate_first(self, rollouts):
        """
        Remove the first timestep on a random subset of
        rollouts.
        """
        do_trunc = np.random.choice(len(rollouts), len(rollouts)//2)
        for idx in do_trunc:
            # The second state isn't stored anywhere.
            rollout = rollouts[idx]
            first_obs = rollout.observations[0]
            outs = self.model.step([first_obs], rollout.start_state)

            rollout.start_state = outs['states']
            rollout.prev_steps = 1
            rollout.rewards = rollout.rewards[1:]
            rollout.model_outs = rollout.model_outs[1:]
            rollout.observations = rollout.observations[1:]
            rollout.infos = rollout.infos[1:]
        return rollouts

def _state_shape_uid(states):
    """
    Get an object that uniquely identifies the shape of
    the model state batch.
    """
    if states is None:
        return 'None'
    elif isinstance(states, tuple):
        return [_state_shape_uid(s) for s in states]
    return np.array(states).shape

class ACTest(unittest.TestCase):
    """
    Test actor-critic models.
    """
    def __init__(self, *args, **kwargs):
        super(ACTest, self).__init__(*args, **kwargs)
        self.session = tf.Session()
        env = TupleCartPole()
        try:
            action_space = env.action_space
            observation_space = env.observation_space
        finally:
            env.close()
        self.action_dist = gym_space_distribution(action_space)
        self.obs_vectorizer = gym_space_vectorizer(observation_space)

    def test_mlp(self):
        """
        Test the MLP model.
        """
        maker = lambda sess: MLP(sess, self.action_dist, self.obs_vectorizer,
                                 layer_sizes=[32])
        ModelTester(self, maker).test_all()

    def test_lstm(self):
        """
        Test an LSTM model.
        """
        maker = lambda sess: RNNCellAC(sess,
                                       self.action_dist,
                                       self.obs_vectorizer,
                                       make_cell=lambda: LSTMCell(32))
        ModelTester(self, maker).test_all()

    def test_multi_rnn(self):
        """
        Test a stacked LSTM with nested tuple state.
        """
        make_cell = lambda: MultiRNNCell([LSTMCell(16), LSTMCell(32)])
        maker = lambda sess: RNNCellAC(sess,
                                       self.action_dist,
                                       self.obs_vectorizer,
                                       make_cell=make_cell)
        ModelTester(self, maker).test_all()

if __name__ == '__main__':
    unittest.main()
