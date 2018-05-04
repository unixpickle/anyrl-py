"""
The core implementation of deep Q-learning.
"""

import time

import tensorflow as tf


class DQN:
    """
    Train TFQNetwork models using Q-learning.
    """

    def __init__(self, online_net, target_net, discount=0.99):
        """
        Create a Q-learning session.

        Args:
          online_net: the online TFQNetwork.
          target_net: the target TFQNetwork.
          discount: the per-step discount factor.
        """
        self.online_net = online_net
        self.target_net = target_net
        self.discount = discount

        obs_shape = (None,) + online_net.obs_vectorizer.out_shape
        self.obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.actions_ph = tf.placeholder(tf.int32, shape=(None,))
        self.rews_ph = tf.placeholder(tf.float32, shape=(None,))
        self.new_obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.terminals_ph = tf.placeholder(tf.bool, shape=(None,))
        self.discounts_ph = tf.placeholder(tf.float32, shape=(None,))
        self.weights_ph = tf.placeholder(tf.float32, shape=(None,))

        losses = online_net.transition_loss(target_net, self.obses_ph, self.actions_ph,
                                            self.rews_ph, self.new_obses_ph, self.terminals_ph,
                                            self.discounts_ph)
        self.losses = self.weights_ph * losses
        self.loss = tf.reduce_mean(self.losses)

        assigns = []
        for dst, src in zip(target_net.variables, online_net.variables):
            assigns.append(tf.assign(dst, src))
        self.update_target = tf.group(*assigns)

    def feed_dict(self, transitions):
        """
        Generate a feed_dict that feeds the batch of
        transitions to the DQN loss terms.

        Args:
          transition: a sequence of transition dicts, as
            defined in anyrl.rollouts.ReplayBuffer.

        Returns:
          A dict which can be fed to tf.Session.run().
        """
        obs_vect = self.online_net.obs_vectorizer
        res = {
            self.obses_ph: obs_vect.to_vecs([t['obs'] for t in transitions]),
            self.actions_ph: [t['model_outs']['actions'][0] for t in transitions],
            self.rews_ph: [self._discounted_rewards(t['rewards']) for t in transitions],
            self.terminals_ph: [t['new_obs'] is None for t in transitions],
            self.discounts_ph: [(self.discount ** len(t['rewards'])) for t in transitions],
            self.weights_ph: [t['weight'] for t in transitions]
        }
        new_obses = []
        for trans in transitions:
            if trans['new_obs'] is None:
                new_obses.append(trans['obs'])
            else:
                new_obses.append(trans['new_obs'])
        res[self.new_obses_ph] = obs_vect.to_vecs(new_obses)
        return res

    def optimize(self, learning_rate=6.25e-5, epsilon=1.5e-4, **adam_kwargs):
        """
        Create a TF Op that optimizes the objective.

        Args:
          learning_rate: the Adam learning rate.
          epsilon: the Adam epsilon.
        """
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon, **adam_kwargs)
        return optim.minimize(self.loss)

    def train(self,
              num_steps,
              player,
              replay_buffer,
              optimize_op,
              train_interval=1,
              target_interval=8192,
              batch_size=32,
              min_buffer_size=20000,
              tf_schedules=(),
              handle_ep=lambda steps, rew: None,
              timeout=None):
        """
        Run an automated training loop.

        This is meant to provide a convenient way to run a
        standard training loop without any modifications.
        You may get more flexibility by writing your own
        training loop.

        Args:
          num_steps: the number of timesteps to run.
          player: the Player for gathering experience.
          replay_buffer: the ReplayBuffer for experience.
          optimize_op: a TF Op to optimize the model.
          train_interval: timesteps per training step.
          target_interval: number of timesteps between
            target network updates.
          batch_size: the size of experience mini-batches.
          min_buffer_size: minimum replay buffer size
            before training is performed.
          tf_schedules: a sequence of TFSchedules that are
            updated with the number of steps taken.
          handle_ep: called with information about every
            completed episode.
          timeout: if set, this is a number of seconds
            after which the training loop should exit.
        """
        sess = self.online_net.session
        sess.run(self.update_target)
        steps_taken = 0
        next_target_update = target_interval
        next_train_step = train_interval
        start_time = time.time()
        while steps_taken < num_steps:
            if timeout is not None and time.time() - start_time > timeout:
                return
            transitions = player.play()
            for trans in transitions:
                if trans['is_last']:
                    handle_ep(trans['episode_step'] + 1, trans['total_reward'])
                replay_buffer.add_sample(trans)
                steps_taken += 1
                for sched in tf_schedules:
                    sched.add_time(sess, 1)
                if replay_buffer.size >= min_buffer_size and steps_taken >= next_train_step:
                    next_train_step = steps_taken + train_interval
                    batch = replay_buffer.sample(batch_size)
                    _, losses = sess.run((optimize_op, self.losses),
                                         feed_dict=self.feed_dict(batch))
                    replay_buffer.update_weights(batch, losses)
                if steps_taken >= next_target_update:
                    next_target_update = steps_taken + target_interval
                    sess.run(self.update_target)

    def _discounted_rewards(self, rews):
        res = 0
        for i, rew in enumerate(rews):
            res += rew * (self.discount ** i)
        return res
