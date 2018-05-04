"""
Use DQN to train a model on the CartPole-v0 environment in
OpenAI Gym.
"""

from anyrl.algos import DQN
from anyrl.models import MLPQNetwork, EpsGreedyQNetwork
from anyrl.rollouts import BasicPlayer, UniformReplayBuffer
from anyrl.spaces import gym_space_vectorizer
import gym
import tensorflow as tf

BUFFER_SIZE = 1024
MIN_BUFFER_SIZE = 256
STEPS_PER_UPDATE = 1
ITERS_PER_LOG = 200
BATCH_SIZE = 64
EPSILON = 0.1
LEARNING_RATE = 0.001


def main():
    """
    Entry-point for the program.
    """
    env = gym.make('CartPole-v0')

    with tf.Session() as sess:
        def make_net(name):
            return MLPQNetwork(sess,
                               env.action_space.n,
                               gym_space_vectorizer(env.observation_space),
                               name,
                               layer_sizes=[32])
        dqn = DQN(make_net('online'), make_net('target'))
        player = BasicPlayer(env, EpsGreedyQNetwork(dqn.online_net, EPSILON),
                             batch_size=STEPS_PER_UPDATE)
        optimize = dqn.optimize(learning_rate=LEARNING_RATE)

        sess.run(tf.global_variables_initializer())

        dqn.train(num_steps=30000,
                  player=player,
                  replay_buffer=UniformReplayBuffer(BUFFER_SIZE),
                  optimize_op=optimize,
                  target_interval=200,
                  batch_size=64,
                  min_buffer_size=200,
                  handle_ep=lambda _, rew: print('got reward: ' + str(rew)))

    env.close()


if __name__ == '__main__':
    main()
