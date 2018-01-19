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
        make_net = lambda name: MLPQNetwork(
            sess, env.action_space.n, gym_space_vectorizer(env.observation_space), name,
            layer_sizes=[32])
        dqn = DQN(make_net('online'), make_net('target'))
        replay_buffer = UniformReplayBuffer(BUFFER_SIZE)
        player = BasicPlayer(env, EpsGreedyQNetwork(dqn.online_net, EPSILON),
                             batch_size=STEPS_PER_UPDATE)

        optimize_op = dqn.optimize(learning_rate=LEARNING_RATE)
        update_target_op = dqn.update_target()

        sess.run(tf.global_variables_initializer())
        sess.run(update_target_op)

        rewards = []
        num_steps = 0
        while True:
            for _ in range(ITERS_PER_LOG):
                transitions = player.play()
                for trans in transitions:
                    if trans['is_last']:
                        rewards.append(trans['total_rew'])
                    num_steps += 1
                    replay_buffer.add_sample(trans)
                if replay_buffer.size > MIN_BUFFER_SIZE:
                    batch = replay_buffer.sample(BATCH_SIZE)
                    sess.run(optimize_op, feed_dict=dqn.feed_dict(batch))
            print('%d steps: mean=%f' % (num_steps, sum(rewards[-10:]) / len(rewards[-10:])))
            sess.run(update_target_op)

if __name__ == '__main__':
    main()
