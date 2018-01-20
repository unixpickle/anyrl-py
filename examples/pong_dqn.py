"""
Use DQN to train a model on the Pong-v0 environment in
OpenAI Gym.
"""

from anyrl.algos import DQN
from anyrl.envs import batched_gym_env
from anyrl.envs.wrappers import DownsampleEnv, FrameStackEnv, GrayscaleEnv
from anyrl.models import NatureQNetwork, EpsGreedyQNetwork
from anyrl.rollouts import BatchedPlayer, UniformReplayBuffer
from anyrl.spaces import gym_space_vectorizer
import gym
import tensorflow as tf

BUFFER_SIZE = 10000
MIN_BUFFER_SIZE = 1000
STEPS_PER_UPDATE = 1
ITERS_PER_LOG = 4096
BATCH_SIZE = 64
EPSILON = 0.1
LEARNING_RATE = 0.0001
NUM_WORKERS = 8

def main():
    """
    Entry-point for the program.
    """
    env = batched_gym_env([make_single_env] * NUM_WORKERS)
    with tf.Session() as sess:
        make_net = lambda name: NatureQNetwork(
            sess, env.action_space.n, gym_space_vectorizer(env.observation_space), name,
            dueling=True)
        dqn = DQN(make_net('online'), make_net('target'))
        replay_buffer = UniformReplayBuffer(BUFFER_SIZE)
        player = BatchedPlayer(env, EpsGreedyQNetwork(dqn.online_net, EPSILON))

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
                        rewards.append(trans['total_reward'])
                    num_steps += 1
                    replay_buffer.add_sample(trans)
                if replay_buffer.size > MIN_BUFFER_SIZE:
                    batch = replay_buffer.sample(BATCH_SIZE)
                    sess.run(optimize_op, feed_dict=dqn.feed_dict(batch))
            print('%d steps: mean=%f' % (num_steps, sum(rewards[-10:]) / len(rewards[-10:])))
            sess.run(update_target_op)

def make_single_env():
    """Make a preprocessed gym.Env."""
    env = gym.make('Pong-v0')
    return FrameStackEnv(GrayscaleEnv(DownsampleEnv(env, 2)), num_images=4)

if __name__ == '__main__':
    main()
