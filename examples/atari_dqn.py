"""
Use DQN to train a model on Atari environments.

For example, train on Pong like:

    $ python atari_dqn.py Pong

"""

import argparse
from functools import partial

from anyrl.algos import DQN
from anyrl.envs import batched_gym_env
from anyrl.envs.wrappers import DownsampleEnv, FrameStackEnv, GrayscaleEnv
from anyrl.models import NatureQNetwork, EpsGreedyQNetwork
from anyrl.rollouts import BatchedPlayer, UniformReplayBuffer
from anyrl.spaces import gym_space_vectorizer
import gym
import tensorflow as tf

def main():
    """
    Entry-point for the program.
    """
    args = _parse_args()
    env = batched_gym_env([partial(make_single_env, args.game)] * args.workers)
    with tf.Session() as sess:
        make_net = lambda name: NatureQNetwork(
            sess, env.action_space.n, gym_space_vectorizer(env.observation_space), name,
            dueling=True)
        dqn = DQN(make_net('online'), make_net('target'))
        replay_buffer = UniformReplayBuffer(args.buffer_size)
        player = BatchedPlayer(env, EpsGreedyQNetwork(dqn.online_net, args.epsilon))

        optimize_op = dqn.optimize(learning_rate=args.lr)
        update_target_op = dqn.update_target()

        sess.run(tf.global_variables_initializer())
        sess.run(update_target_op)

        rewards = []
        num_steps = 0
        while True:
            for _ in range(args.target_interval):
                transitions = player.play()
                for trans in transitions:
                    if trans['is_last']:
                        rewards.append(trans['total_reward'])
                    num_steps += 1
                    replay_buffer.add_sample(trans)
                if replay_buffer.size > args.min_buffer_size:
                    batch = replay_buffer.sample(args.batch_size)
                    sess.run(optimize_op, feed_dict=dqn.feed_dict(batch))
            if rewards:
                print('%d steps: mean=%f' % (num_steps, sum(rewards[-10:]) / len(rewards[-10:])))
            sess.run(update_target_op)

def make_single_env(game):
    """Make a preprocessed gym.Env."""
    env = gym.make(game + '-v0')
    return FrameStackEnv(GrayscaleEnv(DownsampleEnv(env, 2)), num_images=4)

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', help='Adam learning rate', type=float, default=6.25e-5)
    parser.add_argument('--min-buffer-size', help='replay buffer size before training',
                        type=int, default=2000)
    parser.add_argument('--buffer-size', help='replay buffer size', type=int, default=50000)
    parser.add_argument('--workers', help='number of parallel envs', type=int, default=8)
    parser.add_argument('--target-interval', help='training iters per log', type=int, default=512)
    parser.add_argument('--batch-size', help='SGD batch size', type=int, default=32)
    parser.add_argument('--epsilon', help='initial epsilon', type=float, default=0.1)
    parser.add_argument('game', help='game name', default='Pong')
    return parser.parse_args()

if __name__ == '__main__':
    main()
