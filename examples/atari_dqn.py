"""
Use DQN to train a model on Atari environments.

For example, train on Pong like:

    $ python atari_dqn.py Pong

"""

import argparse
from functools import partial

from anyrl.algos import DQN
from anyrl.envs import batched_gym_env
from anyrl.envs.wrappers import BatchedFrameStack, DownsampleEnv, GrayscaleEnv
from anyrl.models import NatureQNetwork, EpsGreedyQNetwork
from anyrl.rollouts import BatchedPlayer, UniformReplayBuffer
from anyrl.spaces import gym_space_vectorizer
import gym
import tensorflow as tf

REWARD_HISTORY = 10


def main():
    """
    Entry-point for the program.
    """
    args = _parse_args()
    env = batched_gym_env([partial(make_single_env, args.game)] * args.workers)

    # Using BatchedFrameStack with concat=False is more
    # memory efficient than other stacking options.
    env = BatchedFrameStack(env, num_images=4, concat=False)

    with tf.Session() as sess:
        def make_net(name):
            return NatureQNetwork(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  name,
                                  dueling=True)
        dqn = DQN(make_net('online'), make_net('target'))
        player = BatchedPlayer(env, EpsGreedyQNetwork(dqn.online_net, args.epsilon))
        optimize = dqn.optimize(learning_rate=args.lr)

        sess.run(tf.global_variables_initializer())

        reward_hist = []
        total_steps = 0

        def _handle_ep(steps, rew):
            nonlocal total_steps
            total_steps += steps
            reward_hist.append(rew)
            if len(reward_hist) == REWARD_HISTORY:
                print('%d steps: mean=%f' % (total_steps, sum(reward_hist) / len(reward_hist)))
                reward_hist.clear()

        dqn.train(num_steps=int(1e7),
                  player=player,
                  replay_buffer=UniformReplayBuffer(args.buffer_size),
                  optimize_op=optimize,
                  target_interval=args.target_interval,
                  batch_size=args.batch_size,
                  min_buffer_size=args.min_buffer_size,
                  handle_ep=_handle_ep)

    env.close()


def make_single_env(game):
    """Make a preprocessed gym.Env."""
    env = gym.make(game + '-v0')
    return GrayscaleEnv(DownsampleEnv(env, 2))


def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', help='Adam learning rate', type=float, default=6.25e-5)
    parser.add_argument('--min-buffer-size', help='replay buffer size before training',
                        type=int, default=2000)
    parser.add_argument('--buffer-size', help='replay buffer size', type=int, default=50000)
    parser.add_argument('--workers', help='number of parallel envs', type=int, default=8)
    parser.add_argument('--target-interval', help='training iters per log', type=int, default=8192)
    parser.add_argument('--batch-size', help='SGD batch size', type=int, default=32)
    parser.add_argument('--epsilon', help='initial epsilon', type=float, default=0.1)
    parser.add_argument('game', help='game name', default='Pong')
    return parser.parse_args()


if __name__ == '__main__':
    main()
