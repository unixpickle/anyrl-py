"""
Use distributed PPO to train an agent on CartPole-v0.

Run this with:

    $ mpirun -n 4 cartpole_mpi.py

"""

from anyrl.algos import PPO
from anyrl.algos.mpi import MPIOptimizer, mpi_ppo
from anyrl.models import MLP
from anyrl.rollouts import BasicRoller, mean_total_reward
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer
import gym
from mpi4py import MPI
import tensorflow as tf

def run_ppo():
    """
    Run a training worker.
    """
    env = gym.make('CartPole-v0')
    action_dist = gym_space_distribution(env.action_space)
    obs_vectorizer = gym_space_vectorizer(env.observation_space)

    with tf.Session() as sess:
        model = MLP(sess, action_dist, obs_vectorizer, layer_sizes=[32])

        # Deal with CartPole-v0 reward scale.
        model.scale_outputs(20)

        roller = BasicRoller(env, model, min_episodes=30)
        ppo = PPO(model)
        optimizer = MPIOptimizer(tf.train.AdamOptimizer(learning_rate=1e-3),
                                 -ppo.objective)

        sess.run(tf.global_variables_initializer())
        for i in range(50):
            rollouts = roller.rollouts()
            # pylint: disable=E1101
            print('batch %d: rank=%d mean=%f' % (i, MPI.COMM_WORLD.Get_rank(),
                                                 mean_total_reward(rollouts)))
            mpi_ppo(ppo, optimizer, rollouts, log_fn=print)

if __name__ == '__main__':
    run_ppo()
