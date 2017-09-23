"""
Use A2C to train a model on the CartPole-v0 environment in
OpenAI Gym.
"""

from anyrl.algos import A2C, PPO
from anyrl.models import MLP
from anyrl.rollouts import BasicRoller, mean_total_reward
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer
import gym
import tensorflow as tf

def main():
    """
    Entry-point for the program.
    """
    for algo in ['ppo', 'a2c']:
        run_algorithm(algo)

def run_algorithm(algo_name):
    """
    Run the specified training algorithm.
    """
    env = gym.make('CartPole-v0')
    action_dist = gym_space_distribution(env.action_space)
    obs_vectorizer = gym_space_vectorizer(env.observation_space)

    with tf.Session() as sess:
        model = MLP(sess, action_dist, obs_vectorizer, layer_sizes=[32])

        # Deal with CartPole-v0 reward scale.
        model.scale_outputs(20)

        roller = BasicRoller(env, model, min_episodes=30)
        inner_loop = algorithm_inner_loop(algo_name, model)

        sess.run(tf.global_variables_initializer())
        print('running algorithm:', algo_name)
        for i in range(50):
            rollouts = roller.rollouts()
            print('batch %d: mean=%f' % (i, mean_total_reward(rollouts)))
            inner_loop(rollouts)

def algorithm_inner_loop(name, model):
    """
    Generate a function which runs a round of training on
    a batch of rollouts.
    """
    if name == 'a2c':
        a2c = A2C(model)
        optimizer = a2c.optimize(learning_rate=1e-2, max_grad_norm=None,
                                 rms_decay=0.9)
        return lambda rollouts: model.session.run(optimizer,
                                                  a2c.feed_dict(rollouts))
    elif name == 'ppo':
        ppo = PPO(model)
        optimizer = ppo.optimize(learning_rate=1e-3)
        return lambda rollouts: ppo.run_optimize(optimizer, rollouts)

if __name__ == '__main__':
    main()
